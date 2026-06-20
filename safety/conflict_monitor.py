"""Independent conflict monitor — the software analogue of the hardware Malfunction
Management Unit (MMU / conflict monitor) required in a NEMA TS-2 signal cabinet.

This is the layer that makes "let it learn" safe to deploy. It sits *downstream* of
whatever controller is running — fixed-time, max-pressure, a neural net, anything — and
independently validates the signal indications that are actually about to be displayed.
It knows nothing about the controller's logic or goals; it only enforces the physical
safety invariants. If it ever sees an unsafe display it *trips*: it forces the
intersection to flashing-red and latches that state until a human resets it. A buggy or
adversarial controller therefore cannot cause a dangerous signal display — the worst it
can do is trip the monitor and degrade the intersection to flashing-red.

Invariants enforced (the safety-critical ones a real MMU checks):
  * No conflicting greens  — two movements that cross may never be green together.
  * Clearance              — a conflicting movement may only go green after the yellow
                             change + red clearance interval has fully elapsed.
  * Minimum green          — a green may not be terminated before its minimum.
"""

from __future__ import annotations


def conflicts_from_phases(phases: list[set[str]]) -> dict[str, set[str]]:
    """Build a conflict map from a phase list: two approaches conflict iff they are
    served by different (non-simultaneous) phases."""
    approaches = sorted({a for p in phases for a in p})
    phase_of = {a: i for i, p in enumerate(phases) for a in p}
    return {a: {b for b in approaches if phase_of[b] != phase_of[a]} for a in approaches}


class ConflictMonitor:
    FLASH = "flash_red"  # the safe latched state

    def __init__(self, conflicts: dict[str, set[str]], min_yellow: float,
                 min_allred: float, min_green: float = 0.0):
        self.conflicts = conflicts
        self.clearance = min_yellow + min_allred
        self.min_green = min_green
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.tripped = False
        self.fault: str | None = None
        self._prev_green: set[str] | None = None  # None until the first observation
        self._green_started: dict[str, float] = {}
        self._green_ended: dict[str, float] = {}

    def _trip(self, reason: str) -> None:
        if not self.tripped:
            self.tripped = True
            self.fault = f"t={self.t:.0f}s: {reason}"

    def check(self, green: set[str], dt: float) -> set[str]:
        """Validate the commanded green set for this step. Returns the set of greens
        that may actually be displayed — the commanded set if safe, or an empty set
        (flashing-red) once the monitor has tripped."""
        if self.tripped:
            self.t += dt
            return set()

        green = set(green)

        # INV1 — no two conflicting movements green at once (always checked)
        for a in green:
            if green & self.conflicts.get(a, set()):
                other = next(iter(green & self.conflicts[a]))
                self._trip(f"conflicting greens displayed: {a} + {other}")
                return set()

        # first observation: accept the greens already on as pre-existing — we can't
        # validate the minimum/clearance of a green whose start we never saw.
        if self._prev_green is None:
            self._prev_green = green
            self.t += dt
            return green

        newly_green = green - self._prev_green
        ended = self._prev_green - green

        # INV2 — a movement may only turn green after conflicts have fully cleared
        for a in newly_green:
            for c in self.conflicts.get(a, set()):
                end = self._green_ended.get(c)
                if end is not None and (self.t - end) < self.clearance - 1e-9:
                    self._trip(f"{a} turned green {self.t - end:.0f}s after conflicting "
                               f"{c} (needs {self.clearance:.0f}s clearance)")
                    return set()

        # INV3 — a green may not end before its minimum (skip greens we never saw start)
        for a in ended:
            start = self._green_started.get(a)
            if start is not None and (self.t - start) < self.min_green - 1e-9:
                self._trip(f"{a} green only {self.t - start:.0f}s "
                           f"(< min {self.min_green:.0f}s)")
                return set()

        for a in newly_green:
            self._green_started[a] = self.t
        for a in ended:
            self._green_ended[a] = self.t
        self._prev_green = green
        self.t += dt
        return green
