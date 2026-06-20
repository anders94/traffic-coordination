"""Signal phase state machine.

This is the *safety envelope* of the system. The learned (or baseline) controller
only ever asks to leave the current green; it can never shorten clearance, skip the
minimum green, or run past the maximum green. In M1 these constraints live here in
simulation; in a field deployment the identical logic is what the conflict monitor /
hardware enforces (see the plan's field-safety wrapper).

States cycle:  GREEN --(switch & min-green met)--> YELLOW --> ALLRED --> next GREEN
"""

from __future__ import annotations

GREEN, YELLOW, ALLRED = "green", "yellow", "allred"


class Signal:
    def __init__(
        self,
        phases: list[set[str]],
        min_green: float,
        max_green: float,
        yellow: float,
        allred: float,
    ):
        self.phases = phases
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow
        self.allred = allred

        self.phase = 0          # index into self.phases
        self.state = GREEN
        self.elapsed_green = 0.0
        self._timer = 0.0       # time in the current yellow/allred interval

    # --- queries -----------------------------------------------------------
    def is_green(self, approach: str) -> bool:
        return self.state == GREEN and approach in self.phases[self.phase]

    @property
    def can_switch(self) -> bool:
        """Controller is *allowed* to end the green (minimum green satisfied)."""
        return self.state == GREEN and self.elapsed_green >= self.min_green

    @property
    def must_switch(self) -> bool:
        """Maximum green reached — the envelope forces termination."""
        return self.state == GREEN and self.elapsed_green >= self.max_green

    # --- advance -----------------------------------------------------------
    def step(self, dt: float, switch_request: bool) -> None:
        if self.state == GREEN:
            self.elapsed_green += dt
            if (switch_request and self.can_switch) or self.must_switch:
                self.state = YELLOW
                self._timer = 0.0
        elif self.state == YELLOW:
            self._timer += dt
            if self._timer >= self.yellow:
                self.state = ALLRED
                self._timer = 0.0
        elif self.state == ALLRED:
            self._timer += dt
            if self._timer >= self.allred:
                self.phase = (self.phase + 1) % len(self.phases)
                self.state = GREEN
                self.elapsed_green = 0.0
                self._timer = 0.0
