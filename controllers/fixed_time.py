"""Fixed-time controller: a fixed green split per phase, cycled forever.

This is the floor baseline — the classic pretimed plan many real intersections still
run. Green durations default to a simple proportional split of an assumed cycle, but
can be set explicitly to a Webster-style plan via `green_s`.
"""

from __future__ import annotations

from .base import Controller


class FixedTime(Controller):
    name = "fixed_time"

    def __init__(self, green_s: list[float] | None = None):
        self.green_s = green_s  # one target green per phase; None -> use cycle/2

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        sig = env.unwrapped_sim.signal
        if self.green_s is not None:
            target = self.green_s[sig.phase % len(self.green_s)]
        else:
            target = sig.max_green * 0.5
        return 1 if sig.elapsed_green >= target else 0
