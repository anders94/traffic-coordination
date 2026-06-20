"""Fully-actuated controller (gap-out / max-out), the realistic NEMA-style baseline.

While the current phase is green, extend it as long as a vehicle keeps arriving in
the loop detection zone within the passage time. Terminate on:
  * gap-out  -- no vehicle in the zone for `passage_time` seconds (and min-green met), or
  * max-out  -- maximum green reached (enforced by the signal envelope).
This reproduces the behaviour the plan's verification step checks for.
"""

from __future__ import annotations

from .base import Controller


class Actuated(Controller):
    name = "actuated"

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        sim = env.unwrapped_sim
        sig = sim.signal
        if not sig.can_switch:
            return 0  # still in minimum green (or mid-transition)
        # gap-out: the green approaches have shown no demand in the zone recently
        green_approaches = [a for a in sig.phases[sig.phase]
                            if a in sim.active_approaches]
        gapped_out = all(sim.gap_timer[a] >= sim.passage_time
                         for a in green_approaches)
        return 1 if gapped_out else 0
