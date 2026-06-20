"""Anticipatory controller parameterized by sensing horizon (Phase 3: perception).

The whole point: hold this single control law fixed and vary only *how far it can see*.
A short horizon mimics a stop-bar loop detector (presence right at the line); a long
horizon mimics a camera that watches the whole approach. The difference in performance
is, by construction, the value of the perception.

Logic (rest-in-green + anticipation):
  * If no vehicle is approaching on the other phase within the horizon, stay green for
    the current phase — never stop anyone for an empty cross street.
  * If the other phase has demand, switch unless a current-phase vehicle is imminent
    (would be stranded). Seeing the other phase's car early (long horizon) lets the
    switch + clearance finish *before* the car arrives, so it meets a green.
"""

from __future__ import annotations

from sim.signal import GREEN

from .base import Controller


class Anticipatory(Controller):
    def __init__(self, horizon: float | None = None, name: str | None = None):
        self.horizon = horizon            # None -> use the scenario's camera horizon
        self.name = name or "anticipatory"

    def act(self, obs, env) -> int:
        sim = env.unwrapped_sim
        sig = sim.signal
        if sig.state != GREEN or not sig.can_switch:
            return 0                      # mid-transition or below min green
        if sig.must_switch:
            return 1                      # max green safety force-off

        H = self.horizon if self.horizon is not None else sim.camera_horizon
        cur = [a for a in sig.phases[sig.phase] if a in sim.active_approaches]
        oth = [a for a in sim.active_approaches if a not in cur]

        other_demand = sum(sim.approaching(a, H) for a in oth)
        if other_demand == 0:
            return 0                      # rest in green: nobody else needs it

        clearance = sig.yellow + sig.allred
        cur_eta = min((sim.nearest_eta(a, H) for a in cur), default=float("inf"))
        cur_demand = sum(sim.approaching(a, H) for a in cur)
        # hold green for an imminent current-phase vehicle rather than strand it just
        # before the line ("don't turn red if a car is right there with nobody after")
        if cur_demand > 0 and cur_eta <= clearance + 2.0:
            return 0
        return 1                          # current approach clear; serve the other phase
