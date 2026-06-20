"""Multi-intersection arterial corridor simulator (Phase 2).

A single light cannot deliver "drive all the way through without stopping" — that is a
*coordination* problem across consecutive lights. This sim models an arterial (a
two-way through road) crossing K signalized intersections, each with a cross street, so
we can test whether coordinating signal **offsets** produces a green wave.

Modeling choices (kept deliberately small, in the spirit of the M1 sim):
* The arterial is 1-D: each through vehicle has a position `x` along the corridor and
  follows the car ahead (per lane) with the same accel/decel + start-up model as M1,
  stopping at the stop line of any intersection that is red for the arterial.
* Cross-street traffic at each intersection is modeled as a queue that arrives (Poisson)
  and discharges at a saturation flow while its phase is green — enough to make the
  cross street compete for green time without simulating every cross vehicle's motion.
* Each intersection has a 2-phase signal (arterial / cross) with the usual min/max green
  and yellow + all-red clearance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from .signal import Signal, GREEN

_ids = count()


@dataclass
class ArtVehicle:
    direction: int            # +1 = eastbound (increasing x), -1 = westbound
    lane: int
    x: float                  # position along the corridor (m)
    entry_time: float
    speed: float = 0.0
    release: float = 0.0
    delay: float = 0.0
    stops: int = 0
    moving: bool = True       # hysteresis flag for counting distinct stops
    vid: int = field(default_factory=lambda: next(_ids))


class CorridorSim:
    def __init__(self, config: dict):
        self.cfg = config
        g = config["geometry"]
        self.K = int(g["n_intersections"])
        self.spacing = float(g["spacing_m"])
        self.v_art = float(g["arterial_speed_ms"])
        self.v_cross = float(g["cross_speed_ms"])
        self.min_gap = float(g["min_gap_m"])
        self.veh_len = float(g["vehicle_length_m"])
        self.reaction_time = float(g.get("startup_reaction_s", 1.5))
        self.accel = float(g.get("accel_ms2", 2.5))
        self.decel = float(g.get("decel_ms2", 3.0))
        self.lanes = int(g.get("arterial_lanes", 2))

        # intersections sit at spacing, 2*spacing, ... with a buffer at both ends
        self.xpos = [self.spacing * (i + 1) for i in range(self.K)]
        self.length = self.spacing * (self.K + 1)

        s = config["signal"]
        self.min_green = float(s["min_green_s"])
        self.max_green = float(s["max_green_s"])
        self.yellow = float(s["yellow_s"])
        self.allred = float(s["allred_s"])
        self.cross_sat = float(s.get("cross_sat_flow_vps", 0.5))

        d = config["demand"]
        # demand may be flat or a time-varying schedule of segments, each:
        #   {until_s, eb_vph, wb_vph, cross_vph}  (last segment covers the rest)
        self.schedule = d.get("schedule")
        if self.schedule is None:
            self._flat = (float(d["eb_vph"]) / 3600.0,
                          float(d["wb_vph"]) / 3600.0,
                          float(d["cross_vph"]) / 3600.0)

        sc = config.get("sim", {})
        self.dt = float(sc.get("dt_s", 1.0))
        self.horizon = float(sc.get("horizon_s", 3600.0))
        self.seed = int(sc.get("seed", 0))
        self.reset()

    # phase 0 = arterial green, phase 1 = cross green
    def _new_signal(self) -> Signal:
        return Signal([{"art"}, {"cross"}], self.min_green, self.max_green,
                      self.yellow, self.allred)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.t = 0.0
        self.signals = [self._new_signal() for _ in range(self.K)]
        # arterial vehicles grouped by direction then lane
        self.art = {+1: [[] for _ in range(self.lanes)],
                    -1: [[] for _ in range(self.lanes)]}
        self.cross_pending = [deque() for _ in range(self.K)]  # arrival times
        self._cross_credit = [0.0] * self.K                    # fractional discharge

        self.total_delay = 0.0
        self.cross_delay = 0.0
        self.completed = []   # (travel_time, stops) for arterial vehicles that exited

    @property
    def done(self) -> bool:
        return self.t >= self.horizon

    def art_green(self, i: int) -> bool:
        return self.signals[i].is_green("art")

    # ---- arterial helpers -------------------------------------------------
    def _entry_lane(self, direction: int) -> int | None:
        """Lane with the most room at the corridor entry for this direction."""
        x_entry = 0.0 if direction > 0 else self.length
        best, best_gap = None, -1.0
        need = self.veh_len + self.min_gap
        for lane in range(self.lanes):
            cars = self.art[direction][lane]
            if not cars:
                return lane
            # forward coordinate s = direction * x; entry is at s = 0
            tail_s = min(direction * c.x for c in cars)  # closest car to entry
            gap = tail_s - 0.0
            if gap >= need and gap > best_gap:
                best, best_gap = lane, gap
        return best

    def _current_rates(self) -> tuple[float, float, float]:
        """(eb, wb, cross) arrival rates in veh/s at the current time."""
        if self.schedule is None:
            return self._flat
        for seg in self.schedule:
            if "until_s" not in seg or self.t < float(seg["until_s"]):
                return (float(seg["eb_vph"]) / 3600.0,
                        float(seg["wb_vph"]) / 3600.0,
                        float(seg["cross_vph"]) / 3600.0)
        last = self.schedule[-1]
        return (float(last["eb_vph"]) / 3600.0, float(last["wb_vph"]) / 3600.0,
                float(last["cross_vph"]) / 3600.0)

    def _spawn(self) -> None:
        rate_eb, rate_wb, rate_cross = self._current_rates()
        for direction, rate in ((+1, rate_eb), (-1, rate_wb)):
            for _ in range(int(self.rng.poisson(rate * self.dt))):
                lane = self._entry_lane(direction)
                if lane is None:
                    continue  # entry blocked this step (oversaturated)
                x0 = 0.0 if direction > 0 else self.length
                self.art[direction][lane].append(
                    ArtVehicle(direction=direction, lane=lane, x=x0,
                               entry_time=self.t, speed=self.v_art))
        for i in range(self.K):
            for _ in range(int(self.rng.poisson(rate_cross * self.dt))):
                self.cross_pending[i].append(self.t)

    def _red_lines_s(self, direction: int) -> list[float]:
        """Forward (s = direction*x) coordinates of stop lines currently red for art."""
        return sorted(direction * self.xpos[i] for i in range(self.K)
                      if not self.art_green(i))

    def _advance_arterial(self) -> float:
        step_delay = 0.0
        for direction in (+1, -1):
            red_s = self._red_lines_s(direction)
            for lane in range(self.lanes):
                cars = self.art[direction][lane]
                # process front-to-back: largest forward coordinate first
                cars.sort(key=lambda c: direction * c.x, reverse=True)
                leader_s = math.inf
                survivors = []
                for c in cars:
                    s = direction * c.x
                    # nearest red stop line the car has not yet crossed (at or ahead);
                    # it stays a constraint until the car actually passes it.
                    line_ahead = math.inf
                    for ls in red_s:
                        if ls >= s - 1e-6:
                            line_ahead = ls
                            break
                    leader_constraint = leader_s - (self.veh_len + self.min_gap)
                    constraint_s = min(leader_constraint, line_ahead)
                    gap = max(0.0, constraint_s - s)

                    if gap > 0.01:
                        c.release += self.dt
                    else:
                        c.release = 0.0
                    if c.release < self.reaction_time:
                        c.speed = 0.0
                    else:
                        v_safe = math.sqrt(2.0 * self.decel * gap)
                        v_target = min(self.v_art, v_safe)
                        v_new = min(v_target, c.speed + self.accel * self.dt)
                        advance = min(max(0.0, v_new) * self.dt, gap)
                        s += advance
                        c.x = direction * s
                        c.speed = advance / self.dt
                    leader_s = s

                    # metrics: delay + distinct-stop counting
                    step_delay += max(0.0, 1.0 - c.speed / self.v_art) * self.dt
                    c.delay += max(0.0, 1.0 - c.speed / self.v_art) * self.dt
                    if c.speed < 0.1 and c.moving:
                        c.stops += 1
                        c.moving = False
                    elif c.speed > 1.0:
                        c.moving = True

                    if s >= self.length - 1e-6:  # exited the far end
                        self.completed.append(
                            (self.t - c.entry_time, c.stops, c.direction))
                    else:
                        survivors.append(c)
                self.art[direction][lane] = survivors
        return step_delay

    def _advance_cross(self) -> float:
        step_delay = 0.0
        for i in range(self.K):
            green = self.signals[i].is_green("cross")
            q = self.cross_pending[i]
            if green:
                self._cross_credit[i] += self.cross_sat * self.dt
                while self._cross_credit[i] >= 1.0 and q:
                    arrival = q.popleft()
                    self._cross_credit[i] -= 1.0
                    self.cross_delay += self.t - arrival  # accumulated wait at exit
            else:
                self._cross_credit[i] = 0.0
            step_delay += len(q) * self.dt  # everyone still queued waits this step
        return step_delay

    def step(self, switch_requests: list[bool]) -> dict:
        for i, sig in enumerate(self.signals):
            sig.step(self.dt, bool(switch_requests[i]))
        self._spawn()
        art_delay = self._advance_arterial()
        cross_delay = self._advance_cross()
        self.total_delay += art_delay + cross_delay
        self.t += self.dt
        return {"step_delay": art_delay + cross_delay}

    # ---- queries used by controllers --------------------------------------
    def cross_queue(self, i: int) -> int:
        return len(self.cross_pending[i])

    def arterial_demand(self, i: int, horizon: float = 120.0) -> int:
        """Through vehicles approaching intersection i within `horizon` metres."""
        n = 0
        x_i = self.xpos[i]
        for direction in (+1, -1):
            for lane in self.art[direction]:
                for c in lane:
                    d_ahead = direction * (x_i - c.x)  # >0 if i is ahead of the car
                    if 0.0 <= d_ahead <= horizon:
                        n += 1
        return n

    def arterial_bins(self, i: int, direction: int, edges: list[float]) -> list[int]:
        """Counts of through vehicles approaching intersection i from `direction`,
        bucketed by distance-to-stop-line — the upstream look-ahead a camera provides
        and the signal needs to anticipate a platoon. edges = upper bounds of each bin."""
        counts = [0] * len(edges)
        x_i = self.xpos[i]
        for lane in self.art[direction]:
            for c in lane:
                d_ahead = direction * (x_i - c.x)   # >0 if i is ahead of this car
                if d_ahead <= 0:
                    continue
                for b, hi in enumerate(edges):
                    if d_ahead <= hi:
                        counts[b] += 1
                        break
        return counts

    def arterial_eta(self, i: int, horizon: float) -> float:
        """Seconds until the nearest through vehicle within `horizon` of intersection i
        (either direction) reaches it at arterial speed. inf if none in view. With a long
        horizon this reveals a platoon the upstream light just released — the look-ahead
        a camera provides and a stop-bar loop cannot."""
        x_i = self.xpos[i]
        best = float("inf")
        for direction in (+1, -1):
            for lane in self.art[direction]:
                for c in lane:
                    d_ahead = direction * (x_i - c.x)
                    if 0.0 < d_ahead <= horizon:
                        best = min(best, d_ahead / self.v_art)
        return best

    def arterial_queue_at(self, i: int) -> int:
        """Stopped through vehicles queued on the approaches to intersection i."""
        x_i = self.xpos[i]
        n = 0
        for direction in (+1, -1):
            lo, hi = (x_i - self.spacing, x_i) if direction > 0 else (x_i, x_i + self.spacing)
            for lane in self.art[direction]:
                for c in lane:
                    if c.speed < 0.1 and lo <= c.x <= hi:
                        n += 1
        return n

    def metrics(self) -> dict:
        st = [s for _, s, _ in self.completed]
        tt = [t for t, _, _ in self.completed]

        def by_dir(direction):
            sd = [s for _, s, d in self.completed if d == direction]
            return {
                "mean_stops": float(np.mean(sd)) if sd else 0.0,
                "frac_no_stop": float(np.mean([s == 0 for s in sd])) if sd else 0.0,
                "n": len(sd),
            }

        return {
            "total_delay_veh_s": self.total_delay,
            "arterial_completed": len(self.completed),
            "mean_travel_time_s": float(np.mean(tt)) if tt else 0.0,
            "free_flow_travel_time_s": self.length / self.v_art,
            "mean_stops_per_vehicle": float(np.mean(st)) if st else 0.0,
            "frac_no_stop": float(np.mean([s == 0 for s in st])) if st else 0.0,
            "cross_delay_veh_s": self.cross_delay,
            "eb": by_dir(+1),
            "wb": by_dir(-1),
        }
