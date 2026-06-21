"""Single-intersection microsimulator.

A deliberately small, transparent, seedable microsim so we own the per-vehicle
delay accounting and the "anticipate the last car through" mechanic end-to-end
(see plan: M1 custom sim, graduate to SUMO in M2).

Model summary
-------------
* Four approaches N/E/S/W, each with a configurable number of lanes.
* Vehicles arrive per approach as a Poisson process at a configured veh/hour rate.
  Arrivals that cannot fit at the upstream entry wait in a `pending` queue and
  accrue delay there too, so oversaturation is represented faithfully.
* Longitudinal motion is a simple safe-following rule: each vehicle advances at
  free-flow speed unless blocked by the vehicle ahead (min gap) or by a non-green
  signal at the stop line.
* Two detection horizons encode the camera advantage: loop-style `detection_zone_m`
  (what a buried loop sees) and the full-approach `camera_horizon_m` (what a camera
  sees). Controllers may use either.
"""

from __future__ import annotations

import math

import numpy as np

from .signal import Signal, GREEN
from .vehicle import Vehicle

APPROACHES = ["N", "E", "S", "W"]


class TrafficSim:
    def __init__(self, config: dict):
        self.cfg = config
        g = config["geometry"]
        self.L = float(g["approach_length_m"])
        # free-flow speed is per-approach (a highway runs faster than a side street),
        # defaulting to the scenario's free_flow_speed_ms where not specified.
        v_default = float(g["free_flow_speed_ms"])
        self.v_free = {a: float(config["approaches"].get(a, {}).get("speed_ms", v_default))
                       for a in APPROACHES}
        self.min_gap = float(g["min_gap_m"])
        self.veh_len = float(g["vehicle_length_m"])
        self.clear = float(g["intersection_clear_m"])
        # start-up reaction: a stopped car only launches after the path ahead has
        # been clear this long, so a queue releases as a wave rather than all at once.
        self.reaction_time = float(g.get("startup_reaction_s", 1.5))
        # comfortable acceleration / braking: cars ease up to speed and slow smoothly
        # to a stop rather than snapping between free-flow and stationary.
        self.accel = float(g.get("accel_ms2", 2.5))
        self.decel = float(g.get("decel_ms2", 3.0))

        self.lanes = {a: int(config["approaches"].get(a, {}).get("lanes", 0))
                      for a in APPROACHES}
        self.active_approaches = [a for a in APPROACHES if self.lanes[a] > 0]

        phases = [set(p["approaches"]) for p in config["phases"]]
        s = config["signal"]
        self.signal = Signal(
            phases=phases,
            min_green=float(s["min_green_s"]),
            max_green=float(s["max_green_s"]),
            yellow=float(s["yellow_s"]),
            allred=float(s["allred_s"]),
        )
        self.passage_time = float(s.get("passage_time_s", 2.5))
        self.detection_zone = float(s.get("detection_zone_m", 60.0))
        self.camera_horizon = float(s.get("camera_horizon_m", self.L))

        d = config["demand"]
        self.rate = {a: float(d.get(a, 0.0)) / 3600.0 for a in APPROACHES}  # veh/s

        sc = config.get("sim", {})
        self.dt = float(sc.get("dt_s", 1.0))
        self.horizon = float(sc.get("horizon_s", 3600.0))
        self.fairness_cap = float(config.get("fairness", {}).get("max_wait_s", 90.0))
        self.emergency_weight = float(config.get("fairness", {}).get("emergency_weight", 10.0))

        self.rng = np.random.default_rng(int(sc.get("seed", 0)))
        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0.0
        # vehicles[approach][lane] = list of Vehicle, ordered closest-to-line first
        self.vehicles: dict[str, list[list[Vehicle]]] = {
            a: [[] for _ in range(self.lanes[a])] for a in self.active_approaches
        }
        self.pending: dict[str, list[float]] = {a: [] for a in self.active_approaches}
        # time since a vehicle was last seen in the loop zone (for actuated control)
        self.gap_timer: dict[str, float] = {a: 0.0 for a in self.active_approaches}

        # cumulative metrics
        self.total_delay = 0.0
        self.cleared = 0
        self.cleared_delay = 0.0
        self.max_wait_seen = {a: 0.0 for a in self.active_approaches}
        self.fairness_excess = 0.0  # cumulative vehicle-seconds spent over the cap
        self.signal.__init__(
            self.signal.phases, self.signal.min_green, self.signal.max_green,
            self.signal.yellow, self.signal.allred,
        )

    @property
    def done(self) -> bool:
        return self.t >= self.horizon

    # -------------------------------------------------------------- spawning
    def _spawn(self) -> None:
        for a in self.active_approaches:
            n = self.rng.poisson(self.rate[a] * self.dt)
            for _ in range(int(n)):
                self.pending[a].append(self.t)
        # admit pending vehicles where there is room at the entry
        for a in self.active_approaches:
            while self.pending[a]:
                lane = self._freest_lane(a)
                if lane is None:
                    break  # no room on any lane this step
                arrival = self.pending[a].pop(0)
                v = Vehicle(approach=a, lane=lane, d=self.L, spawn_time=arrival)
                self.vehicles[a][lane].append(v)  # appended = furthest upstream
                self.vehicles[a][lane].sort(key=lambda x: x.d)

    def _freest_lane(self, a: str) -> int | None:
        best, best_d = None, -1.0
        need = self.veh_len + self.min_gap
        for lane in range(self.lanes[a]):
            cars = self.vehicles[a][lane]
            upstream_gap = self.L if not cars else self.L - cars[-1].d
            if upstream_gap >= need and upstream_gap > best_d:
                best, best_d = lane, upstream_gap
        return best

    # ------------------------------------------------------------------ step
    def step(self, switch_request: bool) -> dict:
        """Advance one timestep. Returns per-step deltas used to form the reward."""
        self.signal.step(self.dt, switch_request)
        self._spawn()

        step_delay = 0.0
        step_excess = 0.0

        # pending (not-yet-admitted) vehicles are fully stopped -> full delay
        for a in self.active_approaches:
            for _ in self.pending[a]:
                step_delay += self.dt

        for a in self.active_approaches:
            green = self.signal.is_green(a)
            zone_occupied = False
            for lane in range(self.lanes[a]):
                cars = self.vehicles[a][lane]
                leader_d = -1e9  # rear constraint imposed by the car ahead
                survivors = []
                for v in cars:  # closest-to-line first
                    # constraint = nearest point the car may not pass: the leader's new
                    # position + min gap, or the stop line on a non-green signal.
                    constraint = leader_d + self.veh_len + self.min_gap
                    if not green and v.d >= 0.0:
                        constraint = max(constraint, 0.0)
                    gap = max(0.0, v.d - constraint)  # room left to advance

                    # start-up reaction: accrue "clear" time while there is room to
                    # advance; only launch once it exceeds the reaction time. A follower
                    # therefore pulls away a beat after its leader, not in lock-step.
                    if gap > 0.01:
                        v.release += self.dt
                    else:
                        v.release = 0.0

                    if v.release < self.reaction_time:
                        v.speed = 0.0  # still reacting / blocked
                    else:
                        # fastest speed from which the car can still brake to a stop at
                        # the constraint -> it decelerates smoothly on the approach.
                        v_safe = math.sqrt(2.0 * self.decel * gap)
                        v_target = min(self.v_free[a], v_safe)
                        # ease up to the target (bounded accel); braking may be harder.
                        v_new = min(v_target, v.speed + self.accel * self.dt)
                        advance = min(max(0.0, v_new) * self.dt, gap)
                        v.d -= advance
                        v.speed = advance / self.dt
                    leader_d = v.d

                    # metrics
                    dly = max(0.0, 1.0 - v.speed / self.v_free[a]) * self.dt
                    w = self.emergency_weight if v.emergency else 1.0
                    step_delay += dly * w
                    v.delay += dly
                    if v.speed < 0.1:
                        v.wait += self.dt
                    self.max_wait_seen[a] = max(self.max_wait_seen[a], v.wait)
                    if v.wait > self.fairness_cap:
                        step_excess += self.dt

                    if 0.0 <= v.d <= self.detection_zone:
                        zone_occupied = True

                    if v.d <= -self.clear:  # fully cleared the intersection
                        self.cleared += 1
                        self.cleared_delay += v.delay
                    else:
                        survivors.append(v)
                self.vehicles[a][lane] = survivors

            # actuated detector bookkeeping: time since the loop zone last had a car
            self.gap_timer[a] = 0.0 if zone_occupied else self.gap_timer[a] + self.dt

        self.total_delay += step_delay
        self.fairness_excess += step_excess
        self.t += self.dt
        return {"step_delay": step_delay, "step_excess": step_excess}

    # ----------------------------------------------------------- observation
    def queue_len(self, a: str) -> int:
        """Number of effectively-stopped vehicles waiting on an approach."""
        n = len(self.pending[a])
        for lane in self.vehicles[a]:
            n += sum(1 for v in lane if v.speed < 0.1 and v.d >= 0.0)
        return n

    def approaching(self, a: str, horizon: float | None = None) -> int:
        """Vehicles within `horizon` metres of the line and not yet crossed."""
        h = self.camera_horizon if horizon is None else horizon
        n = len(self.pending[a])
        for lane in self.vehicles[a]:
            n += sum(1 for v in lane if 0.0 <= v.d <= h)
        return n

    def nearest_eta(self, a: str, horizon: float) -> float:
        """Seconds until the nearest not-yet-crossed vehicle on approach `a` (within
        `horizon` metres) would reach the stop line at free-flow speed. inf if none.
        This is the kind of estimate a camera enables and a loop detector cannot."""
        best = float("inf")
        v = self.v_free[a]
        for n in self.pending[a]:  # waiting to even enter -> treat as far/late
            best = min(best, self.L / v)
        for lane in self.vehicles[a]:
            for veh in lane:
                if 0.0 <= veh.d <= horizon:
                    best = min(best, veh.d / v)
        return best

    def pressure(self, phase_idx: int) -> float:
        """Max-pressure proxy: demand served by a phase (downstream assumed clear)."""
        return sum(self.approaching(a, self.camera_horizon)
                   for a in self.signal.phases[phase_idx]
                   if a in self.active_approaches)

    def metrics(self) -> dict:
        active = [v for a in self.active_approaches
                  for lane in self.vehicles[a] for v in lane]
        in_system = self.cleared + len(active) + sum(len(self.pending[a])
                                                      for a in self.active_approaches)
        return {
            "total_delay_veh_s": self.total_delay,
            "cleared": self.cleared,
            "in_system": in_system,
            "mean_delay_per_veh": self.total_delay / max(1, in_system),
            "throughput_veh_per_h": self.cleared / max(1e-9, self.t) * 3600.0,
            "max_wait_s": max(self.max_wait_seen.values()) if self.max_wait_seen else 0.0,
            "max_wait_by_approach": dict(self.max_wait_seen),
            "fairness_excess_veh_s": self.fairness_excess,
            "fairness_violation": any(w > self.fairness_cap
                                      for w in self.max_wait_seen.values()),
        }
