"""Gymnasium env exposing the intersection sim to controllers and RL.

Action (discrete, 2): 0 = hold current phase, 1 = request switch to the next phase.
The signal's safety envelope (min/max green, clearance) is authoritative — a switch
request before min-green is ignored, and max-green forces a switch regardless.

Observation (per active approach, in fixed N/E/S/W order):
  * queue length (stopped vehicles)            -- also visible to a loop detector
  * approaching count within the camera horizon -- the camera advantage
plus signal context: phase one-hot, signal-state one-hot, normalised elapsed green,
and a can-switch flag.

Reward: -(total delay this step) - beta * (fairness excess this step), scaled.
Minimising the discounted return therefore minimises total vehicle-delay while a
per-vehicle max-wait cap is actively penalised (plan: total delay + fairness cap).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim import TrafficSim, APPROACHES
from sim.signal import GREEN, YELLOW, ALLRED

_STATES = [GREEN, YELLOW, ALLRED]


class IntersectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict, fairness_beta: float = 6.0,
                 delay_scale: float = 10.0):
        super().__init__()
        self.sim = TrafficSim(config)
        self.fairness_beta = fairness_beta
        self.delay_scale = delay_scale
        self.n_phases = len(self.sim.signal.phases)

        self._approaches = self.sim.active_approaches
        obs_dim = 2 * len(self._approaches) + self.n_phases + len(_STATES) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # hold / switch

    # ----------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        return self._obs(), {}

    def step(self, action: int):
        info_step = self.sim.step(switch_request=bool(action))
        reward = -(info_step["step_delay"] / self.delay_scale
                   + self.fairness_beta * info_step["step_excess"])
        terminated = False
        truncated = self.sim.done
        return self._obs(), float(reward), terminated, truncated, self._info()

    # -------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        s = self.sim
        feats: list[float] = []
        # normalise counts by a nominal capacity so the net sees O(1) inputs
        cap = max(1.0, s.L / (s.veh_len + s.min_gap))
        for a in self._approaches:
            feats.append(s.queue_len(a) / cap)
            feats.append(s.approaching(a, s.camera_horizon) / cap)
        phase_oh = [1.0 if i == s.signal.phase else 0.0 for i in range(self.n_phases)]
        state_oh = [1.0 if s.signal.state == st else 0.0 for st in _STATES]
        feats += phase_oh + state_oh
        feats.append(min(1.0, s.signal.elapsed_green / s.signal.max_green))
        feats.append(1.0 if s.signal.can_switch else 0.0)
        return np.asarray(feats, dtype=np.float32)

    def _info(self) -> dict:
        return self.sim.metrics()

    # convenience for baseline controllers that read the sim directly
    @property
    def unwrapped_sim(self) -> TrafficSim:
        return self.sim
