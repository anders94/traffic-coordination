"""Multi-agent environment over the arterial corridor.

One agent per intersection, all sharing a single policy (parameter sharing). Each
agent sees only local state — its own signal, its cross-street queue, the through
traffic approaching from each direction (bucketed by distance, the camera-style
look-ahead), and whether its immediate neighbours are giving the arterial green.
That neighbour + look-ahead information is what lets a *learned* policy discover
coordination (turn green as a platoon arrives) without being told the offsets.

Reward is local: minus the number of vehicles currently queued at this intersection
(through + cross). Minimising local queues everywhere minimises total delay, and the
local signal gives clean credit assignment.
"""

from __future__ import annotations

import numpy as np

from sim import load_scenario  # noqa: F401  (convenience re-export for callers)
from sim.corridor import CorridorSim
from sim.signal import ALLRED, GREEN, YELLOW

EDGES = [100.0, 250.0, 400.0]  # distance bins (m) for approaching-traffic look-ahead


def agent_obs(sim: CorridorSim, i: int) -> np.ndarray:
    sig = sim.signals[i]
    phase_oh = [1.0, 0.0] if sig.phase == 0 else [0.0, 1.0]
    state_oh = [float(sig.state == GREEN), float(sig.state == YELLOW),
                float(sig.state == ALLRED)]
    feats = phase_oh + state_oh + [
        min(1.0, sig.elapsed_green / sig.max_green),
        1.0 if sig.can_switch else 0.0,
        sim.cross_queue(i) / 10.0,
    ]
    feats += [c / 10.0 for c in sim.arterial_bins(i, +1, EDGES)]   # eastbound look-ahead
    feats += [c / 10.0 for c in sim.arterial_bins(i, -1, EDGES)]   # westbound look-ahead
    feats.append(1.0 if (i > 0 and sim.art_green(i - 1)) else 0.0)        # upstream green
    feats.append(1.0 if (i < sim.K - 1 and sim.art_green(i + 1)) else 0.0)  # downstream green
    return np.asarray(feats, dtype=np.float32)


class CorridorMultiEnv:
    def __init__(self, config: dict, reward_scale: float = 10.0):
        self.sim = CorridorSim(config)
        self.reward_scale = reward_scale
        self.K = self.sim.K
        self.obs_dim = len(agent_obs(self.sim, 0))
        self.n_actions = 2  # hold / switch (the signal's envelope enforces min/max)

    def reset(self, seed: int | None = None) -> list[np.ndarray]:
        self.sim.reset(seed=seed)
        return [agent_obs(self.sim, i) for i in range(self.K)]

    def step(self, actions: list[int]):
        self.sim.step([bool(a) for a in actions])
        obs = [agent_obs(self.sim, i) for i in range(self.K)]
        rewards = [-(self.sim.arterial_queue_at(i) + self.sim.cross_queue(i))
                   * self.sim.dt / self.reward_scale for i in range(self.K)]
        done = self.sim.done
        return obs, rewards, done, (self.sim.metrics() if done else {})
