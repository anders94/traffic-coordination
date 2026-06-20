"""DQN learning controller.

Single-agent deep Q-network over the env observation. Deliberately vanilla (MLP +
replay buffer + target network + epsilon-greedy) — M1 only needs to show that a
policy which *learns from the reward signal*, with no hand-coded timing rules, can
match or beat the engineered baselines on total delay while respecting the fairness
cap. The graph/multi-agent machinery (CoLight-style) is M2 work.

torch is an optional dependency (`pip install -e .[learn]`). The baselines and eval
harness run without it.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from .base import Controller

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch installed
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "DQN needs PyTorch. Install it with:  pip install -e '.[learn]'"
        )


if _HAS_TORCH:
    class QNet(nn.Module):
        def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)


class DQN(Controller):
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int = 2, *, lr: float = 1e-3,
                 gamma: float = 0.99, hidden: int = 128, buffer_size: int = 100_000,
                 batch_size: int = 128, target_sync: int = 1000,
                 eps_start: float = 1.0, eps_end: float = 0.02,
                 eps_decay_steps: int = 50_000, device: str | None = None,
                 seed: int = 0):
        _require_torch()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.gamma, self.batch_size, self.target_sync = gamma, batch_size, target_sync
        self.eps_start, self.eps_end, self.eps_decay_steps = (
            eps_start, eps_end, eps_decay_steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        random.seed(seed)

        self.q = QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target = QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.learn_steps = 0
        self.training = True

    # -------------------------------------------------------- policy
    def epsilon(self) -> float:
        frac = min(1.0, self.learn_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray) -> int:
        if self.training and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        return self._greedy(obs)

    def _greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return int(self.q(t).argmax().item())

    # Controller API (greedy, used at evaluation time)
    def act(self, obs, env) -> int:  # noqa: ARG002
        return self._greedy(np.asarray(obs, dtype=np.float32))

    # -------------------------------------------------------- training
    def remember(self, s, a, r, s2, done) -> None:
        self.buffer.append((s, a, r, s2, float(done)))

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s2).max(1, keepdim=True)[0]
            target = r + self.gamma * (1.0 - done) * q_next
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # -------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        torch.save({"obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "state_dict": self.q.state_dict()}, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "DQN":
        _require_torch()
        # our own checkpoint, written by save() below -> safe to fully unpickle
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(int(ckpt["obs_dim"]), int(ckpt["n_actions"]), **kwargs)
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.training = False
        return agent
