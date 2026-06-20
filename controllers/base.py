"""Common controller interface.

A controller maps the current state to a discrete action {0: hold, 1: switch}.
Baseline controllers are allowed to read the sim directly (they emulate the logic a
real cabinet controller would run on its own detector inputs); the learned controller
uses only the observation vector.
"""

from __future__ import annotations

from envs import IntersectionEnv


class Controller:
    name = "base"

    def reset(self) -> None:
        pass

    def act(self, obs, env: IntersectionEnv) -> int:  # noqa: ARG002
        raise NotImplementedError
