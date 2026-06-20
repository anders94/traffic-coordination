"""Run one controller through one episode and return its metrics."""

from __future__ import annotations

from controllers.base import Controller
from envs import IntersectionEnv


def run_episode(env: IntersectionEnv, controller: Controller, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    controller.reset()
    info: dict = {}
    done = False
    while not done:
        action = controller.act(obs, env)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info
