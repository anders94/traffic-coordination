"""Train the DQN controller on a scenario.

Usage:
    python -m experiments.train --scenario divided_highway_side_street \
        --episodes 200 --out runs/dqn.pt

The agent learns purely from the reward (negative total delay + fairness penalty) —
there are no hand-coded timing rules in the policy. After training, evaluate with
`python -m experiments.eval --dqn runs/dqn.pt`.
"""

from __future__ import annotations

import argparse
import os

from controllers import DQN
from envs import IntersectionEnv
from sim import load_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="divided_highway_side_street")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default="runs/dqn.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    config = load_scenario(args.scenario)
    env = IntersectionEnv(config)
    obs_dim = env.observation_space.shape[0]

    # decay exploration over roughly the full training budget
    horizon_steps = int(config["sim"]["horizon_s"] / config["sim"]["dt_s"])
    agent = DQN(obs_dim, n_actions=env.action_space.n, seed=args.seed,
                eps_decay_steps=max(1, args.episodes * horizon_steps // 2))

    print(f"Training DQN on {config['name']}: {args.episodes} episodes "
          f"x {horizon_steps} steps  (device={agent.device})")
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        agent.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            nxt, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.remember(obs, action, reward, nxt, done)
            agent.learn()
            obs = nxt
            ep_reward += reward
        if ep % args.log_every == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  return {ep_reward:10.1f}  "
                  f"total_delay {info['total_delay_veh_s']:12,.0f}  "
                  f"max_wait {info['max_wait_s']:6.1f}  eps {agent.epsilon():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.save(args.out)
    print(f"\nSaved trained policy -> {args.out}")


if __name__ == "__main__":
    main()
