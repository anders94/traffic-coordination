"""Train a shared multi-agent policy over the corridor (parameter sharing).

All intersections share one DQN; each acts on its own local observation and is
rewarded by its own local queues. Coordination is not programmed — the hope is that
it emerges from the upstream look-ahead + neighbour-green features.

    python -m experiments.corridor_train --scenario arterial_corridor \
        --episodes 200 --out runs/corridor_dqn.pt
"""

from __future__ import annotations

import argparse
import os

from controllers.dqn import DQN
from envs.corridor_env import CorridorMultiEnv
from sim import load_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="arterial_corridor")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--horizon", type=float, default=1800.0, help="episode length (s)")
    ap.add_argument("--out", default="runs/corridor_dqn.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    cfg = load_scenario(args.scenario)
    cfg.setdefault("sim", {})["horizon_s"] = args.horizon
    env = CorridorMultiEnv(cfg)
    steps_per_ep = int(args.horizon / env.sim.dt)
    agent = DQN(env.obs_dim, n_actions=env.n_actions, seed=args.seed,
                eps_decay_steps=max(1, args.episodes * steps_per_ep // 2))

    print(f"Training shared corridor policy: {env.K} agents, obs_dim={env.obs_dim}, "
          f"{args.episodes} eps x {steps_per_ep} steps  (device={agent.device})")
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        agent.reset()
        done = False
        ep_reward = 0.0
        while not done:
            actions = [agent.select_action(o) for o in obs]
            nobs, rewards, done, info = env.step(actions)
            for i in range(env.K):
                agent.remember(obs[i], actions[i], rewards[i], nobs[i], done)
            agent.learn()
            agent.learn()  # two updates/step to keep up with K transitions/step
            obs = nobs
            ep_reward += sum(rewards)
        if ep % args.log_every == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  return {ep_reward:9.1f}  "
                  f"EB stops {info['eb']['mean_stops']:.2f}  "
                  f"WB stops {info['wb']['mean_stops']:.2f}  "
                  f"delay {info['total_delay_veh_s']:10,.0f}  eps {agent.epsilon():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent.save(args.out)
    print(f"\nSaved corridor policy -> {args.out}")


if __name__ == "__main__":
    main()
