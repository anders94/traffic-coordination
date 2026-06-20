"""Compare controllers on a scenario over several seeds and print the headline table.

Usage:
    python -m experiments.eval --scenario divided_highway_side_street --seeds 5
    python -m experiments.eval --dqn runs/dqn.pt           # include a trained policy
"""

from __future__ import annotations

import argparse

from controllers import BASELINES, DQN
from envs import IntersectionEnv
from metrics import aggregate, format_table
from sim import load_scenario

from .rollout import run_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="divided_highway_side_street")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dqn", default=None, help="path to a trained DQN checkpoint")
    ap.add_argument("--baseline", default="fixed_time",
                    help="controller to report others relative to")
    args = ap.parse_args()

    config = load_scenario(args.scenario)
    env = IntersectionEnv(config)

    controllers = {name: cls() for name, cls in BASELINES.items()}
    if args.dqn:
        controllers["dqn"] = DQN.load(args.dqn)

    results: dict[str, dict] = {}
    for name, ctrl in controllers.items():
        runs = [run_episode(env, ctrl, seed=s) for s in range(args.seeds)]
        results[name] = aggregate(runs)

    print(f"\nScenario: {config['name']}   seeds: {args.seeds}   "
          f"horizon: {config['sim']['horizon_s']:.0f}s   "
          f"fairness cap: {config['fairness']['max_wait_s']:.0f}s\n")
    print(format_table(results, baseline=args.baseline))
    print("\n(total delay in vehicle-seconds; lower is better. "
          "'fair viol' = % of seeds breaching the max-wait cap.)")


if __name__ == "__main__":
    main()
