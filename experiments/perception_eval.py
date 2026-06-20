"""Phase 3: what does seeing farther (a camera) buy over a loop detector?

Holds one anticipatory control law fixed and varies only its sensing horizon:
  * 15 m  — a stop-bar loop (presence at the line, no anticipation)
  * 60 m  — an advance loop
  * 200 m — a camera watching the whole approach
across traffic levels (we scale the scenario demand). The hypothesis under test: the
camera's advantage is largest in light traffic, where the cost of an unnecessary stop
is high relative to how little opposing traffic there is.

    python -m experiments.perception_eval --seeds 5
"""

from __future__ import annotations

import argparse

from controllers.anticipatory import Anticipatory
from envs import IntersectionEnv
from metrics import aggregate
from sim import load_scenario

from .rollout import run_episode

HORIZONS = [("loop 15m", 15.0), ("loop 60m", 60.0), ("camera 200m", 200.0)]
SCALES = [0.15, 0.30, 0.60, 1.00]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="divided_highway_side_street")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    base = load_scenario(args.scenario)
    base_demand = base["demand"]

    print(f"\nPerception sweep on {base['name']}   seeds: {args.seeds}")
    print("mean delay per vehicle (s); lower is better\n")
    header = (f"{'traffic (EB/SB vph)':<22}" + "".join(f"{n:>14}" for n, _ in HORIZONS)
              + f"{'camera vs loop':>16}")
    print(header)
    print("-" * len(header))

    for scale in SCALES:
        cfg = {**base, "demand": {a: v * scale for a, v in base_demand.items()}}
        env = IntersectionEnv(cfg)
        row = {}
        for name, h in HORIZONS:
            runs = [run_episode(env, Anticipatory(h), seed=s) for s in range(args.seeds)]
            row[name] = aggregate(runs)["mean_delay_per_veh"]
        loop, cam = row["loop 15m"], row["camera 200m"]
        redux = (1 - cam / loop) * 100 if loop > 0 else 0.0
        label = f"{base_demand['E']*scale:.0f}/{base_demand['N']*scale:.0f}"
        print(f"{label:<22}" + "".join(f"{row[n]:>14.1f}" for n, _ in HORIZONS)
              + f"{redux:>15.0f}%")

    print("\n(EB = one highway approach, SB = one side-street approach. "
          "'camera vs loop' = delay reduction of 200 m sight vs 15 m stop-bar loop.)")


if __name__ == "__main__":
    main()
