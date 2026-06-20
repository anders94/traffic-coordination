"""Phase 2: does coordinating signal offsets produce a green wave?

Compares, on the arterial corridor, three timing strategies:
  * uncoordinated fixed-time  — every light cycles in lock-step (offset 0)
  * coordinated fixed-time    — offsets = arterial travel time (a green wave for EB)
  * independent max-pressure  — each light adapts locally, but with no coordination

Reports per-direction stops/vehicle and the fraction of through trips that never stop,
plus mean arterial travel time and total delay.

    python -m experiments.corridor_eval --seeds 5
"""

from __future__ import annotations

import argparse
import statistics as st

from controllers.corridor import (
    CoordinatedAdaptive, FixedTimeCorridor, IndependentMaxPressure, LearnedCorridor,
)
from controllers.dqn import DQN
from sim import load_scenario
from sim.corridor import CorridorSim


def run(sim: CorridorSim, controller, seed: int) -> dict:
    sim.reset(seed=seed)
    controller.reset(sim)
    while not sim.done:
        sim.step(controller.act(sim))
    return sim.metrics()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="arterial_corridor")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--green-art", type=float, default=30.0)
    ap.add_argument("--green-cross", type=float, default=15.0)
    ap.add_argument("--learned", default=None, help="path to a trained corridor policy")
    args = ap.parse_args()

    cfg = load_scenario(args.scenario)
    sim = CorridorSim(cfg)
    ga, gc = args.green_art, args.green_cross

    controllers = {
        "uncoordinated": FixedTimeCorridor(ga, gc, coordinate=False),
        "coordinated": FixedTimeCorridor(ga, gc, coordinate=True),
        "independent_mp": IndependentMaxPressure(),
        "coord_adaptive": CoordinatedAdaptive(ga, gc),
    }
    if args.learned:
        controllers["learned"] = LearnedCorridor(DQN.load(args.learned))

    def mean(runs, *keys):
        vals = [r[keys[0]][keys[1]] if len(keys) == 2 else r[keys[0]] for r in runs]
        return st.mean(vals)

    dem = ("time-varying (EB peak / balanced / WB peak)" if "schedule" in cfg["demand"]
           else f"EB/WB/cross = {cfg['demand']['eb_vph']}/{cfg['demand']['wb_vph']}/"
                f"{cfg['demand']['cross_vph']} vph")
    print(f"\nCorridor: {cfg['geometry']['n_intersections']} intersections, "
          f"{cfg['geometry']['spacing_m']:.0f} m apart   seeds: {args.seeds}   {dem}")
    ff = sim.length / sim.v_art
    print(f"free-flow arterial travel time: {ff:.0f} s\n")

    header = (f"{'strategy':<16}{'EB stops':>10}{'EB no-stop':>12}{'WB stops':>10}"
              f"{'travel s':>10}{'total delay':>14}")
    print(header)
    print("-" * len(header))
    for name, ctrl in controllers.items():
        runs = [run(sim, ctrl, s) for s in range(args.seeds)]
        print(f"{name:<16}"
              f"{mean(runs, 'eb', 'mean_stops'):>10.2f}"
              f"{mean(runs, 'eb', 'frac_no_stop') * 100:>11.0f}%"
              f"{mean(runs, 'wb', 'mean_stops'):>10.2f}"
              f"{mean(runs, 'mean_travel_time_s'):>10.0f}"
              f"{mean(runs, 'total_delay_veh_s'):>14,.0f}")
    print("\n(EB stops = mean stops per eastbound through-trip across 5 lights; "
          "EB no-stop = % of EB trips that never stop.)")


if __name__ == "__main__":
    main()
