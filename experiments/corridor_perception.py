"""Phase 3 on the corridor: does look-ahead matter more across multiple lights?

Sweeps the sensing horizon of a reactive anticipatory controller (which tries to be
green when a platoon arrives and serves the cross street in the gaps). A short horizon
mimics a stop-bar loop; a long one (~the full link) mimics a camera that sees the
platoon the moment the upstream light releases it. For reference we also show the
explicit green wave (coordinated offsets) and lock-step (uncoordinated).

    python -m experiments.corridor_perception --seeds 5
"""

from __future__ import annotations

import argparse
import statistics as st

from controllers.corridor import AnticipatoryCorridor, FixedTimeCorridor
from sim import load_scenario
from sim.corridor import CorridorSim

HORIZONS = [40.0, 100.0, 200.0, 400.0]  # 400 m = one full link (camera sees upstream light)


def run(sim, controller, seed):
    sim.reset(seed=seed)
    controller.reset(sim)
    while not sim.done:
        sim.step(controller.act(sim))
    return sim.metrics()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="arterial_corridor")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_scenario(args.scenario)
    sim = CorridorSim(cfg)

    def stats(ctrl):
        runs = [run(sim, ctrl, s) for s in range(args.seeds)]
        return (st.mean(r["eb"]["mean_stops"] for r in runs),
                st.mean(r["wb"]["mean_stops"] for r in runs),
                st.mean(r["total_delay_veh_s"] for r in runs))

    print(f"\nCorridor look-ahead sweep: {sim.K} lights, {sim.spacing:.0f} m apart, "
          f"seeds {args.seeds}\n")
    header = f"{'controller':<26}{'EB stops':>10}{'WB stops':>10}{'total delay':>14}"
    print(header)
    print("-" * len(header))

    rows = [("anticipatory loop 40m", AnticipatoryCorridor(40.0))]
    rows += [(f"anticipatory {int(h)}m", AnticipatoryCorridor(h)) for h in HORIZONS[1:]]
    for name, ctrl in rows:
        eb, wb, d = stats(ctrl)
        print(f"{name:<26}{eb:>10.2f}{wb:>10.2f}{d:>14,.0f}")
    print("-" * len(header))
    for name, ctrl in (("green wave (offsets)", FixedTimeCorridor(30, 8, coordinate=True)),
                       ("lock-step (no coord)", FixedTimeCorridor(30, 8, coordinate=False))):
        eb, wb, d = stats(ctrl)
        print(f"{name:<26}{eb:>10.2f}{wb:>10.2f}{d:>14,.0f}")

    print("\n(reactive anticipatory control with longer sight should approach the explicit "
          "green wave without any precomputed offsets.)")


if __name__ == "__main__":
    main()
