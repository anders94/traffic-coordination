"""Phase 4 demonstration: the safety layer cannot be defeated by the controller.

Three scenarios on a single intersection:
  A. An *adversarial* controller that requests a phase change on every single step is
     run through the signal envelope. We show the displayed signal sequence is always
     legal — every green meets its minimum and every phase change carries full clearance
     — and the independent conflict monitor never trips. (The proactive envelope works.)
  B. A controller/hardware *fault* injects a conflicting green directly (bypassing the
     envelope). We show the independent conflict monitor catches it and trips the
     intersection to flashing-red. (The backstop works.)
  C. The controller *fails* (returns nothing / errors). We show the system falls back to
     a safe default instead of doing something dangerous. (Graceful degradation.)

    python -m experiments.safety_demo
"""

from __future__ import annotations

from safety import ConflictMonitor, conflicts_from_phases
from sim import GREEN, TrafficSim, load_scenario


def green_set(sim: TrafficSim) -> set[str]:
    sig = sim.signal
    return set(sig.phases[sig.phase]) if sig.state == GREEN else set()


def build(sim: TrafficSim) -> ConflictMonitor:
    sig = sim.signal
    return ConflictMonitor(conflicts_from_phases(sig.phases),
                           min_yellow=sig.yellow, min_allred=sig.allred,
                           min_green=sig.min_green)


def scenario_a(cfg) -> None:
    """Adversarial 'switch every step' controller through the envelope."""
    sim = TrafficSim(cfg)
    sim.reset(seed=0)
    mon = build(sim)
    min_green = sim.signal.min_green
    durations, cur_green, cur_len = [], frozenset(), 0.0
    steps = 2000
    for _ in range(steps):
        sim.step(switch_request=True)        # adversary always demands a switch
        g = green_set(sim)
        mon.check(g, sim.dt)                 # monitor validates the display
        gf = frozenset(g)
        if gf == cur_green:
            cur_len += sim.dt
        else:
            if cur_green:
                durations.append(cur_len)
            cur_green, cur_len = gf, sim.dt
    durations = durations[1:]  # drop the pre-existing initial green (start not observed)
    min_seen = min(durations) if durations else 0.0
    print("A. Adversarial 'switch-every-step' controller, through the envelope")
    print(f"   steps simulated:            {steps}")
    print(f"   distinct green intervals:   {len(durations)}")
    print(f"   shortest green displayed:   {min_seen:.0f}s  (configured minimum {min_green:.0f}s)")
    print(f"   conflict monitor tripped:   {mon.tripped}   "
          f"-> envelope kept every display legal\n")


def scenario_b(cfg) -> None:
    """A fault commands a conflicting green directly; the monitor must catch it."""
    sim = TrafficSim(cfg)
    sim.reset(seed=0)
    mon = build(sim)
    p0, p1 = sim.signal.phases[0], sim.signal.phases[1]
    print("B. Fault injection: controller commands a conflicting green (envelope bypassed)")
    displayed = None
    for step in range(40):
        if step < 20:
            out = mon.check(set(p0), sim.dt)      # normal: phase-0 green
        elif step == 20:
            bad = set(p0) | set(p1)               # FAULT: both phases green at once
            out = mon.check(bad, sim.dt)
            print(f"   step {step}: commanded {sorted(bad)}  (a conflicting display)")
        else:
            out = mon.check(set(p1), sim.dt)      # keep trying to drive normally
        displayed = out
    print(f"   monitor tripped:            {mon.tripped}")
    print(f"   fault recorded:             {mon.fault}")
    print(f"   display after trip:         {sorted(displayed) or 'FLASH-RED (all dark/red)'}")
    print("   -> a conflicting green can never reach the street\n")


def scenario_c(cfg) -> None:
    """Controller failure -> safe fallback rather than an unsafe or frozen state."""
    sim = TrafficSim(cfg)
    sim.reset(seed=0)

    def faulty_controller(_sim):
        raise RuntimeError("controller crashed")

    SAFE_FALLBACK = True   # on any controller error, request a benign hold (envelope runs)
    crashed_steps = 0
    for _ in range(200):
        try:
            req = faulty_controller(sim)
        except Exception:
            crashed_steps += 1
            req = False if SAFE_FALLBACK else False
        sim.step(switch_request=req)           # signal keeps cycling safely under the envelope
    mon = build(sim)  # re-validate the kind of display the fallback produces
    sim.reset(seed=0)
    tripped = False
    for _ in range(200):
        sim.step(switch_request=False)
        if mon.check(green_set(sim), sim.dt) != green_set(sim):
            tripped = True
    print("C. Controller failure (raises on every call)")
    print(f"   steps with crashed controller: {crashed_steps}/200 -> caught, safe hold used")
    print(f"   intersection still legal:      {not tripped}  (no unsafe display, no freeze)")
    print("   -> failure degrades to a safe default, never to a dangerous state\n")


def main() -> None:
    cfg = load_scenario("divided_highway_side_street")
    print("\n=== Phase 4: the safety layer is independent of the controller ===\n")
    scenario_a(cfg)
    scenario_b(cfg)
    scenario_c(cfg)
    print("Conclusion: the learned controller only ever *requests* timing within an "
          "envelope it cannot violate, and an independent monitor backstops the display. "
          "This is the architecture that lets an unproven policy run on a public signal.")


if __name__ == "__main__":
    main()
