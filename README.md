# Adaptive, Camera-Augmented Traffic Signal Control

Make signalized intersections more efficient purely by **re-timing the lights** — using
existing sensors plus added cameras, with controllers that **learn from the reward**
rather than being hand-tuned. This repo is a staged R&D exploration of *what
actually moves the needle* on signal timing, built on a small, transparent, self-contained
traffic simulator. It is a research journal in code, not a product.

<p align="center">
  <img src="img/phase-1.gif" alt="Phase 1: learned single-intersection control" width="420">
  <br>
  <em>Phase 1 — the learned controller holding the highway green and clearing the side
  street just in time (delay cost ticking up top).</em>
</p>

> **Framing.** Camera/AI adaptive control already ships commercially (NoTraffic,
> InSync, Surtrac), and realistic gains from re-timing are bounded (~10% on poorly-tuned,
> under-saturated corridors; ~0 or negative under saturation). The value here is not
> novelty — it is measuring, with adversarial rigor, which levers matter and which
> don't. **The full findings log is [`docs/JOURNAL.md`](docs/JOURNAL.md).**

## Status

| Phase | Question | Status |
|---|---|---|
| 1 — Single intersection | Can a learner beat conventional timing on one light? | ✅ done |
| 2 — Corridor & coordination | Does coordinating offsets produce a green wave? | ✅ done |
| 3 — Perception | What does a camera add over a loop detector? | ✅ done |
| 4 — Interoperability & safety | What makes a learning controller deployable? | ✅ done |
| 5 — Pilot | Shadow-mode on a live intersection | ⬜ planned |

## Headline findings (the through-line)

- **Being demand-responsive is the dominant single-intersection win** (~55% lower delay
  than a fixed split); beating a good adaptive baseline after that is hard.
- **"Minimize total delay" needs a worst-case-wait cap**, or it starves the side street.
  A *learnable* objective can honor delay + fairness at once; a fixed optimal rule can't.
- **Coordination (offsets) is the big corridor lever** — a green wave roughly halves
  arterial stops and delay — and good **split tuning** dissolves the two-way tradeoff.
- **Adaptivity ≠ coordination.** A hand-combined coordinated-actuated controller, *and* a
  multi-agent RL controller, both failed to beat a well-tuned coordinated fixed plan —
  reproducing the real-world result that simulation-strong RL hasn't beaten SCATS/SCOOT.
- **Camera look-ahead is mostly a control-*policy* win, not a sight-distance win.** The
  value of seeing farther saturates at ~one clearance-distance (~100 m) for reactive
  control; long range pays off only for planning/coordination and for jobs loops can't do
  (turning counts, emergency preemption, pedestrians, anomalies).
- **A two-layer safety architecture makes "let it learn" deployable**: the policy can only
  *request* timing within an envelope it cannot violate, and an independent conflict
  monitor trips to flash-red on any unsafe display. The real barriers are institutional
  (standards/procurement/liability), not algorithmic.

## What's here

| Path | Purpose |
|---|---|
| `sim/intersection.py` | Single-intersection microsim: per-approach speeds, accel/decel + start-up, per-vehicle delay |
| `sim/corridor.py` | Multi-intersection arterial corridor (two-way through traffic + cross streets) |
| `sim/signal.py` | The **proactive safety envelope** — min/max green, yellow + all-red clearance |
| `safety/conflict_monitor.py` | The **independent conflict monitor** (software MMU): trips to flash-red on unsafe displays |
| `envs/` | Gymnasium env (single light) + multi-agent corridor env, with camera-horizon look-ahead |
| `controllers/` | Baselines (`fixed_time`, `actuated`, `max_pressure`), `dqn` learner, `anticipatory`, and corridor controllers (coordinated / max-pressure / coordinated-adaptive / learned) |
| `scenarios/` | `divided_highway_side_street`, `arterial_corridor`, `arterial_corridor_varying` |
| `experiments/` | Eval + training + the per-phase studies (see below) |
| `viz/render.py` | Render a rollout to a looping GIF |
| `docs/JOURNAL.md` | The running research journal (findings F1–F17) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # baselines + evals (no torch needed)
pip install -e '.[learn,viz]'    # add DQN training + GIF rendering
```

### Phase 1 — single intersection
```bash
python -m experiments.eval --scenario divided_highway_side_street --seeds 8
python -m experiments.train --episodes 250 --out runs/dqn.pt   # learn the policy
python -m experiments.eval --dqn runs/dqn.pt --seeds 8          # include it
python -m viz.render --controller dqn --dqn runs/dqn.pt --out runs/sim.gif
```

### Phase 2 — corridor & coordination
```bash
python -m experiments.corridor_eval --seeds 5                   # green wave vs lock-step vs adaptive
python -m experiments.corridor_eval --green-cross 8 --seeds 5   # split-tuned coordinated plan
python -m experiments.corridor_train --scenario arterial_corridor_varying \
    --episodes 150 --horizon 3600 --out runs/corridor_dqn.pt    # learned multi-agent
python -m experiments.corridor_eval --learned runs/corridor_dqn.pt --green-cross 8 --seeds 5
```

### Phase 3 — perception (what a camera adds)
```bash
python -m experiments.perception_eval --seeds 5        # sight horizon × traffic, one light
python -m experiments.corridor_perception --seeds 5    # sight horizon on the corridor
```

### Phase 4 — safety
```bash
python -m experiments.safety_demo    # envelope + conflict monitor vs adversarial/faulty controllers
```

## The objective (single intersection)

Reward per step is `-(total delay this step) - beta * (fairness excess this step)`.
Minimising the return minimises aggregate vehicle-delay (Webster's classic objective,
== "sum of every car's wasted time") while a per-approach **max-wait cap** prevents the
side street from being starved — the documented failure mode of every adaptive system
studied. A run that breaches the cap is a fairness failure regardless of its delay number.

## Not a product

This is R&D. There is no field hardware, no real perception pipeline (the camera is
modeled as ground-truth look-ahead), and the corridor sim is trustworthy only in the
under-saturated regime (it does not model gridlock spillback). External validation (e.g.
SUMO) and a shadow-mode pilot are the natural next steps. See the journal for caveats.
