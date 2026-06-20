# Retiming Traffic Lights — A Research Journal

A running log of what we try and what we learn, building toward a writeup on
**strategies for applying better signal timing**. This is a journal of discovery,
not a finished post — entries are dated and additive. The eventual article follows
the project phases below as its sections.

**Phases (article sections):**
1. Single intersection — can a controller beat conventional timing on one light?
2. Realism & coordination — corridors and the "green wave."
3. Perception — what cameras add over loops.
4. Interoperability & safety — getting a learned policy onto real hardware.
5. Pilot — shadow-mode on a live intersection.

The thesis we are testing: *most of the everyday misery at signalized intersections
is bad timing, and a large fraction of it can be recovered just by retiming —
without new roads, without new lanes.*

---

## Grounding (what the literature already establishes)

Before writing any code we checked our assumptions against primary sources (FHWA
Signal Timing Manual & Detector Handbook, Caltrans CA10-0823, VTRC InSync evaluation,
plus the RL literature: RESCO, CoLight).

- **How signals actually decide.** Actuated controllers extend a green while vehicles
  keep arriving and terminate it on **gap-out** (a gap longer than the passage time)
  or **max-out** (hitting the maximum green), bounded below by a **minimum green** and
  separated by fixed **yellow + all-red** clearance. These bounds are safety
  constraints, not tuning knobs.
- **The objective is old.** Minimizing total vehicle delay is Webster-era theory. Our
  framing of "sum of every car's wasted time" is exactly aggregate delay.
- **The known failure mode is fairness.** Every adaptive system studied improves the
  main road partly by **pushing delay onto the side street and left turns**. Any honest
  objective has to bound the worst-case wait, not just the average.
- **The category is not new.** Camera/AI adaptive control already ships (NoTraffic,
  InSync, Surtrac). The opportunity is not novelty; it is *how much timing headroom
  actually exists and how cheaply it can be captured.*
- **Realistic ceiling.** Gains concentrate on moderately congested, poorly-tuned,
  under-saturated corridors (~10%, more if timing is badly out of date) and **vanish
  or go negative under oversaturation** — you cannot hand out green that doesn't exist.
- **The deployment gap is the real story.** After ~20 years, adaptive control runs
  <1% of US signals. Strong simulation results rarely survive to the field. We should
  expect our biggest risk to be sim-to-real, not algorithmic.

---

## Phase 1 — Single intersection

*Entry — 2026-06-19*

**Setup.** A small microsimulator of one intersection: a divided highway (2 lanes
each way, ~45 mph) crossed by a slower single-lane side street (~25 mph) — the everyday
case where a busy, fast road is repeatedly stopped for an almost-empty one. Poisson
arrivals (≈800 veh/h per highway approach, ≈60 on the side street), realistic
car-following with start-up lost time and comfortable acceleration/braking, and a
signal that enforces minimum/maximum green and yellow/all-red clearance.

**The objective we optimize.** Per step, reward = −(total vehicle-delay) with a
penalty whenever any approach's wait exceeds a fairness cap (90 s). Minimizing the
return minimizes aggregate delay *subject to nobody being starved.*

**Strategies compared (the "controllers"):**
- **Fixed-time** — a static split, cycling regardless of who is actually there. The
  status quo at many intersections.
- **Actuated** — the real NEMA logic: extend the green on demand, gap-out / max-out.
- **Max-pressure** — a provably throughput-optimal rule; keep green on the movement
  with the most pressure. The strong baseline to beat.
- **Learned (DQN)** — a controller given *no timing rules at all*, only the reward.
  It learns when to hold and when to switch from experience.

**Findings (8 seeds, 1-hour episodes):**

| Strategy | Total delay (veh·s) | Max wait (s) | Cap breaches | vs fixed-time |
|---|---|---|---|---|
| Fixed-time | 35,931 | 41 | 0% | — |
| Actuated | 16,443 | 63 | 0% | −54% |
| Max-pressure | 15,860 | 82 | 12% | −56% |
| Learned (DQN) | 15,017 | 68 | 0% | **−58%** |

1. **The single biggest lever is simply being demand-responsive.** Going from a static
   split to *any* sensible adaptive rule cut total delay by ~55%. The villain in the
   motivating case isn't subtle — it's a fixed split spending equal green on a road
   carrying ~13× the traffic. This matches the intuition that drove the project: a lone
   side-street car should not stop a full highway.
2. **Beating a good adaptive baseline is hard, and the margin is small.** The learned
   controller edged out actuated and max-pressure, but only by ~1–2 points. This is the
   honest shape of the problem and consistent with the literature: once you're
   demand-responsive, the remaining headroom on a single isolated light is modest.
3. **The learned policy rediscovered proportional green allocation.** With no
   instruction, it settled on giving the highway ~91% of green time and the side street
   ~9% — almost exactly the demand ratio — and learned to terminate the green about when
   the last vehicle clears rather than holding for an arbitrary fixed time.
4. **Fairness is a real tension, and only a *tunable* objective can guarantee it.**
   Max-pressure — provably throughput-optimal — still let the side street wait past the
   cap on ~12% of seeds (the textbook starvation failure, reproduced in our own sim),
   and there is no knob to stop it: it is a fixed rule. The learned controller, by
   contrast, has a fairness penalty we can weight up; doing so drove cap breaches to
   **0% on every seed while it remained the lowest-delay controller**. *Takeaways:
   "minimize total time" alone is not a safe objective — it needs a worst-case wait
   bound; and a learnable objective can honor delay and fairness simultaneously where a
   fixed optimal rule cannot.*
5. **Lost time matters to timing.** Modeling realistic start-up *and* acceleration/
   deceleration (cars launch in a wave and brake over distance, rather than snapping
   between full speed and stopped) raised everyone's absolute delay and shrank the gain
   over fixed-time from ~68% to ~58%. Crucially, this lost time is a roughly fixed tax
   per phase change, so very short greens are disproportionately wasteful — which argues
   against switching too eagerly. Responsiveness and stability genuinely trade off, and a
   good controller has to find the balance rather than just reacting to the latest car.
6. **A faster main road makes the minor road harder to serve fairly.** When we gave the
   highway a higher speed than the side street, the starvation tension sharpened
   markedly: the fast highway clears quickly and keeps drawing green, and a slow
   side-street car can sit through more than one highway green. Holding fairness required
   a noticeably stronger penalty than in the equal-speed case. Speed asymmetry — not just
   volume asymmetry — is a first-class driver of unfairness.

**Open question carried into Phase 2.** A single light *cannot* deliver the experience
the project is really about — "drive all the way through without being stopped." That
is a **coordination** problem across consecutive lights (the green wave), not a
single-intersection one. Phase 2 moves to a corridor and a higher-fidelity simulator to
test whether retiming can produce progression, and to measure where gains evaporate
(saturation).

---

## Phase 2 — Coordination & the green wave

*Entry — 2026-06-19*

**Note on tooling.** The plan called for porting to SUMO here. We judged the SUMO
binary + TraCI plumbing to be high-friction for the question we actually wanted to
answer — *does coordinating signal offsets produce a green wave?* — so we extended our
own corridor simulator instead (fast, self-contained, same accel/decel/start-up model
as M1). SUMO validation for external credibility remains a deferred step.

**Setup.** A 5-intersection arterial, lights 400 m apart, two-way through traffic
(eastbound 700 veh/h, westbound 500) plus a cross street at each light (120 veh/h).
We measure **stops per through-trip** (across all 5 lights), arterial travel time vs.
the 141 s free-flow ideal, and total delay.

**Strategies compared:**
- **Uncoordinated fixed-time** — every light runs the same cycle in lock-step (offset 0).
- **Coordinated fixed-time** — offsets set to the arterial travel time (a green wave).
- **Independent max-pressure** — each light adapts locally, with no coordination.
- **Coordinated-adaptive** — green-wave offsets + an actuated cross street (the
  hand-built "best of both" attempt).

**Findings (5 seeds).** Two cross-green settings: the original 15 s, and a 8 s split
tuned to the light cross demand.

| Strategy | EB stops | WB stops | Travel (s) | Total delay |
|---|---|---|---|---|
| Uncoordinated (cross 15 s) | 3.9 | 3.2 | 210 | 50,072 |
| Coordinated (cross 15 s) | 1.5 | 3.4 | 164 | 19,930 |
| Independent max-pressure | 2.0 | 2.0 | 168 | 18,709 |
| Coordinated-adaptive (hand) | 1.9 | 2.3 | 168 | 27,751 |
| **Coordinated, split-tuned (cross 8 s)** | **1.4** | **1.3** | **161** | **18,909** |

1. **Coordination is a distinct, powerful lever — and it's the one this project is really
   about.** Offsets set to the travel time cut eastbound stops ~3.9 → ~1.5 and pulled
   arterial travel time to within ~15% of the free-flow ideal, versus lock-step; total
   delay more than halved. *No single-intersection cleverness produces this; it is purely
   a between-lights timing relationship.*
2. **Split tuning is the surprise lever, and it dissolves the two-way tradeoff.** With a
   15 s cross green, the green wave helped EB but hurt WB (3.2 → 3.4 stops) — the classic
   "can't wave both ways." But the cross street only had ~2 cars per cycle; cutting its
   green to 8 s (matched to demand) widened the arterial band enough that *both*
   directions rode the wave (WB collapsed 3.4 → 1.3) **and** total delay fell to tie the
   adaptive controller. For steady demand, a well-tuned fixed coordinated plan was the
   best strategy we found — beating clever dynamics.
3. **Adaptivity is not coordination — and naively combining them is hard.** Independent
   max-pressure minimized delay but produced no wave. Our hand-built coordinated-adaptive
   controller (offsets + actuated cross) *underperformed pure coordination on both stops
   and delay*: every dynamic deviation blurred the clean periodic wave, costing more than
   the actuation saved. Getting coordinated-actuated control right by hand is genuinely
   hard — which is exactly the case for *learning* the coordination (the next experiment),
   and a reminder that under steady demand, tuning the plan beats adding dynamics.
4. **"Without stopping" is approached, not achieved.** Even under a green wave, ~0% of
   trips were truly stop-free: random arrivals and a standing queue at the first light
   mean almost everyone stops at least once. The honest claim is *near-free-flow travel
   with far fewer, shorter stops* — not zero stops for everyone.

**Saturation sweep (and an honest limitation).** Sweeping eastbound demand from 500 to
1500 veh/h, the green-wave advantage did *not* collapse — it grew (stop reduction
61%→67%, delay reduction 47%→76%). This **contradicts the well-established field result
that coordination gains vanish near saturation**, and the discrepancy is instructive:
(a) even 1500 veh/h is still below the arterial's ~2000 veh/h capacity, so we never
truly oversaturated; and (b) our corridor sim drops vehicles that can't enter rather
than modeling queue *spillback* (a downstream queue backing through an upstream
intersection and gridlocking it) — which is precisely the mechanism that destroys
coordination benefits in real oversaturation. *Takeaway: our sim is trustworthy in the
under-saturated regime where retiming actually helps, but cannot model the gridlock
regime — a concrete reason the deferred SUMO validation matters.*

**Learned multi-agent coordination (the headline experiment).** We built a learned
controller the modern way: one shared policy (a multi-agent DQN with parameter sharing),
one agent per intersection, each seeing only local state plus an **upstream look-ahead**
(approaching traffic bucketed by distance) and its neighbours' arterial-green status —
features chosen so coordination *could* emerge. Reward is local (minus the vehicles
queued at that intersection). We trained both a steady-demand specialist and a
time-varying generalist.

It learned a sensible policy — comparable to hand-tuned max-pressure, and roughly half
the stops of the naive baseline — but **it did not discover the green wave, and did not
beat the well-tuned coordinated fixed plan** on either steady or time-varying demand:

| Strategy | EB stops | WB stops | Total delay | (demand) |
|---|---|---|---|---|
| Coordinated, split-tuned | 1.4 | 1.3 | 18,909 | steady |
| Independent max-pressure | 2.0 | 2.0 | 18,709 | steady |
| Learned (steady specialist) | 2.0 | 2.9 | 19,604 | steady |
| Coordinated, split-tuned | 1.4 | 1.2 | 20,275 | varying |
| Learned (varying generalist) | 2.4 | 2.7 | 23,939 | varying |

The learned policy minimizes local queues — essentially rediscovering max-pressure —
but never learns the *temporal offset structure* that makes a green wave. Even on
time-varying demand, where we expected adaptivity to pay off, the fixed coordinated
plan with a demand-matched split won.

*Caveats (kept honest):* this is a vanilla multi-agent DQN with a local-queue reward,
not the published state of the art (e.g., CoLight's graph-attention); a more
sophisticated design and reward (explicit progression/stop penalties, shared reward)
might coordinate better. But the result squarely reproduces the real-world picture our
background research surfaced: **strong-in-simulation RL has not been shown to beat a
well-maintained coordinated system (SCATS/SCOOT) in the field.** The robust, dominant,
and far *simpler* levers remain coordination (offsets) + split tuning. Learning's most
promising role may be **maintaining and adapting that tuning over time** and exploiting
richer data — not real-time RL control of the phases.

**Next in Phase 2 (optional):** a spillback-complete model (or SUMO) to test the
saturation regime properly; a stronger RL design (graph attention, progression reward)
if we want to push the learned-control question further.

## Phase 3 — Perception (what a camera adds over a loop)

*Entry — 2026-06-19*

**Hypothesis under test (from the project owner):** noticing cars *before* they reach
the intersection should let the controller make the light green by the time they arrive,
and avoid ending a green prematurely when a trailing car is still coming — with *extreme*
gains in light traffic, where the penalty for "missing a light" (waiting out a red with
no opposing traffic) is high.

**Method.** Hold one anticipatory control law fixed and vary *only its sensing horizon*:
15 m (a stop-bar loop — presence at the line), 60 m (an advance loop), 200 m (a camera
watching the whole approach). The policy rests in green (never stops anyone for an empty
cross street), holds green for an imminent vehicle, and switches early when its approach
is clear. By construction, any performance difference is the value of the perception.
Swept across traffic from very light to busy.

**Findings (5 seeds, single intersection — mean delay per vehicle, s):**

| Light traffic (120/9 vph) | delay/veh |
|---|---|
| Fixed-time (status quo) | 18.0 |
| Actuated, loop gap-out (typical) | 9.7 |
| Anticipatory, 15 m loop sight | 7.8 |
| Anticipatory, 200 m camera sight | **6.8** |

1. **The hypothesis is directionally right — anticipatory camera control gives big
   light-traffic gains versus the status quo:** ~30% lower delay than realistic
   loop-actuated control and ~62% lower than fixed-time. The behaviors that drive it are
   exactly the ones predicted: rest in green (don't stop for an empty cross street) and
   be green by the time the car arrives.
2. **But the gain mostly comes from the *policy*, not the *sight distance*.** Decomposing
   the light-traffic numbers: being responsive at all (fixed → actuated) is −46%; the
   smart rest-in-green + anticipate policy (actuated → 15 m anticipatory) is another −20%;
   and extending sight from a 15 m loop to a 200 m camera, *holding the policy fixed*, is
   only a further ~9–13% (and most of that is already captured by a 60 m advance loop).
3. **Why the extra range buys so little here (honest mechanics).** A 60 m advance loop is
   already enough to anticipate the *slow* side street (60 m ÷ 11 m/s ≈ 5.5 s ≈ the
   clearance time). And the *fast* main road — the one whose 200 m of foresight a loop
   can't match — rarely faces a red at all, because the controller rests it in green. So
   the camera's theoretical edge (anticipating fast approaches) is seldom exercised at an
   isolated, demand-asymmetric intersection.
4. **Where the camera's value actually lives.** (a) It *enables* the smart anticipatory
   policy in the first place — positions and speeds, not just presence; loops give you the
   inputs only if they're well-placed and plentiful. (b) Range matters when approaches are
   fast *and* contended, or where good loops can't be installed. (c) The decisive
   advantages are *qualitative and beyond a delay metric* — turning-movement counts,
   emergency-vehicle preemption, pedestrians, and anomaly detection — which loops cannot
   provide at all and which are central to this project's premise. (d) Corridors may
   amplify look-ahead (a missed light cascades) — worth testing next.

**Does look-ahead matter more on a corridor?** We expected it might — a missed light
cascades. We swept the sight horizon of a *reactive* anticipatory controller (be green
when a platoon arrives, serve the cross street in the gaps) on the 5-light arterial:

| Controller | EB stops | WB stops | Total delay |
|---|---|---|---|
| Anticipatory, 40 m sight | 2.55 | 2.57 | 22,844 |
| Anticipatory, 100 m sight | 2.32 | 2.34 | 20,747 |
| Anticipatory, 200 m sight | 2.32 | 2.34 | 20,747 |
| Anticipatory, 400 m sight | 2.32 | 2.34 | 20,747 |
| Green wave (explicit offsets) | **1.37** | **1.28** | **18,909** |

Two sharp results: (i) **look-ahead value saturates at ~100 m** — 100, 200 and 400 m are
*identical* — because a reactive switch decision only needs about one clearance-distance
of foresight (clearance 5 s × 20 m/s ≈ 100 m); a car seen farther than that changes no
decision, so a camera's extra range is literally *unused* by reactive control. (ii)
**reactive look-ahead does not reproduce the green wave** — even with full-link sight it
sits at ~2.3 stops, well short of explicit offset coordination's ~1.4. Seeing the platoon
coming is not the same as having the whole corridor phased for it.

*Unified takeaway (Phase 3):* the owner's instinct — "stop making cars wait at empty
intersections, be green when they arrive" — is correct and worth real gains versus the
status quo (~30–60% in light traffic), but those gains are a *control-policy* win that
even short-range detection unlocks. Long-range camera sight adds little to *reactive*
switching (it saturates at one clearance-distance). The camera's long range pays off only
when used for *planning* — coordination and prediction across cycles, the dominant lever —
and for the qualitative jobs loops can't do at all (turning counts, emergency preemption,
pedestrians, anomalies). Perception enables good control; it doesn't substitute for it.

## Phase 4 — Interoperability & safety
*(not yet started)*

Planned: the safety envelope (min/max green, clearance, conflict monitor) as the
inviolable layer a learned policy must live inside; NTCIP/NEMA interfaces; the
standards/liability reasons adoption sits below 1%.

## Phase 5 — Pilot
*(not yet started)*

Planned: shadow mode (recommend, don't actuate) on a live intersection, with a proven
fixed-time fallback.

---

## Running list of findings (the spine of the article)

- F1 — The dominant win is demand-responsiveness itself; static splits on asymmetric
  demand are the core problem. (~−55% just from adapting.)
- F2 — Once adaptive, additional gains on a single light are small; strong baselines
  (max-pressure) are hard to beat. Be skeptical of large single-intersection claims.
- F3 — A learned controller, given only the objective, rediscovers proportional green
  and last-car-through termination — no hand timing needed.
- F4 — "Minimize total delay" must be paired with a worst-case wait bound, or it
  starves minor approaches. A fixed optimal rule (max-pressure) can't guarantee the
  bound; a learnable objective can be weighted to honor delay and fairness at once.
- F5 — Lost time (start-up + accel/decel) is a fixed tax per phase change, penalizing
  over-eager switching; responsiveness and stability trade off.
- F6 — Speed asymmetry (a fast main road, slow side street), not just volume asymmetry,
  sharpens starvation: the fast road keeps drawing green while slow minor-road cars wait.
- F7 — Coordination (signal offsets) is a separate, powerful lever from single-light
  responsiveness: a green wave roughly halved arterial stops and delay vs lock-step.
- F8 — The two-way green-wave tradeoff is real at a fixed split but largely *dissolves*
  once the split is tuned: a wider arterial band lets both directions ride the wave.
- F9 — Adaptivity ≠ coordination: local adaptive control minimizes delay but won't
  create progression.
- F10 — Split tuning (matching cross green to actual cross demand) is itself a major
  lever; for steady demand a well-tuned fixed coordinated plan beat every dynamic scheme
  we tried.
- F11 — Hand-combining coordination + actuation underperformed pure coordination —
  dynamic deviations blur the clean wave. Doing it well is hard, motivating *learned*
  coordination.
- F12 — A learned multi-agent controller rediscovered max-pressure but NOT the green
  wave, and did not beat a well-tuned coordinated fixed plan (steady or varying demand) —
  reproducing the real-world result that simulation-strong RL hasn't beaten well-tuned
  coordinated control. Coordination + split tuning is the robust, simpler winner;
  learning's better role is likely adapting the tuning over time, not real-time control.
- F13 — Anticipatory "rest-in-green + be-green-on-arrival" control gives large
  light-traffic gains vs the status quo (~30% vs loop-actuated, ~62% vs fixed-time) — the
  owner's "don't wait at empty intersections" instinct is correct and valuable.
- F14 — But that win is mostly the control *policy*, not camera *sight distance*: holding
  the policy fixed, extending 15 m → 200 m of vision adds only ~10% at an isolated
  intersection (a 60 m advance loop already anticipates the slow cross street; the fast
  main road rests in green and rarely faces red). The camera's real value is enabling the
  smart policy and the capabilities loops lack (turning counts, emergency preemption,
  pedestrians, anomalies) — not raw range at a single light.
- F15 — On a corridor, reactive look-ahead value *saturates at ~one clearance-distance*
  (~100 m): 100/200/400 m of sight give identical results, because a reactive switch only
  needs enough foresight to clear before arrival. And reactive look-ahead does NOT
  reproduce the green wave (~2.3 stops vs ~1.4 for explicit offsets). Long camera range
  pays off only when used for *planning/coordination*, not reactive switching.
- (more as phases progress)
