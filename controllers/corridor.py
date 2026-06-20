"""Corridor (multi-intersection) signal controllers.

These use a different interface from the single-intersection controllers: each decision
returns a list of switch requests, one per intersection. `reset(sim)` is called before a
run (the fixed-time controller uses it to install signal offsets).
"""

from __future__ import annotations

from sim.signal import GREEN


class CorridorController:
    name = "base"

    def reset(self, sim) -> None:  # noqa: ARG002
        pass

    def act(self, sim) -> list[bool]:
        raise NotImplementedError


class FixedTimeCorridor(CorridorController):
    """Fixed cycle at every intersection. With `coordinate=True`, each intersection's
    cycle is offset by the arterial travel time from the corridor start, so an
    eastbound platoon meets a green at each light — a classic green wave. With
    `coordinate=False`, every intersection runs the same cycle in lock-step (offset 0),
    which is the worst case: a platoon released by one light hits the next one red."""

    def __init__(self, green_art: float, green_cross: float, coordinate: bool):
        self.green_art = green_art
        self.green_cross = green_cross
        self.coordinate = coordinate
        self.name = "coordinated" if coordinate else "uncoordinated"

    def reset(self, sim) -> None:
        cycle = (self.green_art + self.green_cross
                 + 2 * sim.yellow + 2 * sim.allred)
        for i, sig in enumerate(sim.signals):
            if self.coordinate:
                # roll the signal so its arterial green *begins* when an EB platoon,
                # released at the corridor entry, arrives at this intersection.
                pre_roll = (-(sim.xpos[i] / sim.v_art)) % cycle
            else:
                pre_roll = 0.0
            for _ in range(int(round(pre_roll / sim.dt))):
                target = self.green_art if sig.phase == 0 else self.green_cross
                sig.step(sim.dt, sig.elapsed_green >= target)

    def act(self, sim) -> list[bool]:
        reqs = []
        for sig in sim.signals:
            target = self.green_art if sig.phase == 0 else self.green_cross
            reqs.append(sig.elapsed_green >= target)
        return reqs


class CoordinatedAdaptive(CorridorController):
    """The "needs both" controller: keep the green-wave offsets for the arterial, but
    run the cross street actuated. Each intersection has a coordinated arterial-green
    band anchored to the offset (so the wave is preserved); outside that band it serves
    the cross street only if cars are waiting, otherwise it leaves the arterial green.

    NOTE (journal F11): in testing this hand-built combination *underperformed* pure
    coordinated fixed-time — the dynamic cross switching blurs the clean periodic wave,
    costing more than the actuation saves. For steady demand, tuning the fixed split
    (see FixedTimeCorridor with a shorter cross green) beat this. Kept as a documented
    negative result and a motivation for learning the coordination."""

    name = "coordinated_adaptive"

    def __init__(self, green_art: float, green_cross: float):
        self.green_art = green_art
        self.green_cross = green_cross

    def reset(self, sim) -> None:
        self.clear = sim.yellow + sim.allred
        self.cycle = self.green_art + self.green_cross + 2 * self.clear
        # arterial green should *begin* when an EB platoon reaches this intersection
        self.offset = [(sim.xpos[i] / sim.v_art) % self.cycle for i in range(sim.K)]
        self.last_cycle = [None] * sim.K   # detect cycle rollovers
        self.served = [False] * sim.K      # cross already served this cycle?

    def act(self, sim) -> list[bool]:
        reqs = []
        for i, sig in enumerate(sim.signals):
            phase_t = sim.t - self.offset[i]
            tau = phase_t % self.cycle                 # position within the cycle
            cyc = int(phase_t // self.cycle)
            if cyc != self.last_cycle[i]:              # new cycle -> cross may serve once
                self.last_cycle[i] = cyc
                self.served[i] = False

            # arterial band (protected for the green wave) runs while tau is small;
            # the rest of the cycle is an actuated cross window, served at most once.
            in_art_band = tau < self.green_art
            if in_art_band:
                desired = 0                            # protect the coordinated band
            elif not self.served[i] and sim.cross_queue(i) > 0:
                desired = 1                            # serve waiting cross traffic
            else:
                desired = 0                            # rest in arterial green (wave)

            # once the cross queue is cleared, don't re-open it this cycle
            if sig.phase == 1 and sim.cross_queue(i) == 0:
                self.served[i] = True
            reqs.append(sig.can_switch and sig.phase != desired)
        return reqs


class LearnedCorridor(CorridorController):
    """Wraps a trained shared multi-agent policy: each intersection greedily picks
    hold/switch from its own local observation (see envs.corridor_env.agent_obs)."""

    name = "learned"

    def __init__(self, agent):
        self.agent = agent  # a trained controllers.dqn.DQN

    def act(self, sim) -> list[bool]:
        from envs.corridor_env import agent_obs
        return [bool(self.agent._greedy(agent_obs(sim, i))) for i in range(sim.K)]


class AnticipatoryCorridor(CorridorController):
    """Reactive look-ahead control: each light watches `horizon` metres of the arterial
    and tries to be green when a platoon arrives, serving the cross street only in the
    gaps between platoons. With a short horizon it cannot see a platoon coming from the
    upstream light until it is nearly at the line (too late to switch); with a long,
    camera-scale horizon it can — producing a green wave *reactively*, with no
    precomputed offsets. Sweeping the horizon isolates the value of seeing farther."""

    def __init__(self, horizon: float, gap_headway: float = 4.0, name: str | None = None):
        self.horizon = horizon
        self.gap = gap_headway          # arterial time-gap that counts as "a gap" for cross
        self.name = name or f"anticipatory_{int(horizon)}m"

    def act(self, sim) -> list[bool]:
        reqs = []
        clearance = sim.yellow + sim.allred
        for i, sig in enumerate(sim.signals):
            if sig.state != GREEN or not sig.can_switch:
                reqs.append(False)
                continue
            if sig.must_switch:
                reqs.append(True)
                continue
            eta = sim.arterial_eta(i, self.horizon)
            cross = sim.cross_queue(i)
            if sig.phase == 0:                       # arterial currently green
                if eta <= self.gap:
                    switch = False                   # platoon still flowing -> hold green
                elif cross > 0:
                    switch = True                    # gap in arterial + cross waiting -> serve cross
                else:
                    switch = False                   # rest in arterial green
            else:                                    # cross currently green
                if eta <= clearance + 1.0:
                    switch = True                    # platoon approaching -> ready the arterial
                elif cross == 0:
                    switch = True                    # cross cleared -> default back to arterial
                else:
                    switch = False                   # keep serving cross in the arterial gap
            reqs.append(switch)
        return reqs


class IndependentMaxPressure(CorridorController):
    """Each intersection independently keeps green on the higher-pressure phase
    (arterial approaching demand vs. cross-street queue). Responsive, but with no
    coordination between lights — the test of whether local adaptivity alone can
    produce corridor progression."""

    name = "independent_mp"

    def act(self, sim) -> list[bool]:
        reqs = []
        for i, sig in enumerate(sim.signals):
            art = sim.arterial_demand(i)
            cross = sim.cross_queue(i)
            desired = 0 if art >= cross else 1
            reqs.append(sig.can_switch and desired != sig.phase)
        return reqs
