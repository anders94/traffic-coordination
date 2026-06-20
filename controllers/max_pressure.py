"""Max-pressure controller — the strong theoretical baseline (the bar to beat).

Each decision, compute the "pressure" of every phase (here, the demand it would
serve, since downstream is open road) and keep green on the highest-pressure phase.
Max-pressure is provably throughput-optimal in theory and is a standard non-learning
RL baseline (RESCO). Switching is gated by the minimum green; max-green still forces
termination via the envelope.
"""

from __future__ import annotations

from .base import Controller


class MaxPressure(Controller):
    name = "max_pressure"

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        sim = env.unwrapped_sim
        sig = sim.signal
        if not sig.can_switch:
            return 0
        pressures = [sim.pressure(i) for i in range(len(sig.phases))]
        best = max(range(len(pressures)), key=lambda i: pressures[i])
        # switch only if another phase is under strictly higher pressure
        return 1 if best != sig.phase and pressures[best] > pressures[sig.phase] else 0
