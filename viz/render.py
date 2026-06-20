"""Render a rollout to a GIF (headless-friendly) for the eyeball check in the plan's
verification step: with a near-empty side street the highway should stay green for
long stretches and the side street should get just enough green to clear.

Usage:
    python -m viz.render --controller max_pressure --seconds 300 --out runs/sim.gif
"""

from __future__ import annotations

import argparse

from controllers import BASELINES, DQN
from envs import IntersectionEnv
from sim import load_scenario
from sim.signal import GREEN, YELLOW

# screen-space direction each approach feeds *from* (dx, dy point toward the centre)
_DIR = {"N": (0, -1), "S": (0, 1), "E": (-1, 0), "W": (1, 0)}
_LANE_OFFSET = {"N": (1, 0), "S": (-1, 0), "E": (0, -1), "W": (0, 1)}


def render_episode(config: dict, controller, seconds: float, out: str,
                   fps: int = 30, substeps: int = 5, warmup: float = 45.0,
                   loop_extra: float = 40.0) -> str:
    """Render to a looping GIF of an already-active intersection.

    Each 1-second sim step is drawn as `substeps` interpolated frames so motion looks
    smooth (the simulation/metrics are unaffected). `warmup` seconds are simulated
    before recording so the road starts busy. We then record `seconds + loop_extra`
    and cut the clip at whichever later frame best matches the opening frame's signal
    phase and queues, so the loop seam is as unobtrusive as possible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    env = IntersectionEnv(config)
    sim = env.unwrapped_sim
    obs, _ = env.reset(seed=0)
    controller.reset()

    L = sim.L
    view = min(L, 70.0)  # zoom in near the junction so queues/launch waves are legible
    substeps = max(1, int(substeps))
    sub_dt = sim.dt / substeps
    lane_w = 6.0
    lanes = sim.lanes  # {approach: n_lanes}
    # draw each car at its body centre; the front bumper stops a few metres behind the
    # stop-line bar so cars don't look like they run into the light.
    stop_gap = 3.0
    draw_setback = sim.veh_len / 2.0 + stop_gap

    # road half-widths: the horizontal (E/W) road's width spans y; the vertical
    # (N/S) road's width spans x. Each direction occupies its own half.
    hw_y = max(1, lanes["E"] + lanes["W"]) * lane_w / 2.0   # horizontal road half-height
    hw_x = max(1, lanes["N"] + lanes["S"]) * lane_w / 2.0   # vertical road half-width
    # distance from centre to where each approach stops (the edge of the road it crosses)
    stop_off = {"E": hw_x, "W": hw_x, "N": hw_y, "S": hw_y}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # asphalt
    ax.add_patch(Rectangle((-L, -hw_y), 2 * L, 2 * hw_y, color="0.82", zorder=0))
    ax.add_patch(Rectangle((-hw_x, -L), 2 * hw_x, 2 * L, color="0.82", zorder=0))
    # yellow centre lines (separating opposing directions), drawn outside the box
    ax.plot([-L, -hw_x], [0, 0], color="gold", lw=1.5, zorder=1)
    ax.plot([hw_x, L], [0, 0], color="gold", lw=1.5, zorder=1)
    ax.plot([0, 0], [-L, -hw_y], color="gold", lw=1.5, zorder=1)
    ax.plot([0, 0], [hw_y, L], color="gold", lw=1.5, zorder=1)
    # white dashed lane dividers within each direction
    for k in range(1, lanes["E"]):
        ax.plot([-L, -hw_x], [-k * lane_w, -k * lane_w], "w--", lw=1, zorder=1)
    for k in range(1, lanes["W"]):
        ax.plot([hw_x, L], [k * lane_w, k * lane_w], "w--", lw=1, zorder=1)
    for k in range(1, lanes["N"]):
        ax.plot([k * lane_w, k * lane_w], [-L, -hw_y], "w--", lw=1, zorder=1)
    for k in range(1, lanes["S"]):
        ax.plot([-k * lane_w, -k * lane_w], [hw_y, L], "w--", lw=1, zorder=1)

    title = ax.set_title("")
    # cars are drawn as lane-aligned rectangles (veh_len long, car_w wide) so they
    # sit inside their lane instead of bleeding across like round markers.
    car_w = min(2.6, lane_w * 0.55)
    car_patches: list = []

    # the signal is drawn as a coloured bar across each approach's stop line, so
    # vehicles visibly stack up *behind* it instead of under a marker.
    _bar_xy = {
        "E": ([-hw_x, -hw_x], [-hw_y, 0]),
        "W": ([hw_x, hw_x], [0, hw_y]),
        "N": ([0, hw_x], [-hw_y, -hw_y]),
        "S": ([-hw_x, 0], [hw_y, hw_y]),
    }
    signal_bars = {}
    for a in sim.active_approaches:
        xs_, ys_ = _bar_xy[a]
        signal_bars[a] = ax.plot(xs_, ys_, lw=5, solid_capstyle="butt", zorder=5)[0]

    def to_xy(approach, lane, d):
        dx, dy = _DIR[approach]
        ox, oy = _LANE_OFFSET[approach]
        base = stop_off[approach] + d + draw_setback  # body centre, behind the line
        lane_center = (lane + 0.5) * lane_w
        return dx * base + ox * lane_center, dy * base + oy * lane_center

    def on_screen(x, y):
        return abs(x) <= view + 6 and abs(y) <= view + 6

    def snapshot() -> dict:
        """Current on-road vehicles: vid -> (approach, lane, d, color)."""
        snap = {}
        for a in sim.active_approaches:
            for lane in sim.vehicles[a]:
                for v in lane:
                    color = ("red" if v.wait > sim.fairness_cap
                             else "orange" if v.speed < 0.1 else "tab:blue")
                    snap[v.vid] = (v.approach, v.lane, v.d, color)
        return snap

    def record() -> dict:
        return {"veh": snapshot(), "phase": sim.signal.phase, "state": sim.signal.state,
                "delay": sim.total_delay}

    # --- pre-simulate: warm up (unrecorded) so the road is already busy, then record
    for _ in range(int(warmup / sim.dt)):
        obs, *_ = env.step(controller.act(obs, env))
    recorded = [record()]
    for _ in range(int((seconds + loop_extra) / sim.dt)):
        obs, *_ = env.step(controller.act(obs, env))
        recorded.append(record())

    # --- choose the loop end: a later state matching the opening phase + queues
    def queues(rec):
        q: dict[str, list[float]] = {a: [] for a in sim.active_approaches}
        for (a, _lane, d, _c) in rec["veh"].values():
            q[a].append(d)
        for a in q:
            q[a].sort()
        return q

    def loop_cost(a_rec, b_rec):
        if a_rec["phase"] != b_rec["phase"] or a_rec["state"] != b_rec["state"]:
            return 1e9  # different point in the cycle -> visible jump
        qa, qb = queues(a_rec), queues(b_rec)
        cost = 0.0
        for ap in sim.active_approaches:
            la, lb = qa[ap], qb[ap]
            cost += 5.0 * abs(len(la) - len(lb))           # match how many are waiting
            for i in range(min(len(la), len(lb))):         # and where they sit
                cost += 0.05 * abs(la[i] - lb[i])
        return cost

    start_idx = max(1, int(seconds / sim.dt))
    end = min(len(recorded) - 1, start_idx)
    best = loop_cost(recorded[0], recorded[end])
    for e in range(start_idx, len(recorded)):
        c = loop_cost(recorded[0], recorded[e])
        if c < best:
            best, end = c, e
    frames = end * substeps  # render macro transitions 0->1 ... (end-1)->end

    # --- play back the recorded states with interpolation + coasting ghosts
    ghosts: list[dict] = []          # {approach, lane, d}
    phases = sim.signal.phases

    def step_and_draw(frame_idx):
        macro = frame_idx // substeps
        sub = frame_idx % substeps
        prev, nxt = recorded[macro]["veh"], recorded[macro + 1]["veh"]

        if sub == 0:  # vehicles present last step but gone now (and past line) -> ghosts
            for vid, (a, lane, d, _c) in prev.items():
                if vid not in nxt and d <= 0.0:
                    ghosts.append({"approach": a, "lane": lane, "d": d})

        for gh in ghosts:
            gh["d"] -= sim.v_free[gh["approach"]] * sub_dt
        ghosts[:] = [gh for gh in ghosts
                     if on_screen(*to_xy(gh["approach"], gh["lane"], gh["d"]))]

        frac = (sub + 1) / substeps
        cars = []  # (x, y, approach, color)
        for vid, (a, lane, d, color) in nxt.items():
            d0 = prev[vid][2] if vid in prev else d  # interpolate if it existed before
            x, y = to_xy(a, lane, d0 + frac * (d - d0))
            cars.append((x, y, a, color))
        for gh in ghosts:  # departed, no longer counted -> drawn faded
            x, y = to_xy(gh["approach"], gh["lane"], gh["d"])
            cars.append((x, y, gh["approach"], "0.6"))

        for p in car_patches:
            p.remove()
        car_patches.clear()
        for x, y, a, color in cars:
            # rectangle aligned with travel direction: long along the road, narrow across
            w, h = (sim.veh_len, car_w) if a in ("E", "W") else (car_w, sim.veh_len)
            rect = Rectangle((x - w / 2.0, y - h / 2.0), w, h, color=color, zorder=3)
            ax.add_patch(rect)
            car_patches.append(rect)

        rec = recorded[macro + 1]
        for a in sim.active_approaches:
            if rec["state"] == GREEN and a in phases[rec["phase"]]:
                color = "limegreen"
            elif rec["state"] == YELLOW and a in phases[rec["phase"]]:
                color = "gold"
            else:
                color = "red"
            signal_bars[a].set_color(color)

        # delay cost accumulated over the clip (starts at 0, climbs, resets on loop)
        d0, d1 = recorded[macro]["delay"], rec["delay"]
        cost = (d0 + frac * (d1 - d0)) - recorded[0]["delay"]
        title.set_text(f"delay cost: {cost:,.0f} vehicle-seconds")
        return [*car_patches, *signal_bars.values(), title]

    anim = animation.FuncAnimation(fig, step_and_draw, frames=frames, blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="divided_highway_side_street")
    ap.add_argument("--controller", default="max_pressure",
                    choices=[*BASELINES.keys(), "dqn"])
    ap.add_argument("--dqn", default=None, help="checkpoint when --controller dqn")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", default="runs/sim.gif")
    ap.add_argument("--substeps", type=int, default=5,
                    help="interpolated frames per sim second (higher = smoother)")
    ap.add_argument("--fps", type=int, default=30, help="playback frames per second")
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="seconds simulated before recording (start with a busy road)")
    ap.add_argument("--loop-extra", type=float, default=40.0,
                    help="extra seconds recorded to search for a clean loop point")
    args = ap.parse_args()

    config = load_scenario(args.scenario)
    if args.controller == "dqn":
        controller = DQN.load(args.dqn)
    else:
        controller = BASELINES[args.controller]()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    path = render_episode(config, controller, args.seconds, args.out,
                          fps=args.fps, substeps=args.substeps,
                          warmup=args.warmup, loop_extra=args.loop_extra)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
