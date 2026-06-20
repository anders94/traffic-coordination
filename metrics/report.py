"""Aggregate per-seed metric dicts and render a comparison table.

The headline objective (plan) is *total vehicle-delay with a fairness cap*: lower
total delay is better, but a run that violates the per-approach max-wait cap is a
fairness failure regardless of its delay.
"""

from __future__ import annotations

import statistics as st


def aggregate(runs: list[dict]) -> dict:
    """Mean (and stdev) of a list of per-seed metric dicts."""
    keys = ["total_delay_veh_s", "mean_delay_per_veh", "throughput_veh_per_h",
            "max_wait_s", "fairness_excess_veh_s"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = st.mean(vals)
        out[k + "_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
    out["fairness_violation_rate"] = st.mean(
        [1.0 if r["fairness_violation"] else 0.0 for r in runs]
    )
    out["n_seeds"] = len(runs)
    return out


def format_table(results: dict[str, dict], baseline: str | None = None) -> str:
    """results: {controller_name: aggregated_metrics}. Renders a fixed-width table."""
    header = (f"{'controller':<14}{'total delay':>14}{'mean/veh':>11}"
              f"{'thru/h':>10}{'max wait':>11}{'fair viol':>11}{'vs base':>9}")
    lines = [header, "-" * len(header)]
    base_delay = results[baseline]["total_delay_veh_s"] if baseline else None
    for name, m in results.items():
        vs = ""
        if base_delay:
            pct = (m["total_delay_veh_s"] - base_delay) / base_delay * 100.0
            vs = f"{pct:+.1f}%"
        lines.append(
            f"{name:<14}"
            f"{m['total_delay_veh_s']:>14,.0f}"
            f"{m['mean_delay_per_veh']:>11.1f}"
            f"{m['throughput_veh_per_h']:>10.0f}"
            f"{m['max_wait_s']:>11.1f}"
            f"{m['fairness_violation_rate']*100:>10.0f}%"
            f"{vs:>9}"
        )
    return "\n".join(lines)
