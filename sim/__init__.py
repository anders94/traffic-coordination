"""Custom single-intersection microsimulator (M1)."""

from pathlib import Path

import yaml

from .intersection import TrafficSim, APPROACHES
from .signal import Signal, GREEN, YELLOW, ALLRED
from .vehicle import Vehicle

_SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def load_scenario(name_or_path: str) -> dict:
    """Load a scenario config by file path or by name from scenarios/."""
    p = Path(name_or_path)
    if not p.exists():
        cand = _SCENARIO_DIR / name_or_path
        p = cand if cand.exists() else _SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


__all__ = [
    "TrafficSim", "APPROACHES", "Signal", "GREEN", "YELLOW", "ALLRED",
    "Vehicle", "load_scenario",
]
