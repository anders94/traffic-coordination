"""Safety layer: the inviolable envelope + independent conflict monitor that make a
learned controller deployable on public signals (Phase 4)."""

from .conflict_monitor import ConflictMonitor, conflicts_from_phases

__all__ = ["ConflictMonitor", "conflicts_from_phases"]
