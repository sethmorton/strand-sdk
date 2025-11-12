"""Scoring Rules (surface).

Rules are simple weights or parameters used by the scoring rule to turn Metrics
into a single score. Implementations may update these over time based on
violations or other signals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class Rules:
    """Container for scoring rule parameters.

    Parameters are stored as a mapping from metric/constraint name to a float
    weight. ``update`` can adjust these values over time.
    """

    init: Mapping[str, float] = field(default_factory=dict)

    def values(self) -> Mapping[str, float]:  # pragma: no cover - surface only
        """Return a read-only view of current rule values."""
        return dict(self.init)

    def update(self, signal: Mapping[str, list[float]]) -> None:
        """Update rule values given per-name signals (e.g., violations).

        Concrete implementations will aggregate signals and adjust ``init``.
        For now, this is a no-op while we implement adaptive dual updates.
        """
        # TODO: Implement adaptive dual variable updates in Phase 5+

