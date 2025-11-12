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
    step_size: float = 0.05
    clip: tuple[float, float] = (0.0, 10.0)

    def values(self) -> Mapping[str, float]:  # pragma: no cover - surface only
        """Return a read-only view of current rule values."""
        return dict(self.init)

    def update(self, signal: Mapping[str, list[float]]) -> None:
        """Adapt rule weights from per-name signals (e.g., violations).

        Simple additive update per name: ``w <- clip(w + step * mean(signal))``.
        Names not present in the current rule set are added on demand.
        """
        values = dict(self.init)
        lo, hi = self.clip
        for name, vals in signal.items():
            if not vals:
                continue
            mean_val = sum(vals) / max(len(vals), 1)
            new_val = values.get(name, 0.0) + self.step_size * mean_val
            if new_val < lo:
                new_val = lo
            elif new_val > hi:
                new_val = hi
            values[name] = new_val
        self.init = values
