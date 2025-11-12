"""Constraint surfaces with clear, typo-safe direction handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Constraint direction for comparisons.

    - ``LE``: metric must be less-than-or-equal to bound
    - ``GE``: metric must be greater-than-or-equal to bound
    """

    LE = "LE"
    GE = "GE"


@dataclass(frozen=True, slots=True)
class BoundedConstraint:
    """A simple bounded constraint on a named metric.

    Violation is always non-negative. Zero means satisfied.
    """

    name: str
    direction: Direction
    bound: float

    def violation(self, value: float) -> float:
        """Return non-negative violation amount for ``value``.

        - LE: max(0, value - bound)
        - GE: max(0, bound - value)
        """
        if self.direction is Direction.LE:
            return max(0.0, value - self.bound)
        return max(0.0, self.bound - value)

