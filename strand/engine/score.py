"""Default scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping

from strand.engine.constraints import BoundedConstraint
from strand.engine.types import Metrics


def default_score(
    metrics: Metrics,
    duals: Mapping[str, float],
    constraints: list[BoundedConstraint],
) -> float:
    """Compute a default score from metrics and rules.

    Formula: `objective − Σ rules[name] × violation(name)`

    This penalizes constraint violations proportionally to their violation amount
    and the associated dual variable weight. Zero violation contributes zero penalty.

    Parameters
    ----------
    metrics : Metrics
        Evaluated metrics with objective and constraint measurements.
    duals : Mapping[str, float]
        Dual weights (Lagrange multipliers) keyed by constraint name.
    constraints : list[BoundedConstraint]
        Constraint definitions with bounds and directions.

    Returns
    -------
    float
        Scalar score: higher is better (objective minus penalties).
    """
    penalty = 0.0
    for constraint in constraints:
        measured_value = metrics.constraints.get(constraint.name, 0.0)
        violation = constraint.violation(measured_value)
        weight = duals.get(constraint.name, 0.0)
        penalty += weight * violation

    return metrics.objective - penalty
