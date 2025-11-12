import logging

import pytest

from strand.engine.constraints import BoundedConstraint, Direction
from strand.engine.score import _MISSING_CONSTRAINT_WARNINGS, default_score
from strand.engine.types import Metrics


@pytest.fixture(autouse=True)
def clear_missing_constraint_warnings():
    _MISSING_CONSTRAINT_WARNINGS.clear()
    yield
    _MISSING_CONSTRAINT_WARNINGS.clear()


def test_default_score_penalizes_violations():
    metrics = Metrics(objective=10.0, constraints={"length": 7.0}, aux={})
    constraint = BoundedConstraint(name="length", direction=Direction.LE, bound=5.0)

    score = default_score(metrics, {"length": 2.0}, [constraint])
    assert score == pytest.approx(6.0)


def test_default_score_warns_once_for_missing_constraint(caplog: pytest.LogCaptureFixture):
    metrics = Metrics(objective=5.0, constraints={}, aux={})
    constraint = BoundedConstraint(name="missing", direction=Direction.LE, bound=1.0)

    with caplog.at_level(logging.WARNING):
        score = default_score(metrics, {}, [constraint])
        assert score == pytest.approx(5.0)

    # Warning should only appear once even if called repeatedly
    with caplog.at_level(logging.WARNING):
        default_score(metrics, {}, [constraint])

    warnings = [rec.message for rec in caplog.records if "missing" in rec.message]
    assert len(warnings) == 1
