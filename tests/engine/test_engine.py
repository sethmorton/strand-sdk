import math

import pytest

from strand.core.sequence import Sequence
from strand.engine import Engine, EngineConfig, default_score
from strand.engine.constraints import BoundedConstraint, Direction
from strand.engine.executors.local import LocalExecutor
from strand.engine.rules import Rules
from strand.engine.strategies.random import RandomStrategy
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


class TestEngineWithRandomStrategy:
    """Integration tests for Engine with RandomStrategy."""

    def _build_engine(self, *, iterations: int = 2, population: int = 6) -> Engine:
        rewards = [RewardBlock.stability(weight=1.0)]
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator, batch_size=population)
        strategy = RandomStrategy(
            alphabet="ACDE",
            min_len=5,
            max_len=10,
            seed=42,
        )
        config = EngineConfig(iterations=iterations, population_size=population, seed=42)
        return Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

    def test_minimal_loop(self):
        """Engine.run returns best candidate, history, and summary."""

        engine = self._build_engine(iterations=3, population=4)
        results = engine.run()

        assert results.best is not None
        assert isinstance(results.best[0], Sequence)
        assert isinstance(results.best[1], float)
        assert len(results.history) == 3
        assert results.summary["total_evals"] == 12
        assert results.summary["iterations_completed"] == 3
        assert results.summary["stopped_early"] is False
        assert math.isfinite(results.summary["best_score"])

        for idx, stats in enumerate(results.history):
            assert stats.iteration == idx
            assert stats.evals == 4
            assert stats.rules == {}
            assert stats.violations == {}
            assert math.isfinite(stats.best)
            assert math.isfinite(stats.mean)
            assert stats.throughput > 0

    def test_stream_respects_max_evals_and_rules(self):
        """Engine.stream stops once max_evals is reached and snapshots rules."""

        rewards = [RewardBlock.stability(weight=1.0)]
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator, batch_size=4)
        strategy = RandomStrategy(alphabet="AC", min_len=4, max_len=4, seed=123)
        config = EngineConfig(
            iterations=10,
            population_size=4,
            max_evals=4,
            early_stop_patience=None,
        )
        rules = Rules(init={"stability": 0.5})
        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
            rules=rules,
        )

        stats_list = list(engine.stream())
        assert len(stats_list) == 1
        stats = stats_list[0]
        assert stats.rules == {"stability": 0.5}
        assert stats.evals == 4
        assert stats.timeouts == 0
        assert stats.errors == 0

    def test_constraints_are_penalized(self, caplog: pytest.LogCaptureFixture):
        """Constraints missing in metrics warn once and contribute zero violation."""

        rewards = [RewardBlock.stability(weight=1.0)]
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator, batch_size=2)
        strategy = RandomStrategy(alphabet="AC", min_len=3, max_len=3, seed=321)
        config = EngineConfig(iterations=1, population_size=2)
        constraint = BoundedConstraint(name="foo", direction=Direction.LE, bound=0.1)
        rules = Rules(init={"foo": 2.0})

        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
            constraints=[constraint],
            rules=rules,
        )

        with caplog.at_level("WARNING"):
            results = engine.run()

        assert results.history[0].violations == {"foo": 0.0}
        warnings = [rec.message for rec in caplog.records if "Constraint" in rec.message]
        assert len(warnings) == 1
