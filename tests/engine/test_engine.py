"""Tests for Engine orchestration (ask/run/score/tell loop)."""

from strand.core.sequence import Sequence
from strand.engine import Engine, EngineConfig, default_score
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies.cem import CEMStrategy
from strand.engine.strategies.random import RandomStrategy
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


class TestEngineWithRandomStrategy:
    """Integration tests for Engine with RandomStrategy."""

    def test_minimal_loop(self):
        """Test a minimal end-to-end optimization loop."""
        rewards = [
            RewardBlock.stability(weight=1.0),
            RewardBlock.novelty(baseline=["ACDE"], weight=0.5),
        ]

        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator)

        strategy = RandomStrategy(
            alphabet="ACDE",
            min_len=5,
            max_len=15,
            seed=42,
        )

        config = EngineConfig(
            iterations=2,
            population_size=8,
            seed=42,
        )

        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

        results = engine.run()

        # Verify results structure
        assert results.best is not None
        assert isinstance(results.best[0], Sequence)
        assert isinstance(results.best[1], float)
        assert len(results.history) == 2
        assert results.history[0].iteration == 0
        assert results.history[1].iteration == 1
        assert results.history[0].evals == 8
        assert results.history[1].evals == 8

    def test_stream_iteration(self):
        """Test that Engine.stream() yields statistics for each iteration."""
        rewards = [RewardBlock.stability(weight=1.0)]
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator)
        strategy = RandomStrategy(alphabet="ACDE", min_len=5, max_len=10, seed=42)

        config = EngineConfig(iterations=3, population_size=4, seed=42)
        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

        stats_list = list(engine.stream())
        assert len(stats_list) == 3
        for i, stats in enumerate(stats_list):
            assert stats.iteration == i
            assert stats.evals == 4


class TestEngineWithCEMStrategy:
    """Integration tests for Engine with CEMStrategy."""

    def test_end_to_end_optimization(self):
        """Test end-to-end optimization with CEM strategy."""
        rewards = [RewardBlock.stability(weight=1.0)]
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator)

        strategy = CEMStrategy(
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            min_len=15,
            max_len=25,
            seed=42,
        )

        config = EngineConfig(
            iterations=3,
            population_size=16,
            seed=42,
            method="cem",
        )

        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

        results = engine.run()

        assert results.best is not None
        best_seq, best_score = results.best
        assert isinstance(best_seq, Sequence)
        assert isinstance(best_score, float)
        assert len(results.history) == 3

