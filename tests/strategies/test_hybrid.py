"""Tests for HybridStrategy enhancements."""

import pytest
from strand.core.sequence import Sequence
from strand.engine.strategies import (
    HybridStrategy,
    RandomStrategy,
    CEMStrategy,
    RLPolicyStrategy,
)
from strand.engine.types import Metrics


class TestHybridStrategyEnhancements:
    """Test hybrid strategy capability merging and batch handling."""

    def test_hybrid_merges_capabilities(self) -> None:
        """Hybrid merges child capabilities correctly."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=50),
            RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50),
        ]
        hybrid = HybridStrategy(strategies=strategies)

        # Should inherit RL's capabilities
        caps = hybrid.strategy_caps()
        assert caps.requires_runtime is True
        assert caps.supports_fine_tuning is True

    def test_hybrid_respects_batch_constraints(self) -> None:
        """Hybrid respects token budget constraints."""
        strategy1 = RandomStrategy(alphabet="ACGT", min_len=10, max_len=50)
        strategy2 = CEMStrategy(alphabet="ACGT", min_len=10, max_len=50)

        hybrid = HybridStrategy(strategies=[strategy1, strategy2])

        # Should still generate sequences
        seqs = hybrid.ask(10)
        assert len(seqs) >= 10

    def test_hybrid_best_sequence_tracking(self) -> None:
        """Hybrid tracks best sequence across all strategies."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=20, seed=42),
            CEMStrategy(alphabet="ACGT", min_len=10, max_len=20, seed=42),
        ]
        hybrid = HybridStrategy(strategies=strategies)

        seqs = hybrid.ask(5)

        # Create scored items
        items = [
            (seq, float(i), Metrics(objective=float(i), constraints={}, aux={}))
            for i, seq in enumerate(seqs)
        ]

        hybrid.tell(items)

        # Should have best sequence
        best = hybrid.best()
        assert best is not None
        assert best[1] == 4.0  # Best score


class TestHybridWithRL:
    """Test hybrid strategies containing RL policy."""

    def test_hybrid_rl_fine_tuning(self) -> None:
        """Hybrid with RL strategy supports fine-tuning."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=50),
            RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50),
        ]
        hybrid = HybridStrategy(strategies=strategies)

        caps = hybrid.strategy_caps()
        assert caps.supports_fine_tuning is True

    def test_hybrid_preserves_rl_kl_regularization(self) -> None:
        """Hybrid inherits RL KL regularization."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=50),
            RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50),
        ]
        hybrid = HybridStrategy(strategies=strategies)

        caps = hybrid.strategy_caps()
        assert caps.kl_regularization == "token"

