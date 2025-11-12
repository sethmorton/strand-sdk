"""Tests for CMAESStrategy implementation."""

import pytest

from strand.engine.strategies.cmaes import CMAESStrategy


class TestCMAESStrategy:
    """CMA-ES strategy tests."""

    def test_initialization(self):
        """Test CMAESStrategy initialization."""
        strategy = CMAESStrategy(
            alphabet="ACDE",
            min_len=10,
            max_len=20,
            seed=42,
            sigma0=0.3,
        )
        assert strategy.alphabet == "ACDE"
        assert strategy.min_len == 10
        assert strategy.max_len == 20
        assert strategy.sigma0 == 0.3

    def test_ask_returns_sequences(self):
        """Test that ask returns valid sequences."""
        strategy = CMAESStrategy(
            alphabet="AC",
            min_len=5,
            max_len=10,
            seed=42,
        )
        sequences = strategy.ask(5)
        assert len(sequences) == 5
        for seq in sequences:
            assert len(seq.tokens) <= 10
            assert all(c in "AC" for c in seq.tokens)

    def test_best_tracking(self):
        """Test that best score is tracked."""
        strategy = CMAESStrategy(
            alphabet="AC",
            min_len=5,
            max_len=10,
            seed=42,
        )
        assert strategy.best() is None

        sequences = strategy.ask(10)
        items = [(seq, 0.5 + i * 0.05, None) for i, seq in enumerate(sequences)]
        strategy.tell(items)

        best_seq, best_score = strategy.best()
        # Best is the last one with highest score
        assert best_seq == sequences[-1]
        assert best_score > 0.9

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="non-empty"):
            CMAESStrategy(alphabet="", min_len=5, max_len=10)

        with pytest.raises(ValueError, match="invalid"):
            CMAESStrategy(alphabet="AC", min_len=0, max_len=10)

        with pytest.raises(ValueError, match="invalid"):
            CMAESStrategy(alphabet="AC", min_len=10, max_len=5)

    def test_state_serialization(self):
        """Test that state is serializable."""
        strategy = CMAESStrategy(
            alphabet="AC",
            min_len=5,
            max_len=10,
            seed=42,
        )
        sequences = strategy.ask(10)
        items = [(seq, 0.5 + i * 0.05, None) for i, seq in enumerate(sequences)]
        strategy.tell(items)

        state = strategy.state()
        assert "best_score" in state
        assert state["best_score"] > 0.9

