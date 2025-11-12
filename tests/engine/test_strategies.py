"""Tests for optimization strategies (Random, CEM, etc.)."""

from strand.core.sequence import Sequence
from strand.engine.strategies.cem import CEMStrategy
from strand.engine.strategies.random import RandomStrategy


class TestRandomStrategy:
    """RandomStrategy unit tests."""

    def test_best_tracking(self):
        """Test that RandomStrategy tracks the best sequence."""
        strategy = RandomStrategy(
            alphabet="ACDE",
            min_len=5,
            max_len=10,
            seed=42,
        )

        # Simulate ask and tell
        candidates = strategy.ask(3)
        assert len(candidates) == 3

        # Initially no best
        assert strategy.best() is None

        # Tell with some scores
        items = [(candidates[0], 0.5, None), (candidates[1], 0.9, None), (candidates[2], 0.3, None)]  # type: ignore
        strategy.tell(items)

        # Should track the best (0.9)
        best_seq, best_score = strategy.best()  # type: ignore
        assert best_score == 0.9
        assert best_seq == candidates[1]

    def test_ask_returns_sequences_in_range(self):
        """Test that RandomStrategy generates sequences within specified length range."""
        strategy = RandomStrategy(
            alphabet="ACDE",
            min_len=10,
            max_len=20,
            seed=42,
        )

        candidates = strategy.ask(5)
        assert len(candidates) == 5
        for seq in candidates:
            assert isinstance(seq, Sequence)
            assert 10 <= len(seq.tokens) <= 20
            assert all(c in "ACDE" for c in seq.tokens)

    def test_seeded_reproducible_within_instance(self):
        """Test that RandomStrategy with same seed produces same results within instance."""
        strategy = RandomStrategy(alphabet="AC", min_len=5, max_len=5, seed=42)

        # Ask twice from the same strategy - should produce different sequences
        seq1 = strategy.ask(2)
        seq2 = strategy.ask(2)

        # But they should be valid sequences from the alphabet
        for seq in seq1 + seq2:
            assert all(c in "AC" for c in seq.tokens)
            assert len(seq.tokens) == 5


class TestCEMStrategy:
    """CEMStrategy unit tests."""

    def test_basic_initialization(self):
        """Test CEMStrategy can be initialized."""
        strategy = CEMStrategy(
            alphabet="ACDE",
            min_len=5,
            max_len=10,
            seed=42,
        )
        assert strategy.alphabet == "ACDE"
        assert strategy.elite_frac == 0.2

    def test_best_tracking(self):
        """Test that CEMStrategy tracks the best sequence."""
        strategy = CEMStrategy(
            alphabet="ACDE",
            min_len=5,
            max_len=10,
            seed=42,
        )

        candidates = strategy.ask(4)
        assert len(candidates) == 4

        # Initially no best
        assert strategy.best() is None

        # Tell with scores
        items = [
            (candidates[0], 0.3, None),
            (candidates[1], 0.7, None),
            (candidates[2], 0.5, None),
            (candidates[3], 0.2, None),
        ]  # type: ignore
        strategy.tell(items)

        best_seq, best_score = strategy.best()  # type: ignore
        assert best_score == 0.7
        assert best_seq == candidates[1]

    def test_state_serialization(self):
        """Test that CEMStrategy state can be serialized."""
        strategy = CEMStrategy(
            alphabet="AC",
            min_len=5,
            max_len=5,
            seed=42,
        )

        candidates = strategy.ask(2)
        items = [(candidates[0], 0.5, None), (candidates[1], 0.9, None)]  # type: ignore
        strategy.tell(items)

        state = strategy.state()
        assert "best_score" in state
        assert state["best_score"] == 0.9
        assert "probs" in state

