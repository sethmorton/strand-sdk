from strand.core.sequence import Sequence
from strand.engine.strategies.cem import CEMStrategy
from strand.engine.strategies.random import RandomStrategy
from strand.engine.types import Metrics


class TestRandomStrategy:
    """RandomStrategy unit tests."""

    def test_best_tracking(self):
        """tell() updates the best sequence with highest score."""

        strategy = RandomStrategy(alphabet="ACDE", min_len=5, max_len=10, seed=42)
        candidates = strategy.ask(3)
        metrics = Metrics(objective=0.0, constraints={}, aux={})
        items = [
            (candidates[0], 0.5, metrics),
            (candidates[1], 0.9, metrics),
            (candidates[2], 0.3, metrics),
        ]

        strategy.tell(items)

        best_seq, best_score = strategy.best()  # type: ignore[misc]
        assert best_score == 0.9
        assert best_seq == candidates[1]

    def test_ask_returns_sequences_in_range(self):
        """RandomStrategy generates valid sequences in the configured range."""

        strategy = RandomStrategy(alphabet="ACDE", min_len=10, max_len=20, seed=99)

        candidates = strategy.ask(5)
        assert len(candidates) == 5
        for seq in candidates:
            assert isinstance(seq, Sequence)
            assert 10 <= len(seq.tokens) <= 20
            assert all(c in "ACDE" for c in seq.tokens)

    def test_seed_reproducible_across_instances(self):
        """Strategies with same seed generate identical batches."""

        strategy_a = RandomStrategy(alphabet="AC", min_len=5, max_len=5, seed=123)
        strategy_b = RandomStrategy(alphabet="AC", min_len=5, max_len=5, seed=123)

        seqs_a = strategy_a.ask(4)
        seqs_b = strategy_b.ask(4)

        assert [seq.tokens for seq in seqs_a] == [seq.tokens for seq in seqs_b]
        assert [seq.id for seq in seqs_a] == ["random_0", "random_1", "random_2", "random_3"]

    def test_state_includes_seed_and_counter(self):
        """state() exposes metadata useful for manifests."""

        strategy = RandomStrategy(alphabet="AC", min_len=3, max_len=3, seed=7)
        strategy.ask(2)

        state = strategy.state()
        assert state["seed"] == 7
        assert state["generated"] == 2


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
        metrics = Metrics(objective=0.0, constraints={}, aux={})
        items = [
            (candidates[0], 0.3, metrics),
            (candidates[1], 0.7, metrics),
            (candidates[2], 0.5, metrics),
            (candidates[3], 0.2, metrics),
        ]
        strategy.tell(items)

        best_seq, best_score = strategy.best()  # type: ignore[misc]
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
        metrics = Metrics(objective=0.0, constraints={}, aux={})
        items = [(candidates[0], 0.5, metrics), (candidates[1], 0.9, metrics)]
        strategy.tell(items)

        state = strategy.state()
        assert "best_score" in state
        assert state["best_score"] == 0.9
        assert "probs" in state
