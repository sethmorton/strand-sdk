"""Tests for constraint satisfaction solver."""

from strand.core.sequence import Sequence
from strand.engine.constraints import ConstraintSolver


class TestConstraintSolver:
    """Constraint solver tests."""

    def test_initialization(self):
        """Test ConstraintSolver initialization."""
        solver = ConstraintSolver(alphabet="ACDE", min_len=5, max_len=20)
        assert solver.alphabet == "ACDE"
        assert solver.min_len == 5
        assert solver.max_len == 20

    def test_is_feasible_valid(self):
        """Test that valid sequences pass feasibility check."""
        solver = ConstraintSolver(alphabet="ACDE", min_len=5, max_len=20)
        seq = Sequence(id="test", tokens="ACDEAC")
        assert solver.is_feasible(seq) is True

    def test_is_feasible_invalid_alphabet(self):
        """Test that invalid alphabet fails."""
        solver = ConstraintSolver(alphabet="AC", min_len=5, max_len=20)
        seq = Sequence(id="test", tokens="ACZDE")  # Z not in alphabet
        assert solver.is_feasible(seq) is False

    def test_is_feasible_invalid_length_too_short(self):
        """Test that too-short sequences fail."""
        solver = ConstraintSolver(alphabet="ACDE", min_len=5, max_len=20)
        seq = Sequence(id="test", tokens="AC")  # Too short
        assert solver.is_feasible(seq) is False

    def test_is_feasible_invalid_length_too_long(self):
        """Test that too-long sequences fail."""
        solver = ConstraintSolver(alphabet="ACDE", min_len=5, max_len=10)
        seq = Sequence(id="test", tokens="A" * 15)  # Too long
        assert solver.is_feasible(seq) is False

    def test_filter_feasible(self):
        """Test filtering sequences."""
        solver = ConstraintSolver(alphabet="AC", min_len=5, max_len=10)
        sequences = [
            Sequence(id="1", tokens="ACACA"),  # Valid
            Sequence(id="2", tokens="AC"),  # Too short
            Sequence(id="3", tokens="ACACACA"),  # Valid
            Sequence(id="4", tokens="ACACAZACA"),  # Invalid alphabet
        ]
        feasible = solver.filter_feasible(sequences)
        assert len(feasible) == 2
        assert feasible[0].id == "1"
        assert feasible[1].id == "3"

    def test_generate_feasible_set(self):
        """Test generating feasible sequences."""
        sequences = ConstraintSolver.generate_feasible_set("AC", length=10, max_size=5)
        assert len(sequences) == 5
        for seq in sequences:
            assert len(seq.tokens) == 10
            assert all(c in "AC" for c in seq.tokens)

