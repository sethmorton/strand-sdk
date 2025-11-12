"""Tests for GCContentReward."""

import pytest

from strand.core.sequence import Sequence
from strand.rewards import RewardBlock


class TestGCContentReward:
    """GCContentReward unit tests."""

    def test_perfect_match(self):
        """Test reward with perfect target match."""
        reward = RewardBlock.gc_content(target=0.5, tolerance=0.1)
        seq = Sequence(id="test", tokens="GCAA")
        assert reward.score(seq) == 1.0

    def test_within_tolerance(self):
        """Test reward when GC content is within tolerance band."""
        reward = RewardBlock.gc_content(target=0.5, tolerance=0.1)
        seq = Sequence(id="test", tokens="GCAAA")
        assert reward.score(seq) == 1.0

    def test_outside_tolerance(self):
        """Test reward when GC content is outside tolerance band."""
        reward = RewardBlock.gc_content(target=0.5, tolerance=0.1)
        seq = Sequence(id="test", tokens="AAAA")
        score = reward.score(seq)
        assert 0.0 <= score < 1.0

    def test_weight_multiplier(self):
        """Test that weight is properly applied."""
        reward_unweighted = RewardBlock.gc_content(target=0.5, tolerance=0.1, weight=1.0)
        reward_weighted = RewardBlock.gc_content(target=0.5, tolerance=0.1, weight=2.0)

        seq = Sequence(id="test", tokens="GCAA")
        score_unweighted = reward_unweighted.score(seq)
        score_weighted = reward_weighted.score(seq)

        assert score_weighted == score_unweighted * 2.0

    def test_empty_sequence(self):
        """Test reward with empty sequence."""
        reward = RewardBlock.gc_content(target=0.5, tolerance=0.1)
        seq = Sequence(id="test", tokens="")
        assert reward.score(seq) == 0.0

    def test_case_insensitive(self):
        """Test that GC content calculation is case-insensitive."""
        reward = RewardBlock.gc_content(target=0.5, tolerance=0.1)

        seq_lower = Sequence(id="test1", tokens="gcaa")
        seq_upper = Sequence(id="test2", tokens="GCAA")
        seq_mixed = Sequence(id="test3", tokens="GcAa")

        assert reward.score(seq_lower) == reward.score(seq_upper) == reward.score(seq_mixed)

    def test_invalid_target(self):
        """Test that invalid target values raise errors."""
        with pytest.raises(ValueError, match="target must be in"):
            RewardBlock.gc_content(target=1.5)

        with pytest.raises(ValueError, match="target must be in"):
            RewardBlock.gc_content(target=-0.1)

    def test_invalid_tolerance(self):
        """Test that invalid tolerance values raise errors."""
        with pytest.raises(ValueError, match="tolerance must be in"):
            RewardBlock.gc_content(tolerance=1.5)

        with pytest.raises(ValueError, match="tolerance must be in"):
            RewardBlock.gc_content(tolerance=-0.1)

    @pytest.mark.parametrize(
        ("target", "tolerance", "expected_ge"),
        [
            (0.5, 0.1, 0.0),  # Should have some score
            (0.4, 0.05, 0.0),  # Custom target
            (0.6, 0.2, 0.0),  # High GC target
        ],
    )
    def test_various_targets(self, target, tolerance, expected_ge):
        """Test reward with various target values."""
        reward = RewardBlock.gc_content(target=target, tolerance=tolerance)
        seq = Sequence(id="test", tokens="GCGCAAAA")
        score = reward.score(seq)
        assert score >= expected_ge

