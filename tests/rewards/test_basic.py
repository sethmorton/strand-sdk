"""Tests for basic reward blocks (Stability, Solubility, Novelty)."""

import pytest

from strand.rewards import RewardBlock


class TestStabilityReward:
    """StabilityReward unit tests."""

    def test_threshold_based_scoring(self):
        """Test that stability uses hydrophobicity as heuristic."""
        reward = RewardBlock.stability(threshold=0.8)
        # Test with existing test case
        from strand.core.sequence import Sequence

        seq = Sequence(id="test", tokens="AILMFWYV")  # All hydrophobic
        score = reward.score(seq)
        assert score > 0


class TestNoveltyReward:
    """NoveltyReward unit tests."""

    def test_requires_baseline(self):
        """Test that novelty requires a baseline."""
        with pytest.raises(ValueError, match="baseline"):
            RewardBlock.novelty(baseline=[])

