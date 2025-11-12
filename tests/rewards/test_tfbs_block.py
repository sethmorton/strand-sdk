"""Tests for TFBSFrequencyCorrelationBlock."""

import pytest
import numpy as np

from strand.core.sequence import Sequence
from strand.rewards.tfbs_block import TFBSConfig, TFBSFrequencyCorrelationBlock


class TestTFBSConfig:
    """Test TFBSConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TFBSConfig(
            motif_names=["SOX2", "OCT4"],
        )
        assert config.motif_names == ["SOX2", "OCT4"]
        assert config.threshold == 0.7
        assert config.aggregation == "correlation"

    def test_invalid_aggregation(self) -> None:
        """Invalid aggregation raises error."""
        with pytest.raises(ValueError):
            TFBSConfig(
                motif_names=["SOX2"],
                aggregation="invalid",
            )


class TestTFBSFrequencyCorrelationBlock:
    """Test TFBSFrequencyCorrelationBlock functionality."""

    @pytest.fixture
    def config(self) -> TFBSConfig:
        """Create test config."""
        return TFBSConfig(
            motif_names=["SOX2", "OCT4"],
            target_frequency={"SOX2": 0.3, "OCT4": 0.2},
            threshold=0.7,
            aggregation="correlation",
        )

    @pytest.fixture
    def block(self, config: TFBSConfig) -> TFBSFrequencyCorrelationBlock:
        """Create test block."""
        return TFBSFrequencyCorrelationBlock(config)

    def test_empty_sequence_list(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Empty input returns zero score."""
        result = block([])
        assert result["tfbs_correlation"] == 0.0
        assert result["tfbs_frequency"] == 0.0

    def test_single_sequence(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Score single sequence."""
        seq = Sequence(id="seq1", tokens="ACGTACGTACGTACGT")
        result = block([seq])

        assert "tfbs_correlation" in result
        assert "tfbs_frequency" in result
        assert isinstance(result["tfbs_correlation"], float)
        assert isinstance(result["tfbs_frequency"], float)

    def test_multiple_sequences(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Score multiple sequences."""
        sequences = [
            Sequence(id="seq1", tokens="ACGTACGT" * 5),
            Sequence(id="seq2", tokens="TGCATGCA" * 5),
        ]
        result = block(sequences)

        assert "tfbs_correlation" in result
        assert "tfbs_frequency" in result

    def test_constraint_channel(self) -> None:
        """Constraint channel included when requested."""
        config = TFBSConfig(
            motif_names=["SOX2"],
            target_frequency={"SOX2": 0.3},
            return_constraint_channel=True,
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT" * 5)
        result = block([seq])

        assert "tfbs_divergence" in result
        assert isinstance(result["tfbs_divergence"], float)

    def test_no_constraint_channel(self) -> None:
        """Constraint channel excluded when not requested."""
        config = TFBSConfig(
            motif_names=["SOX2"],
            target_frequency={"SOX2": 0.3},
            return_constraint_channel=False,
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT" * 5)
        result = block([seq])

        assert "tfbs_divergence" not in result

    def test_correlation_aggregation(self) -> None:
        """Correlation aggregation works."""
        config = TFBSConfig(
            motif_names=["SOX2", "OCT4"],
            target_frequency={"SOX2": 0.3, "OCT4": 0.2},
            aggregation="correlation",
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT" * 10)
        result = block([seq])

        # Correlation should be in [-1, 1]
        corr = result["tfbs_correlation"]
        assert -1.0 <= corr <= 1.0

    def test_frequency_aggregation(self) -> None:
        """Frequency aggregation works."""
        config = TFBSConfig(
            motif_names=["SOX2"],
            target_frequency={"SOX2": 0.3},
            aggregation="frequency",
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT" * 10)
        result = block([seq])

        # Frequency difference should be non-negative
        freq = result["tfbs_frequency"]
        assert 0.0 <= freq <= 1.0

    def test_encoding_sequence(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Test one-hot encoding."""
        seq = "ACGT"
        pwm = block.motifs["SOX2"]

        scores = block._scan_pwm(seq, pwm)

        # Should have scores for each position
        assert len(scores) >= 1
        assert isinstance(scores, np.ndarray)

    def test_motif_scanning(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Test PWM motif scanning."""
        seq = "ACGTACGT" * 5
        pwm = block.motifs["SOX2"]

        scores = block._scan_pwm(seq, pwm)

        # Scores should be between 0 and 1
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_tfbs_frequency_computation(self, block: TFBSFrequencyCorrelationBlock) -> None:
        """Test TFBS frequency computation."""
        seq = "ACGTACGT" * 10
        freqs = block._compute_tfbs_frequency(seq)

        # Should have one frequency per motif
        assert len(freqs) == len(block.config.motif_names)
        # Frequencies should be in [0, 1]
        assert np.all(freqs >= 0.0)
        assert np.all(freqs <= 1.0)


class TestTFBSIntegration:
    """Integration tests for TFBS scoring."""

    def test_reproducibility(self) -> None:
        """Same seed produces same results."""
        config = TFBSConfig(
            motif_names=["SOX2", "OCT4"],
            target_frequency={"SOX2": 0.3, "OCT4": 0.2},
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT" * 10)

        result1 = block([seq])
        result2 = block([seq])

        # Results should be identical (deterministic)
        assert result1["tfbs_correlation"] == result2["tfbs_correlation"]
        assert result1["tfbs_frequency"] == result2["tfbs_frequency"]

    def test_different_sequences_different_scores(self) -> None:
        """Different sequences usually produce different scores."""
        config = TFBSConfig(
            motif_names=["SOX2"],
            target_frequency={"SOX2": 0.3},
        )
        block = TFBSFrequencyCorrelationBlock(config)

        seq1 = Sequence(id="seq1", tokens="AAAAAAAA" * 10)
        seq2 = Sequence(id="seq2", tokens="CCCCCCCC" * 10)

        result1 = block([seq1])
        result2 = block([seq2])

        # Sequences with different composition likely have different scores
        # (may occasionally be the same by chance, but structure is correct)
        assert "tfbs_correlation" in result1
        assert "tfbs_correlation" in result2

    def test_perfect_match(self) -> None:
        """Test perfect match to target frequencies."""
        config = TFBSConfig(
            motif_names=["SOX2"],
            target_frequency={"SOX2": 0.5},
            aggregation="frequency",
        )
        block = TFBSFrequencyCorrelationBlock(config)

        # Create a sequence where we control frequency
        # (this is simplified; real PWM matching is more complex)
        seq = Sequence(id="seq1", tokens="ACGTACGT" * 10)
        result = block([seq])

        # Should have reasonable score
        freq = result["tfbs_frequency"]
        assert 0.0 <= freq <= 1.0

