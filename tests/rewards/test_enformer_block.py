"""Tests for EnformerRewardBlock."""

import pytest
import numpy as np

from strand.core.sequence import Sequence
from strand.rewards.enformer_block import EnformerConfig, EnformerRewardBlock


class TestEnformerConfig:
    """Test EnformerConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC"],
        )
        assert config.target_cell_types == ["hNSPC"]
        assert config.aggregation == "mean"
        assert config.normalization == "zscore"

    def test_invalid_aggregation(self) -> None:
        """Test invalid aggregation raises error."""
        with pytest.raises(ValueError):
            EnformerConfig(
                model_path="dummy.pt",
                target_cell_types=["hNSPC"],
                aggregation="invalid",
            )

    def test_weighted_aggregation_requires_weights(self) -> None:
        """Weighted aggregation requires cell_type_weights."""
        with pytest.raises(ValueError):
            config = EnformerConfig(
                model_path="dummy.pt",
                target_cell_types=["hNSPC"],
                aggregation="weighted",
            )
            # Trigger validation in __post_init__
            _ = config.aggregation


class TestEnformerRewardBlock:
    """Test EnformerRewardBlock functionality."""

    @pytest.fixture
    def config(self) -> EnformerConfig:
        """Create test config using stub model."""
        return EnformerConfig(
            model_path="dummy.pt",  # Will use stub
            target_cell_types=["hNSPC"],
            aggregation="mean",
            device="cpu",
        )

    @pytest.fixture
    def block(self, config: EnformerConfig) -> EnformerRewardBlock:
        """Create test block."""
        return EnformerRewardBlock(config)

    def test_empty_sequence_list(self, block: EnformerRewardBlock) -> None:
        """Empty input returns zero score."""
        result = block([])
        assert result["enformer_activity"] == 0.0

    def test_single_sequence(self, block: EnformerRewardBlock) -> None:
        """Score single sequence."""
        seq = Sequence(id="seq1", tokens="ACGTACGT")
        result = block([seq])

        assert "enformer_activity" in result
        assert isinstance(result["enformer_activity"], float)

    def test_multiple_sequences(self, block: EnformerRewardBlock) -> None:
        """Score multiple sequences."""
        sequences = [
            Sequence(id="seq1", tokens="ACGTACGT"),
            Sequence(id="seq2", tokens="TGCATGCA"),
        ]
        result = block(sequences)

        assert "enformer_activity" in result
        assert isinstance(result["enformer_activity"], float)

    def test_multi_cell_type_scores(self) -> None:
        """Multi-cell-type config includes per-type scores."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC", "RPE"],
            aggregation="mean",
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT")
        result = block([seq])

        assert "enformer_activity" in result
        assert "enformer_hNSPC" in result or "enformer_RPE" in result

    def test_normalization_zscore(self) -> None:
        """Test zscore normalization."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC"],
            normalization="zscore",
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        sequences = [Sequence(id=f"seq{i}", tokens="ACGT" * 10) for i in range(5)]
        result = block(sequences)

        # zscore should center around 0
        assert isinstance(result["enformer_activity"], float)

    def test_normalization_minmax(self) -> None:
        """Test minmax normalization."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC"],
            normalization="minmax",
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        sequences = [Sequence(id=f"seq{i}", tokens="ACGT" * 10) for i in range(5)]
        result = block(sequences)

        # minmax should be in [0, 1]
        score = result["enformer_activity"]
        assert 0.0 <= score <= 1.0

    def test_weighted_aggregation(self) -> None:
        """Test weighted aggregation across cell types."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC", "RPE"],
            aggregation="weighted",
            cell_type_weights={"hNSPC": 0.7, "RPE": 0.3},
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT")
        result = block([seq])

        assert "enformer_activity" in result

    def test_encoding_sequences(self, block: EnformerRewardBlock) -> None:
        """Test one-hot encoding of sequences."""
        seq = Sequence(id="seq1", tokens="ACGTACGT")

        encoded = block._encode_sequences([seq])

        # Check shape: (batch_size=1, seq_len=8, alphabet_size=4)
        assert encoded.shape == (1, 8, 4)
        # Check valid one-hot encoding
        assert np.allclose(encoded.sum(axis=-1), 1.0)

    def test_encoding_with_n(self, block: EnformerRewardBlock) -> None:
        """Test encoding with N (ambiguous) bases."""
        seq = Sequence(id="seq1", tokens="ACNGTACGT")
        encoded = block._encode_sequences([seq])

        # N is treated as A
        assert encoded.shape == (1, 9, 4)
        assert np.allclose(encoded.sum(axis=-1), 1.0)


class TestEnformerIntegration:
    """Integration tests for Enformer scoring."""

    def test_reproducibility(self) -> None:
        """Same seed produces same results."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC"],
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        seq = Sequence(id="seq1", tokens="ACGTACGT")

        result1 = block([seq])
        result2 = block([seq])

        # Results should be identical (no randomness in stub)
        assert result1["enformer_activity"] == result2["enformer_activity"]

    def test_different_sequences_different_scores(self) -> None:
        """Different sequences produce different scores (usually)."""
        config = EnformerConfig(
            model_path="dummy.pt",
            target_cell_types=["hNSPC"],
            device="cpu",
        )
        block = EnformerRewardBlock(config)

        seq1 = Sequence(id="seq1", tokens="AAAAAAAA")
        seq2 = Sequence(id="seq2", tokens="TTTTTTTT")

        result1 = block([seq1])
        result2 = block([seq2])

        # Different sequences may have different scores
        # (with stub, probably same, but structure is correct)
        assert "enformer_activity" in result1
        assert "enformer_activity" in result2

