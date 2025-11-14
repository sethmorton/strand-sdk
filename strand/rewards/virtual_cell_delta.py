"""Virtual cell delta reward using Enformer foundation model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from strand.rewards.base import BaseRewardBlock, BlockType, RewardBlockMetadata, RewardContext

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext

logger = logging.getLogger(__name__)


class VirtualCellDeltaReward(BaseRewardBlock):
    """Reward block that scores variants using Enformer foundation model predictions.

    Computes the delta between reference and alternative sequence predictions
    across cell types and tracks. Higher scores indicate variants that enhance
    regulatory activity.
    """

    def __init__(
        self,
        model_path: str = "enformer-base",
        device: str = "auto",
        target_cell_types: list[str] | None = None,
        weight: float = 1.0,
    ):
        """Initialize VirtualCellDeltaReward.

        Args:
            model_path: HuggingFace model identifier (default: "enformer-base")
            device: Device for inference ("auto", "cpu", "cuda", "cuda:0", etc.)
            target_cell_types: List of cell type names to focus on. If None, uses all.
            weight: Reward weight multiplier

        Raises:
            ImportError: If enformer-pytorch is not installed
        """
        try:
            import enformer_pytorch  # noqa: F401
        except ImportError as e:
            msg = (
                "VirtualCellDeltaReward requires enformer-pytorch. "
                "Install with: pip install strand-sdk[variant-triage]"
            )
            raise ImportError(msg) from e

        super().__init__(
            name="virtual_cell_delta",
            weight=weight,
            metadata=RewardBlockMetadata(
                block_type=BlockType.DETERMINISTIC,
                description="Enformer-based variant effect prediction across cell types",
                requires_context=True,
            ),
        )

        self.model_path = model_path
        self.target_cell_types = target_cell_types or []

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lazy model loading
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load the Enformer model."""
        if self._model is None:
            from enformer_pytorch import Enformer

            logger.info(f"Loading Enformer model: {self.model_path}")
            self._model = Enformer.from_pretrained(self.model_path)
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence to one-hot tensor.

        Args:
            sequence: DNA sequence string

        Returns:
            One-hot encoded tensor of shape (seq_len, 4) for Enformer compatibility
        """
        # Convert to uppercase and handle N bases
        seq = sequence.upper().replace("N", "A")  # Treat N as A for simplicity

        # Map nucleotides to indices
        nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        indices = [nuc_to_idx.get(nuc, 0) for nuc in seq]  # Default to A for unknowns

        # Create one-hot encoding: (seq_len, 4) shape for Enformer
        seq_len = len(seq)
        one_hot = torch.zeros(seq_len, 4, dtype=torch.float32)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0

        return one_hot

    def _pad_sequence(self, seq_tensor: torch.Tensor, target_length: int = 196608) -> torch.Tensor:
        """Pad or truncate sequence to target length.

        Args:
            seq_tensor: One-hot encoded sequence (seq_len, 4)
            target_length: Target sequence length (Enformer expects 196608)

        Returns:
            Padded/truncated tensor (target_length, 4)
        """
        seq_len, _ = seq_tensor.shape

        if seq_len < target_length:
            # Pad with zeros at the end
            padding = torch.zeros(target_length - seq_len, 4, dtype=seq_tensor.dtype, device=seq_tensor.device)
            return torch.cat([seq_tensor, padding], dim=0)
        elif seq_len > target_length:
            # Truncate center to preserve variant position
            start = (seq_len - target_length) // 2
            return seq_tensor[start:start + target_length]
        else:
            return seq_tensor

    def _predict_sequence(self, sequence: str) -> torch.Tensor:
        """Run Enformer prediction on a single sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            Predictions tensor of shape (n_tracks,)
        """
        # Encode and pad sequence to Enformer's expected shape (seq_len, 4)
        seq_tensor = self._encode_sequence(sequence)
        seq_tensor = self._pad_sequence(seq_tensor)  # Shape: (196608, 4)
        
        # Add batch dimension: (1, 196608, 4) - Enformer expects (batch, seq_len, 4)
        seq_tensor = seq_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(seq_tensor)

        # Extract predictions (assuming output has 'predictions' key)
        if isinstance(output, dict) and "predictions" in output:
            predictions = output["predictions"]
        else:
            predictions = output

        # Enformer output shape varies by implementation
        # Common shapes: (batch, n_tracks) or (batch, n_cell_types, n_tracks)
        if predictions.dim() == 3:
            # Average across cell types: (batch, n_cell_types, n_tracks) -> (batch, n_tracks)
            predictions = predictions.mean(dim=1)
        
        # Return per-track predictions for single sequence: (n_tracks,)
        return predictions.squeeze(0)

    def score_context(
        self,
        context: SequenceContext,
        *,
        reward_context: RewardContext | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score variant effect using Enformer predictions.

        Args:
            context: SequenceContext with ref/alt sequences
            reward_context: Optional iteration context (unused)

        Returns:
            Tuple of (objective_delta, auxiliary_metrics)
        """
        # Get predictions for ref and alt sequences
        ref_pred = self._predict_sequence(context.ref_seq.tokens)
        alt_pred = self._predict_sequence(context.alt_seq.tokens)

        # Compute delta (alt - ref)
        delta = alt_pred - ref_pred

        # Objective: mean delta across all tracks
        objective = float(delta.mean())

        # Auxiliary metrics: per-track deltas
        aux = {}
        for i, track_delta in enumerate(delta):
            aux[f"enformer_track_{i}"] = float(track_delta)

        # If target cell types specified, could filter here
        # For now, return all tracks

        return objective, aux

    def _score(self, sequence, context: RewardContext) -> float:
        """Fallback scoring for non-context sequences."""
        # For single sequences without context, just return the mean prediction
        pred = self._predict_sequence(sequence.tokens)
        return float(pred.mean())
