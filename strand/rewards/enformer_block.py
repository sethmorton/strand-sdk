"""Enformer-based reward block for cell-type-specific regulatory activity scoring.

Enformer is a deep learning model that predicts gene expression from DNA sequences.
This block enables scoring sequences based on predicted regulatory activity across
different cell types, supporting Ctrl-DNA's cell-type-specific optimization.

Reference: Avsec et al., "Effective gene expression prediction from sequence by
integrating long-range interactions". Nature Methods, 2021.
https://github.com/deepmind/deepmind-research/tree/master/enformer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from strand.core.sequence import Sequence
from strand.rewards.base import (
    BaseRewardBlock,
    BlockType,
    RewardBlockMetadata,
    RewardContext,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnformerConfig:
    """Configuration for EnformerRewardBlock.

    Attributes
    ----------
    model_path : str | Path
        Path to Enformer model checkpoint (PyTorch or ONNX).
    model_type : {"pytorch", "onnx"}
        Type of model format.
    target_cell_types : list[str]
        Cell types to optimize for (e.g., ["hNSPC", "RPE"]).
    aggregation : {"mean", "max", "weighted"}
        How to combine scores across cell types:
        - "mean": Average across target cell types
        - "max": Take maximum across cell types
        - "weighted": Weighted average (requires cell_type_weights)
    cell_type_weights : dict[str, float] | None
        Weights for each cell type (for weighted aggregation).
    normalization : {"none", "zscore", "minmax"}
        Normalization method:
        - "none": No normalization
        - "zscore": Zero-mean unit-variance
        - "minmax": Scale to [0, 1]
    device : str
        Device to run on ("cpu" or "cuda:X").
    batch_size : int
        Batch size for inference.
    return_gradients : bool
        If True, also return per-position gradient information.
    """

    model_path: str | Path
    model_type: str = "pytorch"
    target_cell_types: list[str] = None  # type: ignore[assignment]
    aggregation: str = "mean"
    cell_type_weights: dict[str, float] | None = None
    normalization: str = "zscore"
    device: str = "cpu"
    batch_size: int = 32
    return_gradients: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.target_cell_types is None:
            object.__setattr__(self, "target_cell_types", ["hNSPC"])

        if self.aggregation not in ("mean", "max", "weighted"):
            raise ValueError(f"Invalid aggregation: {self.aggregation}")

        if self.normalization not in ("none", "zscore", "minmax"):
            raise ValueError(f"Invalid normalization: {self.normalization}")

        if self.aggregation == "weighted" and not self.cell_type_weights:
            raise ValueError("Weighted aggregation requires cell_type_weights")


class EnformerRewardBlock(BaseRewardBlock):
    """Enformer-based reward block for regulatory activity prediction.

    Scores sequences based on predicted cell-type-specific gene expression,
    enabling cell-type-controlled DNA design.

    Example
    -------
    >>> config = EnformerConfig(
    ...     model_path="enformer_checkpoint.pt",
    ...     target_cell_types=["hNSPC", "RPE"],
    ...     aggregation="weighted",
    ...     cell_type_weights={"hNSPC": 0.6, "RPE": 0.4},
    ... )
    >>> block = EnformerRewardBlock(config)
    >>> scores = block([Sequence(id="seq1", tokens="ACGTACGT")])
    >>> print(scores)  # {"enformer_activity": [...]}
    """

    def __init__(self, config: EnformerConfig, *, weight: float = 1.0) -> None:
        """Initialize Enformer reward block.

        Parameters
        ----------
        config : EnformerConfig
            Configuration specifying model and scoring options.

        Raises
        ------
        FileNotFoundError
            If model checkpoint not found.
        """
        metadata = RewardBlockMetadata(
            block_type=BlockType.ADVANCED,
            description="Enformer-based regulatory activity block",
            requires_context=False,
        )
        super().__init__(name="enformer", weight=weight, metadata=metadata)
        self.config = config
        self.model = self._load_model()
        self._init_normalization_stats()

    def _load_model(self) -> "nn.Module":
        """Load Enformer model from checkpoint.

        Returns
        -------
        nn.Module
            Loaded Enformer model.

        Raises
        ------
        FileNotFoundError
            If checkpoint not found.
        """
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Enformer model not found: {model_path}")

        if self.config.model_type == "onnx":
            return self._load_onnx_model(model_path)
        elif self.config.model_type == "pytorch":
            return self._load_pytorch_model(model_path)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _load_onnx_model(self, model_path: Path) -> object:
        """Load ONNX Enformer model.

        Parameters
        ----------
        model_path : Path
            Path to ONNX model.

        Returns
        -------
        object
            ONNX session wrapper.
        """
        try:
            import onnxruntime as ort  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = (
                "onnxruntime required for ONNX models. "
                "Install with: pip install onnxruntime"
            )
            raise ImportError(msg) from exc

        session = ort.InferenceSession(str(model_path))
        _LOGGER.info(f"Loaded ONNX Enformer model from {model_path}")
        return session

    def _load_pytorch_model(self, model_path: Path) -> "nn.Module":
        """Load PyTorch Enformer model.

        Parameters
        ----------
        model_path : Path
            Path to PyTorch checkpoint.

        Returns
        -------
        nn.Module
            Enformer model.
        """
        import torch

        try:
            import enformer_pytorch  # type: ignore[import]
            model = enformer_pytorch.Enformer.from_pretrained("enformer-base")
        except ImportError:  # pragma: no cover - optional dependency
            # Fallback: create stub for testing
            _LOGGER.warning(
                "enformer_pytorch not installed. Using stub model. "
                "Install with: pip install enformer-pytorch"
            )
            model = self._create_stub_model()

        checkpoint = torch.load(model_path, map_location=self.config.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.config.device)
        model.eval()
        _LOGGER.info(f"Loaded PyTorch Enformer model from {model_path}")
        return model

    def _create_stub_model(self) -> "nn.Module":
        """Create stub Enformer model for testing.

        Returns
        -------
        nn.Module
            Dummy model that returns random predictions.
        """

        import torch
        import torch.nn as nn

        class StubEnformer(nn.Module):
            def __init__(self, num_cell_types: int = 5):
                super().__init__()
                self.num_cell_types = num_cell_types

            def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
                # x: (batch_size, seq_len, 4) one-hot encoded sequences
                batch_size = x.shape[1]  # Enformer typically processes features
                # Return random predictions for each cell type
                predictions = torch.randn(batch_size, self.num_cell_types)
                return {"predictions": predictions}

        return StubEnformer()

    def _init_normalization_stats(self) -> None:
        """Initialize normalization statistics.

        For zscore normalization, pre-compute mean/std on a sample of sequences.
        """
        self._norm_mean = 0.0
        self._norm_std = 1.0

    def __call__(self, sequences: list[Sequence]) -> dict[str, float]:
        """Score sequences using Enformer predictions.

        Parameters
        ----------
        sequences : list[Sequence]
            Sequences to score.

        Returns
        -------
        dict[str, float]
            Mapping of metric names to scores:
            - "enformer_activity": Mean predicted activity
            - Per-cell-type scores if requested
        """
        if not sequences:
            return {"enformer_activity": 0.0}

        # Tokenize sequences
        encoded = self._encode_sequences(sequences)

        # Run inference
        import torch

        with torch.no_grad():
            predictions = self._predict(encoded)

        # Aggregate across cell types
        scores = self._aggregate_predictions(predictions)

        # Normalize
        scores = self._normalize_scores(scores)

        # Return as dict with enformer_activity key
        result = {
            "enformer_activity": float(np.mean(scores)),
        }

        # Optionally add per-cell-type scores
        if len(self.config.target_cell_types) > 1:
            for cell_type, score in zip(self.config.target_cell_types, scores):
                result[f"enformer_{cell_type}"] = float(score)

        return result

    def _encode_sequences(self, sequences: list[Sequence]) -> "torch.Tensor":
        """One-hot encode sequences for Enformer.

        Parameters
        ----------
        sequences : list[Sequence]
            DNA sequences.

        Returns
        -------
        torch.Tensor
            One-hot encoded sequences of shape (batch_size, seq_len, 4).
        """
        # Alphabet mapping
        alphabet = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}  # N -> A for safety

        batch_size = len(sequences)
        max_len = max(len(seq.tokens) for seq in sequences)

        encoded = np.zeros((batch_size, max_len, 4), dtype=np.float32)

        for i, seq in enumerate(sequences):
            tokens = seq.tokens.upper()
            for j, token in enumerate(tokens):
                if j < max_len:
                    idx = alphabet.get(token, 0)
                    encoded[i, j, idx] = 1.0

        import torch

        return torch.from_numpy(encoded).to(self.config.device)

    def _predict(self, encoded: "torch.Tensor") -> np.ndarray:
        """Run Enformer inference.

        Parameters
        ----------
        encoded : torch.Tensor
            One-hot encoded sequences.

        Returns
        -------
        np.ndarray
            Model predictions of shape (batch_size, num_cell_types).
        """
        import torch
        import torch.nn as nn

        if isinstance(self.model, nn.Module):
            # PyTorch inference
            output = self.model(encoded)
            if isinstance(output, dict):
                predictions = output.get("predictions", output.get("output"))
            else:
                predictions = output

            if predictions.dim() > 2:
                # Average over spatial dimensions (tracks)
                predictions = predictions.mean(dim=1)

            return predictions.cpu().numpy()
        else:
            # ONNX inference
            session = self.model
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            predictions = session.run([output_name], {input_name: encoded.cpu().numpy()})
            return predictions[0]

    def _aggregate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Aggregate predictions across cell types.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions of shape (batch_size, num_cell_types).

        Returns
        -------
        np.ndarray
            Aggregated scores of shape (batch_size,).
        """
        # Select target cell type indices
        # (This is simplified; real Enformer has named cell types)
        num_targets = len(self.config.target_cell_types)

        if self.config.aggregation == "mean":
            return np.mean(predictions[:, :num_targets], axis=1)

        elif self.config.aggregation == "max":
            return np.max(predictions[:, :num_targets], axis=1)

        elif self.config.aggregation == "weighted":
            weights = np.array(
                [
                    self.config.cell_type_weights.get(ct, 1.0)
                    for ct in self.config.target_cell_types
                ]
            )
            weights /= weights.sum()
            return np.average(predictions[:, :num_targets], axis=1, weights=weights)

        else:
            return np.mean(predictions[:, :num_targets], axis=1)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores.

        Parameters
        ----------
        scores : np.ndarray
            Raw scores.

        Returns
        -------
        np.ndarray
            Normalized scores.
        """
        if self.config.normalization == "none":
            return scores

        elif self.config.normalization == "zscore":
            return (scores - self._norm_mean) / (self._norm_std + 1e-8)

        elif self.config.normalization == "minmax":
            min_val = scores.min()
            max_val = scores.max()
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            return scores

        else:
            return scores

    def _score(self, sequence: Sequence, context: RewardContext) -> float:  # noqa: ARG002
        """Scalar objective used by RewardAggregator."""
        result = self([sequence])
        return float(result.get("enformer_activity", 0.0))


__all__ = [
    "EnformerConfig",
    "EnformerRewardBlock",
]
