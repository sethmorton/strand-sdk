"""Policy head implementations for RL strategies.

This module provides pluggable policy head architectures that can be injected
into RL strategies (e.g., RLPolicyStrategy). Each head converts a representation
into token logits suitable for categorical distribution sampling.

Available Heads
===============

1. **PerPositionLogitsHead**: Per-position token logits (baseline, no conditioning).
2. **HyenaDNAHead**: Uses HyenaDNA backbone to generate context-aware logits.
3. **TransformerHead**: Transformer-based policy head for dependencies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)


class PolicyHead(ABC, nn.Module):
    """Base class for policy heads.

    A policy head takes sequence context and produces token logits at each position.
    Subclasses implement specific architectures.
    """

    @abstractmethod
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate logits for token sampling.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch data. Common keys:
            - "input_ids": (batch_size, seq_len) token IDs
            - "attention_mask": (batch_size, seq_len) attention weights
            - "lengths": (batch_size,) actual sequence lengths
        **kwargs : object
            Additional arguments (e.g., temperature, top_k).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size).
        """

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size for this head.

        Returns
        -------
        int
            Vocabulary size (e.g., 4 for DNA).
        """


@dataclass(frozen=True, slots=True)
class PerPositionLogitsHeadConfig:
    """Configuration for PerPositionLogitsHead.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    vocab_size : int
        Vocabulary size (e.g., 4 for DNA).
    """

    max_seq_len: int
    vocab_size: int


class PerPositionLogitsHead(PolicyHead):
    """Per-position logits head (no conditioning on context).

    This is the simplest policy head: each position has independent learnable
    logits. It's equivalent to the baseline approach in RLPolicyStrategy.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    vocab_size : int
        Vocabulary size.
    logits : nn.Parameter
        Learnable logits of shape (max_seq_len, vocab_size).
    """

    def __init__(self, config: PerPositionLogitsHeadConfig) -> None:
        """Initialize per-position logits head.

        Parameters
        ----------
        config : PerPositionLogitsHeadConfig
            Configuration with max_seq_len and vocab_size.
        """
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size

        # Initialize learnable logits per position
        self.logits = nn.Parameter(
            torch.randn(config.max_seq_len, config.vocab_size) * 0.01
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:
        """Return per-position logits.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch data (used only for batch size).
        **kwargs : object
            Ignored.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("batch must contain 'input_ids'")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Expand logits to batch size: (seq_len, vocab_size) -> (batch_size, seq_len, vocab_size)
        return self.logits[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


@dataclass(frozen=True, slots=True)
class HyenaDNAHeadConfig:
    """Configuration for HyenaDNA-based policy head.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    vocab_size : int
        Vocabulary size.
    backbone_model : nn.Module
        Pre-trained HyenaDNA backbone (or compatible autoregressive model).
    backbone_hidden_size : int
        Hidden size of backbone (embedding dimension).
    freeze_backbone : bool
        If True, don't update backbone weights during training.
    """

    max_seq_len: int
    vocab_size: int
    backbone_model: nn.Module
    backbone_hidden_size: int
    freeze_backbone: bool = True


class HyenaDNAHead(PolicyHead):
    """Policy head using HyenaDNA as backbone.

    Uses the HyenaDNA model's contextualized embeddings to generate position-aware
    logits. This allows the policy to be conditioned on the sequence content.

    Attributes
    ----------
    config : HyenaDNAHeadConfig
        Configuration.
    backbone : nn.Module
        HyenaDNA model (frozen if requested).
    proj_head : nn.Linear
        Projects backbone output to vocab_size logits.
    """

    def __init__(self, config: HyenaDNAHeadConfig) -> None:
        """Initialize HyenaDNA-based policy head.

        Parameters
        ----------
        config : HyenaDNAHeadConfig
            Configuration with backbone and projection settings.
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size

        # Register backbone (may be frozen)
        self.backbone = config.backbone_model
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            _LOGGER.info("HyenaDNA backbone frozen")

        # Projection head: backbone_hidden_size -> vocab_size
        self.proj_head = nn.Linear(config.backbone_hidden_size, config.vocab_size)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate logits using HyenaDNA backbone.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch data with:
            - "input_ids": (batch_size, seq_len) token IDs
            - "attention_mask": (batch_size, seq_len) optional
        **kwargs : object
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        if input_ids is None:
            raise ValueError("batch must contain 'input_ids'")

        # Get backbone embeddings
        with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
            backbone_output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states (typically the last layer)
        if hasattr(backbone_output, "hidden_states"):
            hidden_states = backbone_output.hidden_states[-1]
        elif hasattr(backbone_output, "last_hidden_state"):
            hidden_states = backbone_output.last_hidden_state
        else:
            # Assume backbone_output itself is the hidden states
            hidden_states = backbone_output

        # Project to logits
        logits = self.proj_head(hidden_states)  # (batch_size, seq_len, vocab_size)

        return logits

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


@dataclass(frozen=True, slots=True)
class TransformerHeadConfig:
    """Configuration for Transformer-based policy head.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    vocab_size : int
        Vocabulary size.
    hidden_size : int
        Hidden dimension (embedding dimension).
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Feedforward inner dimension.
    """

    max_seq_len: int
    vocab_size: int
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512


class TransformerHead(PolicyHead):
    """Transformer-based policy head.

    Uses transformer encoder layers to capture dependencies and generate context-aware logits.
    """

    def __init__(self, config: TransformerHeadConfig) -> None:
        """Initialize Transformer policy head.

        Parameters
        ----------
        config : TransformerHeadConfig
            Configuration with dimensions and architecture.
        """
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embedding
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projection
        self.proj_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate logits using Transformer encoder.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch data with "input_ids" of shape (batch_size, seq_len).
        **kwargs : object
            Additional arguments (ignored).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        if input_ids is None:
            raise ValueError("batch must contain 'input_ids'")

        batch_size, seq_len = input_ids.shape

        # Embedding + positional encoding
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, hidden_size)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = self.pos_embedding(positions)
        x = embeddings + positions

        # Create attention mask if provided
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Project to logits
        logits = self.proj_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


def create_policy_head(
    head_type: str,
    **kwargs: object,
) -> PolicyHead:
    """Factory function to create a policy head.

    Parameters
    ----------
    head_type : str
        Type of head: "per-position", "hyenadna", "transformer".
    **kwargs : object
        Configuration parameters passed to the head's config dataclass.

    Returns
    -------
    PolicyHead
        Instantiated policy head.

    Raises
    ------
    ValueError
        If head_type is unknown.
    """
    if head_type == "per-position":
        config = PerPositionLogitsHeadConfig(**kwargs)  # type: ignore[arg-type]
        return PerPositionLogitsHead(config)

    if head_type == "hyenadna":
        config = HyenaDNAHeadConfig(**kwargs)  # type: ignore[arg-type]
        return HyenaDNAHead(config)

    if head_type == "transformer":
        config = TransformerHeadConfig(**kwargs)  # type: ignore[arg-type]
        return TransformerHead(config)

    raise ValueError(
        f"Unknown head_type: {head_type}. "
        "Choose from: per-position, hyenadna, transformer"
    )


__all__ = [
    "PolicyHead",
    "PerPositionLogitsHead",
    "PerPositionLogitsHeadConfig",
    "HyenaDNAHead",
    "HyenaDNAHeadConfig",
    "TransformerHead",
    "TransformerHeadConfig",
    "create_policy_head",
]

