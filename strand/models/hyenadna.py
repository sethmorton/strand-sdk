"""HyenaDNA foundation model loader for sequence generation.

HyenaDNA is a state-of-the-art autoregressive model for DNA sequences.
This module provides utilities to load pre-trained models and tokenizers
from local paths or the Hugging Face Model Hub.

Reference: https://github.com/HyenaDNA/HyenaDNA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from filelock import FileLock

_LOGGER = logging.getLogger(__name__)

# Default model configurations
DEFAULT_HUGGINGFACE_ORG = "hyena"
DEFAULT_MODEL_NAME = "hyenadna-tiny-1k"
DEFAULT_HF_MODEL_ID = f"{DEFAULT_HUGGINGFACE_ORG}/{DEFAULT_MODEL_NAME}"


@dataclass(frozen=True, slots=True)
class HyenaDNAConfig:
    """Configuration for HyenaDNA model loading.

    Attributes
    ----------
    model_id : str
        Hugging Face model ID (e.g., "hyena/hyenadna-tiny-1k") or local path.
    checkpoint_path : str | Path | None
        Optional local checkpoint file path. If provided, loads from local
        path instead of Hugging Face Hub.
    device : str
        Device to load model on ("cpu" or "cuda:X").
    dtype : torch.dtype
        Data type for model weights (e.g., torch.float32, torch.bfloat16).
    trust_remote_code : bool
        If True, trust remote code from Hugging Face (for custom architectures).
    cache_dir : str | Path | None
        Cache directory for downloaded models (uses HF default if None).
    """

    model_id: str = DEFAULT_HF_MODEL_ID
    checkpoint_path: str | Path | None = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    trust_remote_code: bool = True
    cache_dir: str | Path | None = None


@dataclass(frozen=True, slots=True)
class HyenaDNAModel:
    """Loaded HyenaDNA model and tokenizer pair.

    Attributes
    ----------
    model : nn.Module
        The autoregressive HyenaDNA model.
    tokenizer : object
        Tokenizer for encoding/decoding DNA sequences.
    config : HyenaDNAConfig
        Configuration used to load the model.
    vocab_size : int
        Size of the vocabulary (for this DNA model, typically 4 for A/C/G/T).
    max_seq_len : int
        Maximum sequence length the model can handle.
    """

    model: nn.Module
    tokenizer: object
    config: HyenaDNAConfig
    vocab_size: int
    max_seq_len: int


def load_hyenadna(config: HyenaDNAConfig) -> HyenaDNAModel:
    """Load a HyenaDNA model and tokenizer.

    Supports loading from:
    1. Local checkpoint (if checkpoint_path is provided)
    2. Hugging Face Model Hub (default)

    Parameters
    ----------
    config : HyenaDNAConfig
        Configuration specifying model source and loading options.

    Returns
    -------
    HyenaDNAModel
        Loaded model with tokenizer and metadata.

    Raises
    ------
    FileNotFoundError
        If local checkpoint path doesn't exist.
    ValueError
        If model loading fails or configuration is invalid.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library is required to load HyenaDNA models. "
            "Install with: pip install transformers"
        )

    _LOGGER.info(f"Loading HyenaDNA model: {config.model_id}")

    # Determine source: local checkpoint or Hugging Face Hub
    if config.checkpoint_path is not None:
        model_source = Path(config.checkpoint_path)
        if not model_source.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_source}")
        _LOGGER.info(f"Loading from local checkpoint: {model_source}")
    else:
        model_source = config.model_id

    # Load tokenizer with thread-safe caching
    try:
        if config.cache_dir:
            cache_dir = str(config.cache_dir)
        else:
            cache_dir = None

        # Use file lock to prevent concurrent downloads
        if cache_dir:
            lock_file = Path(cache_dir) / f"{config.model_id.replace('/', '_')}.lock"
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(lock_file))
        else:
            lock = FileLock("/tmp/hyenadna_download.lock")

        with lock:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_source),
                trust_remote_code=config.trust_remote_code,
                cache_dir=cache_dir,
            )
            _LOGGER.info("Tokenizer loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {model_source}: {e}")

    # Load model
    try:
        with lock:
            model = AutoModel.from_pretrained(
                str(model_source),
                trust_remote_code=config.trust_remote_code,
                cache_dir=cache_dir,
                torch_dtype=config.dtype,
                device_map=None,  # Manual device placement below
            )
            _LOGGER.info("Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_source}: {e}")

    # Move model to device
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    _LOGGER.info(f"Model moved to device: {device}")

    # Extract model metadata
    vocab_size = _get_vocab_size(tokenizer, model)
    max_seq_len = _get_max_seq_len(model)

    _LOGGER.info(f"Model vocab_size: {vocab_size}, max_seq_len: {max_seq_len}")

    return HyenaDNAModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def load_hyenadna_from_hub(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HyenaDNAModel:
    """Convenience function to load HyenaDNA from Hugging Face Hub.

    Parameters
    ----------
    model_name : str
        Model name (e.g., "hyenadna-tiny-1k", "hyenadna-small-32k").
    device : str
        Device to load on ("cpu" or "cuda:X").
    dtype : torch.dtype
        Data type for model weights.

    Returns
    -------
    HyenaDNAModel
        Loaded model with tokenizer.
    """
    model_id = f"{DEFAULT_HUGGINGFACE_ORG}/{model_name}"
    config = HyenaDNAConfig(model_id=model_id, device=device, dtype=dtype)
    return load_hyenadna(config)


def load_hyenadna_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HyenaDNAModel:
    """Load HyenaDNA from a local checkpoint directory.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to checkpoint directory containing model weights and tokenizer.
    device : str
        Device to load on.
    dtype : torch.dtype
        Data type for model weights.

    Returns
    -------
    HyenaDNAModel
        Loaded model with tokenizer.

    Raises
    ------
    FileNotFoundError
        If checkpoint path doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    config = HyenaDNAConfig(
        model_id=str(checkpoint_path),
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
    )
    return load_hyenadna(config)


def _get_vocab_size(tokenizer: object, model: nn.Module) -> int:
    """Extract vocabulary size from tokenizer or model.

    Parameters
    ----------
    tokenizer : object
        Tokenizer instance.
    model : nn.Module
        Model instance.

    Returns
    -------
    int
        Vocabulary size.
    """
    # Try tokenizer first
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size

    # Try model config
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return model.config.vocab_size

    # Default for DNA (A, C, G, T + special tokens)
    _LOGGER.warning("Could not determine vocab_size; defaulting to 4 (DNA bases)")
    return 4


def _get_max_seq_len(model: nn.Module) -> int:
    """Extract maximum sequence length from model config.

    Parameters
    ----------
    model : nn.Module
        Model instance.

    Returns
    -------
    int
        Maximum sequence length.
    """
    if hasattr(model, "config"):
        # Try common config attribute names
        for attr in ["max_position_embeddings", "max_seq_len", "max_length"]:
            if hasattr(model.config, attr):
                return getattr(model.config, attr)

    # Default based on HyenaDNA typical context window
    _LOGGER.warning("Could not determine max_seq_len; defaulting to 1024")
    return 1024


__all__ = [
    "HyenaDNAConfig",
    "HyenaDNAModel",
    "load_hyenadna",
    "load_hyenadna_from_hub",
    "load_hyenadna_from_checkpoint",
]

