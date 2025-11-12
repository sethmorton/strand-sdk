"""Sequence datasets for supervised fine-tuning and evaluation.

This module provides utilities to load and manage DNA/protein sequences for
supervised fine-tuning workflows in RL strategies. Supports FASTA, CSV, and
streaming from URLs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from strand.core.sequence import Sequence

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SequenceDatasetConfig:
    """Configuration for SequenceDataset.

    Attributes
    ----------
    data_path : str | Path
        Path to dataset file (FASTA, CSV, or JSON).
    tokenizer : object
        Tokenizer for encoding sequences (e.g., from HyenaDNA).
    max_seq_len : int | None
        Filter out sequences longer than this. None means no limit.
    min_seq_len : int | None
        Filter out sequences shorter than this. None means no limit.
    cell_type_column : str | None
        For CSV files, column name containing cell type labels.
    validation_split : float
        Fraction of data to reserve for validation (0.0-1.0).
    random_seed : int
        Random seed for train/val split.
    """

    data_path: str | Path
    tokenizer: object
    max_seq_len: int | None = None
    min_seq_len: int | None = None
    cell_type_column: str | None = None
    validation_split: float = 0.1
    random_seed: int = 42


@dataclass(slots=True)
class SequenceBatch:
    """A batch of sequences ready for model input.

    Attributes
    ----------
    input_ids : torch.Tensor
        Token IDs of shape (batch_size, seq_len).
    attention_mask : torch.Tensor
        Attention mask of shape (batch_size, seq_len).
    sequences : list[Sequence]
        Original Sequence objects.
    labels : torch.Tensor | None
        Optional labels for supervised learning.
    cell_types : list[str] | None
        Optional cell type labels.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    sequences: list[Sequence]
    labels: torch.Tensor | None = None
    cell_types: list[str] | None = None


class SequenceDataset(Dataset):
    """Sequence dataset for supervised fine-tuning.

    Loads DNA/protein sequences from FASTA or CSV files and provides PyTorch
    DataLoader-compatible batching.

    Supports:
    - FASTA format (standard)
    - CSV format (with optional cell type labels)
    - JSON format (list of {"sequence": "ACGT", "label": "..."} dicts)
    - Filtering by sequence length
    - Train/validation splits
    - Tokenization for autoregressive models

    Example
    -------
    >>> config = SequenceDatasetConfig(
    ...     data_path="sequences.fasta",
    ...     tokenizer=hyenadna_tokenizer,
    ...     max_seq_len=1024,
    ... )
    >>> dataset = SequenceDataset(config)
    >>> train_loader = dataset.train_loader(batch_size=32)
    >>> for batch in train_loader:
    ...     # batch is a SequenceBatch
    ...     input_ids, attn_mask = batch.input_ids, batch.attention_mask
    """

    def __init__(self, config: SequenceDatasetConfig) -> None:
        """Initialize dataset.

        Parameters
        ----------
        config : SequenceDatasetConfig
            Dataset configuration.
        """
        self.config = config
        self.sequences: list[Sequence] = []
        self.labels: list[str | None] = []
        self.cell_types: list[str] = []

        self._load_sequences()
        self._split_train_val()

    def _load_sequences(self) -> None:
        """Load sequences from file.

        Raises
        ------
        ValueError
            If file format is not recognized.
        FileNotFoundError
            If data file doesn't exist.
        """
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".fasta" or suffix == ".fa":
            self._load_fasta(path)
        elif suffix == ".csv":
            self._load_csv(path)
        elif suffix == ".json":
            self._load_json(path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported: .fasta, .fa, .csv, .json"
            )

        _LOGGER.info(
            f"Loaded {len(self.sequences)} sequences from {path}. "
            f"Filtered to {len(self.sequences)} after length filtering."
        )

    def _load_fasta(self, path: Path) -> None:
        """Load sequences from FASTA file."""
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError(
                "BioPython is required for FASTA parsing. "
                "Install with: pip install biopython"
            )

        for i, record in enumerate(SeqIO.parse(path, "fasta")):
            seq_str = str(record.seq).upper()

            if not self._passes_length_filter(seq_str):
                continue

            sequence = Sequence(id=record.id, tokens=seq_str)
            self.sequences.append(sequence)
            self.labels.append(None)
            self.cell_types.append("unknown")

    def _load_csv(self, path: Path) -> None:
        """Load sequences from CSV file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for CSV parsing. "
                "Install with: pip install pandas"
            )

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            seq_str = str(row.get("sequence", "")).upper()

            if not self._passes_length_filter(seq_str):
                continue

            seq_id = str(row.get("id", len(self.sequences)))
            sequence = Sequence(id=seq_id, tokens=seq_str)
            self.sequences.append(sequence)

            # Extract label and cell type
            label = row.get("label")
            self.labels.append(label if label is not None else None)

            cell_type = str(row.get(self.config.cell_type_column, "unknown")) \
                if self.config.cell_type_column else "unknown"
            self.cell_types.append(cell_type)

    def _load_json(self, path: Path) -> None:
        """Load sequences from JSON file (list of dicts)."""
        import json

        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of sequence records")

        for i, record in enumerate(data):
            seq_str = str(record.get("sequence", "")).upper()

            if not self._passes_length_filter(seq_str):
                continue

            seq_id = str(record.get("id", i))
            sequence = Sequence(id=seq_id, tokens=seq_str)
            self.sequences.append(sequence)

            label = record.get("label")
            self.labels.append(label if label is not None else None)

            cell_type = str(record.get("cell_type", "unknown"))
            self.cell_types.append(cell_type)

    def _passes_length_filter(self, seq: str) -> bool:
        """Check if sequence passes length constraints."""
        seq_len = len(seq)
        if self.config.min_seq_len is not None and seq_len < self.config.min_seq_len:
            return False
        if self.config.max_seq_len is not None and seq_len > self.config.max_seq_len:
            return False
        return True

    def _split_train_val(self) -> None:
        """Split dataset into train and validation sets."""
        import random

        random.seed(self.config.random_seed)
        indices = list(range(len(self.sequences)))
        random.shuffle(indices)

        split_idx = int(len(indices) * (1 - self.config.validation_split))
        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]

        _LOGGER.info(
            f"Train/val split: {len(self.train_indices)} / {len(self.val_indices)}"
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Get a single sample.

        Parameters
        ----------
        idx : int
            Index.

        Returns
        -------
        dict[str, object]
            Dictionary with keys: "sequence", "input_ids", "attention_mask", etc.
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        cell_type = self.cell_types[idx]

        # Tokenize
        encoding = self.config.tokenizer(
            sequence.tokens,
            return_tensors="pt",
            max_length=self.config.max_seq_len,
            truncation=True,
            padding="max_length" if self.config.max_seq_len else "longest",
        )

        return {
            "sequence": sequence.tokens,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding.get("attention_mask", torch.ones_like(encoding["input_ids"])).squeeze(0),
            "label": label,
            "cell_type": cell_type,
        }

    def train_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get DataLoader for training split.

        Parameters
        ----------
        batch_size : int
            Batch size.
        shuffle : bool
            Whether to shuffle.
        num_workers : int
            Number of data loading workers.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for training data.
        """
        subset = torch.utils.data.Subset(self, self.train_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_sequences,
        )

    def val_loader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get DataLoader for validation split.

        Parameters
        ----------
        batch_size : int
            Batch size.
        num_workers : int
            Number of data loading workers.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for validation data.
        """
        subset = torch.utils.data.Subset(self, self.val_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_sequences,
        )


def _collate_sequences(batch: list[dict]) -> SequenceBatch:
    """Collate function for DataLoader.

    Parameters
    ----------
    batch : list[dict]
        List of items from dataset.

    Returns
    -------
    SequenceBatch
        Batched sequences.
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Collect sequences
    sequences = [Sequence(id=f"seq_{i}", tokens=item["sequence"]) for i, item in enumerate(batch)]

    # Collect cell types
    cell_types = [item.get("cell_type", "unknown") for item in batch]

    return SequenceBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequences=sequences,
        cell_types=cell_types,
    )


__all__ = [
    "SequenceDatasetConfig",
    "SequenceBatch",
    "SequenceDataset",
]

