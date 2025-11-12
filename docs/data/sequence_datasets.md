# SequenceDataset Overview

`strand.data.sequence_dataset.SequenceDataset` provides a typed, tokenization-aware way to load Ctrl-DNA (or general sequence) datasets for supervised fine-tuning and evaluation.

## Supported Formats

- **FASTA / FA** — standard `>id` headers with raw sequences.
- **CSV** — expects at least a `sequence` column; optional `id`, `label`, and cell-type columns.
- **JSON** — list of `{ "sequence": "...", "label": "..." }` dicts.

## Configuration Fields

| Field | Description |
| --- | --- |
| `data_path` | Path to FASTA/CSV/JSON file.
| `tokenizer` | Any object exposing `__call__` that returns `input_ids` + `attention_mask` (HyenaDNA tokenizer works great).
| `max_seq_len` / `min_seq_len` | Length filtering to keep dataset in the model’s context window.
| `cell_type_column` | Column name in CSV files for cell-type labels.
| `validation_split` | Fraction held out for validation (default 0.1).
| `random_seed` | Deterministic split control.

## Basic Usage

```python
from strand.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig
from strand.models.hyenadna import load_hyenadna_from_hub

hyena = load_hyenadna_from_hub(model_name="hyenadna-tiny-1k", device="cuda")

dataset = SequenceDataset(
    SequenceDatasetConfig(
        data_path="data/promoters/mock_promoters.fasta",
        tokenizer=hyena.tokenizer,
        max_seq_len=1024,
        validation_split=0.1,
    )
)

train_loader = dataset.train_loader(batch_size=32, shuffle=True)
val_loader = dataset.val_loader(batch_size=32)

for batch in train_loader:
    print(batch.input_ids.shape, batch.attention_mask.shape)
```

`SequenceBatch` objects expose `input_ids`, `attention_mask`, `sequences` (original `Sequence` objects), optional `labels`, and optional `cell_types`.

## Dataset Preparation Script

Run the helper to generate starter datasets:

```bash
python scripts/datasets/ctrl_dna/download_promoters.py --output-dir data/promoters --source mock
```

The script currently ships with a mock generator; hooks for GSE/TCGA downloads are scaffolded but not implemented yet.

## Integration Tips

- Use `SequenceDataset` alongside `Engine(..., sft=SFTConfig(...))` so strategies that implement `warm_start` (like `RLPolicyStrategy`) automatically pre-train before RL.
- Attach `SequenceDataset` (or just its config path) to your MLflow run so you can trace provenance.
- When mixing multiple assays, create one dataset per assay and register them in your config files so strategies can check `needs_sft_dataset` before running.

## Limitations

- The dataset module does not currently stream extremely large corpora; it loads everything into memory. Add chunked iterators if you need bigger-than-RAM datasets.
- Label semantics are free-form today. Future releases will tighten this schema once we standardize multi-cell SFT datasets.
