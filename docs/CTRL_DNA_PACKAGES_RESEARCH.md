# Ctrl-DNA Dependencies (Alpha)

This note tracks the non-standard packages required for the Ctrl-DNA parity plan.

## Core (installed via `requirements.txt` / editable install)

- `torch`, `torchvision`, `torchaudio`
- `accelerate`
- `numpy`, `scipy`, `pandas`

## Extras

| Extra | Packages | Purpose |
| --- | --- | --- |
| `models` | `transformers`, `tokenizers`, `filelock` | HyenaDNA loaders + tokenizers |
| `rl-training` | `torchrl`, `pytorch-lightning` | Future high-level RL / SFT helpers |
| `inference` | `onnx`, `onnxruntime` | Enformer ONNX support |
| `bio` | `biopython`, `JASPAR2024` | TFBS motif parsing |
| `logging` | `mlflow`, `tensorboard` | Experiment tracking |
| `dev` | `pytest`, `pytest-xdist`, `pytest-benchmark`, `black`, `ruff` | Contributor workflow |

## Installation Cheatsheet

```bash
# Foundation models + advanced rewards
pip install -e .[models,inference,bio]

# Full-stack RL + dev tooling
pip install -e .[models,rl-training,inference,bio,logging,dev]
```

Keep this file up to date as new packages land so contributors know which extras to install when following the Ctrl-DNA tutorial.
