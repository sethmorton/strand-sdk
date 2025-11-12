# Ctrl-DNA Pipeline (Alpha Status)

This tutorial assembles the Ctrl-DNA building blocks that already exist in the SDK: HyenaDNA loaders, SequenceDataset utilities, advanced reward blocks (Enformer + TFBS), RL strategies, and dual-variable logging. It also calls out what is **not** finished yet so you have an accurate picture.

## Pipeline at a Glance

1. **Load a foundation model** (HyenaDNA tokenizer + autoregressive backbone)
2. **Prepare a dataset** (FASTA/CSV/JSON via `SequenceDataset` or the Ctrl-DNA downloader script)
3. **Stack rewards** (Enformer + TFBS + heuristics)
4. **Configure dual variables & constraints**
5. **Run RLPolicyStrategy** with evaluator + executor inside `Engine`
6. **Log results** via MLflow + manifests

A reference config lives at `configs/examples/ctrl_dna/hyenadna_rl.yaml`. Use it as a starting point for your own Hydra/OmegaConf workflows.

## 1. Load HyenaDNA

```python
import torch
from strand.models.hyenadna import load_hyenadna_from_hub

hyena = load_hyenadna_from_hub(
    model_name="hyenadna-tiny-1k",
    device="cuda",
    dtype=torch.bfloat16,
)

tokenizer = hyena.tokenizer
model = hyena.model
print("Max context:", hyena.config.max_seq_len)
```

You can also load local checkpoints with `load_hyenadna_from_checkpoint("path/to/ckpt")`.

## 2. Prepare Sequence Data

```python
from pathlib import Path
from strand.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig

config = SequenceDatasetConfig(
    data_path=Path("data/promoters/mock_promoters.fasta"),
    tokenizer=tokenizer,
    max_seq_len=1024,
    validation_split=0.1,
)
dataset = SequenceDataset(config)

train_loader = dataset.train_loader(batch_size=32, shuffle=True)
val_loader = dataset.val_loader(batch_size=32)
```

Need sample data? Run `python scripts/datasets/ctrl_dna/download_promoters.py --output-dir data/promoters` to create a mock FASTA library.

Pass this dataset into `Engine(..., sft=SFTConfig(dataset, epochs=3, batch_size=32))` and `RLPolicyStrategy.warm_start` will tokenize, fine-tune, and log SFT metrics automatically before RL starts.

## 3. Stack Reward Blocks

```python
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward
from strand.rewards.advanced import EnformerRewardBlock, EnformerConfig, TFBSFrequencyCorrelationBlock, TFBSConfig

reward_blocks = [
    EnformerRewardBlock(
        EnformerConfig(
            cell_types=["hNSPC"],
            weight=0.6,
            backend="onnx",
            model_path="weights/enformer_hnspc.onnx",
        )
    ),
    TFBSFrequencyCorrelationBlock(
        TFBSConfig(
            motifs=["CEBPB", "STAT1"],
            target_profile_path="configs/targets/hnspc_tfbs.json",
            weight=0.3,
        )
    ),
    GCContentReward(target=0.52, tolerance=0.05, weight=0.1),
]

rewards = RewardAggregator(reward_blocks)
```

Install extra dependencies with `pip install -e .[models,inference]` before using Enformer/TFBS blocks.

## 4. Constraints & Dual Variables

```python
from strand.engine.constraints import BoundedConstraint, Direction
from strand.engine.constraints.dual import DualVariableSet

constraints = [
    BoundedConstraint(name="off_target", direction=Direction.LE, bound=0.15),
    BoundedConstraint(name="length", direction=Direction.LE, bound=600.0),
]

dual_vars = DualVariableSet()
dual_vars.add_constraint("off_target", init_weight=1.0, adaptive_step=0.2)
dual_vars.add_constraint("length", init_weight=0.3, adaptive_step=0.1)

# Example: update after each iteration using recorded violations
violations = {"off_target": 0.18, "length": 580.0}
weights = dual_vars.update_all(violations)
print("Updated dual weights", weights)

> Ensure your evaluator populates `metrics.constraints["off_target"]` (for example, by exposing the TFBS divergence metric) so `BoundedConstraint` and dual updates receive real measurements.
```

Use the summaries emitted by `dual_vars.log_summary()` to debug constraint drift in MLflow or console logs.

## 5. Run the RL Engine

```python
from strand.engine.engine import Engine, EngineConfig
from strand.engine.executors.local import LocalExecutor
from strand.engine.runtime import BatchConfig, DeviceConfig
from strand.engine.strategies.rl.rl_policy import RLPolicyStrategy
from strand.engine.score import default_score
from strand.evaluators.composite import CompositeEvaluator

device_cfg = DeviceConfig(target="cuda", mixed_precision="bf16")
batch_cfg = BatchConfig(eval_size=64, train_size=16, max_tokens=2048)

evaluator = CompositeEvaluator(
    rewards=rewards,
    include_length=True,
    include_gc=False,
)

engine = Engine(
    config=EngineConfig(
        iterations=75,
        population_size=128,
        method="rl-policy",
        device=device_cfg,
        batching=batch_cfg,
    ),
    strategy=RLPolicyStrategy(alphabet="ACGT", min_len=200, max_len=600, seed=7),
    evaluator=evaluator,
    executor=LocalExecutor(evaluator=evaluator, batch_size=64),
    score_fn=default_score,
    constraints=constraints,
)

results = engine.run()
dual_vars.log_summary()
print("Best sequence:", results.best)
```

Tips:
- Switch to `LocalPoolExecutor` if your reward stack is CPU-bound and parallel-friendly.
- Use `TorchExecutor` when your evaluator is GPU-heavy and you have a `ModelRuntime` handy.

## 6. Logging & Reproducibility

```python
from strand.logging.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(experiment_name="ctrl-dna-alpha", tracking_uri="./mlruns")
tracker.start_run(run_name="hnspc_rl")
tracker.log_config(engine._config)  # private access, but convenient inside scripts

for step, stats in enumerate(results.history):
    tracker.log_iteration_stats(step, stats)

for epoch in range(rl_epochs := 3):  # replace with your actual SFT loop stats
    tracker.log_sft_metrics(epoch=epoch, loss=0.42, accuracy=0.91, kl=0.06)

tracker.log_results(results)
tracker.end_run()
```

Call `tracker.log_sft_metrics` (and optionally `tracker.log_sft_checkpoint`) from your warm-start loop so SFT loss/accuracy/KL sit alongside RL metrics.

## Current Limitations (Explicit)

| Area | Status |
| --- | --- |
| SequenceDataset streaming | Entire dataset is loaded into memory. Chunk massive corpora manually or extend the loader with streaming. |
| Ctrl-DNA dataset downloader | Only emits mock FASTA files today. Fill in the GSE/TCGA hooks if you need real promoters. |
| Foundation model checkpoints | HyenaDNA/Enformer weights are not bundled; reference your own checkpoints or Hugging Face repositories in configs. |

Keeping these gaps visible makes it clear what remains for full Ctrl-DNA parity. Contributions welcome!
