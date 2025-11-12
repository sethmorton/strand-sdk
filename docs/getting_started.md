# Getting Started with Strand SDK

The Strand SDK packages strategies, evaluators, and runtime helpers so you can optimize DNA (or protein) sequences with either simple heuristics or foundation models. Everything below reflects the code that ships in `main` today.

## Quick Setup

1. **Create a Python 3.11+ environment.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the SDK.**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Your First Optimization

### Simple: Evolutionary search + GC content

```python
from strand.engine.engine import Engine, EngineConfig
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies import CEMStrategy
from strand.evaluators.composite import CompositeEvaluator
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward

strategy = CEMStrategy(alphabet="ACGT", min_len=50, max_len=120, seed=42)
rewards = RewardAggregator([GCContentReward(target=0.52, tolerance=0.08)])
evaluator = CompositeEvaluator(rewards=rewards, include_gc=True)
executor = LocalExecutor(evaluator=evaluator, batch_size=32)

engine = Engine(
    config=EngineConfig(iterations=10, population_size=64, method="cem"),
    strategy=strategy,
    evaluator=evaluator,
    executor=executor,
    score_fn=lambda metrics, *_: metrics.objective,
)

results = engine.run()
print("Best candidate:", results.best)
```

### RL Policy + Device Context

All strategies can declare capabilities via `strategy_caps()`. When a strategy sets `requires_runtime=True`, the engine will pass a `StrategyContext` containing `DeviceConfig`, `BatchConfig`, and a lazily-built `ModelRuntime` powered by Hugging Face Accelerate.

```python
from strand.engine.engine import EngineConfig
from strand.engine.runtime import DeviceConfig, BatchConfig
from strand.engine.strategies import RLPolicyStrategy

config = EngineConfig(
    iterations=50,
    population_size=128,
    device=DeviceConfig(target="cuda", mixed_precision="bf16"),
    batching=BatchConfig(eval_size=64, train_size=16, max_tokens=2048),
)
strategy = RLPolicyStrategy(alphabet="ACGT", min_len=50, max_len=200)
```

`Engine` automatically calls `strategy.prepare(context)`, runs `warm_start(...)` once if you pass an `SFTConfig`, and invokes `train_step(...)` each iteration so the policy can blend SFT + RL without extra plumbing.

### Supervised Datasets & Scripts

- `strand.data.sequence_dataset.SequenceDataset` loads FASTA/CSV/JSON datasets, applies min/max length filters, tokenizes sequences, and exposes PyTorch dataloaders.
- `scripts/datasets/ctrl_dna/download_promoters.py` fetches or synthesizes promoter libraries so you can bootstrap experiments without public GSE access.
- Pair the dataset loader with the HyenaDNA tokenizer returned by `strand.models.hyenadna.load_hyenadna_from_hub` when preparing Ctrl-DNA inputs.

## Project Structure

```
strand/
├── engine/strategies/     # Random, evolutionary, RL, Hybrid
├── engine/runtime.py      # DeviceConfig, BatchConfig, StrategyCaps
├── engine/executors/      # Local, LocalPool, Torch executors + factory
├── evaluators/            # RewardAggregator + composite evaluators
├── rewards/               # Basic + advanced reward blocks
├── models/                # HyenaDNA loader + configs
├── data/                  # SequenceDataset + batching helpers
├── engine/constraints/    # DualVariableManager utilities
└── scripts/datasets/      # Ctrl-DNA dataset downloaders
```

## Key Features

- ✅ **Strategies**: Random, CEM, GA, CMA-ES, RL Policy, Hybrid
- ✅ **Runtime context**: DeviceConfig + BatchConfig passed to strategies that ask for it
- ✅ **Reward Blocks**: GC content, novelty, stability, Enformer, TFBS correlation
- ✅ **Foundation Models**: HyenaDNA loader plus policy head implementations (manual wiring today)
- ✅ **Constraints**: Dual variable managers with summaries/logging
- ✅ **Datasets**: SequenceDataset + Ctrl-DNA download script
- ✅ **Tracking**: MLflow tracker, manifests, checkpoint helpers

## Next Steps

- **Examples**: `examples/engine_basic.py`, `engine_with_tracking.py`, `engine_ctrl_dna_hybrid.py`
- **Datasets**: `scripts/datasets/ctrl_dna/download_promoters.py` to create mock promoters
- **Docs**: `docs/tutorial/ctrl_dna_pipeline.md` for the Ctrl-DNA playbook (with SFT + RL warm-start flow)
- **CLI**: Use `strand run path/to/config.yaml` to launch full experiments from a single file
