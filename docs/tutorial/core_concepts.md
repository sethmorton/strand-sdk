# Core Concepts

The Strand SDK revolves around a small set of composable building blocks. This document ties those blocks back to the concrete modules that ship in the repository today so you know exactly what is implemented.

## 1. Sequence

`strand.core.sequence.Sequence` is an immutable dataclass with `id`, `tokens`, and optional metadata. Strategies produce `Sequence` objects, evaluators consume them, and manifests serialize them.

```python
from strand.core.sequence import Sequence

seq = Sequence(id="candidate_001", tokens="ACGTACGT")
print(len(seq))  # 8
```

## 2. Reward Blocks & Evaluators

- Basic reward blocks live under `strand.rewards` (stability, solubility, novelty, GC content, length penalty).
- Advanced reward blocks (`EnformerRewardBlock`, `TFBSFrequencyCorrelationBlock`) live under `strand.rewards.advanced` and require extra dependencies (`onnxruntime`, `JASPAR2024`, `biopython`).
- `strand.evaluators.reward_aggregator.RewardAggregator` wraps any list of reward blocks and already satisfies the `Evaluator` protocol.
- `strand.evaluators.composite.CompositeEvaluator` combines the aggregated objective with optional constraint metrics (length, GC, novelty) so rule managers have structured inputs.

```python
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward

rewards = RewardAggregator([
    GCContentReward(target=0.5, tolerance=0.08, weight=0.7),
])
```

## 3. Strategies & StrategyCaps

Strategies adhere to the `strand.engine.interfaces.Strategy` protocol: `ask`, `tell`, `best`, `state`, with optional `prepare`, `train_step`, and `strategy_caps` methods.

- Evolutionary strategies are under `strand.engine.strategies.evolutionary` (Random, CEM, GA, CMA-ES, CMA-ES variable length).
- RL strategy lives at `strand.engine.strategies.rl.rl_policy.RLPolicyStrategy` and declares `StrategyCaps(requires_runtime=True, supports_fine_tuning=True, kl_regularization="token")`.
- `StrategyCaps` (defined in `strand.engine.runtime`) let the engine know whether to build a `ModelRuntime`, how to size batches, and whether the strategy needs supervised data.

```python
strategy = RLPolicyStrategy(alphabet="ACGT", min_len=50, max_len=200)
caps = strategy.strategy_caps()
assert caps.requires_runtime is True
```

## 4. StrategyContext & ModelRuntime

When a strategy requests a runtime, the engine calls `build_strategy_context` (also in `strand.engine.runtime`). The context includes:

- `DeviceConfig` — target device (`cpu`/`cuda`), mixed precision, gradient accumulation.
- `BatchConfig` — optional evaluator/train batch sizes and token budgets.
- `ModelRuntime` — thin wrapper around Hugging Face Accelerate. Use it to `prepare_module`, `autocast`, `backward`, and `wait`.

`strand.engine.strategies.runtime_adapter.StrategyRuntimeAdapter` is a helper you can instantiate with the runtime to standardize module preparation, checkpointing, and mixed-precision contexts for custom strategies.

## 5. Executors

Executors satisfy `strand.engine.interfaces.Executor` and parallelize evaluators while preserving sequence order.

- `LocalExecutor` — sequential baseline with batch-size control.
- `LocalPoolExecutor` — thread/process pool parallelism while keeping order.
- `TorchExecutor` — cooperates with a `ModelRuntime` to enforce autocast, token budgets, and gradient-friendly batching.

Use the executors directly for now; `ExecutorFactory` exists but still targets an older constructor signature (see “Known Gaps”).

## 6. Engine Loop & Scoring

`strand.engine.engine.Engine` composes everything. It repeatedly:

1. Calls `strategy.ask(population_size)`
2. Uses the executor to evaluate batches
3. Builds `Metrics` objects (objective + constraints + aux)
4. Calls `score_fn(metrics, rules, constraints)` (defaults to `strand.engine.score.default_score`)
5. Invokes `strategy.tell(...)` and optional `strategy.train_step(...)`
6. Tracks `IterationStats` for manifests/logging

## 7. Constraints & Dual Variables

- Declare static bounds with `strand.engine.constraints.BoundedConstraint` and optionally attach `strand.engine.rules.Rules` to maintain weights.
- Use `strand.engine.constraints.dual.DualVariableManager` (or `DualVariableSet`) when you want CBROP-style adaptive penalties. Each manager logs violation histories and exposes summaries for MLflow/manifests.

## 8. SequenceDataset & Data Scripts

`strand.data.sequence_dataset.SequenceDataset` converts FASTA/CSV/JSON files into PyTorch-ready batches that can be passed straight into `Engine` via `SFTConfig`. It handles:

- Train/validation splits
- Length filtering
- Optional cell-type labels
- Tokenization via any tokenizer (e.g., HyenaDNA)

Pair it with the `scripts/datasets/ctrl_dna/download_promoters.py` helper to create mock promoter libraries before training. Pass the dataset to `Engine(..., sft=SFTConfig(dataset, epochs=3))` and any strategy that implements `warm_start` (e.g., `RLPolicyStrategy`) will pre-train automatically before RL begins.

## 9. Results, Manifests, and Tracking

- `Engine.run()` returns `EngineResults` (best sequence, per-iteration history, summary dictionary).
- MLflow support lives in `strand.logging.mlflow_tracker.MLflowTracker`. It logs engine configs, per-iteration stats, summary metrics, artifacts (JSON manifests, checkpoints), and now exposes `log_sft_metrics`/`log_sft_checkpoint` helpers for supervised warm-starts.
- Export manifests however you like (JSON/TOML) by serializing the `EngineResults.summary` block together with your own metadata.

## 10. Known Gaps (Documented on Purpose)

- `SequenceDataset` currently loads entire datasets into memory; use smaller FASTA/CSV chunks or extend it with streaming if you need >RAM corpora.
- `scripts/datasets/ctrl_dna/download_promoters.py` only ships a mock generator today (no automatic GSE/TCGA fetch). Fill in the downloader hooks if you have access to those repositories.
- HyenaDNA/Enformer weights are not bundled for licensing reasons—point the configs to your own checkpoints or Hugging Face references.

Documenting these limitations here keeps the rest of the docs honest and gives you a checklist if you want to help land the remaining features.
