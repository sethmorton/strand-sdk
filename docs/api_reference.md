# API Reference Map

Use this as a directory of implemented modules. Full docstrings live in the source files.

## Engine Core

- `strand.engine.engine` — `Engine`, `EngineConfig`, `EngineResults`, `IterationStats`.
- `strand.engine.score` — `default_score` helper.
- `strand.engine.interfaces` — `Strategy`, `Evaluator`, `Executor` protocols.
- `strand.engine.runtime` — `DeviceConfig`, `BatchConfig`, `StrategyCaps`, `StrategyContext`, `ModelRuntime`.
- `strand.engine.types` — `Metrics` dataclass.

## Strategies

- `strand.engine.strategies.random.RandomStrategy`
- `strand.engine.strategies.evolutionary` — `CEMStrategy`, `GAStrategy`, `CMAESStrategy`, `CMAESVarLenStrategy`.
- `strand.engine.strategies.hybrid.HybridStrategy`
- `strand.engine.strategies.rl.rl_policy.RLPolicyStrategy`
- `strand.engine.strategies.rl.policy_heads` — `PerPositionLogitsHead`, `HyenaDNAHead`, `TransformerHead`.
- `strand.engine.strategies.runtime_adapter.StrategyRuntimeAdapter`

## Evaluators & Rewards

- `strand.evaluators.reward_aggregator.RewardAggregator`
- `strand.evaluators.composite.CompositeEvaluator`
- `strand.rewards.*` — Stability, Solubility, Novelty, Length, GC blocks
- `strand.rewards.advanced` — Enformer + TFBS blocks

## Data & Models

- `strand.data.sequence_dataset.SequenceDataset` and `SequenceDatasetConfig`
- `strand.models.hyenadna` — `HyenaDNAConfig`, loader helpers

## Constraints & Rules

- `strand.engine.constraints.BoundedConstraint`, `Direction`
- `strand.engine.constraints.dual.DualVariableManager`, `DualVariableSet`
- `strand.engine.rules.Rules`

## Executors

- `strand.engine.executors.local.LocalExecutor`
- `strand.engine.executors.pool.LocalPoolExecutor`
- `strand.engine.executors.torch.TorchExecutor`
- `strand.engine.executors.factory.ExecutorFactory` (experimental, pending API sync)

## Logging & CLI

- `strand.logging.mlflow_tracker.MLflowTracker`
- `strand.cli.cli` — config-driven `strand run` command

## Examples

- `examples/engine_basic.py`
- `examples/engine_with_tracking.py`
- `examples/engine_ctrl_dna_hybrid.py`

Refer to the inline type hints and docstrings for detailed parameter descriptions until a generated reference site is published.
