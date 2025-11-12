# Quick Start

This guide shows the smallest possible Strand optimization loop using the **real** SDK surfaces that ship today. Copy the snippets into a fresh virtual environment after installing the SDK in editable mode (`pip install -e .`).

## 1. Minimal Engine Run

```python
from strand.engine.engine import Engine, EngineConfig
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies import RandomStrategy
from strand.engine.score import default_score
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward

strategy = RandomStrategy(alphabet="ACGT", min_len=48, max_len=64, seed=1337)
rewards = RewardAggregator([GCContentReward(target=0.5, tolerance=0.1, weight=1.0)])
executor = LocalExecutor(evaluator=rewards, batch_size=32)

engine = Engine(
    config=EngineConfig(iterations=5, population_size=64, method="random"),
    strategy=strategy,
    evaluator=rewards,
    executor=executor,
    score_fn=default_score,
)

results = engine.run()
print("Best sequence:", results.best)
```

Why this works:
- `RewardAggregator` bundles any set of reward blocks and already behaves like an evaluator.
- `LocalExecutor` simply iterates over batches, so you can stay in pure Python while experimenting.
- `default_score` sums `metrics.objective` and subtracts rule penalties when present.

## 2. Add Constraints & Rules

```python
from strand.engine.strategies import CEMStrategy
from strand.evaluators.composite import CompositeEvaluator
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward
from strand.engine.constraints import BoundedConstraint, Direction
from strand.engine.rules import Rules

constraints = [
    BoundedConstraint(name="length", direction=Direction.LE, bound=60.0),
]
rules = Rules(init={"length": 0.2})
rewards = RewardAggregator([GCContentReward(target=0.5)])

engine = Engine(
    config=EngineConfig(iterations=20, population_size=96, method="cem"),
    strategy=CEMStrategy(alphabet="ACGT", min_len=48, max_len=80, seed=8),
    evaluator=CompositeEvaluator(
        rewards=rewards,
        include_length=True,
    ),
    executor=LocalExecutor(evaluator=rewards, batch_size=48),
    score_fn=default_score,
    constraints=constraints,
    rules=rules,
)
```

Constraints live entirely in the engine: evaluators just need to surface per-sequence measurements in `metrics.constraints` (here, sequence length). Rule weights can be updated externally or left static.

## 3. Batching & Device Hints

Strategies that declare `requires_runtime=True` or `supports_fine_tuning=True` receive a `StrategyContext` with device and batch hints. You configure those via `EngineConfig`:

```python
from strand.engine.runtime import BatchConfig, DeviceConfig

config = EngineConfig(
    iterations=60,
    population_size=256,
    batching=BatchConfig(eval_size=64, train_size=16, max_tokens=2048),
    device=DeviceConfig(target="cuda", mixed_precision="bf16", gradient_accumulation_steps=2),
)
```

- **BatchConfig** limits executor/evaluator workload by item count or total tokens.
- **DeviceConfig** describes where the shared `ModelRuntime` should live and whether to use mixed precision.
- `StrategyRuntimeAdapter` (in `strand.engine.strategies.runtime_adapter`) wraps the `ModelRuntime` methods you need to prepare `nn.Module`s, checkpoints, and autocast contexts for custom strategies.

## 4. When to Use RL Policy or HyenaDNA

- Use `strand.engine.strategies.RLPolicyStrategy` when you want an online policy-gradient loop. The strategy exposes `train_step` **and** a `warm_start(dataset, *, epochs, batch_size, context)` hook, so you can pass an `SFTConfig` to `Engine` and let it pre-train automatically before the RL loop.
- Load a HyenaDNA backbone via `strand.models.hyenadna.load_hyenadna_from_hub` (returns tokenizer + model). Plug it into the policy-head utilities under `strand.engine.strategies.rl.policy_heads` if you are crafting your own RL strategy variant.

## 5. Next Steps

- [Core Concepts](./core_concepts.md) — Deep dive on strategies, evaluators, manifests, and constraint handling.
- [Ctrl-DNA Pipeline](./ctrl_dna_pipeline.md) — Foundation-model workflow + current limitations.
- [SequenceDataset Overview](../data/sequence_datasets.md) — Expected file formats for SFT workflows.
- [StrategyContext Guide](../architecture/strategy_context.md) — How runtimes/device configs reach your strategy.

> **Tip:** Use `strand run configs/examples/ctrl_dna/hyenadna_rl.yaml` to execute a config-driven run without writing a script.
