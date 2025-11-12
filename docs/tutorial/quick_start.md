# Quick Start (Surfaces)

This walkthrough shows the smallest optimization loop with the new engine surfaces. Implementations are landing next; consider this your API preview.

## Installation

```bash
git clone https://github.com/sethmorton/strand-sdk.git
cd strand-sdk
pip install -e .
```

## Your First Optimization (API sketch)

```python
from strand.core.sequence import Sequence
from strand.rewards import RewardBlock
from strand.engine import (Engine, EngineConfig, BoundedConstraint, Direction, Rules, default_score)
from strand.engine.executors.local import LocalExecutor
from strand.evaluators.reward_aggregator import RewardAggregator

# Sequences and rewards
sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLQY"]
rewards = [RewardBlock.stability(), RewardBlock.novelty(baseline=["MKT..."], weight=0.5)]

# Evaluator and executor
evaluator = RewardAggregator(reward_blocks=rewards)
executor = LocalExecutor(evaluator=evaluator, mode="auto", num_workers="auto", batch_size=64)

# Optional constraints and rules
constraints = [BoundedConstraint(name="novelty", direction=Direction.GE, bound=0.3)]
rules = Rules(init={"novelty": 0.2})

# Strategy placeholder implementing the Strategy Protocol
class RandomStrategy:
    ...

config = EngineConfig(iterations=25, population_size=128, seed=1337)
engine = Engine(
    config=config,
    strategy=RandomStrategy(),
    evaluator=evaluator,
    executor=executor,
    score_fn=default_score,  # or: lambda m, r, cs: m.objective
    constraints=constraints,
    rules=rules,
)

results = engine.run()  # surface placeholder today
```

**Important:** Constraint names must match `Metrics.constraints` keys. Missing keys will be treated as zero and warned once.

### Batching & Device Hints

Power users can keep this minimal surface and still opt into accelerator-aware training by supplying two optional dataclasses:

```python
from strand.engine.runtime import BatchConfig, DeviceConfig

config = EngineConfig(
    iterations=100,
    population_size=512,
    batching=BatchConfig(eval_size=64, train_size=16, max_tokens=2048),
    device=DeviceConfig(target="cuda", mixed_precision="bf16", gradient_accumulation_steps=2),
)
```

- `BatchConfig` caps executor batch sizes (count + total tokens) without rewriting your strategy.
- `DeviceConfig` feeds the shared `ModelRuntime` (PyTorch + Accelerate) used by strategies/executors that advertise `strategy_caps()`.
- If a strategy ignores these hints, the engine silently falls back to today’s behaviour.

## Next Steps

- 📖 Mental Model (README) — Understand Strategy/Evaluator/Executor and the loop
- 🔧 API Reference (coming) — Protocols and dataclasses
- 💾 Examples (coming) — CEM/GA/CMA‑ES implementations

## Troubleshooting

### Import Errors

Install in development mode from the repo root:

```bash
pip install -e .
```

### Performance

Prefer `num_workers="auto"` and `mode="auto"` on the LocalExecutor. Use threads for GPU models and processes for CPU‑bound evaluators.
