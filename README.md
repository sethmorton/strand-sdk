# Strand SDK

> Status: Pre‑Alpha — surfaces stabilized; implementations landing next.

Strand is a modular optimization engine for biological sequences. You compose a strategy (how to propose candidates), an evaluator (how to measure them), and an executor (how to run evaluations in parallel). A tiny scoring function blends metrics and constraints into a single number to optimize. The engine then runs an iterative ask → evaluate → score → tell loop and records iteration summaries for reproducibility.

## Mental Model

- Strategy proposes candidates: `ask(n) -> [Sequence]`
- Executor runs the Evaluator in parallel and preserves order
- Evaluator returns `Metrics` per sequence: `objective`, `constraints`, `aux`
- score_fn computes a scalar score from `Metrics` plus rule weights and constraints
- Strategy ingests feedback: `tell([(seq, score, metrics)])` and updates its state
- Optional rule manager updates weights from constraint violations

## Quick Start (Surfaces)

```bash
# from strand-sdk root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

```python
from strand.core.sequence import Sequence
from strand.rewards import RewardBlock
from strand.engine import (
    Engine, EngineConfig, Metrics,
    Strategy, Evaluator, Executor,
    BoundedConstraint, Direction, Rules, default_score,
)
from strand.engine.executors.local import LocalExecutor
from strand.evaluators.reward_aggregator import RewardAggregator

# Sequences and rewards (heuristics today)
sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLQY"]
rewards = [RewardBlock.stability(), RewardBlock.novelty(baseline=["MKT..."], weight=0.5)]

# Evaluator and executor
evaluator: Evaluator = RewardAggregator(reward_blocks=rewards)
executor: Executor = LocalExecutor(evaluator=evaluator, batch_size=64)

# Optional constraints and rules
constraints = [BoundedConstraint(name="novelty", direction=Direction.GE, bound=0.3)]
rules = Rules(init={c.name: 0.2 for c in constraints})

# Strategy and engine config (example placeholders)
class RandomStrategy:  # implements Strategy Protocol
    ...

config = EngineConfig(iterations=50, population_size=256, seed=1337)
engine = Engine(
    config=config,
    strategy=RandomStrategy(),
    evaluator=evaluator,
    executor=executor,
    score_fn=default_score,  # or a trivial lambda m, r, cs: m.objective
    constraints=constraints,
    rules=rules,
)

results = engine.run()  # surface placeholder today
```

## Extending

- Implement a custom Strategy (ask/tell) with any algorithm
- Wrap ML models or heuristics in an Evaluator (batched), then scale with an Executor
- Add constraints by declaring `BoundedConstraint` instances; use `AdditiveLagrange` to enforce them

## Accelerator-Aware Runs

- `EngineConfig` accepts optional `batching` and `device` dataclasses so you can cap evaluation batch sizes (by count or token budget) and describe the desired compute target/mixed precision without touching existing call sites.
- Strategies that expose `strategy_caps()` automatically receive a `StrategyContext` containing a lazily-initialized `ModelRuntime` (backed by PyTorch + Hugging Face Accelerate) so they can fine-tune large models without prop drilling.
- Use `strand.engine.executors.torch.TorchExecutor` when running heavy evaluators: it pad-safely batches sequences according to both element count and total token budget while reusing the shared `ModelRuntime` autocast context.

Reward blocks in `strand/rewards/` remain usable via the `RewardAggregator` evaluator. Current reward blocks are simple heuristics (e.g., hydrophobicity proxy for "stability", polar-residue fraction for "solubility"). The `model` parameter on these blocks is a provenance label only.

**Important:** Constraint names must match `Metrics.constraints` keys; missing keys will be treated as zero (and warned once).

## Contributing

Please review `CONTRIBUTING.md` for coding standards, testing expectations, and contribution guidelines.

## 📄 License

This project is licensed under the terms of the MIT license - see [LICENSE](LICENSE) for details.
