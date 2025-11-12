# Strand SDK

> Status: **Alpha** — Full Ctrl-DNA implementation complete with evolutionary strategies, RL optimization, foundation models, and reward blocks.

Strand is a production-ready optimization engine for biological sequences. Compose a **strategy** (how to propose candidates), an **evaluator** (how to measure quality), and an **executor** (how to parallelize). The engine runs an iterative **ask → evaluate → score → tell** loop with full reproducibility via manifests.

**Highlights:**
- 6 built-in strategies: Random, CEM, GA, CMA-ES, RL Policy, Hybrid
- Foundation model support: HyenaDNA with pluggable policy heads
- Advanced reward blocks: GC content, Enformer, TFBS correlation
- Adaptive constraints: Dual variable managers for feasibility
- Full SFT support: Supervised fine-tuning warm-start for RL
- Reproducibility: Manifests, checkpoints, MLflow integration

## Mental Model

- Strategy proposes candidates: `ask(n) -> [Sequence]`
- Executor runs the Evaluator in parallel and preserves order
- Evaluator returns `Metrics` per sequence: `objective`, `constraints`, `aux`
- score_fn computes a scalar score from `Metrics` plus rule weights and constraints
- Strategy ingests feedback: `tell([(seq, score, metrics)])` and updates its state
- Optional rule manager updates weights from constraint violations

## Quick Start

**Start simple, add complexity as needed:**

```bash
# from strand-sdk root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

**Level 1 — Basic (5 minutes):**
```python
from strand.engine.strategies import RandomStrategy
from strand.rewards.basic import GCContentReward
from strand.evaluators.composite import CompositeEvaluator
from strand.engine.executors.local import LocalExecutor
from strand.engine.engine import Engine, EngineConfig

strategy = RandomStrategy(alphabet="ACGT", min_len=50, max_len=500)
evaluator = CompositeEvaluator([GCContentReward(target=0.5)])
executor = LocalExecutor(evaluator)
engine = Engine(strategy, executor, EngineConfig(iterations=5, population_size=32))
results = engine.run()
```

**Level 2+ — Advanced:**
See [docs/getting_started.md](docs/getting_started.md) for intermediate strategies (CEM, GA) and advanced rewards (Enformer, TFBS with foundation models).

See [docs/tutorial/core_concepts.md](docs/tutorial/core_concepts.md) for detailed examples with constraints, RL fine-tuning, and adaptive optimization.

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
