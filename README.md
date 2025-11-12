# Strand SDK


> Status: **Alpha** — Engine, CLI orchestration, supervised warm-starts, and logging are implemented; surfaces may still move as Ctrl-DNA parity hardens.

Strand is a production-ready optimization engine for biological sequences. Compose a **strategy** (how to propose candidates), an **evaluator** (how to measure quality), and an **executor** (how to parallelize). The engine runs an iterative **ask → evaluate → score → tell** loop with full reproducibility via manifests.

**Highlights:**
- Six built-in strategies (Random, CEM, GA, CMA-ES, RL Policy, Hybrid) with declared `StrategyCaps`
- Device-aware runtimes via `StrategyContext` plus a shared `StrategyRuntimeAdapter`
- Foundation-model loader (`strand/models/hyenadna.py`) and pluggable RL policy heads for HyenaDNA/Transformer backbones
- Reward blocks ranging from heuristics (GC, stability, novelty) to Enformer and TFBS correlation
- Adaptive constraints powered by `DualVariableManager` + manifest-friendly logging
- SequenceDataset utilities, dataset prep scripts, and built-in supervised warm-start hooks for Ctrl-DNA style workflows
- Config-driven CLI (`strand run ctrl-dna.yaml`) to assemble engines without bespoke scripts
- MLflow tracker, manifests, and config examples for reproducible runs

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
from strand.engine.engine import Engine, EngineConfig
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies import RandomStrategy
from strand.evaluators.composite import CompositeEvaluator
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward

strategy = RandomStrategy(alphabet="ACGT", min_len=50, max_len=500, seed=7)
rewards = RewardAggregator([GCContentReward(target=0.5, tolerance=0.1)])
evaluator = CompositeEvaluator(rewards=rewards, include_gc=True)
executor = LocalExecutor(evaluator=evaluator, batch_size=32)
engine = Engine(
    config=EngineConfig(iterations=5, population_size=32, method="random"),
    strategy=strategy,
    evaluator=evaluator,
    executor=executor,
    score_fn=lambda metrics, *_: metrics.objective,
)
results = engine.run()
print(results.best)
```

**Level 2+ — Advanced:**
See [docs/getting_started.md](docs/getting_started.md) for intermediate strategies (CEM, GA) and advanced rewards (Enformer, TFBS with foundation models).

See [docs/tutorial/core_concepts.md](docs/tutorial/core_concepts.md) for detailed examples with constraints, RL training, and adaptive optimization.

## Extending

- Implement a custom Strategy (ask/tell) with any algorithm
- Wrap ML models or heuristics in an Evaluator (batched), then scale with an Executor
- Add constraints by declaring `BoundedConstraint` instances and hook them into `DualVariableManager`

## Accelerator-Aware Runs

- `EngineConfig` accepts optional `batching` and `device` dataclasses so you can cap evaluation batch sizes (by count or total tokens) and pick the accelerator/mixed precision without new call sites.
- Strategies that expose `strategy_caps()` automatically receive a `StrategyContext` containing a lazily-initialized `ModelRuntime` (PyTorch + Accelerate) so they can fine-tune large models without prop drilling.
- Use `strand.engine.executors.torch.TorchExecutor` when running heavy evaluators: pass the shared `ModelRuntime` and optionally a token budget to control padding + memory.

## Supervised Data & Foundation Models

- Load HyenaDNA (or compatible) checkpoints with `strand.models.hyenadna.load_*` helpers, which return both the tokenizer and autoregressive backbone.
- Tokenize Ctrl-DNA style promoter datasets via `strand.data.sequence_dataset.SequenceDataset` (FASTA/CSV/JSON) and the `scripts/datasets/ctrl_dna/download_promoters.py` helper.
- RL strategies expose both online `train_step` updates and a built-in `warm_start(dataset, epochs, batch_size, context, tracker)` hook powered by `SequenceDataset`. Run SFT once, then resume RL without rebuilding everything.

## CLI

The CLI now supports hydrated runs:

```bash
strand run configs/examples/ctrl_dna/hyenadna_rl.yaml
```

Config files describe strategies, rewards, executors, constraints, device/batch hints, SFT datasets, and dual-variable settings so you can launch experiments without editing Python.

Reward blocks in `strand/rewards/` remain usable via the `RewardAggregator` evaluator. Current reward blocks are simple heuristics (e.g., hydrophobicity proxy for "stability", polar-residue fraction for "solubility"). The `model` parameter on these blocks is a provenance label only.

**Important:** Constraint names must match `Metrics.constraints` keys; missing keys will be treated as zero (and warned once).

## Contributing

Please review `CONTRIBUTING.md` for coding standards, testing expectations, and contribution guidelines.

## 📄 License

This project is licensed under the terms of the MIT license - see [LICENSE](LICENSE) for details.
