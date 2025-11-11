# Strand SDK

> **⚠️ Status: Pre-Alpha**
>
> Strand is under active development. The public API surface will continue to change until after the first design-partner runs ship.

Strand is an early-stage Python SDK that brokers between generative biology models and wet-lab programs. Teams plug in model proposals, define biological objectives, and let Strand run constrained search to surface a handful of sequences worth testing. Every run emits a manifest with the model versions, reward blocks, and search parameters so results stay repeatable and defensible.

This repository tracks the open-source optimization engine. Managed cloud and on-prem packages for regulated teams will layer on the same primitives once we prove out the tracing and optimization stack with design partners.

## Status

- **Current Phase**: Optimization engine + tracing layer under active development with design partners
- **Focus**: Model adapters, reward-block graph, reproducible manifests, CLI tooling
- **Stability**: Pre-alpha — expect breaking changes between commits
- **Near-Term Milestones**:
  - **Mid December 2025**: first closed design-partner optimization runs connected to wet-lab programs
  - **Q1 2026**: tagged open-source release of the engine + manifest tooling
  - **Q2 2026**: managed cloud and on-prem deployments for regulated workloads

## SDK Scope

1. **Model Inputs**: adapters for plugging foundation or in-house generative models straight into Strand’s search loop
2. **Search Engines**: interchangeable algorithms (CEM, CMA-ES, genetic, random) tuned for sequence space exploration under constraints
3. **Reward Blocks**: composable scoring functions (stability, solubility, novelty, manufacturability) plus extension points for lab-specific metrics
4. **Provenance + CLI**: manifests that log parameters, model versions, and reward graphs so every run is auditable, plus CLI tooling for submitting and tracing jobs

## Quick Start (Pre-Alpha)

```bash
# from strand-sdk root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

```python
from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock

optimizer = Optimizer(
    sequences=["MKT..."] ,
    reward_blocks=[
        RewardBlock.stability(weight=1.0),
        RewardBlock.novelty(baseline=["MKP..."], metric="hamming", weight=0.5),
    ],
    method="cem",
    iterations=25,
)

results = optimizer.run()
print(results.top(5))
```

## Repo Layout

See `GITHUB_REPO_STRUCTURE.md` in the design workspace for the authoritative target layout. All directories listed there are present here with placeholder implementations, docs, examples, tests, and benchmarks so contributors can iterate per vertical slice.

## Contributing

Please review `CONTRIBUTING.md` for coding standards, testing expectations, and contribution guidelines.

## 📄 License

This project is licensed under the terms of the MIT license - see [LICENSE](LICENSE) for details.
