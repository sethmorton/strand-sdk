# Strand SDK Documentation

Strand is a production-ready optimization engine for biological sequences. Compose strategies (search algorithms), rewards (objectives), and executors (parallelization) to explore sequence space under constraints.

**Designed with progressive disclosure**: start with a random strategy plus GC content, then grow into RL, dual variables, and foundation models when you need them.

## Quick Links

- **Install + run** → [Getting Started](./getting_started.md)
- **Minimal code sample** → [Quick Start](./tutorial/quick_start.md)
- **Architecture & runtime context** → [StrategyContext guide](./architecture/strategy_context.md)
- **Datasets & preprocessing** → [SequenceDataset overview](./data/sequence_datasets.md)
- **Ctrl-DNA walkthrough** → [Full pipeline tutorial](./tutorial/ctrl_dna_pipeline.md)

## Core Concepts

| Level | What | Time |
|-------|------|------|
| **1** | Basic optimization with RandomStrategy + GC content | 5 min |
| **2** | Better algorithms (CEM, GA, CMA-ES, Hybrid) | 20 min |
| **3** | Advanced rewards (Enformer, TFBS, foundation models) | 30 min |
| **4** | RL with supervised fine-tuning + HyenaDNA | 1 hour |
| **5** | Adaptive constraints with dual variables | 1.5 hours |

## Features

✅ **Strategies**: Random, CEM, GA, CMA-ES, RL Policy, Hybrid (each declares `StrategyCaps`)  
✅ **Device-aware runtimes**: `StrategyContext` hands your strategy `ModelRuntime`, `DeviceConfig`, and `BatchConfig` when needed  
✅ **Reward blocks**: GC content, novelty, stability + advanced Enformer and TFBS correlation modules  
✅ **Foundation models**: HyenaDNA loader + policy head implementations ready for custom RL loops  
✅ **Datasets**: `SequenceDataset` for FASTA/CSV/JSON + Ctrl-DNA dataset downloader script  
✅ **Constraints**: CBROP-inspired `DualVariableManager` with logging helpers  
✅ **CLI + Reproducibility**: `strand run` configs, MLflow tracker, manifests, Hyena/Enformer examples  

## Navigation

- [Getting Started](./getting_started.md) — Installation & first run
- [Quick Start](./tutorial/quick_start.md) — Minimal code sample
- [Core Concepts](./tutorial/core_concepts.md) — Mental model + manifests
- [StrategyContext](./architecture/strategy_context.md) — Runtime/device hand-off
- [Sequence Datasets](./data/sequence_datasets.md) — SFT-ready data formats
- [Reward Blocks](./reward_blocks.md) — Scoring modules & dependencies
- [Optimization Methods](./optimization_methods.md) — Strategy cheat sheet
- [Examples](./examples.md) — Script summaries
- [FAQ](./faq.md) — Common questions & roadmap

## Philosophy

Start with what you need today. Strand's architecture lets you:
- Begin with simple strategies and basic rewards
- Add complexity (advanced rewards, RL, constraints) when you need it
- Switch algorithms without rewriting evaluation code
- Reproduce any run via manifests
