# Strand SDK Documentation

Strand is a production-ready optimization engine for biological sequences. Compose strategies (search algorithms), rewards (objectives), and executors (parallelization) to explore sequence space under constraints.

**Designed with progressive disclosure**: start simple, discover advanced features as needed.

## Quick Links

- **New to Strand?** → [Getting Started](./getting_started.md) (5 minutes)
- **Quick reference** → [Quick Start](./tutorial/quick_start.md)
- **Deep dive** → [Core Concepts](./tutorial/core_concepts.md)

## Core Concepts

| Level | What | Time |
|-------|------|------|
| **1** | Basic optimization with RandomStrategy + GC content | 5 min |
| **2** | Better algorithms (CEM, GA, CMA-ES, Hybrid) | 20 min |
| **3** | Advanced rewards (Enformer, TFBS, foundation models) | 30 min |
| **4** | RL with supervised fine-tuning + HyenaDNA | 1 hour |
| **5** | Adaptive constraints with dual variables | 1.5 hours |

## Features

✅ **6 strategies**: Random, CEM, GA, CMA-ES, RL Policy, Hybrid  
✅ **Basic rewards**: GC content, length, novelty, stability  
✅ **Advanced rewards**: Enformer (cell-type activity), TFBS (binding sites)  
✅ **Foundation models**: HyenaDNA with pluggable policy heads  
✅ **RL support**: SFT warm-start, policy gradients, KL regularization  
✅ **Adaptive constraints**: Dual variable managers for feasibility  
✅ **Full reproducibility**: Manifests, checkpoints, MLflow integration  

## Navigation

- [Getting Started](./getting_started.md) - Installation & first run
- [Progressive Disclosure](./PROGRESSIVE_DISCLOSURE.md) - Self-paced learning path
- [API Reference](./api_reference.md) - All modules and classes
- [Reward Blocks](./reward_blocks.md) - Available scoring functions
- [Optimization Methods](./optimization_methods.md) - Search algorithms
- [FAQ](./faq.md) - Common questions

## Philosophy

Start with what you need today. Strand's architecture lets you:
- Begin with simple strategies and basic rewards
- Add complexity (advanced rewards, RL, constraints) when you need it
- Switch algorithms without rewriting evaluation code
- Reproduce any run via manifests
