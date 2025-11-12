# Getting Started with Strand SDK

The Strand SDK provides a complete toolkit for DNA sequence optimization using evolutionary strategies, reinforcement learning, and foundation models.

## Quick Setup

1. **Create a Python 3.11+ environment.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the SDK.**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Your First Optimization

### Simple: Use an Evolutionary Algorithm

```python
from strand.engine.strategies import CEMStrategy, RandomStrategy
from strand.engine.engine import Engine, EngineConfig
from strand.engine.executors.local import LocalExecutor
from strand.rewards.gc_content import GCContentBlock
from strand.evaluators.composite import CompositeEvaluator

# Define strategies
strategy = CEMStrategy(alphabet="ACGT", min_len=50, max_len=500, seed=42)

# Create reward function
evaluator = CompositeEvaluator([GCContentBlock(target=0.5)])

# Run optimization
executor = LocalExecutor(evaluator)
engine = Engine(
    strategy=strategy,
    executor=executor,
    config=EngineConfig(iterations=10, population_size=32),
)
results = engine.run()
print(f"Best: {results.best_sequence.tokens}")
```

### Advanced: Use RL with Foundation Models

See `docs/tutorial/ctrl_dna_pipeline.md` for a complete SFT + RL example with HyenaDNA and Enformer reward blocks.

## Project Structure

```
strand/
├── engine/strategies/
│   ├── evolutionary/      # CEM, GA, CMA-ES
│   ├── rl/                # RL Policy + Policy Heads
│   ├── ensemble/          # Hybrid strategies
│   └── random.py          # Basic random search
├── rewards/               # Evaluators (GC content, Enformer, TFBS)
├── models/                # Foundation models (HyenaDNA)
├── data/                  # Dataset utilities
└── constraints/           # Constraint management (Dual variables)
```

## Key Features

- ✅ **Multiple Strategies**: Random, CEM, GA, CMA-ES, RL Policy, Hybrid
- ✅ **Foundation Models**: HyenaDNA with pluggable policy heads
- ✅ **Reward Blocks**: GC content, Enformer, TFBS correlation
- ✅ **Adaptive Constraints**: Dual variable managers for feasibility
- ✅ **SFT Support**: Supervised fine-tuning warm-start for RL
- ✅ **Device Flexibility**: CPU/GPU with mixed-precision support

## Next Steps

- **Examples**: Check `examples/` for end-to-end workflows
- **Tests**: Run `pytest tests/` to verify everything works
- **Docs**: See `docs/tutorial/quick_start.md` for minimal example
