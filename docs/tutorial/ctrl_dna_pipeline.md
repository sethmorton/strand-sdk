# Ctrl-DNA End-to-End Pipeline Tutorial

This guide walks through a complete Ctrl-DNA workflow: supervised fine-tuning (SFT) warm-start followed by constrained RL optimization with dual variable management.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Load Foundation Model (HyenaDNA)                            │
│     - Tokenizer + autoregressive backbone                       │
│     - Policy head (per-position, HyenaDNA, or Transformer)      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│  2. Supervised Fine-Tuning (Optional)                           │
│     - Load training dataset (FASTA/CSV with labels)             │
│     - warm_start(): Pre-train policy on supervised data         │
│     - Checkpoint best model                                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│  3. Constrained RL Loop                                         │
│     - ask(): Generate sequences (autoregressive sampling)       │
│     - evaluate: Score with reward blocks + constraints          │
│     - tell(): Update policy (REINFORCE + KL regularization)     │
│     - Update dual variables for constraint management           │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│  4. Evaluation & Deployment                                    │
│     - Export best sequences and model                           │
│     - Log results to MLflow                                     │
│     - Create reproducible manifest                              │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Configuration

Create a YAML config file combining all stages:

```yaml
# configs/ctrl_dna_full.yaml
engine:
  iterations: 100
  population_size: 64
  seed: 42

strategy:
  type: "rl-policy"
  alphabet: "ACGT"
  min_len: 50
  max_len: 500
  policy_head:
    type: "hyenadna"
    model_id: "hyena/hyenadna-tiny-1k"
    freeze_backbone: true

device:
  target: "cuda"
  mixed_precision: "bf16"

batch:
  eval_size: 32
  train_size: 8
  max_tokens: 1024

# SFT warm-start configuration
sft:
  enabled: true
  dataset_path: "data/promoters.fasta"
  epochs: 3
  val_split: 0.1

# Reward blocks and constraints
evaluation:
  reward_blocks:
    - type: "enformer"
      weight: 0.5
      cell_types: ["hNSPC"]
    - type: "gc_content"
      weight: 0.3
      target: 0.5
  
  constraints:
    - name: "off_target"
      min: 0.0
      max: 0.2  # Minimize off-target activity

logging:
  mlflow_uri: "local"
  experiment: "ctrl-dna-full"
  checkpoint_dir: "./checkpoints/ctrl-dna-best"
```

## Step 2: Load Foundation Model

```python
from strand.models.hyenadna import load_hyenadna_from_hub
from strand.engine.runtime import DeviceConfig

# Load HyenaDNA
config = load_hyenadna_from_hub(
    model_name="hyenadna-tiny-1k",
    device="cuda",
    dtype=torch.bfloat16,
)

print(f"Model vocab_size: {config.vocab_size}")
print(f"Max context: {config.max_seq_len}")
```

## Step 3: Optional Supervised Fine-Tuning (SFT)

### Why SFT?

- **Warm-start policy**: Initialize policy with real sequence patterns
- **Reduce RL variance**: Policy starts well-calibrated instead of random
- **Faster convergence**: RL loop can focus on optimization vs. learning basics
- **Constraint awareness**: Pre-train on diverse, realistic sequences

### SFT Workflow

```python
from strand.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig
from strand.engine.strategies import RLPolicyStrategy
from strand.engine.runtime import DeviceConfig, build_strategy_context

# Configure dataset
dataset_config = SequenceDatasetConfig(
    data_path="data/promoters.fasta",
    tokenizer=model.tokenizer,
    max_seq_len=1024,
    validation_split=0.1,
)
dataset = SequenceDataset(dataset_config)

# Build strategy with device context
device = DeviceConfig(target="cuda", mixed_precision="bf16")
context = build_strategy_context(
    device=device,
    batch=None,
    require_runtime=True,
)

# Create strategy
strategy = RLPolicyStrategy(
    alphabet="ACGT",
    min_len=50,
    max_len=500,
)

# Prepare with runtime
strategy.prepare(context)

# SFT warm-start
strategy.warm_start(dataset=dataset, epochs=3)

print("✓ SFT warm-start complete")
```

### SFT Metrics & Logging

During SFT, the following are logged to MLflow:

- **loss**: Cross-entropy loss per epoch
- **accuracy**: Token accuracy on train/val splits
- **kl_divergence**: KL between policy and reference (if applicable)
- **best_val_loss**: Best validation loss across epochs
- **checkpoint**: Model weights at best epoch

## Step 4: Constrained RL Loop

### Strategy Capabilities & Context

The RLPolicy strategy declares:

```python
caps = strategy.strategy_caps()
assert caps.requires_runtime is True  # Needs device/autocast support
assert caps.supports_fine_tuning is True  # Can do warm_start()
assert caps.kl_regularization == "token"  # Per-token KL penalty
```

### RL Loop

```python
from strand.engine.engine import Engine, EngineConfig
from strand.evaluators.composite import CompositeEvaluator
from strand.engine.executors.local import LocalExecutor

# Create evaluator (combines reward blocks)
evaluator = CompositeEvaluator(
    reward_blocks=[
        EnformerRewardBlock(cell_types=["hNSPC"]),
        GCContentBlock(target=0.5),
    ],
    constraint_blocks=[
        OffTargetConstraint(max_divergence=0.2),
    ],
)

# Create executor for parallel evaluation
executor = LocalExecutor(evaluator, num_workers=4)

# Configure engine
engine_config = EngineConfig(
    iterations=100,
    population_size=64,
    seed=42,
)

# Run optimization
engine = Engine(
    strategy=strategy,
    executor=executor,
    config=engine_config,
)
results = engine.run()

print(f"Best score: {results.best_score}")
print(f"Best sequence: {results.best_sequence.tokens}")
```

### Dual Variable Management (Constraint Handling)

For adaptive constraint enforcement, use dual variables:

```python
from strand.engine.constraints.dual import DualVariableManager

# Create managers for each constraint
dual_managers = {
    "off_target": DualVariableManager(
        init_weight=1.0,
        max_weight=100.0,
        adaptive_step=0.1,
    ),
}

# In RL strategy (inside tell()):
for constraint_name, manager in dual_managers.items():
    violation = compute_constraint_violation(...)
    new_weight = manager.update(violation)
    log_metric(f"dual_{constraint_name}_weight", new_weight)
```

## Step 5: Evaluation & Deployment

### Export Best Model

```python
from strand.engine.strategies.runtime_adapter import StrategyRuntimeAdapter

adapter = StrategyRuntimeAdapter(strategy._runtime)

# Save checkpoint
adapter.save_checkpoint(
    strategy._policy_module,
    strategy._optimizer,
    path="checkpoints/best_policy.pt",
    metadata={
        "step": results.best_iteration,
        "score": results.best_score,
        "date": datetime.now().isoformat(),
    },
)

# Save sequences
with open("results/optimized_sequences.fasta", "w") as f:
    for seq in results.all_sequences[:100]:
        f.write(f">{seq.id}\n{seq.tokens}\n")
```

### Create Reproducible Manifest

```python
import json
from strand.manifests import EngineManifest

manifest = EngineManifest(
    config=engine_config,
    strategy=strategy.state(),
    results=results,
    model_checkpoint="checkpoints/best_policy.pt",
    mlflow_run_id=experiment_run_id,
)

with open("results/manifest.json", "w") as f:
    json.dump(manifest.to_dict(), f, indent=2)
```

## Troubleshooting

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch.eval_size` or `batch.train_size`
2. Enable gradient accumulation: `device.gradient_accumulation_steps=4`
3. Use `device.mixed_precision="bf16"` (bfloat16 uses half memory)
4. Use a smaller policy head (e.g., "per-position" instead of "hyenadna")

### Dataset Schema Mismatch

**Problem**: `ValueError: batch must contain 'input_ids'`

**Causes**:
- Tokenizer returns different keys (e.g., `token_ids` vs `input_ids`)
- FASTA parsing failed (check file format)

**Solutions**:
1. Verify FASTA format: `head -4 data/promoters.fasta`
2. Check tokenizer output: `print(tokenizer("ACGT"))`
3. Use BioPython for robust FASTA: `pip install biopython`

### Constraint Divergence

**Problem**: Dual variables exploding (weight → ∞)

**Causes**:
- Constraint too restrictive (impossible to satisfy)
- Reward blocks contradicting constraints

**Solutions**:
1. Relax constraint bounds
2. Check reward blocks don't conflict
3. Reduce `constraint_penalty` parameter
4. Log constraint violations: `log_metric("constraint_violation", violation)`

### Slow RL Convergence

**Problem**: Policy not improving over iterations

**Causes**:
- Learning rate too low
- SFT not initialized properly
- Reward signal too noisy

**Solutions**:
1. Increase `strategy.learning_rate` (default 0.1)
2. Extend SFT `epochs` (more pre-training)
3. Aggregate reward blocks (reduce noise): `aggregation="mean"`
4. Increase `population_size` for better signal

## Extension: Custom Reward Blocks

To add domain-specific rewards:

```python
from strand.rewards.base import RewardBlock

class CustomBlock(RewardBlock):
    def __call__(self, sequences: list[Sequence]) -> dict[str, float]:
        rewards = []
        for seq in sequences:
            # Your custom scoring
            score = score_my_metric(seq.tokens)
            rewards.append(score)
        return {"custom_reward": float(np.mean(rewards))}
```

## References

- **Architecture**: See `docs/architecture/strategies.md`
- **Reward Blocks**: See `docs/rewards/enformer_tfbs.md`
- **Constraint Management**: See `docs/constraints/dual_variables.md`
- **HyenaDNA**: https://github.com/HyenaDNA/HyenaDNA
- **Ctrl-DNA Paper**: https://arxiv.org/abs/2505.20578

