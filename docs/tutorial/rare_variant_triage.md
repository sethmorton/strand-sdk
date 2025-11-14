# Tutorial: Rare Variant Triage

This tutorial demonstrates how to use Strand SDK for **rare variant triage** - the process of identifying genomic variants that disrupt or enhance regulatory element function. You'll learn to combine foundation models, motif analysis, and conservation scoring to optimize variants for therapeutic applications.

## Overview

Rare variant triage involves:
1. **Loading variants** from VCF files with genomic context from reference FASTA
2. **Scoring variants** using multiple complementary approaches:
   - **Virtual Cell**: Foundation model predictions of regulatory activity
   - **Motif Analysis**: Transcription factor binding site preservation
   - **Conservation**: Evolutionary constraint analysis
3. **Optimizing sequences** to maximize regulatory impact while preserving function

## Installation

```bash
uv pip install strand-sdk[variant-triage]
```

This installs the core SDK plus dependencies for:
- `enformer-pytorch`: Foundation model predictions
- `pyjaspar` + `MOODS-python`: Motif analysis
- `pyBigWig` + `pysam`: Genomic data handling

## Quick Start: 5 Variants

Let's start with a minimal example using synthetic data:

```python
from strand.data.variant_dataset import VariantDataset
from strand.evaluators.variant_composite import VariantCompositeEvaluator
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.virtual_cell_delta import VirtualCellDeltaReward

# Create synthetic VCF and FASTA for demo (replace with real files)
# For this tutorial, assume you have variants.vcf.gz and reference.fa

# Load variants with 1kb context windows
dataset = VariantDataset(
    vcf_path="variants.vcf.gz",
    fasta_path="reference.fa",
    window_size=1000
)

# Single reward: Enformer delta predictions
rewards = RewardAggregator([
    VirtualCellDeltaReward(
        model_path="enformer-base",
        device="cpu",  # Use "cuda" for GPU
        weight=1.0
    )
])

# Basic evaluator (no constraints yet)
evaluator = VariantCompositeEvaluator(rewards=rewards)

# Evaluate first 5 variants
contexts = list(dataset)[:5]
metrics = evaluator.evaluate_batch_with_context(contexts)

for i, (context, metric) in enumerate(zip(contexts, metrics)):
    print(f"Variant {i+1}: {context.metadata.chrom}:{context.metadata.pos} "
          f"{context.metadata.ref}>{context.metadata.alt} "
          f"Score: {metric.objective:.4f}")
```

## Deep Dive: Understanding Each Reward Block

### 1. Virtual Cell Delta (Enformer)

**What it does**: Predicts how a variant changes chromatin accessibility and gene expression across 5,313 cell types using the Enformer foundation model.

**When to use**: When you want to identify variants that enhance regulatory activity or disrupt enhancers/silencers.

**Key parameters**:
- `model_path`: HuggingFace model ID ("enformer-base")
- `device`: "cpu", "cuda", or "auto"
- `target_cell_types`: List of cell types to focus on (empty = all)
- `weight`: Relative importance in final score

```python
from strand.rewards.virtual_cell_delta import VirtualCellDeltaReward

reward = VirtualCellDeltaReward(
    model_path="enformer-base",
    device="cuda",  # GPU recommended
    target_cell_types=["hNSPC", "H1-hESC"],  # Focus on relevant cell types
    weight=0.6
)
```

**Output**: Returns `objective` (mean delta across tracks) and auxiliary metrics for each of ~5,000 tracks.

### 2. Motif Delta (JASPAR + MOODS)

**What it does**: Scans for transcription factor binding motifs and measures how variants affect motif presence.

**When to use**: When TF binding disruption/enhancement is your therapeutic mechanism.

**Key parameters**:
- `tf_list`: JASPAR accession IDs (e.g., ["MA0001", "MA0002"])
- `threshold`: Log-odds threshold for motif matches
- `background_freq`: Nucleotide background frequencies
- `weight`: Relative importance

```python
from strand.rewards.motif_delta import MotifDeltaReward

reward = MotifDeltaReward(
    tf_list=["MA0001", "MA0002", "MA0003"],  # Common TFs
    threshold=6.0,  # Conservative threshold
    weight=0.3
)
```

**Output**: Returns total motif delta plus per-TF metrics and disruption flags.

### 3. Conservation (PhyloP/GERP)

**What it does**: Scores variants based on evolutionary conservation from bigWig tracks.

**When to use**: When you want to prioritize variants in conserved regulatory regions.

**Key parameters**:
- `bw_path`: Path to conservation bigWig file
- `agg_method`: "mean", "max", "sum", "min"
- `weight`: Relative importance

```python
from strand.rewards.conservation import ConservationReward

reward = ConservationReward(
    bw_path="phylop.bw",  # Or GERP, CADD tracks
    agg_method="mean",   # Average conservation in window
    weight=0.1
)
```

**Output**: Conservation scores for reference and alternative windows.

## Full Pipeline: Multi-Objective Optimization

Combine all three approaches for comprehensive variant assessment:

```python
from strand.evaluators.variant_composite import VariantCompositeEvaluator

# Multi-objective rewards
rewards = RewardAggregator([
    VirtualCellDeltaReward(model_path="enformer-base", weight=0.5),
    MotifDeltaReward(tf_list=["MA0001", "MA0002"], weight=0.3),
    ConservationReward(bw_path="phylop.bw", weight=0.2),
])

# Add constraints
evaluator = VariantCompositeEvaluator(
    rewards=rewards,
    include_length=True,           # Penalize length changes
    include_gc=True,               # Maintain GC content
    include_motif_disruption=True, # Flag motif loss
    include_conservation_windows=True,  # Include conservation metrics
)

# Use with optimization strategy
from strand.engine.strategies import CMAESStrategy
from strand.engine import Engine, EngineConfig

strategy = CMAESStrategy(alphabet="ACGT", min_len=100, max_len=100)  # Fixed length
engine = Engine(
    config=EngineConfig(iterations=50, population_size=100),
    strategy=strategy,
    evaluator=evaluator,
    executor=executor,  # From earlier
)
```

## Compute Expectations

**Hardware Requirements**:
- **CPU**: 16GB RAM minimum, 32GB recommended
- **GPU**: RTX 3080+ (8GB VRAM) for Enformer inference
- **Storage**: 50GB for models + datasets

**Runtime Estimates** (per variant, batch_size=4):
- VirtualCellDelta: 2-5 seconds (GPU), 30-60 seconds (CPU)
- MotifDelta: 0.1-0.5 seconds
- Conservation: 0.01-0.1 seconds

**Total for 1000 variants**: 30-60 minutes with GPU acceleration

## Advanced Usage: Custom Reward Composition

### Disease-Specific TF Selection

```python
# For hematopoiesis-focused optimization
hematopoiesis_tfs = [
    "MA0001",  # TAL1
    "MA0002",  # GATA1
    "MA0003",  # RUNX1
    # Add more disease-relevant TFs
]

motif_reward = MotifDeltaReward(
    tf_list=hematopoiesis_tfs,
    threshold=5.0,  # More sensitive for rare variants
    weight=0.4
)
```

### Cell-Type-Specific Enformer

```python
# Focus on relevant cell types
cell_types = [
    "hNSPC",      # Neural stem cells
    "H1-hESC",    # Embryonic stem cells
    "GM12878",    # Blood cells
]

enformer_reward = VirtualCellDeltaReward(
    model_path="enformer-base",
    target_cell_types=cell_types,
    weight=0.6
)
```

### Multi-Track Conservation

```python
# Combine multiple conservation metrics
rewards = RewardAggregator([
    ConservationReward(bw_path="phylop.bw", agg_method="mean", weight=0.1),
    ConservationReward(bw_path="gerp.bw", agg_method="max", weight=0.1),
])
```

## Configuration-Driven Pipeline

For reproducible research, use the YAML configuration:

```yaml
# rare_variant_triage.yaml
dataset:
  type: "variant_vcf"
  vcf_path: "variants.vcf.gz"
  fasta_path: "reference.fa"
  window_size: 1000

rewards:
  - type: "virtual_cell_delta"
    config:
      model_path: "enformer-base"
      weight: 0.5

  - type: "motif_delta"
    config:
      tf_list: ["MA0001", "MA0002"]
      threshold: 6.0
      weight: 0.3

evaluator:
  type: "variant_composite"
  include_gc: true
  include_motif_disruption: true

engine:
  method: "cmaes"
  iterations: 50
  population_size: 100
```

Run with: `strand run-variant-triage rare_variant_triage.yaml --device cuda`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `uv pip install strand-sdk[variant-triage]`
2. **CUDA OOM**: Reduce batch_size in executor config
3. **Slow Inference**: Use GPU device for Enformer
4. **No Motifs Found**: Check JASPAR accession IDs are valid
5. **BigWig Errors**: Ensure conservation tracks match genome build

### Performance Tips

- **Batch Processing**: Use batch_size=4-8 for Enformer
- **GPU**: CUDA significantly faster than CPU
- **Caching**: VariantDataset caches sequences by default
- **Parallel**: Use multiple workers for motif scanning

## Next Steps

- **Custom Rewards**: Extend base classes for disease-specific scoring
- **RL Optimization**: Use RLPolicyStrategy for sequence generation
- **Multi-Modal**: Combine with expression/functional data
- **Clinical Integration**: Connect to variant interpretation pipelines

## References

- [Enformer Paper](https://www.nature.com/articles/s41592-021-01252-x)
- [JASPAR Database](https://jaspar.genereg.net/)
- [PhyloP Conservation](https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=cons100way)
- [Variant Triage Methods](https://www.nature.com/articles/s41576-020-00277-3)
