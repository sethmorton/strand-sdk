# Reward Blocks

Reward blocks are small scoring units that produce float scores given a `Sequence` and optional context. You can mix and match them inside a `RewardAggregator`.

## Basic Blocks (always available)

| Block | Module | Description | Key Args |
| --- | --- | --- | --- |
| Stability | `strand.rewards.stability.StabilityReward` | Hydrophobicity heuristic, favors higher stability | `threshold`, `weight`
| Solubility | `strand.rewards.solubility.SolubilityReward` | Polar residue fraction heuristic | `weight`
| Novelty | `strand.rewards.novelty.NoveltyReward` | Distance vs. baseline sequences (`hamming`/`levenshtein`) | `baseline`, `metric`, `weight`
| Length Penalty | `strand.rewards.length_penalty.LengthPenaltyReward` | Soft clamp around a target length | `target_length`, `tolerance`, `weight`
| GC Content | `strand.rewards.gc_content.GCContentReward` | Penalizes deviations from target GC ratio | `target`, `tolerance`, `weight`
| Custom | `strand.rewards.custom.CustomReward` | Wrap any callable scorer | `fn`, `name`, `weight`

## Advanced Blocks (extra dependencies)

| Block | Module | Description | Extra Requirements |
| --- | --- | --- | --- |
| EnformerRewardBlock | `strand.rewards.enformer_block` | Runs Enformer (ONNX or PyTorch) to predict per-cell-type activity | `onnx`, `onnxruntime` **or** PyTorch Enformer weights |
| TFBSFrequencyCorrelationBlock | `strand.rewards.tfbs_block` | Loads TF motifs (JASPAR), computes positional frequencies, correlates with targets | `biopython`, `JASPAR2024`, `numpy`, `scipy`

## Advanced Blocks (Variant-Aware)

These blocks require sequence context (reference + alternative variants) and are designed for genomic optimization tasks.

| Block | Module | Description | Use Case |
| --- | --- | --- | --- |
| VirtualCellDelta | `strand.rewards.virtual_cell_delta` | Enformer ref/alt delta across cell types | Regulatory element optimization |
| MotifDelta | `strand.rewards.motif_delta` | JASPAR+MOODS TF motif presence delta | TF binding site preservation |
| Conservation | `strand.rewards.conservation` | pyBigWig conservation track scoring | Evolutionary constraint analysis |

**Installation:** `uv pip install strand-sdk[variant-triage]`

**Example Configuration:**
```yaml
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

  - type: "conservation"
    config:
      bw_path: "phylop.bw"
      agg_method: "mean"
      weight: 0.2
```

Use extras to install dependencies, e.g. `uv pip install strand-sdk[variant-triage]`.

## Example: Aggregating Blocks

```python
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards.gc_content import GCContentReward
from strand.rewards.advanced import EnformerRewardBlock, EnformerConfig

rewards = RewardAggregator([
    GCContentReward(target=0.52, tolerance=0.05, weight=0.3),
    EnformerRewardBlock(EnformerConfig(cell_types=["hNSPC"], weight=0.7)),
])
```

## Constraints vs. Rewards

- Rewards influence `metrics.objective` (what you maximize).
- Constraints live in `metrics.constraints`. Use `CompositeEvaluator` or your own evaluator to add GC, length, or motif constraints separately.

## Extending the Registry

Register custom factories with `strand.rewards.registry.RewardRegistry.register("my_block", Callable)` so they are available via `RewardBlock.from_registry("my_block", **params)`.

Remember to document any heavy dependencies in your README or config files so teammates know which extras to install.
