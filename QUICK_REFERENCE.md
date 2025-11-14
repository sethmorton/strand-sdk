# Variant Triage Quick Reference

**TL;DR:** Everything you need to know to start implementing the variant-triage plan.

---

## 📦 What's Being Built

| Component | Purpose | Status |
|-----------|---------|--------|
| `VirtualCellDeltaReward` | Enformer ref/alt variant effects | TODO |
| `MotifDeltaReward` | JASPAR/MOODS motif changes | TODO |
| `ConservationReward` | pyBigWig conservation scoring | TODO |
| `VariantDataset` | VCF + FASTA loader → SequenceContext | TODO |
| `VariantCompositeEvaluator` | Variant-aware composite evaluator | TODO |
| `variant-triage` config example | Full YAML example | TODO |
| Docs + tutorial | Rare variant triage guide | TODO |

---

## 🎯 Implementation Order (Recommended)

```
1. Phase 1: Update pyproject.toml (optional dependencies)
   ↓
2. Phase 2: Enhance base.py + registry.py (metadata + factory)
   ↓
3. Phase 2: Implement 3 new reward blocks (Virtual Cell, Motif, Conservation)
   ↓
4. Phase 3: Define SequenceContext + VariantMetadata in types.py
   ↓
5. Phase 3: Implement VariantDataset loader
   ↓
6. Phase 4: Update RewardAggregator for context support
   ↓
7. Phase 4: Implement VariantCompositeEvaluator
   ↓
8. Phase 5: Extend CLI + create example config
   ↓
9. Phase 6-8: Documentation, tests, logging
```

---

## 📋 25 Todo Items at a Glance

**Phase 1: Dependencies** (1 item)
- [ ] Update pyproject.toml with variant-triage extras

**Phase 2: Reward Blocks** (5 items)
- [ ] Enhance base.py with ObjectiveType, BlockType enums
- [ ] Expand registry.py with factory pattern
- [ ] Implement virtual_cell_delta.py (Enformer)
- [ ] Implement motif_delta.py (JASPAR+MOODS)
- [ ] Implement conservation.py (pyBigWig)

**Phase 3: Data Layer** (2 items)
- [ ] Define VariantMetadata, SequenceContext in types.py
- [ ] Implement variant_dataset.py (pysam/PyRanges)

**Phase 4: Evaluators** (2 items)
- [ ] Update RewardAggregator with context parameter
- [ ] Implement VariantCompositeEvaluator

**Phase 5: CLI & Config** (3 items)
- [ ] Extend strand/cli/cli.py
- [ ] Update Hydra config schema
- [ ] Create rare_variant_triage.yaml example

**Phase 6: Documentation** (4 items)
- [ ] Update docs/reward_blocks.md
- [ ] Create docs/tutorial/rare_variant_triage.md
- [ ] Update README.md
- [ ] Update docs/getting_started.md

**Phase 7: Testing** (6 items)
- [ ] tests/rewards/test_virtual_cell_delta.py
- [ ] tests/rewards/test_motif_delta.py
- [ ] tests/rewards/test_conservation.py
- [ ] tests/data/test_variant_dataset.py
- [ ] tests/evaluators/test_variant_composite.py
- [ ] tests/integration/test_variant_triage_pipeline.py

**Phase 8: Logging** (2 items)
- [ ] Enhance MLflow tracker with aux channels
- [ ] Add nested run support

**Cleanup** (3 items)
- [ ] Linting (ruff, mypy)
- [ ] Test coverage (80%+)
- [ ] Architecture docs

---

## 🔑 Key Classes & Modules

### New Files to Create
```
strand/rewards/virtual_cell_delta.py    (200-300 lines)
strand/rewards/motif_delta.py           (200-300 lines)
strand/rewards/conservation.py          (150-200 lines)
strand/data/variant_dataset.py          (300-400 lines)
strand/evaluators/variant_composite.py  (150-200 lines)

configs/examples/rare_variant_triage/rare_variant_triage.yaml

docs/tutorial/rare_variant_triage.md    (Tutorial)

tests/rewards/test_virtual_cell_delta.py
tests/rewards/test_motif_delta.py
tests/rewards/test_conservation.py
tests/data/test_variant_dataset.py
tests/evaluators/test_variant_composite.py
tests/integration/test_variant_triage_pipeline.py
```

### Files to Modify
```
pyproject.toml                          (add variant-triage extras)
strand/rewards/base.py                  (add metadata enums)
strand/rewards/registry.py              (add factory pattern)
strand/engine/types.py                  (add VariantMetadata, SequenceContext)
strand/evaluators/reward_aggregator.py  (add context parameter)
strand/cli/cli.py                       (add variant commands)
strand/logging/mlflow_tracker.py        (add aux channel logging)
docs/reward_blocks.md                   (document new blocks)
README.md                               (mention variants)
docs/getting_started.md                 (variant quick start)
```

---

## 💻 Code Snippets to Implement

### 1. Metadata in base.py
```python
from enum import Enum

class ObjectiveType(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

class BlockType(Enum):
    HEURISTIC = "heuristic"
    ADVANCED = "advanced"
    DETERMINISTIC = "deterministic"

@dataclass(slots=True)
class BaseRewardBlock:
    name: str
    weight: float = 1.0
    block_type: BlockType = BlockType.HEURISTIC
    description: str = ""
    requires_context: bool = False
    # ... rest
```

### 2. Factory in registry.py
```python
class RewardRegistry:
    @classmethod
    def create(cls, config: dict | str, **kwargs):
        if isinstance(config, str):
            # Backward compat: just name
            name = config
        else:
            # Config dict: {"type": "virtual_cell_delta", "config": {...}}
            name = config.get("type")
            kwargs = {**config.get("config", {}), **kwargs}

        if name not in cls._registry:
            raise KeyError(f"Unknown reward: {name}")
        return cls._registry[name](**kwargs)
```

### 3. VariantMetadata in types.py
```python
@dataclass(frozen=True, slots=True)
class VariantMetadata:
    chrom: str
    pos: int
    ref: str
    alt: str
    rsid: str | None = None
    annotations: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SequenceContext:
    ref_seq: Sequence
    alt_seq: Sequence
    metadata: VariantMetadata
    ref_window: tuple[int, int]
    alt_window: tuple[int, int]
```

### 4. VariantDataset loader
```python
class VariantDataset:
    def __init__(self, vcf_path: str, fasta_path: str, window_bp: int = 1000):
        self.vcf = pysam.VariantFile(vcf_path)
        self.fasta = pysam.FastaFile(fasta_path)
        self.window = window_bp

    def __iter__(self):
        for variant in self.vcf:
            context = self._build_context(variant)
            yield context

    def _build_context(self, variant) -> SequenceContext:
        # Extract ref/alt sequences with flanking windows
        # Return SequenceContext object
        pass
```

### 5. VariantCompositeEvaluator
```python
@dataclass
class VariantCompositeEvaluator(Evaluator):
    rewards: RewardAggregator
    context: SequenceContext | None = None
    include_gc: bool = False
    include_length: bool = False

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        # Call rewards with optional context
        # Aggregate constraints (GC, length, motif disruption)
        # Return Metrics with objective, constraints, aux
        pass
```

### 6. Example Config
```yaml
dataset:
  type: "variant_vcf"
  vcf_path: "variants.vcf"
  fasta_path: "reference.fa"
  window_size: 1000

rewards:
  - type: "virtual_cell_delta"
    model_path: "enformer-base"
    weight: 0.5

  - type: "motif_delta"
    tf_list: ["MA0001"]
    weight: 0.3

  - type: "conservation"
    bw_path: "phylop.bw"
    weight: 0.2

evaluator:
  type: "variant_composite"
  include_gc: true
```

---

## ✅ Success Criteria (Checkpoints)

### After Phase 2 (Reward Blocks)
- [ ] Can import `VirtualCellDeltaReward` without errors
- [ ] Can import `MotifDeltaReward` without errors
- [ ] Can import `ConservationReward` without errors
- [ ] All new blocks register in RewardRegistry
- [ ] Tests mock heavy dependencies (Enformer, JASPAR, BigWig)

### After Phase 3 (Data Layer)
- [ ] Can create SequenceContext from VCF+FASTA
- [ ] VariantDataset produces correct ref/alt sequences
- [ ] Window extraction works (±1000bp)

### After Phase 4 (Evaluators)
- [ ] RewardAggregator accepts SequenceContext
- [ ] VariantCompositeEvaluator composes all metrics
- [ ] Metrics include objective, constraints, aux

### After Phase 5 (CLI)
- [ ] `strand run config.yaml` works
- [ ] Example config parses without errors

### After Phase 7 (Tests)
- [ ] pytest runs without import errors
- [ ] All new modules have 85%+ coverage
- [ ] Tests include dependency guards
- [ ] Integration test runs end-to-end

---

## 🚀 Running the Implementation

### Start
```bash
cd /Users/sethmorton/Desktop/codingplayground/strand-sdk
git checkout -b feat/variant-triage
```

### Phase by Phase
```bash
# Phase 1: Update deps
# Edit pyproject.toml, add variant-triage extras

# Phase 2: Reward blocks
# Create 3 new files: virtual_cell_delta.py, motif_delta.py, conservation.py
# Modify: base.py, registry.py

# Phase 3: Data layer
# Create: variant_dataset.py
# Modify: types.py

# Phase 4: Evaluators
# Create: variant_composite.py
# Modify: reward_aggregator.py

# Phase 5: CLI
# Modify: cli.py
# Create: rare_variant_triage.yaml

# Phase 6-8: Docs, tests, logging

# Cleanup
pytest --cov=strand tests/
ruff check strand/ --fix
mypy strand/ --strict
```

### PR/Review
```bash
git add -A
git commit -m "feat: variant-triage support (phases 1-8)"
git push origin feat/variant-triage
# Open PR for review
```

---

## 📊 Dependency Check

```bash
# Verify extras are installable
pip install -e ".[variant-triage]"

# Verify imports work
python -c "import enformer_pytorch, pysam, pyjaspar, MOODS, pyBigWig, pyranges"

# Verify linting
ruff check strand/
mypy strand/

# Run tests
pytest tests/ -v
```

---

## 📞 Troubleshooting

### Import Errors
```python
# Use dependency guards
try:
    import enformer_pytorch
except ImportError:
    raise ImportError("Install with: pip install strand-sdk[variant-triage]")
```

### VCF Errors
```bash
# Ensure VCF is bgzipped and indexed
bcftools view variants.vcf -O z > variants.vcf.bgz
bcftools index variants.vcf.bgz
```

### BigWig Access
```python
# Use try/finally for file cleanup
bw = pyBigWig.open(path)
try:
    stats = bw.stats(...)
finally:
    bw.close()
```

### GPU Out of Memory
```python
# Reduce batch size in config
executor:
  batch_size: 2  # was 4
```

---

## 🎓 Learning Resources

- **Enformer**: [Nature Methods Paper](https://www.nature.com/articles/s41592-021-01252-x)
- **JASPAR**: [Database](https://jaspar.genereg.net/)
- **MOODS**: [GitHub](https://github.com/ajawahar7/MOODS)
- **pyBigWig**: [GitHub](https://github.com/dpryan79/pyBigWig)
- **pysam**: [Docs](https://pysam.readthedocs.io/)
- **PyRanges**: [Docs](https://pyranges.readthedocs.io/)

---

## 📌 Important Files to Review First

1. **EXECUTION_PLAN.md** — Full detailed plan with all 25 items
2. **API_RESEARCH.md** — Package APIs, code patterns, best practices
3. **ARCHITECTURE_PATTERNS.md** — Design principles from httpx
4. **This file** — Quick reference

---

**Status:** Ready for Implementation ✅
**Total Work:** ~10 days focused development
**Todo Items:** 25
**Coverage Target:** 80%+ overall, 85%+ new modules
