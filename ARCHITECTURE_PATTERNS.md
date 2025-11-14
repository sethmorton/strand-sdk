# Strand SDK Architecture Patterns: Inspired by httpx

**Inspired by:** [httpx Repository](https://github.com/encode/httpx)  
**Focus:** Progressive Disclosure, Clear Naming, Modular Composition

These notes translate httpx’s design habits into concrete rules for strand-sdk. The variant-triage campaign is simply the first consumer of these rules: the same patterns should make it trivial to plug in new evaluators, new datasets, or entirely different campaigns without rewriting the engine. Treat this file as the guardrail for *any* medium-to-large feature.

---

## 🎯 Key Principles

The strand-sdk variant-triage implementation adopts architectural principles from httpx, a modern HTTP client that balances simplicity with power:

### 1. **Progressive Disclosure**
Complex features are available but not forced. Users start simple and layer on complexity as needed.

**httpx Example:** Basic `httpx.get()` works out of the box; advanced features like HTTP/2, connection pooling, and custom auth are available through optional parameters or dedicated classes.

**Strand Example (Campaign-Agnostic):**
```python
# Level 1: Basic usage (no variants)
from strand.engine.engine import Engine
engine = Engine(...)
results = engine.run()

# Level 2: With additional context (opt-in)
from strand.data.variant_dataset import VariantDataset
from strand.evaluators.variant_composite import VariantCompositeEvaluator
dataset = VariantDataset("variants.vcf", "reference.fa")
engine = Engine(..., evaluator=VariantCompositeEvaluator(...))

# Swap VariantDataset for another context loader for different campaigns without touching Engine.
```

---

### 2. **Clear Naming Conventions**
Names **immediately convey purpose**. No ambiguity about what a component does.

**httpx Example:**
- `Client` ← You know it's a client
- `Request`, `Response` ← Obvious purposes
- `AsyncClient` ← Immediately recognizable as async variant
- `Auth` base class, `BasicAuth`, `DigestAuth` ← Clear inheritance

**Strand Example:**
```python
# Clear naming for reward blocks
GCContentReward         # Heuristic: score by GC%
StabilityReward         # Heuristic: score by hydrophobicity
NoveltyReward           # Heuristic: score by distance

VirtualCellDeltaReward  # Advanced: Enformer ref/alt delta
MotifDeltaReward        # Advanced: JASPAR/MOODS TF presence delta
ConservationReward      # Advanced: pyBigWig window scores

# Clear evaluator names
CompositeEvaluator      # Basic: rewards + optional constraints
VariantCompositeEvaluator  # Advanced: rewards + variant context
```

---

### 3. **Modular Composition (Not Inheritance Hell)**
Prefer composition over deep inheritance hierarchies. Mix and match components.

**httpx Example:**
```python
# httpx doesn't force inheritance chains
client = httpx.Client(auth=BasicAuth(...), timeout=10, limits=Limits(...))
# Each feature is independently composable
```

**Strand Example:**
```python
# Each reward is independent, composed together
rewards = [
    GCContentReward(target=0.5, tolerance=0.05, weight=0.2),
    VirtualCellDeltaReward(model_path="...", weight=0.5),
    MotifDeltaReward(tf_list=["MA0001"], weight=0.3),
    ConservationReward(bw_path="phylop.bw", weight=0.2),
]

# Use with any evaluator
evaluator = CompositeEvaluator(
    rewards=RewardAggregator(rewards),
    include_gc=True,
    include_length=True,
)
```

---

### 4. **Config-First Design**
Enable declarative composition via YAML/config, not just programmatic.

**httpx Example:**
- Supports via Hydra/structured configs (not native, but pattern is there)
- Users can compose clients declaratively in many frameworks

**Strand Example:**
```yaml
# configs/examples/rare_variant_triage/rare_variant_triage.yaml
dataset:
  type: "variant_vcf"
  vcf_path: "variants.vcf"
  fasta_path: "reference.fa"
  window_size: 1000

rewards:
  - type: "gc_content"
    target: 0.5
    tolerance: 0.05
    weight: 0.2

  - type: "virtual_cell_delta"
    model_path: "enformer-base"
    target_cell_types: ["hNSPC"]
    weight: 0.5

  - type: "motif_delta"
    tf_list: ["MA0001", "MA0002"]
    threshold: 6.0
    weight: 0.3

  - type: "conservation"
    bw_path: "phylop.bw"
    agg_method: "mean"
    weight: 0.2

evaluator:
  type: "variant_composite"
  include_gc: true
  include_length: true
  constraints:
    - type: "motif_disruption"
      max_disruption: 2

engine:
  method: "cmaes"
  iterations: 50
  population_size: 100

executor:
  type: "torch"
  device: "cuda:0"
  batch_size: 4
```

**Usage:**
```bash
strand run configs/examples/rare_variant_triage/rare_variant_triage.yaml
```

---

### 5. **Type Safety with Rich Metadata**
Use type annotations + dataclasses to make intent clear and catch errors early.

**httpx Example:**
```python
@dataclass
class Request:
    method: str
    url: str
    headers: dict[str, str] | None = None
    content: bytes | None = None
```

**Strand Example:**
```python
from dataclasses import dataclass
from enum import Enum

class ObjectiveType(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

class BlockType(Enum):
    HEURISTIC = "heuristic"
    ADVANCED = "advanced"
    DETERMINISTIC = "deterministic"

@dataclass(frozen=True, slots=True)
class RewardBlockMetadata:
    block_type: BlockType
    description: str
    requires_context: bool = False

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

---

### 6. **Optional Extras Strategy**
Heavy dependencies are optional. Core functionality works without them.

**httpx Example:**
```toml
[project.optional-dependencies]
http2 = ["h2"]
cli = ["rich", "click"]
```

**Strand Example:**
```toml
[project.optional-dependencies]
variant-triage = [
    "enformer-pytorch>=0.1.0",
    "pyjaspar>=3.0.0",
    "MOODS-python>=1.9.4",
    "pyBigWig>=0.3.21",
    "pysam>=0.23.0",
    "PyRanges>=0.13.0",
]
```

**Installation Tiers:**
```bash
# Minimal: Just core strategies and heuristic rewards
pip install strand-sdk

# With models: Add foundation model support
pip install strand-sdk[models]

# With variants: Add variant triage support
pip install strand-sdk[variant-triage]

# Full: Everything
pip install strand-sdk[variant-triage,models,rl-training]
```

---

### 7. **Clear Error Messages with Dependency Guards**
When dependencies are missing, provide **actionable error messages**.

**httpx Example:**
```python
try:
    import h2
except ImportError:
    raise ImportError("HTTP/2 support requires: pip install httpx[http2]")
```

**Strand Example:**
```python
def load_virtual_cell_delta(config):
    try:
        import enformer_pytorch
    except ImportError:
        raise ImportError(
            "VirtualCellDelta requires enformer-pytorch. "
            "Install with: pip install strand-sdk[variant-triage]"
        )
    # ... implementation
```

**In Tests:**
```python
import pytest

pytest.importorskip("enformer_pytorch")

def test_virtual_cell_delta():
    # Test only runs if enformer_pytorch is installed
    pass
```

---

### 8. **Module Organization Mirrors Use Cases**
Directory structure reflects user workflows, not just technical layering.

**httpx Structure:**
```
httpx/
  ├── _api/           # Core sync API
  ├── _async/         # Core async API
  ├── _models/        # Request/Response types
  ├── _auth/          # Auth strategies
  ├── _transports/    # HTTP implementations
  └── _client/        # Client implementations
```

**Strand Structure (with variant enhancements):**
```
strand/
  ├── core/
  │   └── sequence.py          # Base sequence type
  ├── data/
  │   ├── sequence_dataset.py  # Basic loader
  │   └── variant_dataset.py   # Variant-aware loader (NEW)
  ├── engine/
  │   ├── types.py             # Enhanced with VariantMetadata, SequenceContext
  │   ├── engine.py            # Core engine (unchanged)
  │   ├── strategies/          # Optimization strategies
  │   ├── executors/           # Parallel executors
  │   └── interfaces.py        # Evaluator interface
  ├── rewards/
  │   ├── base.py              # Enhanced with metadata
  │   ├── registry.py          # Expanded with factory pattern
  │   ├── gc_content.py        # Heuristic
  │   ├── stability.py         # Heuristic
  │   ├── virtual_cell_delta.py    # Advanced (NEW)
  │   ├── motif_delta.py           # Advanced (NEW)
  │   └── conservation.py          # Advanced (NEW)
  ├── evaluators/
  │   ├── reward_aggregator.py     # Enhanced with context support
  │   ├── composite.py             # Basic composite
  │   └── variant_composite.py     # Variant-aware (NEW)
  ├── logging/
  │   └── mlflow_tracker.py    # Enhanced with aux channels
  ├── cli/
  │   ├── cli.py               # Enhanced with variant commands
  │   └── commands/
  └── models/
      └── hyenadna.py          # Foundation model loader
```

**User Mental Model:**
- Start with `sequence.py` and simple rewards
- Move to `variant_dataset.py` for variant work
- Use `variant_composite.py` evaluator for full stack
- Config file in `configs/examples/rare_variant_triage/`
- Run via CLI: `strand run config.yaml`

---

### 9. **Defensive Imports and Lazy Loading**
Only import heavy dependencies when needed.

**Pattern:**
```python
# Don't do this (always imports, even if unused):
import enformer_pytorch
import pyBigWig
import pysam

# Do this instead:
def load_enformer_model(path: str):
    try:
        import enformer_pytorch
        return enformer_pytorch.Enformer.from_pretrained(path)
    except ImportError:
        raise ImportError("enformer-pytorch not installed")
```

---

### 10. **Documentation Mirrors Usage Tiers**

**httpx Docs:**
1. "Getting Started" — basic requests
2. "Advanced Usage" — sessions, auth, async
3. "API Reference" — detailed types
4. "Ecosystem" — integrations

**Strand Docs (with variants):**
1. "Getting Started" — basic optimization
2. "Reward Blocks" — available scoring functions
3. "Tutorial: Rare Variant Triage" — variant workflow
4. "Advanced Usage" — custom evaluators, constraints
5. "API Reference" — types and interfaces
6. "ARCHITECTURE.md" — design patterns (this file!)

---

## Implementation Checklist

Using these patterns, the variant-triage implementation will:

- [ ] **Progressive Disclosure:** Core engine unchanged; variant features are additive
- [ ] **Clear Naming:** Block types are self-documenting (VirtualCellDelta, MotifDelta, etc.)
- [ ] **Modular Composition:** Each reward is independent; compose them as needed
- [ ] **Config-First:** YAML configs drive behavior; CLI is thin wrapper
- [ ] **Type Safety:** Rich metadata in base classes; Pydantic validation on configs
- [ ] **Optional Extras:** Variant-triage is separate dependency group
- [ ] **Dependency Guards:** Clear error messages when packages are missing
- [ ] **Module Organization:** Structure reflects workflows (data/ has variant_dataset.py)
- [ ] **Lazy Loading:** Import heavy packages only when needed
- [ ] **Documentation:** Tutorial, examples, ARCHITECTURE.md, API reference

> **Pattern Adoption Roadmap:** when we add a *non-variant* feature (e.g., RNA structural rewards, protein docking datasets) we should check off the same boxes. The list is campaign-neutral on purpose.

---

## Comparison: Before and After

### Before (Hypothetical)
```python
# Confusing: unclear whether this is for variants
evaluator = ComplexEvaluator(...)

# Forced to write lots of boilerplate
# No clear path from VCF to optimization
```

### After (With Patterns)
```python
# Crystal clear: this is for variant optimization
evaluator = VariantCompositeEvaluator(
    rewards=RewardAggregator([
        VirtualCellDeltaReward(...),
        MotifDeltaReward(...),
        ConservationReward(...),
    ]),
    include_gc=True,
    include_length=True,
)

# Config-driven: YAML file does the work
$ strand run configs/examples/rare_variant_triage/rare_variant_triage.yaml
```

---

## Why These Patterns Matter

1. **Discoverability:** New users find the right components via names and examples
2. **Maintainability:** Clear structure makes changes easier; no hidden dependencies
3. **Extensibility:** Adding new reward types follows the same pattern
4. **Testability:** Modular components are easier to mock and test
5. **Reliability:** Type safety catches errors early; dependency guards prevent failures
6. **Usability:** Progressive disclosure means users don't need to understand everything upfront
7. **Repeatability:** Future campaigns inherit the same scaffolding; we avoid one-off subsystems.

---

## References

- [httpx GitHub Repository](https://github.com/encode/httpx)
  - File structure and module organization
  - Optional extras strategy in pyproject.toml
  - Dependency guard patterns

- [Python Packaging Best Practices](https://packaging.python.org/)
  - Optional dependencies and extras
  - Metadata and classifiers

- [Pydantic Documentation](https://docs.pydantic.dev/)
  - Type safety and validation

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
  - Single Responsibility: Each reward does one thing
  - Open/Closed: Registry pattern allows extension
  - Liskov Substitution: All rewards implement RewardBlockProtocol

---

**Last Updated:** November 2025
**Pattern Source:** [httpx](https://github.com/encode/httpx)
**Implementation Guide:** See EXECUTION_PLAN.md
