# Examples Directory

All scripts live under `examples/`. Run them from the repo root after installing the SDK in editable mode.

| Script | What it demonstrates |
| --- | --- |
| `basic_optimization.py` | Legacy heuristic rewards (stability, novelty) across Random, CEM, GA. Good for quick smoke tests. |
| `engine_basic.py` | Minimal engine loop using GC content + LocalExecutor. Mirrors the Quick Start but with CLI arguments. |
| `engine_with_tracking.py` | Adds MLflow tracking, manifest exports, and checkpoint logging to the basic run. |
| `engine_varlen_cmaes.py` | Demonstrates CMA-ES for variable-length sequences, including how to clamp candidates and update length statistics. |
| `engine_ctrl_dna_hybrid.py` | Hybrid coordination of RLPolicy + evolutionary strategies, logging dual variables and mock constraint violations. |

Tips:
- Use `python examples/engine_with_tracking.py --help` to see runtime flags.
- Point the examples at your own FASTA/CSV files once comfortable; all reward/evaluator hooks are surfaced via arguments so nothing is hard-coded.
