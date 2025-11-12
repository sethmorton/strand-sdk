# FAQ

### What stage is Strand in right now?
Alpha. The engine, strategies, evaluators, datasets, logging utilities, supervised warm-start hooks, and CLI orchestration are ready for local experimentation. Expect some API movement while Ctrl-DNA parity solidifies.

### When can I run real optimization jobs that feed a wet lab?
You can run simulated Ctrl-DNA loops today (Random/CEM/RLPolicy + GC/Enformer/TFBS rewards). Wet-lab integrations and verified HyenaDNA checkpoints are being vetted with partners through early 2026; tagged open-source releases will follow those runs.

### Which Python versions are supported?
Python 3.11+ during the alpha period. The tooling uses `typing` features and dataclass slots that require 3.11.

### How will the open-source SDK and managed offering relate?
The open-source SDK (this repo) includes strategies, evaluators, datasets, manifests, and logging. The managed service layers on cluster orchestration, data governance, and workflow automation. Anything related to model training and reward math will remain OSS.

### What makes Strand different from other sequence-design tools?
Strand treats optimization + tracing as the neutral layer between many generative models and the wet lab. You plug in any sequence generator or foundation model, define constraints as reward blocks, and Strand runs iterative ask→evaluate→score→tell loops while emitting manifests/MLflow runs for reproducibility.

### Where is the CLI?
Use `strand run path/to/config.yaml` to launch runs directly from declarative configs. The CLI wires up strategies, rewards, executors, devices, SFT datasets, and dual managers without writing new scripts.

### Does the RL policy support supervised warm-starts?
Yes. Pass `Engine(..., sft=SFTConfig(dataset, epochs, batch_size))` and `Engine` will call `strategy.warm_start(...)` before the RL loop. `RLPolicyStrategy` ships with a built-in warm-start implementation that logs SFT metrics via `MLflowTracker`.
