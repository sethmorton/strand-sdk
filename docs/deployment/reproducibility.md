# Reproducibility & Deployment

Guide for creating reproducible, shareable Ctrl-DNA runs.

## Run Manifests

Export complete run state:

```python
import json
from strand.manifests import EngineManifest

manifest = EngineManifest(
    config=engine_config,
    strategy=strategy.state(),
    results=results,
    model_checkpoint="checkpoints/best_policy.pt",
    mlflow_run_id="abc123",
)

with open("run_manifest.json", "w") as f:
    json.dump(manifest.to_dict(), f, indent=2)
```

## Sharing Results

Package for sharing:

```
my_run/
├── config.yaml
├── best_policy.pt
├── results.json
├── run_manifest.json
└── best_sequences.fasta
```

## Verification

Verify checkpoint integrity:

```python
import hashlib

def checksum_file(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

sha = checksum_file("best_policy.pt")
# Record and share this hash
```

## Reproduction

Load and reproduce:

```python
import json
with open("run_manifest.json") as f:
    manifest = json.load(f)

# Recreate exact config
engine_config = EngineConfig(**manifest["config"])
```

## Best Practices

1. **Version everything**: Config, code, models
2. **Use MLflow**: Automatic artifact management
3. **Export manifests**: Share reproducible runs
4. **Document seeds**: Random seeds for determinism
5. **Track hashes**: Verify file integrity

