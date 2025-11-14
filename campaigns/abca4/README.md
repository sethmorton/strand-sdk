# ABCA4 Variant Intelligence Campaign

This folder contains all work related to the ABCA4 rare-variant pipeline so the
`strand-sdk` framework can remain clean and reusable. Everything in here is
self-contained:

- `src/` – pipeline scripts (data ingestion, annotation, feature computation, reporting).
- `docs/` – research notes and orchestration plans.
- `notebooks/` – Marimo notebooks for exploratory analysis.
- `data_raw/`, `data_processed/` – campaign-specific artifacts (ignored by git).
- `tasks.py` – Invoke entrypoint for campaign automation.

## Quick Start

Run tasks from the repo root as usual:

```bash
invoke -l                        # list available tasks
invoke download-data             # fetch ClinVar/gnomAD/SpliceAI/AlphaMissense
invoke run-pipeline              # execute end-to-end pipeline
invoke run-optimization          # rank variants + log to MLflow
invoke generate-report           # write data_processed/reports/* snapshot files
```

All scripts assume paths relative to this folder, so nothing leaks into the
framework modules under `strand/` or `src/`.


get an example fasta sequence related to abca4  
```bash
https://rest.uniprot.org/uniprotkb/P78363.fasta
```