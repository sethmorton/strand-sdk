# Reward Blocks

| Block | Description | Config |
| --- | --- | --- |
| Stability | Hydrophobicity-ratio heuristic (no pretrained model) | `model` (label only), `threshold`, `weight` |
| Solubility | Polar-residue fraction heuristic | `model` (label only), `weight` |
| Novelty | Distance metrics vs. baseline | `baseline`, `metric`, `weight` |
| Length Penalty | Soft clamp around target length | `target_length`, `tolerance`, `weight` |
| Custom | User callable | `name`, `fn`, `weight` |
