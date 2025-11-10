# Reward Blocks

| Block | Description | Config |
| --- | --- | --- |
| Stability | Hydrophobicity proxy via ESMFold placeholder | `model`, `threshold`, `weight` |
| Solubility | Polar residue ratio | `model`, `weight` |
| Novelty | Distance metrics vs. baseline | `baseline`, `metric`, `weight` |
| Length Penalty | Soft clamp around target length | `target_length`, `tolerance`, `weight` |
| Custom | User callable | `name`, `fn`, `weight` |
