"""Export manifest and CSV example."""

from pathlib import Path

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock

if __name__ == "__main__":
    optimizer = Optimizer(
        sequences=["ACDEFG"],
        reward_blocks=[RewardBlock.stability(), RewardBlock.solubility()],
        iterations=2,
    )
    results = optimizer.run()
    results.export_csv(Path("artifacts") / "results.csv")
    manifest = results.to_manifest()
    if manifest:
        manifest.save(Path("artifacts") / "manifest.json")
