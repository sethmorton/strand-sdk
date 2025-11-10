"""Cloud API integration placeholder."""

import json
from pathlib import Path

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def push_results(payload: dict) -> None:
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/cloud_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    optimizer = Optimizer(
        sequences=["ACDE"],
        reward_blocks=[RewardBlock.stability(), RewardBlock.solubility()],
        method="cem",
    )
    payload = {
        "experiment": "cloud-prototype",
        "results": [
            {"id": seq.id, "score": score}
            for seq, score in optimizer.run().top()
        ],
    }
    push_results(payload)
