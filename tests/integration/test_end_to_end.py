from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def test_end_to_end_manifest_roundtrip(tmp_path):
    optimizer = Optimizer(
        sequences=["ACDEFGHIK"],
        reward_blocks=[RewardBlock.stability(), RewardBlock.novelty(baseline=["ACDEFGHIK"])],
        method="ga",
        iterations=2,
    )
    results = optimizer.run()
    manifest = results.to_manifest()
    assert manifest is not None
    path = manifest.save(tmp_path / "manifest.json")
    assert path.exists()
