from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def test_optimizer_runs_and_builds_manifest(tmp_path):
    optimizer = Optimizer(
        sequences=["ACDEFG"],
        reward_blocks=[RewardBlock.stability(), RewardBlock.solubility()],
        iterations=2,
        population_size=2,
    )
    results = optimizer.run()
    assert results.scores
    manifest = results.to_manifest()
    assert manifest is not None
    path = manifest.save(tmp_path / "manifest.json")
    assert path.exists()
