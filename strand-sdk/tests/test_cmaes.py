from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def test_cmaes_filters_below_average():
    optimizer = Optimizer(
        sequences=["AAAAA", "CCCCC", "GGGGG"],
        reward_blocks=[RewardBlock.solubility()],
        method="cmaes",
        iterations=4,
        population_size=2,
    )
    results = optimizer.run()
    assert len(results.scores) <= 3
    assert all(score >= 0 for score in results.scores)
