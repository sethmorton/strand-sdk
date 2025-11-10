from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def test_cem_optimizer_path():
    optimizer = Optimizer(
        sequences=["AAAAA", "CCCCC"],
        reward_blocks=[RewardBlock.stability()],
        method="cem",
        iterations=3,
    )
    assert optimizer.run().scores
