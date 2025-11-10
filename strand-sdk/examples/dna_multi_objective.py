"""DNA multi-objective example."""

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock

if __name__ == "__main__":
    sequences = ["ATGCGTACGTAGCTAGCTAG", "ATGCGTACGTAGTTTTTTTT"]
    rewards = [
        RewardBlock.length_penalty(target_length=20, tolerance=2),
        RewardBlock.custom(
            name="gc-content",
            fn=lambda seq, _: seq.tokens.count("G") / len(seq),
            weight=0.5,
        ),
    ]
    optimizer = Optimizer(sequences=sequences, reward_blocks=rewards, method="ga", iterations=6)
    print(optimizer.run().top())
