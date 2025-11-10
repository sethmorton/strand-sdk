"""Basic optimization example."""

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock

if __name__ == "__main__":
    sequences = ["ACDEFGHIKLMNPQRSTVWY"]
    rewards = [RewardBlock.stability(), RewardBlock.solubility()]
    optimizer = Optimizer(sequences=sequences, reward_blocks=rewards, iterations=5)
    results = optimizer.run()
    for seq, score in results.top():
        print(seq.id, score)
