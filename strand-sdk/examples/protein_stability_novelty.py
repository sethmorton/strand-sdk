"""Protein stability + novelty demo."""

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock

if __name__ == "__main__":
    baseline = ["MKTAYIAKQRQISFVKSHFSRQDILD"]
    sequences = [
        "MKTVYIAKQRQISFVKSHFSRQDILD",
        "GHHHHHHHHHHHHHHHHHHHHHHHHH",
    ]
    rewards = [
        RewardBlock.stability(threshold=0.6, weight=0.7),
        RewardBlock.novelty(baseline=baseline, metric="levenshtein", weight=0.3),
    ]
    optimizer = Optimizer(sequences=sequences, reward_blocks=rewards, method="cmaes", iterations=8)
    print(optimizer.run().top())
