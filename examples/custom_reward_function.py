"""Custom reward block example."""

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


def aromatic_ratio(sequence, _context):
    aromatics = {"F", "Y", "W"}
    return sum(residue in aromatics for residue in sequence.tokens.upper()) / len(sequence)


if __name__ == "__main__":
    rewards = [RewardBlock.custom(name="aromatic", fn=aromatic_ratio)]
    optimizer = Optimizer(sequences=["FWYFWY"], reward_blocks=rewards, method="random", iterations=3)
    print(optimizer.run().top())
