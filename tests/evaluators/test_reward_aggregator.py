from strand.core.sequence import Sequence
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


def test_reward_aggregator_weighted_sum():
    rewards = [
        RewardBlock.stability(weight=1.0),
        RewardBlock.length_penalty(target_length=4, tolerance=0, weight=0.5),
    ]
    evaluator = RewardAggregator(reward_blocks=rewards)

    seq = Sequence(id="test", tokens="ACDE")
    metrics = evaluator.evaluate_batch([seq])[0]

    # All reward blocks already apply weights internally, so objective is additive
    stability_score = rewards[0].score(seq)
    length_score = rewards[1].score(seq)

    assert metrics.objective == stability_score + length_score
    assert metrics.constraints == {}
    assert metrics.aux == {}


def test_reward_aggregator_handles_empty_batch():
    evaluator = RewardAggregator(reward_blocks=[RewardBlock.stability(weight=1.0)])
    assert evaluator.evaluate_batch([]) == []
