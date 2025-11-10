from strand.core.sequence import Sequence
from strand.rewards import RewardBlock


def test_stability_reward_threshold():
    block = RewardBlock.stability(threshold=0.2)
    sequence = Sequence(id="seq", tokens="AAAA")
    assert block.score(sequence) == 1.0


def test_novelty_requires_baseline():
    try:
        RewardBlock.novelty(baseline=[], weight=1.0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when baseline missing")
