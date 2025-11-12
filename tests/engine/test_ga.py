from strand.core.sequence import Sequence
from strand.engine.strategies.ga import GAStrategy
from strand.engine.types import Metrics


def test_ga_strategy_initialization_and_ask():
    ga = GAStrategy(alphabet="AC", min_len=5, max_len=7, seed=123)
    pop = ga.ask(6)
    assert len(pop) == 6
    assert all(isinstance(s, Sequence) for s in pop)
    # fixed length midpoint
    assert all(len(s.tokens) == (5 + 7) // 2 for s in pop)


def test_ga_strategy_tell_updates_best():
    ga = GAStrategy(alphabet="AC", min_len=4, max_len=4, seed=1)
    pop = ga.ask(4)
    metrics = Metrics(objective=0.0, constraints={}, aux={})
    items = [
        (pop[0], 0.2, metrics),
        (pop[1], 0.8, metrics),
        (pop[2], 0.5, metrics),
        (pop[3], 0.3, metrics),
    ]
    ga.tell(items)
    best = ga.best()
    assert best is not None and best[1] == 0.8

