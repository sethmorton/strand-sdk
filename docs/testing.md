# Testing Guide

This document describes how tests are organized and best practices for writing tests in the Strand SDK.

## Test Organization

Tests are organized by **feature** rather than by module. This makes it easier to find related tests and understand the test coverage for each feature area.

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_utils.py            # Utility functions
├── test_manifests.py        # Manifest serialization
│
├── engine/                  # Engine feature tests
│   ├── __init__.py
│   ├── test_strategies.py   # Strategy tests (Random, CEM)
│   └── test_engine.py       # Engine orchestration tests
│
├── rewards/                 # Reward block tests
│   ├── __init__.py
│   ├── test_basic.py        # Basic rewards (Stability, Solubility)
│   └── test_gc_content.py   # GC Content reward
│
└── core/                    # Core type tests
    └── (future)
```

## Test Patterns

### 1. Using Pytest Classes

Group related tests into test classes for better organization:

```python
class TestRandomStrategy:
    """RandomStrategy unit tests."""

    def test_best_tracking(self):
        """Test that RandomStrategy tracks the best sequence."""
        # ...

    def test_ask_returns_sequences_in_range(self):
        """Test that sequences are in the correct length range."""
        # ...
```

### 2. Shared Fixtures

Use `conftest.py` to define shared fixtures. This avoids duplication across test files:

```python
# tests/conftest.py
@pytest.fixture
def basic_rewards():
    """Basic reward blocks for testing."""
    return [
        RewardBlock.stability(weight=1.0),
        RewardBlock.solubility(weight=0.5),
    ]

# Usage in test files
def test_something(basic_rewards):
    evaluator = RewardAggregator(reward_blocks=basic_rewards)
    # ...
```

### 3. Parametrized Tests

Use `@pytest.mark.parametrize` to test multiple inputs:

```python
@pytest.mark.parametrize("target,tolerance,expected_ge", [
    (0.5, 0.1, 0.0),
    (0.4, 0.05, 0.0),
    (0.6, 0.2, 0.0),
])
def test_various_targets(self, target, tolerance, expected_ge):
    """Test reward with various target values."""
    reward = RewardBlock.gc_content(target=target, tolerance=tolerance)
    seq = Sequence(id="test", tokens="GCGCAAAA")
    score = reward.score(seq)
    assert score >= expected_ge
```

## Test Layers

Tests are organized by layer of abstraction:

### Unit Tests
Test individual components in isolation.

```python
# tests/engine/test_strategies.py
def test_best_tracking(self):
    """Test that RandomStrategy tracks the best sequence."""
    strategy = RandomStrategy(alphabet="ACDE", min_len=5, max_len=10, seed=42)
    candidates = strategy.ask(3)
    items = [(candidates[0], 0.5, None), ...]
    strategy.tell(items)
    best_seq, best_score = strategy.best()
    assert best_score == 0.9
```

### Integration Tests
Test how components work together.

```python
# tests/engine/test_engine.py
def test_minimal_loop(self):
    """Test a minimal end-to-end optimization loop."""
    rewards = [RewardBlock.stability(weight=1.0), ...]
    evaluator = RewardAggregator(reward_blocks=rewards)
    executor = LocalExecutor(evaluator=evaluator)
    strategy = RandomStrategy(...)
    engine = Engine(...)
    results = engine.run()
    assert results.best is not None
```

## Naming Conventions

- Test files: `test_<feature>.py`
- Test classes: `Test<Component>`
- Test methods: `test_<what_is_being_tested>`

Example: `tests/rewards/test_gc_content.py::TestGCContentReward::test_perfect_match`

## Running Tests

### Run all tests
```bash
source .venv/bin/activate
pytest tests/
```

### Run specific test file
```bash
pytest tests/engine/test_strategies.py
```

### Run specific test class
```bash
pytest tests/engine/test_strategies.py::TestRandomStrategy
```

### Run specific test
```bash
pytest tests/engine/test_strategies.py::TestRandomStrategy::test_best_tracking
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=strand --cov-report=html
```

## Best Practices

1. **Clear test names**: Use descriptive names that explain what is being tested
   - ✓ `test_best_tracking` - Clear what's being tested
   - ✗ `test_stuff` - Too vague

2. **One assertion per concept**: Each test should verify one logical thing
   - ✓ Test that `best()` returns the right sequence
   - ✗ Test `best()`, `ask()`, and `tell()` all at once

3. **Use fixtures for common setup**: Avoid repeating setup code

4. **Clear arrange/act/assert**: Structure tests with clear phases:
   ```python
   def test_something(self):
       # Arrange - set up inputs
       strategy = RandomStrategy(...)
       candidates = strategy.ask(3)
       
       # Act - perform the action
       strategy.tell([(candidates[0], 0.5, None), ...])
       best_seq, best_score = strategy.best()
       
       # Assert - verify the result
       assert best_score == 0.5
   ```

5. **Test error cases**: Don't just test the happy path
   ```python
   def test_invalid_target(self):
       """Test that invalid target values raise errors."""
       with pytest.raises(ValueError, match="target must be in"):
           RewardBlock.gc_content(target=1.5)
   ```

6. **Use pytest features**: Leverage parametrize, fixtures, marks:
   ```python
   @pytest.mark.parametrize("input,expected", [...])
   def test_something(self, input, expected):
       assert process(input) == expected
   ```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Pushes to develop branch

All tests must pass before merging.

## Coverage Goals

Target test coverage:
- Core engine: >90%
- Strategies: >85%
- Rewards: >85%
- Overall: >80%

Check coverage with:
```bash
pytest tests/ --cov=strand --cov-report=term-missing
```

