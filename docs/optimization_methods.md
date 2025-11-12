# Optimization Methods

The SDK ships with six strategy implementations. Each implements the `Strategy` protocol and declares its `StrategyCaps` so the engine can provide runtimes or warn about missing inputs.

| Strategy | Module | When to use | Caps / Notes |
| --- | --- | --- | --- |
| RandomStrategy | `strand.engine.strategies.random` | Baseline sampling, smoke tests, evaluator debugging | No special caps. Extremely fast, zero state. |
| CEMStrategy | `strand.engine.strategies.evolutionary.cem` | Continuous improvement on structured search spaces | Maintains elite set; deterministic with `seed`. |
| GAStrategy | `strand.engine.strategies.evolutionary.ga` | Classic selection + crossover for discrete alphabets | Supports min/max length; no runtime requirements. |
| CMAESStrategy / CMAESVarLenStrategy | `strand.engine.strategies.evolutionary.cmaes*` | Real-valued encodings and variable-length optimization | Requires translating sequences to numeric vectors; still CPU-friendly. |
| RLPolicyStrategy | `strand.engine.strategies.rl.rl_policy` | Ctrl-DNA style policy-gradient optimization with constraints | `StrategyCaps(requires_runtime=True, supports_fine_tuning=True, kl_regularization="token")`. Includes online `train_step` and `warm_start` hooks plus pluggable policy heads. |
| HybridStrategy | `strand.engine.strategies.hybrid` | Blend multiple child strategies (e.g., RL + CEM) | Aggregates child `StrategyCaps`; enforces lowest token budget and merges batch hints. |

## Adding Your Own Strategy

1. Implement `ask`, `tell`, `best`, and `state`.
2. (Optional) Implement `strategy_caps` to request runtimes, token budgets, or KL settings.
3. (Optional) Implement `prepare(context: StrategyContext)` to capture device/batch hints.
4. (Optional) Implement `train_step(items, context)` if you need per-iteration learning.

Use `StrategyRuntimeAdapter` when dealing with PyTorch modules so you avoid repeating runtime prep/checkpoint logic.
