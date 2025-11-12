# StrategyContext & Runtime Adapter

Strategies can opt into accelerator support, batching hints, and runtime helpers by declaring capabilities via `StrategyCaps`. This document explains how the context is constructed and how to use it safely.

## When the Context Appears

1. You build an `Engine` with `EngineConfig(device=..., batching=...)`.
2. The engine inspects `strategy.strategy_caps()` (or the `_CAPS` attribute).
3. If `requires_runtime` or `supports_fine_tuning` is `True`, the engine calls `build_strategy_context(...)`.
4. The engine calls `strategy.prepare(context)` (if defined) before the first `ask`.
5. Each iteration, `strategy.train_step(items, context)` is invoked when the strategy supports fine-tuning.

## What the Context Contains

```python
from strand.engine.runtime import StrategyContext

context.device  # DeviceConfig(target, mixed_precision, grad_accum)
context.batch   # BatchConfig(eval_size, train_size, max_tokens)
context.runtime # ModelRuntime | None
```

- `DeviceConfig` defaults to CPU with no mixed precision. Override via `EngineConfig(device=...)`.
- `BatchConfig` provides hints to both the executor and strategies (train batches) without prop drilling.
- `ModelRuntime` wraps Hugging Face Accelerate and exposes `.prepare_module`, `.autocast()`, `.backward(loss)`, and `.wait()`.

## Using StrategyRuntimeAdapter

`strand.engine.strategies.runtime_adapter.StrategyRuntimeAdapter` centralizes the boilerplate involved in preparing PyTorch modules and optimizers.

```python
from strand.engine.strategies.runtime_adapter import StrategyRuntimeAdapter

def prepare(self, context: StrategyContext) -> None:
    if context.runtime is None:
        return
    self._adapter = StrategyRuntimeAdapter(context.runtime)
    policy = PolicyNet(...)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    result = self._adapter.prepare_module(policy, opt)
    self._policy = result.module
    self._optimizer = result.optimizer

def train_step(..., context: StrategyContext) -> None:
    with self._adapter.autocast_context():
        loss = ...
    self._adapter.backward(loss)
    self._optimizer.step()
```

The adapter also includes convenience methods for checkpointing (`save_checkpoint` / `load_checkpoint`) that unwrap distributed wrappers before serializing states.

## Declaring Capabilities

`StrategyCaps` (in `strand.engine.runtime`) exposes these fields:

| Field | Meaning |
| --- | --- |
| `requires_runtime` | Needs a `ModelRuntime` (PyTorch modules, accelerators) |
| `supports_fine_tuning` | Strategy wants `train_step` calls each iteration |
| `needs_sft_dataset` | Emit a warning if no supervised data is configured |
| `kl_regularization` | Requested KL penalty mode (`"none"`, `"token"`, `"sequence"`) |
| `max_tokens_per_batch` | Token budget for evaluator/executor |
| `prefers_autocast` | Hint for using mixed precision |

Always default to `StrategyCaps()` unless you truly need extras. Smaller caps mean easier reuse in CPU-only or low-resource settings.

## Troubleshooting

- If `context.runtime` is `None`, double-check that your strategy actually set `requires_runtime=True`.
- If Accelerate raises errors, verify that `torch.cuda.is_available()` matches the `DeviceConfig.target` you selected.
- Log the resolved caps via `resolve_strategy_caps(strategy)` when debugging hybrid strategies; the helper will merge child caps for you.
