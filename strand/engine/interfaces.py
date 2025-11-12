"""Engine protocol surfaces (no implementations).

These Protocols define the modular boundaries for the optimization engine.
They are intentionally small and easy to reason about.

Runtime Context
===============

Strategies that need accelerator support, fine-tuning, or advanced device management
can opt into a ``StrategyContext`` via the engine. The context carries:

- **DeviceConfig**: Target device (cpu/cuda), mixed-precision settings, gradient
  accumulation steps.
- **BatchConfig**: Batch-size hints and token budget for memory-efficient operations.
- **ModelRuntime**: Optional wrapper around :class:`accelerate.Accelerator` for
  device-aware model preparation, autocast, and backward passes.

Example: An RL policy strategy declares ``supports_fine_tuning=True`` in its
``StrategyCaps`` and receives a context with a ``ModelRuntime``. It can then:

1. Prepare its policy module via ``runtime.prepare_module(policy, optimizer)``
2. Use autocast context during training: ``with runtime.autocast(): ...``
3. Perform backward passes: ``runtime.backward(loss)``

Capability Declaration
======================

Every strategy *can* declare capabilities via a ``strategy_caps()`` method
returning ``StrategyCaps``:

- ``requires_runtime``: Signals that the strategy needs a ``ModelRuntime``.
- ``supports_fine_tuning``: Indicates the strategy supports a warm-start hook.
- ``needs_sft_dataset``: Strategy requires supervised data for initialization.
- ``kl_regularization``: Level of KL penalty ("none", "token", "sequence").
- ``max_tokens_per_batch``: Token budget; engine enforces batch splits.
- ``prefers_autocast``: Autocast context available during forward/backward.

The engine (``Engine.prepare``) inspects these and warns on mismatches
(e.g., "strategy needs SFT dataset but none provided").
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from strand.core.sequence import Sequence
from strand.engine.types import Metrics


class Strategy(Protocol):
    """Proposes candidates and ingests feedback (ask/tell).

    A Strategy owns its internal state and may implement any algorithm (e.g.,
    CEM, CMA-ES, GA, constrained RL). The engine treats it as a black box.

    Optional Advanced Methods
    ==========================

    Strategies that support fine-tuning or advanced runtimes may implement:

    - ``strategy_caps() -> StrategyCaps``: Declare capabilities and requirements.
    - ``prepare(context: StrategyContext) -> None``: Initialize strategy with
      device/batch config and optional runtime. Called once before ask/tell loop.
    - ``warm_start(dataset, *, epochs=1, batch_size=None, context=None, **kwargs) -> None``:
      Optional supervised pre-training hook invoked once before the RL loop.
    """

    def ask(self, n: int) -> list[Sequence]:
        """Return ``n`` candidate sequences to evaluate."""

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest evaluated candidates as ``(sequence, score, metrics)``."""

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best known candidate and its score, if any."""

    def state(self) -> Mapping[str, object]:
        """Return a serializable snapshot of strategy state (for manifests)."""


class Evaluator(Protocol):
    """Pure evaluator that maps sequences to structured metrics.

    Evaluators should be side-effect free and thread/process safe when used with
    executors that parallelize calls to ``evaluate_batch``.
    """

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Return metrics for a batch of sequences.

        Implementations may compute objectives with reward blocks or call
        external predictors. They should not perform parallelism internally; use
        an Executor for that.
        """


class Executor(Protocol):
    """Parallel wrapper around an Evaluator.

    Executors handle concurrency (threads/processes/remote) and timeouts while
    delegating actual scoring to an Evaluator.
    """

    def prepare(self) -> None:  # pragma: no cover - surface only
        """Optional heavy initialization (e.g., model warmup)."""

    def run(self, seqs: list[Sequence], *, timeout_s: float | None = None, batch_size: int = 64) -> list[Metrics]:
        """Evaluate ``seqs`` and return metrics in the same order."""

    def close(self) -> None:  # pragma: no cover - surface only
        """Optional cleanup hook for releasing resources."""
