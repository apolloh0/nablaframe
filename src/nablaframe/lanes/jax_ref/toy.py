"""G0.6 toy model: a 2-layer, d=128 GPT used to measure mode equality under jit.

Spec: docs/specs/phase-0.md, G0.6 row and "Mode-equality oracle".

Implement ``ToyGPT`` so it satisfies ``nablaframe.oracle.mode_equality.InstrumentedModel``:

* ``fast(params, tokens)`` returns logits only, from a ``jax.jit``-compiled function.
* ``instrumented(params, tokens, taps)`` returns ``(logits, {tap: tensor})`` from a
  separately ``jit``-compiled function that also outputs the requested taps. Taps
  are recorded as extra outputs, never by host callbacks or side effects.
* Tap names come from ``nablaframe.trace.registry.tap_names(2)``; requesting any
  other name raises KeyError.
* Pre-norm GPT-2 block: LayerNorm, causal attention, residual add, LayerNorm, MLP
  with GELU, residual add; learned positional embeddings; untied unembedding.
* float32 throughout; parameters come only from ``init(seed)``, deterministically.

G0.6 then runs both functions on 3 seeds and records whether the logits are
bitwise equal, and the max abs difference if not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

N_LAYERS = 2
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
VOCAB = 256
SEQ = 64


class ToyGPT:
    def init(self, seed: int) -> Any:
        """Parameters as a pytree, deterministic in ``seed``."""
        raise NotImplementedError

    def fast(self, params: Any, tokens: Any) -> Any:
        raise NotImplementedError

    def instrumented(self, params: Any, tokens: Any, taps: Sequence[str]) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError
