"""GPT-2, matching Levanter's architecture, with every canonical tap built in.

Spec: docs/specs/phase-1.md, "The reference model".

Architecture, pinned by G0.3's recorded parameter counts: tied input and output
embeddings, learned position embeddings, biases on every linear layer, pre-norm
blocks with two layer norms each, and a final layer norm. The GELU variant,
layer-norm epsilon and attention scaling follow Levanter's GPT-2 at the pinned
version; G1.2 verifies them against Levanter itself. float32 throughout. The
implementation uses Haliax named axes internally but returns plain arrays.

``params`` is a pytree of float32 arrays; ``init_params`` is deterministic in its
seed. ``GPT2Model`` satisfies the oracle's ``InstrumentedModel`` protocol:

* ``fast(params, tokens)`` returns logits ``(Batch, Pos, Vocab)`` from a
  ``jit``-compiled function.
* ``instrumented(params, tokens, taps, interventions=None)`` returns the logits and
  a dict with exactly the requested taps, shaped as ``registry.tap_spec`` says, from
  a separately compiled function that outputs them directly, never through host
  callbacks. A name outside ``registry.tap_names(n_layers)`` raises KeyError.
* ``interventions`` maps tap names to pure JAX functions, each applied at its tap
  point. The recorded tap is the value after the intervention, which is what flows
  downstream. It exists for oracle discrimination tests now and for Phase 3's
  interventions later.

``load_hf_gpt2(path, config)`` loads a Hugging Face GPT-2 export into our params, for G1.2.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Intervention = Callable[[Any], Any]


@dataclass(frozen=True)
class GPT2Config:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    seq: int

    @property
    def d_head(self) -> int:
        raise NotImplementedError

    def n_params(self) -> int:
        """Exact parameter count for this architecture, including tied embeddings once."""
        raise NotImplementedError

    @classmethod
    def from_size(cls, size: str, vocab: int) -> GPT2Config:
        """Size "S": 4 layers, d_model 256, 4 heads, d_mlp 1024, seq 256.
        Size "M": 4 layers, d_model 512, 8 heads, d_mlp 2048, seq 256."""
        raise NotImplementedError


def init_params(config: GPT2Config, seed: int) -> Any:
    raise NotImplementedError


class GPT2Model:
    config: GPT2Config

    def __init__(self, config: GPT2Config) -> None:
        raise NotImplementedError

    def fast(self, params: Any, tokens: Any) -> Any:
        raise NotImplementedError

    def instrumented(self, params: Any, tokens: Any, taps: Sequence[str],
                     interventions: Mapping[str, Intervention] | None = None) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError


def load_hf_gpt2(path: str | Path, config: GPT2Config) -> Any:
    raise NotImplementedError
