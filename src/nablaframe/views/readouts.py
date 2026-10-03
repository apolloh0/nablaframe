"""Readout views: exact decodes of traced tensors.

Spec: docs/specs/phase-2.md, "Views". A readout's meter is the hash check: every
tensor comes from ``TraceStore.load_tensor``, which verifies its SHA-256.

* ``TraceReader(root)`` opens the store read-only. ``steps()`` is the sorted
  tuple of steps with current events. ``tensor(step, tap)`` returns the current
  event's tensor for that step and tap (KeyError if there is none).
  ``batch_hash(step)`` is the probe hash of that step's events; ValueError if they
  disagree.
* ``attention_pattern(reader, step, layer, head, sequence)``: the traced
  ``blocks.<layer>.attn.hook_pattern`` for that sequence and head, (QPos, KPos),
  exactly as traced.
* ``residual_norms(reader, step, n_layers)``: (n_layers + 1, Batch, Pos) float32.
  Row l is the L2 norm over Embed of ``blocks.<l>.hook_resid_pre`` for l below
  n_layers, and the last row that of ``blocks.<n_layers - 1>.hook_resid_post``.
  Computed in float64, returned as float32.
* ``per_token_loss(reader, step, probe)``: (Batch, Pos) float32. Position t holds
  ``-log softmax(logits[b, t])[probe[b, t + 1]]``, computed in float64 from the
  traced ``logits``. The last position has no next token and holds NaN. Raises
  ValueError unless ``batch_hash(probe)`` equals the events' batch hash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TraceReader:
    def __init__(self, root: str | Path) -> None:
        raise NotImplementedError

    def steps(self) -> tuple[int, ...]:
        raise NotImplementedError

    def tensor(self, step: int, tap: str) -> np.ndarray:
        raise NotImplementedError

    def batch_hash(self, step: int) -> str:
        raise NotImplementedError


def attention_pattern(reader: TraceReader, step: int, layer: int, head: int, sequence: int) -> np.ndarray:
    raise NotImplementedError


def residual_norms(reader: TraceReader, step: int, n_layers: int) -> np.ndarray:
    raise NotImplementedError


def per_token_loss(reader: TraceReader, step: int, probe: np.ndarray) -> np.ndarray:
    raise NotImplementedError
