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

from ..data.corpus import batch_hash as probe_batch_hash
from ..trace.store import TraceStore


class TraceReader:
    def __init__(self, root: str | Path) -> None:
        self._store = TraceStore(root, mode="r")
        self._events: dict[tuple[int, str], list] = {}
        self._steps = set()
        for event in self._store.current():
            if event.tensor_ref is not None:
                self._steps.add(event.step)
                self._events.setdefault((event.step, event.tap), []).append(event)

    def steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._steps))

    def tensor(self, step: int, tap: str) -> np.ndarray:
        matches = self._events.get((step, tap), [])
        if not matches:
            raise KeyError((step, tap))
        if len(matches) > 1:
            raise ValueError(f"step {step} has {len(matches)} current events for tap {tap!r}")
        return self._store.load_tensor(matches[0])

    def batch_hash(self, step: int) -> str:
        hashes = {e.batch_hash for (s, _), events in self._events.items() if s == step for e in events}
        if not hashes:
            raise KeyError(step)
        if len(hashes) > 1:
            raise ValueError(f"step {step} has events recorded on different batches: {sorted(hashes)}")
        return hashes.pop()


def attention_pattern(reader: TraceReader, step: int, layer: int, head: int, sequence: int) -> np.ndarray:
    pattern = reader.tensor(step, f"blocks.{layer}.attn.hook_pattern")
    return np.array(pattern[sequence, head])


def residual_norms(reader: TraceReader, step: int, n_layers: int) -> np.ndarray:
    taps = [f"blocks.{layer}.hook_resid_pre" for layer in range(n_layers)]
    taps.append(f"blocks.{n_layers - 1}.hook_resid_post")
    rows = [np.linalg.norm(reader.tensor(step, tap).astype(np.float64), axis=-1) for tap in taps]
    return np.stack(rows).astype(np.float32)


def per_token_loss(reader: TraceReader, step: int, probe: np.ndarray) -> np.ndarray:
    probe = np.asarray(probe)
    if probe_batch_hash(probe) != reader.batch_hash(step):
        raise ValueError(f"probe does not match the batch step {step} was recorded on")
    logits = reader.tensor(step, "logits").astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = np.full(probe.shape, np.nan, dtype=np.float32)
    nll = -np.take_along_axis(log_probs[:, :-1], probe[:, 1:, None].astype(np.intp), axis=-1)[..., 0]
    loss[:, :-1] = nll
    return loss
