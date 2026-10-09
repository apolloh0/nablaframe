"""Behavioural ground truth, from logits alone (D30).

* ``TWIN_SEED = 1``. ``twin_params(config)``: ``init_params(config, seed=TWIN_SEED)``, the
  random-weights twin. JAX is imported inside it, so this module imports without JAX.
* ``next_token_accuracy(logits, windows, positions)``: the fraction of (sequence, position)
  pairs where the argmax of ``logits[b, p]`` (ties to the lowest id) equals ``windows[b, p + 1]``.
  ``logits`` is (Batch, Pos, Vocab) and ``windows`` (Batch, Pos + 1); anything else raises
  ValueError.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TWIN_SEED = 1


def twin_params(config: Any) -> Any:
    from ..model.gpt2 import init_params

    return init_params(config, seed=TWIN_SEED)


def next_token_accuracy(logits: np.ndarray, windows: np.ndarray, positions: Any) -> float:
    logits, windows = np.asarray(logits), np.asarray(windows)
    if logits.ndim != 3 or windows.ndim != 2:
        raise ValueError("logits must be (Batch, Pos, Vocab) and windows (Batch, Pos + 1)")
    if windows.shape != (logits.shape[0], logits.shape[1] + 1):
        raise ValueError(f"windows {windows.shape} don't match logits {logits.shape}: want (Batch, Pos + 1)")
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        raise ValueError("no positions")
    if pos.min() < 0 or pos.max() >= logits.shape[1]:
        raise ValueError(f"positions must lie in [0, {logits.shape[1]})")
    predicted = np.argmax(logits[:, pos], axis=-1)  # argmax returns the first maximum: ties to the lowest id
    return float(np.mean(predicted == windows[:, pos + 1]))
