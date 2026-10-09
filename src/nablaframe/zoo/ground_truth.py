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
    raise NotImplementedError


def next_token_accuracy(logits: np.ndarray, windows: np.ndarray, positions: Any) -> float:
    raise NotImplementedError
