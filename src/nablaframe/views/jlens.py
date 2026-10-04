"""The summed Jacobian of D17, exactly as anthropics/jacobian-lens computes it (jlens/fitting.py at 581d398).

Spec: docs/specs/phase-2.md, "The J-lens"; ruling: D17. Caches follow D11, with D18's code identity.

* ``SKIP_FIRST = 16``. ``valid_positions(seq_len, skip_first=SKIP_FIRST)``: positions
  ``skip_first`` to ``seq_len - 2`` as an int array, as the reference's
  ``valid_position_mask``. Raises ValueError if none remain.
* ``block_map(model, params, layer, n_layers)``: a pure JAX function from one sequence's
  ``blocks.<layer>.hook_resid_post`` (T, d) to the last block's ``hook_resid_post`` (T, d),
  through the model's intervention hook, which replaces that tap's value. The later blocks
  see only the residual stream, so one function serves every sequence. For the last
  layer it is the identity.
* ``summed_jacobian(f, xs, skip_first=SKIP_FIRST)``: for each sequence in ``xs`` (N, T, d)
  and each output dimension i, one reverse-mode pass with a one-hot cotangent on dimension
  i at every valid position at once. The gradient at valid source position p is then the
  sum over valid p' >= p of d f(x)[p'] / d x[p], dimension i; row i of that sequence's J is
  its mean over the valid source positions. J is the mean over sequences, each weighted
  equally, as the reference's ``fit`` averages prompts. Gradients in float32, averages in
  float64, returned as float32 (d, d). Deterministic.
* ``code_identity()``: the git blob hash of this module's source file, exactly as
  ``git hash-object`` prints it (D18), so an unrelated commit doesn't invalidate a cache.
* ``jacobian_key(*, checkpoint_sha256, fit_sha256, layer, code)``:
  ``cache_key({"checkpoint": ..., "fit_sample": ..., "layer": str(layer)}, code)``.
* ``save_jacobian(root, key, J)`` writes ``<root>/<key>.npy`` with ``numpy.save``,
  through a temporary file and a rename, and returns the path. Saving identical content
  again is a no-op; different content under an existing key raises ValueError.
  ``load_jacobian(root, key)`` returns the float32 array, or None.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SKIP_FIRST = 16


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST) -> np.ndarray:
    raise NotImplementedError


def block_map(model: Any, params: Any, layer: int, n_layers: int) -> Callable[[Any], Any]:
    raise NotImplementedError


def summed_jacobian(f: Callable[[Any], Any], xs: np.ndarray, skip_first: int = SKIP_FIRST) -> np.ndarray:
    raise NotImplementedError


def code_identity() -> str:
    raise NotImplementedError


def jacobian_key(*, checkpoint_sha256: str, fit_sha256: str, layer: int, code: str) -> str:
    raise NotImplementedError


def save_jacobian(root: str | Path, key: str, J: np.ndarray) -> Path:
    raise NotImplementedError


def load_jacobian(root: str | Path, key: str) -> np.ndarray | None:
    raise NotImplementedError
