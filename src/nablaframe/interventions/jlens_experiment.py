"""The J-lens experiment's directions and steering scales (Phase 3's first experiment).

Spec: docs/specs/phase-3.md, "The first experiment".

* ``jlens_direction(J, wte, token)``: the unit vector along ``J.T @ wte[token]``, computed in
  float64 and returned as float32: the residual direction J carries most strongly onto the
  token's unembedding. With J the identity it equals ``logit_lens_direction``.
* ``logit_lens_direction(wte, token)``: the unit vector along ``wte[token]``, float32.
* ``SCALE_FACTORS = (0.25, 0.5, 1.0)``. ``steering_scales(resid)``: each factor times the
  mean, over every leading index, of the residual stream's L2 norm over its last axis,
  computed in float64, as plain floats.
"""

from __future__ import annotations

import numpy as np

SCALE_FACTORS = (0.25, 0.5, 1.0)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalise a zero or non-finite direction")
    return (v / norm).astype(np.float32)


def jlens_direction(J: np.ndarray, wte: np.ndarray, token: int) -> np.ndarray:
    J64 = np.asarray(J, dtype=np.float64)
    w = np.asarray(wte, dtype=np.float64)[token]
    return _unit(J64.T @ w)


def logit_lens_direction(wte: np.ndarray, token: int) -> np.ndarray:
    return _unit(np.asarray(wte, dtype=np.float64)[token])


def steering_scales(resid: np.ndarray) -> tuple[float, ...]:
    mean_norm = float(np.linalg.norm(np.asarray(resid, dtype=np.float64), axis=-1).mean())
    return tuple(float(f * mean_norm) for f in SCALE_FACTORS)
