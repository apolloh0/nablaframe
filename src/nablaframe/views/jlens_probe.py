"""The same-position Jacobian, exact and estimated: the measurements behind D15 (G2.5).

Spec: docs/specs/phase-2.md, "The J-lens" and "The estimator G2.5 tests". ``f`` maps
one sequence's residual stream (T, d) to the final residual stream (T, d) and is a
pure JAX function.

* ``exact_same_position_jacobian(f, x)``: the mean over positions t of the d-by-d
  Jacobian of ``f(x)[t]`` with respect to ``x[t]``, computed exactly with JAX
  autodiff. For small tests and G2.5's reference only; its cost grows with T * d.
* ``estimate_same_position_jacobian(f, xs, n_draws, seed)``: for each basis
  direction i and each draw, perturb every position of every sequence in ``xs``
  (N, T, d) along e_i with an independent random sign (from
  ``numpy.random.default_rng(seed)``), take one forward-mode product, multiply each
  position's output change by its own sign, and average over sequences, positions
  and draws. Column i of the result estimates the mean same-position Jacobian's
  column i. Deterministic in ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def exact_same_position_jacobian(f: Callable[[Any], Any], x: np.ndarray) -> np.ndarray:
    import jax

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be one sequence of shape (T, d), got {x.shape}")
    full = np.asarray(jax.jacfwd(f)(x), dtype=np.float64)  # (T, d, T, d): d f(x)[t, :] / d x[s, :]
    positions = np.arange(x.shape[0])
    return full[positions, :, positions, :].mean(axis=0)


def estimate_same_position_jacobian(f: Callable[[Any], Any], xs: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"xs must be sequences of shape (N, T, d), got {xs.shape}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    n, t, d = xs.shape

    @jax.jit
    def column(xs, signs, i):
        # One forward-mode product per sequence along signs * e_i at every position;
        # each position's output change times its own sign, averaged over sequences
        # and positions. Other positions' contributions carry independent signs.
        tangent = signs[..., None] * jax.nn.one_hot(i, d, dtype=xs.dtype)
        _, change = jax.vmap(lambda x, v: jax.jvp(f, (x,), (v,)))(xs, tangent)
        return (change * signs[..., None]).mean(axis=(0, 1))

    rng = np.random.default_rng(seed)
    xs_device = jnp.asarray(xs)
    estimate = np.zeros((d, d), dtype=np.float64)
    # Signs are drawn direction by direction, draw by draw, one (N, T) array each.
    for i in range(d):
        for _ in range(n_draws):
            signs = (2 * rng.integers(0, 2, size=(n, t)) - 1).astype(xs.dtype)
            estimate[:, i] += np.asarray(column(xs_device, jnp.asarray(signs), i), dtype=np.float64)
    return estimate / n_draws
