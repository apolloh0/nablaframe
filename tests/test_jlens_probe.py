"""Phase 2 work package 7: the same-position Jacobian, exact and estimated (D15, G2.5).

Spec: docs/specs/phase-2.md, "The estimator G2.5 tests". Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from nablaframe.views.jlens_probe import estimate_same_position_jacobian, exact_same_position_jacobian  # noqa: E402

D, T = 6, 16
A = np.random.default_rng(0).standard_normal((D, D)).astype(np.float32)
B = 2.0 * np.random.default_rng(1).standard_normal((D, D)).astype(np.float32)  # a strong cross-position term


def f(x):
    """y[t] = A x[t] + B x[t-1]: the same-position Jacobian is exactly A at every position."""
    shifted = jnp.concatenate([jnp.zeros((1, D), x.dtype), x[:-1]], axis=0)
    return x @ A.T + shifted @ B.T


def test_estimator_matches_exact_jacobian_on_linear_model():
    xs = np.random.default_rng(2).standard_normal((64, T, D)).astype(np.float32)
    exact = exact_same_position_jacobian(f, xs[0])
    assert exact.shape == (D, D) and np.allclose(exact, A, atol=1e-5)
    estimate = estimate_same_position_jacobian(f, xs, n_draws=8, seed=0)
    assert estimate.shape == (D, D)
    assert np.linalg.norm(estimate - A) / np.linalg.norm(A) <= 0.05, "the cross-position term must average away"
    again = estimate_same_position_jacobian(f, xs, n_draws=8, seed=0)
    assert np.array_equal(estimate, again), "deterministic in its seed"
