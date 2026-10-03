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
    raise NotImplementedError


def estimate_same_position_jacobian(f: Callable[[Any], Any], xs: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    raise NotImplementedError
