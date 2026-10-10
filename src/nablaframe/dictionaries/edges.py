"""The edge-sparsity meter between two dictionaries, and the Jacobian across an MLP (D42, D43).

Spec: docs/specs/phase-5.md, D42, D43 and "The edge-sparsity meter". JAX is imported inside the
functions that need it, so this module imports without it. Nothing here caches or judges.

A side is a ``(dictionary, site)`` pair. Two kinds of pair are measured:

* **Across an MLP:** the upstream side reads and reconstructs ``blocks.<l>.ln2.hook_normalized``
  and the downstream side ``blocks.<l>.hook_mlp_out`` of the same block, with ``weights`` that
  block's MLP from ``train.mlp_weights``. Two TopK SAEs, or a Jacobian SAE's sites 0 and 1.
* **Across a block:** the upstream side reads and reconstructs ``blocks.<l>.hook_resid_post`` and
  the downstream side ``blocks.<l + 1>.hook_resid_post``. Two TopK SAEs, or a Staircase SAE's
  sites ``l`` and ``l + 1``.

Any other pair, or sides whose widths don't meet, raises ValueError. In both, ``u`` is the
upstream side's ``encode`` of its tap's values on the unintervened pass over ``tokens``, every
position, and "the downstream code" is the downstream side's encoding, computed as ``encode``
computes it, of what its tap would hold. Every matrix product in a derivative or an ablation runs
at float32 with the highest precision.

* ``EDGE_POINTS = 14``, ``IG_STEPS = 5``, ``SETTLE = 0.1``.
* ``edge_counts(n_edges, points=EDGE_POINTS)``: the ablation curve's edge counts: the distinct
  values of ``round(v)`` for ``v`` in ``numpy.logspace(0, log10(n_edges), points)``, with
  ``n_edges`` itself, ascending. ValueError unless both are positive ints.
* ``kl(clean_logits, logits)``: the mean, over every window and position, of the KL divergence
  from the clean distribution to the other, ``Σ p (log p − log q)``, from log-softmaxes in
  float64, as a plain float. ValueError unless the shapes match.
* ``mlp_jacobian(up, down, weights, codes)``: per row of upstream ``codes`` (T, m_up):
  ``(jac, idx_in, idx_out)``. ``idx_in`` (T, k_up) holds the row's ``k`` largest codes' latent ids,
  ties to the lower id. The downstream pre-activations are computed, as ``encode`` computes them,
  on ``mlp(weights, decode(up, codes))``, and ``idx_out`` (T, k_down) holds the ``k`` that
  ``encode`` would keep. ``jac`` (T, k_up, k_down), float32, is the derivative of the downstream
  pre-activation at ``idx_out[t, j]`` with respect to the upstream code at ``idx_in[t, i]``, at
  ``jac[t, i, j]``, and zero where that pre-activation is not positive. For a Jacobian SAE's sites
  0 and 1 it equals ``train.pair_jacobian``.
* ``jacobian_stats(jac)``: ``{"entries_for_90", "share_above_0.01", "mean_abs"}`` over the rows
  of ``jac`` (T, a, b), as plain floats: the median over rows of the fewest entries whose
  absolute values sum to at least 90% of the row's total (0 for a row of zeros); the mean over
  rows of the share of entries whose absolute value is above 0.01; the mean absolute entry.
* ``mlp_edge_scores(model, params, tokens, up, down, weights, *, ig_steps=IG_STEPS)``: float64
  (m_up, m_down). Each edge ``(i, j)`` is scored by integrated gradients from the origin, per
  position of every window: with ``α`` at the midpoints ``(s + 0.5) / ig_steps``, its attribution
  is ``u[i]`` times the mean over ``α`` of the derivative of downstream code ``j`` with respect to
  upstream code ``i`` at ``α · u``, the derivative ``mlp_jacobian`` gives (zero for a downstream
  latent ``encode`` wouldn't keep there). The score is the root mean square of the attribution
  over every position of every window.
* ``block_edge_scores(model, params, tokens, up, down, *, ig_steps=IG_STEPS)``: float64 (m_up,
  m_down). Per window, with the upstream tap holding ``decode(up, α · u)`` at every position, let
  ``f_j`` be downstream code ``j`` summed over positions. Edge ``(i, j)``'s attribution is the sum
  over positions ``p`` of ``u[p, i]`` times the mean over ``α`` of ``∂f_j / ∂u[p, i]`` at ``α · u``.
  The score is the root mean square of the attribution over windows.
* ``mlp_curve(model, params, tokens, up, down, weights, scores, *, points=EDGE_POINTS)`` and
  ``block_curve(model, params, tokens, up, down, scores, *, points=EDGE_POINTS)``:
  ``{"curve", "pair_kl", "scored_edges"}``. ``scored_edges`` counts the scores above zero, and
  ``curve`` is a list of ``[n, kl]`` for each ``n`` in ``edge_counts(scored_edges, points)``:

  - keep the ``n`` highest-scoring edges, ties to the lower upstream id, then the lower
    downstream id
  - every downstream latent with a kept edge takes, at every position, its downstream code when
    the upstream code is ``u`` with every latent not on one of its kept edges zeroed: across an
    MLP, through ``mlp`` at each position; across a block, through the model with the upstream
    tap holding that code's ``decode`` at every position. Every other downstream latent is zero
  - the downstream side's ``decode`` of those codes replaces its tap at every position, and
    ``kl`` is taken from the clean logits

  ``pair_kl`` is ``kl`` with the pair spliced in whole: the downstream tap holds the downstream
  side's ``reconstruct`` of what it holds when the upstream tap holds ``reconstruct`` of ``u``'s
  input (across an MLP, ``mlp`` of the upstream reconstruction). Every replacement enters the
  compiled program as an argument (PR #38). ValueError for ``scores`` not shaped (m_up, m_down),
  or with no score above zero.
* ``meter(curve, pair_kl)``: the edges the computation needs: the fewest ``n`` in ``curve`` from
  which every later point's KL, its own included, is at most ``floor + SETTLE · (kl₁ − floor)``,
  where ``kl₁`` is the first point's KL and ``floor`` is the lower of ``pair_kl`` and the last
  point's KL, the term ``kl₁ − floor`` taken as 0 when negative. None if no point qualifies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EDGE_POINTS, IG_STEPS, SETTLE = 14, 5, 0.1


def edge_counts(n_edges: int, points: int = EDGE_POINTS) -> list[int]:
    raise NotImplementedError


def kl(clean_logits: Any, logits: Any) -> float:
    raise NotImplementedError


def meter(curve: Sequence[Sequence[float]], pair_kl: float) -> int | None:
    raise NotImplementedError


def mlp_jacobian(up: Any, down: Any, weights: Mapping[str, Any], codes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError


def jacobian_stats(jac: Any) -> dict[str, float]:
    raise NotImplementedError


def mlp_edge_scores(model: Any, params: Any, tokens: Any, up: Any, down: Any, weights: Mapping[str, Any], *,
                    ig_steps: int = IG_STEPS) -> np.ndarray:
    raise NotImplementedError


def block_edge_scores(model: Any, params: Any, tokens: Any, up: Any, down: Any, *, ig_steps: int = IG_STEPS) -> np.ndarray:
    raise NotImplementedError


def mlp_curve(model: Any, params: Any, tokens: Any, up: Any, down: Any, weights: Mapping[str, Any], scores: Any, *,
              points: int = EDGE_POINTS) -> dict[str, Any]:
    raise NotImplementedError


def block_curve(model: Any, params: Any, tokens: Any, up: Any, down: Any, scores: Any, *,
                points: int = EDGE_POINTS) -> dict[str, Any]:
    raise NotImplementedError
