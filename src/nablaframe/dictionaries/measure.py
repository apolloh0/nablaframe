"""Measuring a dictionary on a model: its meters, its selectivity and its patches (D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo". JAX is imported inside the
functions that need it, so this module imports without it. Nothing here caches or judges: the
gates choose windows, sites and variables, and hold the results to the bars.

* ``activations(model, params, tokens, taps, *, chunk=256)``: each tap's values on the
  unintervened pass, as float32 numpy arrays shaped like the tap, (batch, pos, ...).
  ``model.instrumented`` runs on ``chunk`` sequences at a time, in order, the last chunk
  possibly shorter. ValueError for a ``chunk`` that isn't a positive int.
* ``fvu(dictionary, site, x, y)``: the fraction of variance unexplained, ``Σ(y − ŷ)² / Σ(y − ȳ)²``
  with ``ŷ = reconstruct(dictionary, site, x)`` and ``ȳ`` the mean row of ``y``, summed in
  float64 over every row and feature, as a plain float. ``x`` and ``y`` are (N, d) with one N.
  ValueError for rows that don't line up, or a ``y`` with no variance.
* ``dead_share(dictionary, site, x)``: the share of the site's ``active_width`` latents that
  fire on no row of ``x``, as a plain float.
* ``readout_loss(logits, windows, positions)``: ``(loss, accuracy)``, plain floats, over every
  window and every position ``p`` in ``positions``, the target being ``windows[:, p + 1]``. The
  loss is the mean negative log probability of the target in nats, from a log-softmax in
  float64; the accuracy is how often the argmax is the target, ties to the lower id. ``logits``
  is (batch, pos, vocab) and ``windows`` (batch, at least max(positions) + 2). ValueError
  otherwise, or for an empty ``positions``.
* ``spliced_logits(model, params, tokens, dictionary, site)``: the logits, float32 (batch, pos,
  vocab), of the pass in which the site's output tap holds, at every position,
  ``reconstruct`` of the site's input tap's unintervened value. The replacement enters the
  compiled program as an argument (PR #38).
* ``ablated_logits(model, params, tokens, tap, *, n_layers)``: the logits, float32, of
  ``run_intervened`` with one ``mean`` edit at ``tap``, every position, whose reference batch is
  ``tokens`` itself: a per-position mean ablation over the windows being measured.
* ``f1_table(codes, values, n_values, *, min_count=5)``: ``(f1, value_ids)``. ``codes`` is (N,
  m) and ``values`` (N,) ints in ``[0, n_values)``, one per row. ``value_ids`` holds, ascending,
  the values that occur at least ``min_count`` times. ``f1`` is float64 (m, len(value_ids)):
  for latent ``i`` and value ``v``, ``2·|i active ∧ value = v| / (|i active| + |value = v|)``,
  a latent being active where its code is above zero. ValueError for mismatched or
  out-of-range inputs, or if no value occurs ``min_count`` times.
* ``mean_best_f1(f1)``: the mean over values (columns) of the best latent's F1, as a plain float.
* ``assigned(tables, name, *, floor=0.5)``: the latents assigned to the variable ``name``, as a
  sorted list of ints. ``tables`` maps each variable at one position to its ``f1`` from
  ``f1_table``, every table with the same number of latents. A latent is assigned to the
  variable holding its best F1 over every variable and value, the first variable in ``tables``'s
  order on a tie, if that F1 is at least ``floor``. ValueError for a ``name`` not in ``tables``.
* ``probe_accuracy(features, values, n_values, *, ridge=1e-2)``: how well a linear probe reads
  the variable, as a plain float. In float64: rows at even indices fit, rows at odd indices
  test. Features are centred on the fitting rows' mean. With ``G`` the fitting rows' Gram
  matrix, the weights solve ``(G + ridge · max(trace G, 1e-12) / n_features · I) W = Xᵀ Y``,
  ``Y`` the one-hot values. The accuracy is how often the argmax of ``X W`` on a test row is its
  value, ties to the lower id. ValueError for fewer than two rows or mismatched inputs.
* ``patch_predictions(model, params, tokens, sources, *, tap, positions, read, n_layers,
  dictionary=None, latents=None)``: each pass's argmax at position ``read``, int64 (batch,),
  ties to the lower id, keyed by pass, every pass through ``run_intervened``:

  - ``"residual"``: one ``patch`` edit at ``tap`` and ``positions`` from ``sources``
  - ``"code"``, with a ``dictionary``: one ``dpatch`` edit there instead
  - ``"latents"``, with a ``dictionary`` and ``latents``, a mapping from each position in
    ``positions`` to its latent ids: one ``dpatch`` edit per position, in ``positions``'s order,
    each at that position alone with that position's ``latents``

  ``resolve`` holds only ``tokens``, ``sources`` and the dictionary. ValueError for ``latents``
  without a dictionary, or one whose keys aren't exactly ``positions``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import Dictionary


def activations(model: Any, params: Any, tokens: Any, taps: Sequence[str], *, chunk: int = 256) -> dict[str, np.ndarray]:
    raise NotImplementedError


def fvu(dictionary: Dictionary, site: int, x: Any, y: Any) -> float:
    raise NotImplementedError


def dead_share(dictionary: Dictionary, site: int, x: Any) -> float:
    raise NotImplementedError


def readout_loss(logits: Any, windows: Any, positions: Sequence[int]) -> tuple[float, float]:
    raise NotImplementedError


def spliced_logits(model: Any, params: Any, tokens: Any, dictionary: Dictionary, site: int) -> np.ndarray:
    raise NotImplementedError


def ablated_logits(model: Any, params: Any, tokens: Any, tap: str, *, n_layers: int) -> np.ndarray:
    raise NotImplementedError


def f1_table(codes: Any, values: Any, n_values: int, *, min_count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def mean_best_f1(f1: Any) -> float:
    raise NotImplementedError


def assigned(tables: Mapping[str, Any], name: str, *, floor: float = 0.5) -> list[int]:
    raise NotImplementedError


def probe_accuracy(features: Any, values: Any, n_values: int, *, ridge: float = 1e-2) -> float:
    raise NotImplementedError


def patch_predictions(model: Any, params: Any, tokens: Any, sources: Any, *, tap: str, positions: Sequence[int],
                      read: int, n_layers: int, dictionary: Dictionary | None = None,
                      latents: Mapping[int, Sequence[int]] | None = None) -> dict[str, np.ndarray]:
    raise NotImplementedError
