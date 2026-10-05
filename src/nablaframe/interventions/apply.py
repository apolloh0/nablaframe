"""Applying an intervention spec to a forward pass (D23, D25, D26).

Spec: docs/specs/phase-3.md. JAX is imported inside the functions that need it, so this
module imports without it.

* ``editable_taps(n_layers)``: ``tap_names(n_layers)`` without the ``attn.hook_pattern``
  taps, whose axes are queries and keys rather than positions (D23).
* ``swap_component(params, other, component)``: ``params`` with one component replaced by
  ``other``'s: ``embeddings`` (``wte`` and ``wpe``), ``ln_final`` (``ln_f``), ``blocks.<l>``,
  ``blocks.<l>.attn`` or ``blocks.<l>.mlp``. ValueError for any other name or layer.
* ``run_intervened(model, params, tokens, spec, *, n_layers, resolve, params_at=None,
  taps=())``: ``(logits, recorded)`` from ``model.instrumented`` with the spec applied.

  - Swaps apply first, in order, each taking its component from ``params_at(step)``.
  - Every other edit applies at its tap, through the model's intervention hook, in order,
    to the positions it selects along the Pos axis; a tap outside ``editable_taps`` raises
    ValueError before anything runs.
  - ``zero`` writes zeros. ``add`` adds ``alpha`` times the vector. ``patch`` writes the
    source batch's value at that tap and those positions. ``mean`` writes the reference
    batch's mean at that tap, per position, taken in float64 and cast back. ``resample``
    writes, for each sequence, the value of a reference window drawn with
    ``numpy.random.default_rng(seed).integers(0, len(reference), size=batch)``.
  - Values for ``patch``, ``mean`` and ``resample`` come from ``model.instrumented`` on the
    source or reference tokens with the post-swap parameters, requesting only that tap.
  - ``resolve(hash)`` returns the array a hash names; ``run_intervened`` recomputes the hash
    (``batch_hash`` for token batches, ``tensor_sha256`` for vectors) and raises ValueError
    if it differs.
  - ``recorded`` holds the requested taps' values after the edits.
* ``code_identity()``: this module's git blob hash (D18, D24).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .spec import InterventionSpec


def editable_taps(n_layers: int) -> tuple[str, ...]:
    raise NotImplementedError


def swap_component(params: Any, other: Any, component: str) -> Any:
    raise NotImplementedError


def run_intervened(model: Any, params: Any, tokens: np.ndarray, spec: InterventionSpec, *, n_layers: int,
                   resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None = None,
                   taps: Sequence[str] = ()) -> tuple[Any, dict[str, Any]]:
    raise NotImplementedError


def code_identity() -> str:
    raise NotImplementedError
