"""The view builder: every derived view, for every traced step, written for the viewer.

Spec: docs/specs/phase-2.md, "The scrubber" and D11. The payloads below are the contract
the viewer reads. All floats are plain Python floats; NaN is written as null.

* ``VIEWS = ("per_token_loss", "logit_lens", "attribution", "layer_structure")``,
  ``TOP_K = 5``, and ``BADGES``: for ``logit_lens`` and ``attribution``, the sentence
  stating which discrimination levels the view has passed (D14).
* ``view_payloads(reader, step, params, probe, n_layers)``: one payload per view for that
  step, keyed by view name. Every payload has ``"step"`` and ``"meter"``:

  - ``per_token_loss``: ``"loss"`` is ``per_token_loss(reader, step, probe)`` as nested
    lists (Batch, Pos), null at the last position. Meter:
    ``{"kind": "hash_check", "tensor_sha256": <the logits event's sha256>}``.
  - ``logit_lens``: ``"layers"`` names the residual stream each reading decodes, in
    ``lens_readings`` order. ``"top_tokens"`` (layer, Batch, Pos, TOP_K) are each
    reading's highest-probability tokens, best first, ties broken by the lower token id;
    ``"top_probs"`` are their softmax probabilities, computed in float64. Meter:
    ``lens_meter(readings, traced logits)``. ``"badge"``: ``BADGES["logit_lens"]``.
  - ``attribution``: ``"parts"`` lists the contribution names in
    ``attribution_from_trace``'s order; ``"contributions"`` (part, Batch, Pos), null at the
    last position. Meter: ``{"kind": "decomposition_error", "max_abs": <its error>}``.
    ``"badge"``: ``BADGES["attribution"]``.
  - ``layer_structure``: ``"residual_norm_mean"`` (n_layers + 1) is the mean of
    ``residual_norms`` over Batch and Pos. ``"attention_entropy"`` (layer, head) is the
    mean over Batch and query positions of ``-sum p ln p`` over keys, in float64, with
    ``0 ln 0 = 0``. ``"lens_kl"`` is the logit lens meter's ``kl_mean`` per reading.
    Meter: ``{"kind": "composite", "residual_norm_mean": "hash_check",
    "attention_entropy": "hash_check", "lens_kl": "logit_lens"}``.

* ``build_views(reader, out_dir, *, params_for, probe, n_layers, run_id, vocab=None,
  val_loss_curve=None, progress=None)``: for each step in ``reader.steps()``, writes each
  payload with ``export_view(out_dir, f"{view}-step-{step}", payload)``, then calls
  ``progress(done, total)`` once per step. ``params_for(step)`` returns that step's
  parameters. Last, it writes ``index.json`` as canonical JSON through a temporary file
  and a rename: ``{"run_id", "steps", "views", "files": {view: {str(step): file name}},
  "probe", "n_layers", "vocab", "val_loss_curve"}``. The index isn't a view, so it has no
  meter. Returns every path written, the index last. Two builds from the same inputs are
  byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

VIEWS = ("per_token_loss", "logit_lens", "attribution", "layer_structure")
TOP_K = 5
BADGES = {
    "logit_lens": ("Discrimination: recovers planted structure at decoder level, and is empty on random "
                   "weights (G2.3). Planted structure at model level waits for Phase 4's zoo."),
    "attribution": ("Discrimination: recovers planted structure at decoder level, and is an exact "
                    "decomposition whose error is its meter (G2.4). Planted structure at model level waits "
                    "for Phase 4's zoo."),
}


def view_payloads(reader: Any, step: int, params: Any, probe: np.ndarray, n_layers: int) -> dict[str, dict[str, Any]]:
    raise NotImplementedError


def build_views(reader: Any, out_dir: str | Path, *, params_for: Callable[[int], Any], probe: np.ndarray,
                n_layers: int, run_id: str, vocab: Sequence[str] | None = None,
                val_loss_curve: Sequence[dict[str, Any]] | None = None,
                progress: Callable[[int, int], None] | None = None) -> list[Path]:
    raise NotImplementedError
