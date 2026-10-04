"""The view builder: every derived view, for every traced step, written for the viewer.

Spec: docs/specs/phase-2.md, "The scrubber" and D11. The payloads below are the contract
the viewer reads. All floats are plain Python floats; NaN is written as null.

* ``VIEWS = ("per_token_loss", "logit_lens", "attribution", "layer_structure")``,
  ``TOP_K = 5``, and ``BADGES``: for ``logit_lens`` and ``attribution``, the sentence
  stating which discrimination tests the view is held to and which gate checks them (D14).
  Whether they passed is in that gate's report; a view that fails its gate doesn't ship.
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

from ..trace.schema import canonical_json
from .attribution import attribution_from_trace
from .cache import _write_atomic
from .export import export_view
from .lens import lens_meter, lens_readings
from .readouts import per_token_loss, residual_norms

VIEWS = ("per_token_loss", "logit_lens", "attribution", "layer_structure")
TOP_K = 5
BADGES = {
    "logit_lens": ("Discrimination tests: recovers planted structure at decoder level, and is empty on random "
                   "weights, checked by G2.3. Planted structure at model level waits for Phase 4's zoo."),
    "attribution": ("Discrimination tests: recovers planted structure at decoder level, and is an exact "
                    "decomposition whose error is its meter, checked by G2.4. Planted structure at model level "
                    "waits for Phase 4's zoo."),
}


def _nulled(array: np.ndarray) -> list:
    """Nested lists of plain floats, with NaN written as None (null in JSON)."""
    values = np.asarray(array, dtype=np.float64)
    return np.where(np.isnan(values), None, values).tolist()


def _lens_layers(n_layers: int) -> list[str]:
    return ["blocks.0.hook_resid_pre"] + [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]


def _top_tokens(readings: np.ndarray) -> tuple[list, list]:
    """Each reading's TOP_K most probable tokens, best first, ties to the lower id, and their probabilities."""
    tokens, probs = [], []
    for reading in readings:
        shifted = reading.astype(np.float64)
        shifted -= shifted.max(axis=-1, keepdims=True)
        p = np.exp(shifted)
        p /= p.sum(axis=-1, keepdims=True)
        # A stable sort of the negated probabilities keeps equal ones in token-id order.
        top = np.argsort(-p, axis=-1, kind="stable")[..., :TOP_K]
        tokens.append(top.tolist())
        probs.append(np.take_along_axis(p, top, axis=-1).tolist())
    return tokens, probs


def _attention_entropy(reader: Any, step: int, n_layers: int) -> list[list[float]]:
    rows = []
    for layer in range(n_layers):
        pattern = reader.tensor(step, f"blocks.{layer}.attn.hook_pattern").astype(np.float64)  # (B, H, Q, K)
        entropy = -(pattern * np.log(np.where(pattern > 0, pattern, 1.0))).sum(axis=-1)
        rows.append(entropy.mean(axis=(0, 2)).tolist())
    return rows


def view_payloads(reader: Any, step: int, params: Any, probe: np.ndarray, n_layers: int) -> dict[str, dict[str, Any]]:
    probe = np.asarray(probe)
    loss = {
        "step": step,
        "loss": _nulled(per_token_loss(reader, step, probe)),
        "meter": {"kind": "hash_check", "tensor_sha256": reader.event(step, "logits").sha256},
    }

    readings = lens_readings(reader, step, params, n_layers)
    meter = lens_meter(readings, reader.tensor(step, "logits"))
    top_tokens, top_probs = _top_tokens(readings)
    lens = {
        "step": step,
        "layers": _lens_layers(n_layers),
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "meter": meter,
        "badge": BADGES["logit_lens"],
    }

    contributions, error = attribution_from_trace(reader, step, params, probe, n_layers)
    parts = list(contributions)
    attr = {
        "step": step,
        "parts": parts,
        "contributions": [_nulled(contributions[part]) for part in parts],
        "meter": {"kind": "decomposition_error", "max_abs": error},
        "badge": BADGES["attribution"],
    }

    norms = residual_norms(reader, step, n_layers).astype(np.float64)
    layers = {
        "step": step,
        "residual_norm_mean": norms.mean(axis=(1, 2)).tolist(),
        "attention_entropy": _attention_entropy(reader, step, n_layers),
        "lens_kl": [reading["kl_mean"] for reading in meter],
        "meter": {"kind": "composite", "residual_norm_mean": "hash_check",
                  "attention_entropy": "hash_check", "lens_kl": "logit_lens"},
    }
    return {"per_token_loss": loss, "logit_lens": lens, "attribution": attr, "layer_structure": layers}


def build_views(reader: Any, out_dir: str | Path, *, params_for: Callable[[int], Any], probe: np.ndarray,
                n_layers: int, run_id: str, vocab: Sequence[str] | None = None,
                val_loss_curve: Sequence[dict[str, Any]] | None = None,
                progress: Callable[[int, int], None] | None = None) -> list[Path]:
    out = Path(out_dir)
    probe = np.asarray(probe)
    steps = reader.steps()
    files: dict[str, dict[str, str]] = {view: {} for view in VIEWS}
    paths: list[Path] = []
    for done, step in enumerate(steps, start=1):
        payloads = view_payloads(reader, step, params_for(step), probe, n_layers)
        for view in VIEWS:
            path = export_view(out, f"{view}-step-{step}", payloads[view])
            files[view][str(step)] = path.name
            paths.append(path)
        if progress is not None:
            progress(done, len(steps))
    index = {
        "run_id": run_id,
        "steps": list(steps),
        "views": list(VIEWS),
        "files": files,
        "probe": probe.tolist(),
        "n_layers": n_layers,
        "vocab": None if vocab is None else list(vocab),
        "val_loss_curve": None if val_loss_curve is None else [dict(point) for point in val_loss_curve],
    }
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    _write_atomic(index_path, canonical_json(index).encode("ascii"))
    paths.append(index_path)
    return paths
