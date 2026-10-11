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
    The `logit_lens` payload gains `target_rank` and `target_prob`, from
    `jlens_view.target_ranks(readings, probe)`.
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

  `build_views` takes a keyword `jacobians_for=None`. When given, each step also writes
  `jlens_payload(reader, step, params_for(step), jacobians_for(step), probe, n_layers)` as
  `jlens-step-<step>`, and the index's `views` is `VIEWS` followed by `"jlens"`. Without
  it, nothing changes.
  ``view_payloads`` takes ``next_token=None`` and passes it to ``per_token_loss``, ``target_ranks`` and ``attribution_from_trace``. ``build_views`` takes ``next_token=None`` and ``curves=None``. It passes ``next_token`` to every payload, the J-lens's included. When it's given, the index gains ``"next_token"``, a list of ints. When ``curves`` is given, the index gains ``"curves"``: a list of ``{"name", "unit", "points"}``, ``name`` and ``unit`` non-empty strings, each point ``{"step", "value"}`` with an int step and a finite value or null. Anything else raises ValueError before a file is written. Without either keyword, the index has exactly its earlier keys and every file is byte-identical to before, so earlier runs' builds don't change.

  `build_views` takes `features=None`: payloads from `views.features.features_payload`. Each is written with `export_view` as `features-<dictionary>-site-<site>`, after every step's views and before the index, and the index gains `"features"`: one entry per payload, in the given order, `{"dictionary", "kind", "site", "taps", "step", "file"}`, the first five copied from the payload and `file` its file name. Given as an empty list, the index gains `"features": []`. A payload that isn't a dict holding those five keys and a `"meter"`, two payloads with the same dictionary and site, or a payload whose step isn't one of the reader's raises ValueError before a file is written. Without it, the index and every file are byte-identical to a build without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..trace.schema import canonical_json
from .attribution import attribution_from_trace
from .cache import _write_atomic
from .export import export_view
from .jlens_view import jlens_payload, target_ranks
from .lens import lens_meter, lens_readings
from .readouts import checked_next_token, per_token_loss, residual_norms

VIEWS = ("per_token_loss", "logit_lens", "attribution", "layer_structure")
TOP_K = 5
BADGES = {
    "logit_lens": ("Discrimination tests: recovers planted structure at decoder level, and is empty on random "
                   "weights, checked by G2.3. Its model-level result is in the badge below (D38)."),
    "attribution": ("Discrimination tests: recovers planted structure at decoder level, and is an exact "
                    "decomposition whose error is its meter, checked by G2.4. Its model-level result is in the "
                    "badge below (D38)."),
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


def _checked_curves(curves: Any) -> list[dict[str, Any]] | None:
    """``curves`` as plain Python values in the index's form, or None; ValueError if it isn't well formed."""
    if curves is None:
        return None
    if isinstance(curves, str | bytes | Mapping) or not isinstance(curves, Sequence):
        raise ValueError(f"curves must be a list of curves, got {type(curves).__name__}")
    checked = []
    for i, curve in enumerate(curves):
        if not isinstance(curve, Mapping) or set(curve) != {"name", "unit", "points"}:
            raise ValueError(f"curve {i} must be a dict with exactly the keys 'name', 'unit' and 'points'")
        for key in ("name", "unit"):
            if not isinstance(curve[key], str) or not curve[key]:
                raise ValueError(f"curve {i}: {key!r} must be a non-empty string")
        points = curve["points"]
        if isinstance(points, str | bytes | Mapping) or not isinstance(points, Sequence):
            raise ValueError(f"curve {i}: 'points' must be a list")
        clean = []
        for j, point in enumerate(points):
            if not isinstance(point, Mapping) or set(point) != {"step", "value"}:
                raise ValueError(f"curve {i}, point {j} must be a dict with exactly the keys 'step' and 'value'")
            step, value = point["step"], point["value"]
            if isinstance(step, bool) or not isinstance(step, int | np.integer):
                raise ValueError(f"curve {i}, point {j}: 'step' must be an int, got {step!r}")
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
                    raise ValueError(f"curve {i}, point {j}: 'value' must be a finite number or null, got {value!r}")
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"curve {i}, point {j}: 'value' must be finite or null, got {value!r}")
            clean.append({"step": int(step), "value": value})
        checked.append({"name": curve["name"], "unit": curve["unit"], "points": clean})
    return checked


def view_payloads(reader: Any, step: int, params: Any, probe: np.ndarray, n_layers: int, *,
                  next_token: Any = None) -> dict[str, dict[str, Any]]:
    probe = np.asarray(probe)
    loss = {
        "step": step,
        "loss": _nulled(per_token_loss(reader, step, probe, next_token=next_token)),
        "meter": {"kind": "hash_check", "tensor_sha256": reader.event(step, "logits").sha256},
    }

    readings = lens_readings(reader, step, params, n_layers)
    meter = lens_meter(readings, reader.tensor(step, "logits"))
    top_tokens, top_probs = _top_tokens(readings)
    target_rank, target_prob = target_ranks(readings, probe, next_token=next_token)
    lens = {
        "step": step,
        "layers": _lens_layers(n_layers),
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "target_rank": target_rank,
        "target_prob": target_prob,
        "meter": meter,
        "badge": BADGES["logit_lens"],
    }

    contributions, error = attribution_from_trace(reader, step, params, probe, n_layers,
                                                    next_token=next_token)
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
                progress: Callable[[int, int], None] | None = None,
                jacobians_for: Callable[[int], Mapping[int, np.ndarray]] | None = None,
                next_token: Any = None, curves: Sequence[dict[str, Any]] | None = None) -> list[Path]:
    out = Path(out_dir)
    probe = np.asarray(probe)
    next_token = checked_next_token(next_token, probe)
    curves = _checked_curves(curves)
    steps = reader.steps()
    views = VIEWS if jacobians_for is None else VIEWS + ("jlens",)
    files: dict[str, dict[str, str]] = {view: {} for view in views}
    paths: list[Path] = []
    for done, step in enumerate(steps, start=1):
        params = params_for(step)
        payloads = view_payloads(reader, step, params, probe, n_layers, next_token=next_token)
        if jacobians_for is not None:
            payloads["jlens"] = jlens_payload(reader, step, params, jacobians_for(step), probe, n_layers,
                                            next_token=next_token)
        for view in views:
            path = export_view(out, f"{view}-step-{step}", payloads[view])
            files[view][str(step)] = path.name
            paths.append(path)
        if progress is not None:
            progress(done, len(steps))
    index = {
        "run_id": run_id,
        "steps": list(steps),
        "views": list(views),
        "files": files,
        "probe": probe.tolist(),
        "n_layers": n_layers,
        "vocab": None if vocab is None else list(vocab),
        "val_loss_curve": None if val_loss_curve is None else [dict(point) for point in val_loss_curve],
    }
    if next_token is not None:
        index["next_token"] = next_token.tolist()
    if curves is not None:
        index["curves"] = curves
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    _write_atomic(index_path, canonical_json(index).encode("ascii"))
    paths.append(index_path)
    return paths
