"""Direct logit attribution: an exact decomposition of the target token's logit.

Spec: docs/specs/phase-2.md, "Views", and D14.

* ``attribution(parts, final_resid, ln_weight, ln_bias, wte, targets)``: for each
  part (B, T, d), its contribution to the logit of ``targets`` (B, T): the part,
  centered by its own mean over Embed, times the final layer norm's scale at that
  position (computed from ``final_resid``, epsilon ``LN_EPS``), times ``ln_weight``,
  dotted with ``wte[target]``. An extra entry ``"ln_bias"`` holds
  ``ln_bias . wte[target]``. Because the parts sum to ``final_resid``, the
  contributions sum to the lens logit of the target, exactly in exact arithmetic.
* ``parts_from_trace(reader, step, n_layers)``: ``(parts, final_resid)``. The parts
  are ``hook_embed``, ``hook_pos_embed`` and each block's ``hook_attn_out`` and
  ``hook_mlp_out``, keyed by tap name; ``final_resid`` is the last block's
  ``hook_resid_post``.
* ``attribution_from_trace(reader, step, params, probe, n_layers)``:
  ``(contributions, error)``. The target at position t is ``probe[:, t + 1]``; the
  last position has none, and its contributions are NaN. ``error`` is the largest
  absolute gap, over the other positions, between the summed contributions and the
  traced logit of the target: the decomposition error, a plain Python float.
  Raises ValueError unless ``batch_hash(probe)`` equals the events' batch hash.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..data.corpus import batch_hash as probe_batch_hash
from .lens import LN_EPS, lens_params


def _f64(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def attribution(parts: Mapping[str, np.ndarray], final_resid: np.ndarray, ln_weight: np.ndarray,
                ln_bias: np.ndarray, wte: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    if "ln_bias" in parts:
        raise ValueError("'ln_bias' is reserved for the layer norm's bias term and can't name a part")
    final = _f64(final_resid)
    centered_final = final - final.mean(axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt((centered_final * centered_final).mean(axis=-1) + LN_EPS)  # (B, T)
    unembed = _f64(wte)[np.asarray(targets, dtype=np.intp)]  # (B, T, d)
    direction = unembed * _f64(ln_weight)
    contributions = {}
    for name, part in parts.items():
        part = _f64(part)
        centered = part - part.mean(axis=-1, keepdims=True)
        contributions[name] = scale * np.einsum("btd,btd->bt", centered, direction)
    contributions["ln_bias"] = unembed @ _f64(ln_bias)
    return contributions


def parts_from_trace(reader: Any, step: int, n_layers: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    taps = ["hook_embed", "hook_pos_embed"]
    for layer in range(n_layers):
        taps += [f"blocks.{layer}.hook_attn_out", f"blocks.{layer}.hook_mlp_out"]
    parts = {tap: reader.tensor(step, tap) for tap in taps}
    return parts, reader.tensor(step, f"blocks.{n_layers - 1}.hook_resid_post")


def attribution_from_trace(reader: Any, step: int, params: Any, probe: np.ndarray,
                           n_layers: int) -> tuple[dict[str, np.ndarray], float]:
    probe = np.asarray(probe)
    if probe_batch_hash(probe) != reader.batch_hash(step):
        raise ValueError(f"probe does not match the batch step {step} was recorded on")
    ln_w, ln_b, wte = lens_params(params)
    parts, final = parts_from_trace(reader, step, n_layers)
    targets = probe[:, 1:]
    inner = attribution({name: part[:, :-1] for name, part in parts.items()}, final[:, :-1],
                        ln_w, ln_b, wte, targets)
    contributions = {}
    for name, values in inner.items():
        padded = np.full(probe.shape, np.nan, dtype=np.float64)
        padded[:, :-1] = values
        contributions[name] = padded
    logits = reader.tensor(step, "logits")
    target_logits = np.take_along_axis(_f64(logits[:, :-1]), targets[..., None].astype(np.intp), axis=-1)[..., 0]
    error = float(np.abs(sum(inner.values()) - target_logits).max())
    return contributions, error
