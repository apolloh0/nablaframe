"""The J-lens view: D17's summed Jacobian read through the final layer norm and unembedding.

Spec: docs/specs/phase-2.md, "The J-lens"; rulings D17 and D20. The J's come from G2.9's cache.

* ``JLENS_BADGE``: the sentence stating which discrimination tests the view is held to and
  which gate checks them (D14).
* ``jlens_lens(resid, J, ln_weight, ln_bias, wte)``: ``logit_lens(resid @ J.T, ...)``, in
  float32: J carries the residual stream to a predicted final residual stream, which the
  logit lens then reads. With J the identity, it equals ``logit_lens(resid, ...)`` exactly.
* ``jlens_readings(reader, step, params, jacobians, n_layers)``: (n_layers, Batch, Pos,
  Vocab). Reading l is ``jlens_lens`` of ``blocks.<l>.hook_resid_post`` with
  ``jacobians[l]``, for l below the last block; the last reading is the logit lens of the
  last block's output, whose J is the identity. A missing J raises KeyError.
* ``lens_divergence(a, b)``: one dict per reading, in order: ``{"layer": i, "js_mean": float,
  "top1_agreement": float}``. ``js_mean`` is the mean over positions of the Jensen-Shannon
  divergence between the two readings' softmax distributions, in nats, computed in float64;
  ``top1_agreement`` is the fraction of positions whose top-1 tokens agree. Plain floats.
* ``target_ranks(readings, probe)``: ``(ranks, probs)``, each nested lists (reading, Batch,
  Pos). At position t the target is ``probe[:, t + 1]``; its rank is its 0-based place by
  descending softmax probability, ties going to the lower token id (as the payloads' top
  tokens are ordered), and its probability is computed in float64. The last position has
  no target: both are null there.
  ``target_ranks`` and ``jlens_payload`` take the same keyword ``next_token=None``. When given, the last position has a rank and a probability like any other, under the same tie rule; without it, both stay null there.
* ``jlens_payload(reader, step, params, jacobians, probe, n_layers)``: raises ValueError
  unless ``batch_hash(probe)`` equals the step's events' batch hash. Otherwise
  ``{"step", "layers", "top_tokens", "top_probs", "target_rank", "target_prob", "meter",
  "comparison", "badge"}``: ``layers`` names each reading's residual stream; the top tokens
  and probabilities follow the logit lens payload's rules; ``meter`` is
  ``lens_meter(readings, traced logits)``; ``comparison`` is ``lens_divergence`` of the
  readings against the logit lens readings of the same block outputs; ``badge`` is
  ``JLENS_BADGE``. JSON-ready: plain Python values, null for NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..data.corpus import batch_hash as probe_batch_hash
from .lens import lens_meter, lens_params, logit_lens
from .readouts import checked_next_token

JLENS_BADGE = ("Discrimination tests: recovers planted structure at decoder level (a J that carries planted "
               "directions to tokens), and is empty on random weights, checked by G2.10. Its definition matches "
               "anthropics/jacobian-lens to 1.1e-7 (G2.8). Its model-level result is in the badge below (D38).")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float64, computed as the logit lens payload's top tokens compute it."""
    p = logits.astype(np.float64)
    p -= p.max(axis=-1, keepdims=True)
    p = np.exp(p)
    p /= p.sum(axis=-1, keepdims=True)
    return p


def _json_ready(value: Any) -> Any:
    """Plain Python values, with float NaN written as None (null in JSON)."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def jlens_lens(resid: np.ndarray, J: np.ndarray, ln_weight: np.ndarray, ln_bias: np.ndarray,
               wte: np.ndarray) -> np.ndarray:
    predicted = np.asarray(resid, dtype=np.float32) @ np.asarray(J, dtype=np.float32).T
    return logit_lens(predicted, ln_weight, ln_bias, wte)


def jlens_readings(reader: Any, step: int, params: Any, jacobians: Mapping[int, np.ndarray],
                   n_layers: int) -> np.ndarray:
    ln_w, ln_b, wte = lens_params(params)
    readings = []
    for layer in range(n_layers):
        resid = reader.tensor(step, f"blocks.{layer}.hook_resid_post")
        if layer == n_layers - 1:
            readings.append(logit_lens(resid, ln_w, ln_b, wte))  # the last block's J is the identity
        else:
            readings.append(jlens_lens(resid, jacobians[layer], ln_w, ln_b, wte))
    return np.stack(readings)


def lens_divergence(a: np.ndarray, b: np.ndarray) -> list[dict[str, float]]:
    if np.shape(a) != np.shape(b):
        raise ValueError(f"readings have different shapes: {np.shape(a)} and {np.shape(b)}")
    rows = []
    for layer, (x, y) in enumerate(zip(a, b)):
        p, q = _softmax(x), _softmax(y)
        m = (p + q) / 2
        # 0 ln 0 = 0: a token with no probability under one reading adds nothing to its term.
        kl_pm = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / m), 0.0).sum(axis=-1)
        kl_qm = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0) / m), 0.0).sum(axis=-1)
        js = 0.5 * kl_pm + 0.5 * kl_qm
        rows.append({
            "layer": layer,
            "js_mean": float(js.mean()),
            "top1_agreement": float((x.argmax(axis=-1) == y.argmax(axis=-1)).mean()),
        })
    return rows


def target_ranks(readings: np.ndarray, probe: np.ndarray, *, next_token: Any = None) -> tuple[list, list]:
    readings = np.asarray(readings)
    probe = np.asarray(probe)
    if probe.shape != readings.shape[1:3]:
        raise ValueError(f"probe shape {probe.shape} does not match the readings' (Batch, Pos) {readings.shape[1:3]}")
    next_token = checked_next_token(next_token, probe)
    if next_token is None:
        targets = probe[:, 1:, None].astype(np.intp)  # (Batch, Pos - 1, 1): positions with a next token
        tail: list = [None]
    else:
        targets = np.concatenate([probe[:, 1:], next_token[:, None]], axis=1)[..., None].astype(np.intp)
        tail = []
    n_targets = targets.shape[1]
    ids = np.arange(readings.shape[-1])
    ranks_out, probs_out = [], []
    for reading in readings:  # one reading at a time keeps the float64 copy small
        p = _softmax(reading)[:, :n_targets]
        target_p = np.take_along_axis(p, targets, axis=-1)
        # 0-based rank by descending probability; an equally probable token ranks above the target only if its
        # id is lower, as the payloads' top tokens are ordered.
        ranks = (p > target_p).sum(axis=-1) + ((p == target_p) & (ids < targets)).sum(axis=-1)
        ranks_out.append([row + tail for row in ranks.tolist()])
        probs_out.append([row + tail for row in target_p[..., 0].tolist()])
    return ranks_out, probs_out


def jlens_payload(reader: Any, step: int, params: Any, jacobians: Mapping[int, np.ndarray], probe: np.ndarray,
                  n_layers: int, *, next_token: Any = None) -> dict[str, Any]:
    from .build import _top_tokens  # imported here: build imports this module

    probe = np.asarray(probe)
    if probe_batch_hash(probe) != reader.batch_hash(step):
        raise ValueError(f"probe does not match the batch step {step} was recorded on")
    readings = jlens_readings(reader, step, params, jacobians, n_layers)
    ln_w, ln_b, wte = lens_params(params)
    logit = np.stack([logit_lens(reader.tensor(step, f"blocks.{layer}.hook_resid_post"), ln_w, ln_b, wte)
                      for layer in range(n_layers)])
    top_tokens, top_probs = _top_tokens(readings)
    target_rank, target_prob = target_ranks(readings, probe, next_token=next_token)
    return _json_ready({
        "step": step,
        "layers": [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)],
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "target_rank": target_rank,
        "target_prob": target_prob,
        "meter": lens_meter(readings, reader.tensor(step, "logits")),
        "comparison": lens_divergence(readings, logit),
        "badge": JLENS_BADGE,
    })
