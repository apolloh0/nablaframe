"""Effects of an intervention against the unintervened pass, their meter, and their cache (D24, D25).

Spec: docs/specs/phase-3.md, "Effects and their meters".

* ``IDENTITIES = ("null edit", "steering at zero", "self-swap", "full patch", "full swap")``.
* ``effects(base_logits, intervened_logits, tokens)``: float64 arrays (Batch, Pos):
  ``delta_logit``, the change in the actual next token's logit; ``delta_loss``, the change
  in its loss, ``-log softmax``; both NaN at the last position, which has no next token. And
  ``kl``, KL(unintervened || intervened) per position, at every position.
* ``effect_payload(spec, base_logits, intervened_logits, tokens)``: JSON-ready, null for
  NaN: ``{"spec_hash", "spec": spec_json(spec), "delta_logit", "delta_loss", "kl",
  "kl_mean", "meter": {"kind": "exactness", "identities": list(IDENTITIES), "checked_by":
  "G3.1"}}``.
* ``intervention_key(*, checkpoint_sha256, batch_hash, spec_hash, code)``:
  ``cache_key({"checkpoint": ..., "batch": ..., "spec": ...}, code)``.
* ``save_result(root, key, array)`` and ``load_result(root, key)``: as ``save_jacobian`` and
  ``load_jacobian`` in ``views/jlens.py``, for any float32 array: a key never changes meaning.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.interventions.spec import spec_hash, spec_json
from nablaframe.views.cache import _check_key, _write_atomic, cache_key

IDENTITIES = ("null edit", "steering at zero", "self-swap", "full patch", "full swap")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log softmax over the vocabulary axis, in float64."""
    x = np.asarray(logits, dtype=np.float64)
    top = x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x - top).sum(axis=-1, keepdims=True)) - top


def effects(base_logits: np.ndarray, intervened_logits: np.ndarray, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """Per-position effects of an intervention against the unintervened pass, in float64."""
    base, moved, tokens = np.asarray(base_logits), np.asarray(intervened_logits), np.asarray(tokens)
    if base.ndim != 3 or base.shape != moved.shape or tokens.shape != base.shape[:2]:
        raise ValueError(f"logits must be two (Batch, Pos, Vocab) arrays of one shape and tokens (Batch, Pos), "
                         f"got {base.shape}, {moved.shape} and {tokens.shape}")
    targets = tokens[:, 1:, None].astype(np.intp)
    delta_logit = np.full(tokens.shape, np.nan, dtype=np.float64)
    delta_logit[:, :-1] = (np.take_along_axis(moved[:, :-1].astype(np.float64), targets, axis=-1)
                           - np.take_along_axis(base[:, :-1].astype(np.float64), targets, axis=-1))[..., 0]
    log_base, log_moved = _log_softmax(base), _log_softmax(moved)
    delta_loss = np.full(tokens.shape, np.nan, dtype=np.float64)
    delta_loss[:, :-1] = (np.take_along_axis(log_base[:, :-1], targets, axis=-1)
                          - np.take_along_axis(log_moved[:, :-1], targets, axis=-1))[..., 0]
    kl = (np.exp(log_base) * (log_base - log_moved)).sum(axis=-1)
    return {"delta_logit": delta_logit, "delta_loss": delta_loss, "kl": kl}


def _nulled(array: np.ndarray) -> list:
    """Nested lists of plain floats, with NaN written as None (null in JSON)."""
    values = np.asarray(array, dtype=np.float64)
    return np.where(np.isnan(values), None, values).tolist()


def effect_payload(spec: Any, base_logits: np.ndarray, intervened_logits: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
    """The JSON-ready effect of ``spec``, with its exactness meter."""
    out = effects(base_logits, intervened_logits, tokens)
    return {
        "spec_hash": spec_hash(spec),
        "spec": spec_json(spec),
        "delta_logit": _nulled(out["delta_logit"]),
        "delta_loss": _nulled(out["delta_loss"]),
        "kl": _nulled(out["kl"]),
        "kl_mean": float(out["kl"].mean()),
        "meter": {"kind": "exactness", "identities": list(IDENTITIES), "checked_by": "G3.1"},
    }


def intervention_key(*, checkpoint_sha256: str, batch_hash: str, spec_hash: str, code: str) -> str:
    return cache_key({"checkpoint": checkpoint_sha256, "batch": batch_hash, "spec": spec_hash}, code)


def _path(root: str | Path, key: str) -> Path:
    _check_key(key)
    return Path(root) / f"{key}.npy"


def save_result(root: str | Path, key: str, array: np.ndarray) -> Path:
    """Store ``array`` under ``key`` and return its path; a key never changes meaning."""
    array = np.asarray(array)
    if array.dtype != np.float32:
        raise ValueError(f"a result must be a float32 array, got {array.dtype}")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    data = buffer.getvalue()
    path = _path(root, key)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} holds a different result; a cache key never changes meaning")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, data)
    return path


def load_result(root: str | Path, key: str) -> np.ndarray | None:
    path = _path(root, key)
    if not path.exists():
        return None
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise ValueError(f"{path} holds {array.dtype}, not float32")
    return array
