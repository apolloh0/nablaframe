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

from pathlib import Path
from typing import Any

import numpy as np

IDENTITIES = ("null edit", "steering at zero", "self-swap", "full patch", "full swap")


def effects(base_logits: np.ndarray, intervened_logits: np.ndarray, tokens: np.ndarray) -> dict[str, np.ndarray]:
    raise NotImplementedError


def effect_payload(spec: Any, base_logits: np.ndarray, intervened_logits: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
    raise NotImplementedError


def intervention_key(*, checkpoint_sha256: str, batch_hash: str, spec_hash: str, code: str) -> str:
    raise NotImplementedError


def save_result(root: str | Path, key: str, array: np.ndarray) -> Path:
    raise NotImplementedError


def load_result(root: str | Path, key: str) -> np.ndarray | None:
    raise NotImplementedError
