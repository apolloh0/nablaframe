"""Trace schema v0: the contract shared by both lanes.

Spec: docs/specs/phase-0.md, section "Trace schema v0".

Three records, all frozen dataclasses with a JSON form, and one rule: any tensor
in a trace can be recomputed from its manifest and a checkpoint.

Encoding rules, fixed here so both lanes agree byte for byte:

* Canonical JSON is ``json.dumps(obj, sort_keys=True, separators=(",", ":"),
  ensure_ascii=True, allow_nan=False)`` encoded as UTF-8. NaN and infinity are
  never serialized.
* Hashes are SHA-256, reported as 64 lowercase hex characters.
* A tensor's hash covers its bytes in C order and little-endian byte order,
  nothing else. Its dtype and shape are pinned by the tap's TapSpec.

Records are immutable after construction. Mapping-valued fields are not copied
defensively; never mutate them after building a record. Records holding mappings
are not hashable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

SITES: tuple[str, ...] = ("embedding", "residual", "attention", "mlp", "final_norm")
EVENT_KINDS: tuple[str, ...] = ("forward", "grad", "update", "checkpoint", "intervention")
MODES: tuple[str, ...] = ("fast", "instrumented")


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` with the canonical encoding described in the module docstring.

    Raises ValueError for NaN or infinity anywhere in ``obj``.
    """
    raise NotImplementedError


def sha256_hex(data: bytes) -> str:
    """SHA-256 of ``data`` as 64 lowercase hex characters."""
    raise NotImplementedError


def tensor_sha256(array: np.ndarray) -> str:
    """SHA-256 of the array's bytes in C order and little-endian byte order.

    Non-contiguous arrays and big-endian dtypes are converted first, so two arrays
    with equal values, dtype and shape always hash equal.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class Seeds:
    """Every source of randomness in a run."""

    param_init: int
    data_order: int
    dropout: int


@dataclass(frozen=True)
class RunManifest:
    """One per run. Everything needed to reproduce the run's trace.

    ``run_id`` is the SHA-256 of the canonical JSON of ``to_dict()`` with
    ``created_at`` removed. Two runs with one ``run_id`` must produce identical
    traces; gate G0.2 is the first test of that promise.

    ``model_config`` and ``optimizer`` must be JSON-compatible mappings.
    ``optimizer`` covers the optimizer and its schedule. ``created_at`` is
    ISO-8601 UTC with a trailing ``Z``.
    """

    model_config: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    seeds: Seeds
    corpus_id: str
    corpus_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    git_commit: str
    git_dirty: bool
    env_lock_sha256: str
    backend: str
    xla_flags: str
    checkpoint_schedule: tuple[int, ...]
    created_at: str

    @property
    def run_id(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field, including ``created_at``.

        ``seeds`` becomes a nested dict and ``checkpoint_schedule`` a list.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        raise NotImplementedError

    def to_json(self) -> str:
        """Canonical JSON of ``to_dict()``."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        raise NotImplementedError


@dataclass(frozen=True)
class Axis:
    """One named tensor axis with its size."""

    name: str
    size: int


@dataclass(frozen=True)
class TapSpec:
    """One per tap point.

    ``name`` is a TransformerLens hook name (see ``registry.py``). ``axes`` are
    ordered: the tuple order is the tensor's axis order and must survive a JSON
    round trip. ``dtype`` is a numpy dtype name such as ``"float32"``. ``site``
    is one of ``SITES``.

    Construction raises ValueError when ``site`` is unknown, ``dtype`` is not a
    numpy dtype name, an axis size is not a positive int, or axis names repeat.
    """

    name: str
    axes: tuple[Axis, ...]
    dtype: str
    site: str

    def __post_init__(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """``axes`` becomes a list of ``{"name": ..., "size": ...}`` objects, in order."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TapSpec:
        raise NotImplementedError

    def to_json(self) -> str:
        """Canonical JSON of ``to_dict()``. ``from_json(s).to_json() == s`` must hold."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, text: str) -> TapSpec:
        raise NotImplementedError


@dataclass(frozen=True)
class TraceEvent:
    """One per recorded tensor or trace fact.

    Callers set the first six fields. The store fills the rest:

    * ``sha256``: hash of the tensor, when the event carries one (``tensor_sha256``).
    * ``supersedes``: the ``event_id`` this event replaces; set only by
      ``TraceStore.supersede``.
    * ``event_id``: ``identity_hash()`` of the completed event.
    * ``tensor_ref``: ``"shards/<event_id>.safetensors"`` when the event carries a
      tensor, else None.

    ``kind`` is one of ``EVENT_KINDS``; ``mode`` is one of ``MODES``.
    """

    run_id: str
    step: int
    kind: str
    tap: str
    batch_hash: str
    mode: str
    sha256: str | None = None
    supersedes: str | None = None
    event_id: str | None = None
    tensor_ref: str | None = None

    def identity_hash(self) -> str:
        """SHA-256 of the canonical JSON of every field except ``event_id`` and ``tensor_ref``.

        Two events with the same content, including ``sha256`` and ``supersedes``,
        share an identity. That is what lets the store reject rewrites.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceEvent:
        raise NotImplementedError
