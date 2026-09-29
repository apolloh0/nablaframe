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

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

SITES: tuple[str, ...] = ("embedding", "residual", "attention", "mlp", "final_norm")
EVENT_KINDS: tuple[str, ...] = ("forward", "grad", "update", "checkpoint", "intervention")
MODES: tuple[str, ...] = ("fast", "instrumented")


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` with the canonical encoding described in the module docstring.

    Raises ValueError for NaN or infinity anywhere in ``obj``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_hex(data: bytes) -> str:
    """SHA-256 of ``data`` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    """SHA-256 of the array's bytes in C order and little-endian byte order.

    Non-contiguous arrays and big-endian dtypes are converted first, so two arrays
    with equal values, dtype and shape always hash equal.
    """
    array = np.asarray(array)
    little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return sha256_hex(little.tobytes(order="C"))


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
        identity = self.to_dict()
        del identity["created_at"]
        return sha256_hex(canonical_json(identity).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field, including ``created_at``.

        ``seeds`` becomes a nested dict and ``checkpoint_schedule`` a list.
        """
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["model_config"] = dict(self.model_config)
        out["optimizer"] = dict(self.optimizer)
        out["seeds"] = {f.name: getattr(self.seeds, f.name) for f in fields(Seeds)}
        out["checkpoint_schedule"] = list(self.checkpoint_schedule)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        values = {f.name: data[f.name] for f in fields(cls)}
        values["model_config"] = dict(values["model_config"])
        values["optimizer"] = dict(values["optimizer"])
        values["seeds"] = Seeds(**{f.name: values["seeds"][f.name] for f in fields(Seeds)})
        values["checkpoint_schedule"] = tuple(values["checkpoint_schedule"])
        return cls(**values)

    def to_json(self) -> str:
        """Canonical JSON of ``to_dict()``."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> RunManifest:
        return cls.from_dict(json.loads(text))


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

    `dtype` must be the canonical numpy name, that is `np.dtype(dtype).name == dtype`, so `"float32"` is accepted and `"f4"` is not.

    `axes` must be a tuple of `Axis`, each with a non-empty str name. Anything else raises ValueError.
    """

    name: str
    axes: tuple[Axis, ...]
    dtype: str
    site: str

    def __post_init__(self) -> None:
        if self.site not in SITES:
            raise ValueError(f"unknown site {self.site!r}; expected one of {SITES}")
        try:
            dtype_name = np.dtype(self.dtype).name
        except TypeError:
            dtype_name = None
        if not isinstance(self.dtype, str) or dtype_name != self.dtype:
            raise ValueError(f"{self.dtype!r} is not a numpy dtype name")
        if not isinstance(self.axes, tuple):
            raise ValueError(f"axes must be a tuple of Axis, got {type(self.axes).__name__}")
        for axis in self.axes:
            if not isinstance(axis, Axis):
                raise ValueError(f"axes must be a tuple of Axis, got an element {axis!r}")
            if not isinstance(axis.name, str) or not axis.name:
                raise ValueError(f"axis name must be a non-empty str, got {axis.name!r}")
            if not isinstance(axis.size, int) or isinstance(axis.size, bool) or axis.size < 1:
                raise ValueError(f"axis {axis.name!r} has size {axis.size!r}; expected a positive int")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"axis names repeat: {names}")

    def to_dict(self) -> dict[str, Any]:
        """``axes`` becomes a list of ``{"name": ..., "size": ...}`` objects, in order."""
        return {
            "name": self.name,
            "axes": [{"name": axis.name, "size": axis.size} for axis in self.axes],
            "dtype": self.dtype,
            "site": self.site,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TapSpec:
        return cls(
            name=data["name"],
            axes=tuple(Axis(name=a["name"], size=a["size"]) for a in data["axes"]),
            dtype=data["dtype"],
            site=data["site"],
        )

    def to_json(self) -> str:
        """Canonical JSON of ``to_dict()``. ``from_json(s).to_json() == s`` must hold."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> TapSpec:
        return cls.from_dict(json.loads(text))


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

    Construction raises ValueError when `kind` is not in `EVENT_KINDS` or `mode` is not in `MODES`. `tap` is not checked: checkpoint markers have none, and valid names depend on layer count.
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

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {EVENT_KINDS}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")

    def identity_hash(self) -> str:
        """SHA-256 of the canonical JSON of every field except ``event_id`` and ``tensor_ref``.

        Two events with the same content, including ``sha256`` and ``supersedes``,
        share an identity. That is what lets the store reject rewrites.
        """
        identity = self.to_dict()
        del identity["event_id"]
        del identity["tensor_ref"]
        return sha256_hex(canonical_json(identity).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceEvent:
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
