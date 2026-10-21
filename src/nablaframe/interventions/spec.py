"""Intervention specs (D22): declarative, hashable lists of edits.

Spec: docs/specs/phase-3.md, D22.

* ``OPERATIONS = ("zero", "mean", "resample", "patch", "add", "swap")``. Their arguments,
  exactly these keys and no others: ``zero``: none; ``mean``: ``reference``; ``resample``:
  ``reference`` and a non-negative int ``seed``; ``patch``: ``source``; ``add``: ``vector`` and
  a float ``alpha``; ``swap``: a non-negative int ``step``. ``reference``, ``source`` and
  ``vector`` are content hashes, 64 lowercase hex characters, never values.
* ``swap`` edits name a component instead of a tap: ``embeddings``, ``ln_final``,
  ``blocks.<l>``, ``blocks.<l>.attn`` or ``blocks.<l>.mlp``; their positions must be ``"all"``.
* Positions are ``"all"``, a list of distinct non-negative ints, or ``{"range": [a, b]}`` with
  ``0 <= a <= b``. A repeated position is refused: an ``add`` there would apply twice.
* ``edit(tap, positions, op, **args)`` validates and returns an ``Edit``; anything invalid
  raises ValueError. ``InterventionSpec(edits)`` holds a tuple of edits, applied in order.
* ``spec_json(spec)``: ``{"edits": [{"tap", "positions", "op", "args"}, ...]}``, JSON-ready.
  ``spec_from_json`` inverts it. ``spec_hash(spec)``: SHA-256 of ``canonical_json(spec_json(spec))``.
* ``position_index(positions, seq_len)``: the int array of positions selected; ValueError for
  any position outside ``[0, seq_len)`` or any other form.
* ``OPERATIONS`` gains ``jswap``, taking ``J`` (a content hash) and the token ids ``a`` and ``b``
  (non-negative ints) (D33). ``zero`` takes either no arguments or exactly ``head``, a
  non-negative int: zeroing one attention head's output.
* ``OPERATIONS`` gains ``dpatch`` (D45), taking ``dictionary`` and ``source`` (content hashes) and,
  in its other form, ``latents`` as well: a list of distinct non-negative ints, possibly empty,
  stored sorted, so two specs that list the same latents in another order hash equal.
* ``OPERATIONS`` gains ``dsteer`` (D48), taking ``dictionary``, ``remove``, ``add`` and
  ``amounts``, each a content hash: the dictionary's ``content_hash``, and each array's
  ``array_hash``.
* ``array_hash(array)``: the SHA-256 of the canonical JSON of ``{"dtype", "shape", "bytes"}``:
  the array's numpy dtype name (``"int32"``, ``"float32"``), its shape as a list, and its
  ``tensor_sha256``. Arrays equal in dtype, shape and values hash equal, whatever their byte
  order. Unlike the byte hash alone, it tells apart arrays whose bytes agree, such as an empty
  ``remove`` and an empty ``amounts``. ValueError for an array that isn't of integers or floats.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..trace.schema import canonical_json, sha256_hex

OPERATIONS = ("zero", "mean", "resample", "patch", "add", "swap", "jswap", "dpatch")

# Each operation's arguments: name -> kind. Exactly these keys, no others.
_ARGS: dict[str, dict[str, str]] = {
    "zero": {},
    "mean": {"reference": "hash"},
    "resample": {"reference": "hash", "seed": "int"},
    "patch": {"source": "hash"},
    "add": {"vector": "hash", "alpha": "float"},
    "swap": {"step": "int"},
    "jswap": {"J": "hash", "a": "int", "b": "int"},
    "dpatch": {"dictionary": "hash", "source": "hash"},
}
# The other form an operation's arguments may take: ``zero`` of one head, ``dpatch`` of some latents.
_ALTERNATIVE: dict[str, dict[str, str]] = {
    "zero": {"head": "int"},
    "dpatch": {"dictionary": "hash", "source": "hash", "latents": "latents"},
}
_HASH = re.compile(r"[0-9a-f]{64}")
_COMPONENT = re.compile(r"embeddings|ln_final|blocks\.(0|[1-9][0-9]*)(\.attn|\.mlp)?")


@dataclass(frozen=True)
class Edit:
    tap: str
    positions: Any
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionSpec:
    edits: tuple[Edit, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.edits, Edit) or not isinstance(self.edits, Iterable):
            raise ValueError("edits must be a sequence of Edit")
        edits = tuple(self.edits)
        if not all(isinstance(e, Edit) for e in edits):
            raise ValueError("edits must be a sequence of Edit")
        object.__setattr__(self, "edits", edits)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _positions(positions: Any) -> Any:
    """``positions`` in its JSON form, or ValueError."""
    if isinstance(positions, str):
        if positions == "all":
            return "all"
        raise ValueError(f"positions must be 'all', a list or a range, not {positions!r}")
    if isinstance(positions, (list, tuple)):
        if not all(_is_int(p) and p >= 0 for p in positions):
            raise ValueError(f"positions must be non-negative ints: {positions!r}")
        if len(set(int(p) for p in positions)) != len(positions):
            raise ValueError(f"positions must be distinct: {positions!r}")
        return [int(p) for p in positions]
    if isinstance(positions, Mapping) and set(positions) == {"range"}:
        bounds = positions["range"]
        if (isinstance(bounds, (list, tuple)) and len(bounds) == 2 and all(_is_int(b) for b in bounds)
                and 0 <= bounds[0] <= bounds[1]):
            return {"range": [int(bounds[0]), int(bounds[1])]}
        raise ValueError(f"a range is [a, b] with 0 <= a <= b: {bounds!r}")
    raise ValueError(f"positions must be 'all', a list or a range, not {positions!r}")


def _argument(op: str, name: str, kind: str, value: Any) -> Any:
    if kind == "hash":
        if isinstance(value, str) and _HASH.fullmatch(value):
            return str(value)
        raise ValueError(f"{op}: {name} must be a content hash, 64 lowercase hex characters")
    if kind == "int":
        if _is_int(value) and value >= 0:
            return int(value)
        raise ValueError(f"{op}: {name} must be a non-negative int, not {value!r}")
    if kind == "latents":
        if not isinstance(value, (list, tuple)) or not all(_is_int(i) and i >= 0 for i in value):
            raise ValueError(f"{op}: {name} must be a list of non-negative ints, not {value!r}")
        if len(set(int(i) for i in value)) != len(value):
            raise ValueError(f"{op}: {name} must be distinct: {value!r}")
        return sorted(int(i) for i in value)  # sorted, so the same latents in another order hash equal
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        if math.isfinite(value):
            return float(value)
    raise ValueError(f"{op}: {name} must be a finite float, not {value!r}")


def _edit(tap: Any, positions: Any, op: Any, args: Mapping[str, Any]) -> Edit:
    if not isinstance(op, str) or op not in OPERATIONS:
        raise ValueError(f"unknown operation {op!r}; expected one of {OPERATIONS}")
    if not isinstance(tap, str) or not tap:
        raise ValueError(f"tap must be a non-empty string, not {tap!r}")
    if not isinstance(args, Mapping):
        raise ValueError(f"{op}: args must be a mapping")
    expected = _ARGS[op]
    if op in _ALTERNATIVE and set(args) == set(_ALTERNATIVE[op]):
        expected = _ALTERNATIVE[op]
    if set(args) != set(expected):
        forms = " or ".join(str(sorted(form)) for form in (_ARGS[op], _ALTERNATIVE.get(op)) if form is not None)
        raise ValueError(f"{op} takes exactly {forms}, got {sorted(map(str, args))}")
    positions = _positions(positions)
    if op == "swap":
        if not _COMPONENT.fullmatch(tap):
            raise ValueError(f"swap names a component (embeddings, ln_final, blocks.<l>[.attn|.mlp]), not {tap!r}")
        if positions != "all":
            raise ValueError("a swap replaces a whole component, so its positions must be 'all'")
    checked = {name: _argument(op, name, expected[name], args[name]) for name in sorted(expected)}
    return Edit(str(tap), positions, op, checked)


def edit(tap: str, positions: Any, op: str, **args: Any) -> Edit:
    return _edit(tap, positions, op, args)


def spec_json(spec: InterventionSpec) -> dict[str, Any]:
    if not isinstance(spec, InterventionSpec):
        raise ValueError("spec_json takes an InterventionSpec")
    edits = []
    for e in spec.edits:
        checked = _edit(e.tap, e.positions, e.op, e.args)  # an Edit built by hand is checked too
        edits.append({"tap": checked.tap, "positions": checked.positions, "op": checked.op,
                      "args": dict(checked.args)})
    return {"edits": edits}


def spec_from_json(data: Mapping[str, Any]) -> InterventionSpec:
    if not isinstance(data, Mapping) or set(data) != {"edits"} or not isinstance(data["edits"], list):
        raise ValueError('a spec\'s JSON form is {"edits": [...]}')
    edits = []
    for item in data["edits"]:
        if not isinstance(item, Mapping) or set(item) != {"tap", "positions", "op", "args"}:
            raise ValueError("each edit's JSON form has exactly tap, positions, op and args")
        edits.append(_edit(item["tap"], item["positions"], item["op"], item["args"]))
    return InterventionSpec(tuple(edits))


def spec_hash(spec: InterventionSpec) -> str:
    return sha256_hex(canonical_json(spec_json(spec)).encode("ascii"))


def position_index(positions: Any, seq_len: int) -> np.ndarray:
    if not _is_int(seq_len) or seq_len < 0:
        raise ValueError(f"seq_len must be a non-negative int, not {seq_len!r}")
    positions = _positions(positions)
    if positions == "all":
        return np.arange(seq_len, dtype=np.int64)
    if isinstance(positions, dict):
        a, b = positions["range"]
        if b > seq_len:
            raise ValueError(f"range {[a, b]} reaches past sequence length {seq_len}")
        return np.arange(a, b, dtype=np.int64)
    if any(p >= seq_len for p in positions):
        raise ValueError(f"positions {positions} reach past sequence length {seq_len}")
    return np.asarray(positions, dtype=np.int64)
