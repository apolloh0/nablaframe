"""Intervention specs (D22): declarative, hashable lists of edits.

Spec: docs/specs/phase-3.md, D22.

* ``OPERATIONS = ("zero", "mean", "resample", "patch", "add", "swap")``. Their arguments,
  exactly these keys and no others: ``zero``: none; ``mean``: ``reference``; ``resample``:
  ``reference`` and an int ``seed``; ``patch``: ``source``; ``add``: ``vector`` and a float
  ``alpha``; ``swap``: an int ``step``. ``reference``, ``source`` and ``vector`` are content
  hashes, 64 lowercase hex characters, never values.
* ``swap`` edits name a component instead of a tap: ``embeddings``, ``ln_final``,
  ``blocks.<l>``, ``blocks.<l>.attn`` or ``blocks.<l>.mlp``; their positions must be ``"all"``.
* Positions are ``"all"``, a list of non-negative ints, or ``{"range": [a, b]}`` with
  ``0 <= a <= b``.
* ``edit(tap, positions, op, **args)`` validates and returns an ``Edit``; anything invalid
  raises ValueError. ``InterventionSpec(edits)`` holds a tuple of edits, applied in order.
* ``spec_json(spec)``: ``{"edits": [{"tap", "positions", "op", "args"}, ...]}``, JSON-ready.
  ``spec_from_json`` inverts it. ``spec_hash(spec)``: SHA-256 of ``canonical_json(spec_json(spec))``.
* ``position_index(positions, seq_len)``: the int array of positions selected; ValueError for
  any position outside ``[0, seq_len)`` or any other form.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

OPERATIONS = ("zero", "mean", "resample", "patch", "add", "swap")


@dataclass(frozen=True)
class Edit:
    tap: str
    positions: Any
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionSpec:
    edits: tuple[Edit, ...] = ()


def edit(tap: str, positions: Any, op: str, **args: Any) -> Edit:
    raise NotImplementedError


def spec_json(spec: InterventionSpec) -> dict[str, Any]:
    raise NotImplementedError


def spec_from_json(data: Mapping[str, Any]) -> InterventionSpec:
    raise NotImplementedError


def spec_hash(spec: InterventionSpec) -> str:
    raise NotImplementedError


def position_index(positions: Any, seq_len: int) -> np.ndarray:
    raise NotImplementedError
