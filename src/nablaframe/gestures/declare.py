"""What each run declares to the gestures: descriptions, position classes, planted names, predictions and relations.

Spec: docs/specs/phase-6.md, D49, D50 and "What each run declares". The zoo tasks' declarations
live with the tasks, in ``zoo/declarations.py``; this module holds what every run shares, and the
reference run's declaration, which is nothing but that.

* ``DESCRIPTIONS = ("token(p)", "token(p-1)", "token(p-2)", "token(p-3)", "token(p+1)")`` and
  ``MIN_COUNT = 5``.
* ``token_descriptions(windows, seq)``: each of ``DESCRIPTIONS`` as an int64 array (B, seq), for
  ``windows`` an integer array (B, seq + 1) of non-negative tokens, the model reading the first
  ``seq``. At position ``p``, ``token(p-k)`` is ``windows[:, p - k]`` where ``p >= k`` and -1, for
  nothing, where ``p < k``; ``token(p+1)`` is ``windows[:, p + 1]``. ValueError for windows of
  another shape or with a negative token, or a ``seq`` that isn't a positive int.
* ``values_at(values, positions, *, min_count=MIN_COUNT)``: the values that occur at least
  ``min_count`` times in ``values[:, positions]``, ignoring -1, as a sorted int64 array.
  ``counts_at(values, positions, *, min_count=MIN_COUNT)``: whether there are at least two such
  values. A description counts at a class only where ``counts_at`` holds: a constant names
  nothing.
* ``Place(description, block, cls)``: a frozen dataclass, a description read after a block, at a
  position class. ``Planted(place, judged)``: a planted name, judged by a gate or recorded.
  ``Relation(name, up, down, judged)``: two places a run relates; their positions and values are
  paired by the declaration's ``pairs`` and ``value_map``.
* ``Declaration``: a frozen dataclass with

  - ``run``: a non-empty string; ``seq``: a positive int
  - ``classes``: a mapping from class names to positions, each stored as a sorted tuple of
    distinct ints in ``range(seq)``, none empty; stored read-only, so assigning into it raises
    TypeError
  - ``variables``: the names of the task's planted variables, beyond ``DESCRIPTIONS``
  - ``planted``: a tuple of ``Planted``; ``relations``: a tuple of ``Relation``, names distinct
  - ``describe_variables(windows)``: the variables as int64 arrays (B, seq), -1 for nothing
  - ``predict(description, window, position, w)``: ``(read position, token)``, the run's
    prediction when ``description`` is moved to ``w`` at ``position`` of ``window``, or None
    where the run declares none
  - ``pairs(relation, window)``: the relation's ``(upstream position, downstream position)``
    pairs in that window, a tuple
  - ``value_map(relation, window, v)``: the downstream value for upstream value ``v``, or None

  Construction raises ValueError for a bad field, a place whose description isn't one of
  ``DESCRIPTIONS`` or ``variables`` or whose class isn't declared, a negative block, or a
  variable named like one of ``DESCRIPTIONS``. ``describe(windows)`` returns
  ``token_descriptions`` and ``describe_variables`` together, ValueError if the variables come
  back named or shaped other than declared. ``relation(name)`` returns the relation so named,
  ValueError for none.
* ``reference(seq=256)``: the reference run's declaration. ``run`` is ``"reference"``; one class,
  ``"every"``, every position; no variables, planted names or relations. ``predict`` gives
  ``(p, w)`` for ``token(p+1)`` at any position ``p`` and None for every other description.
  ``pairs`` and ``value_map`` raise ValueError, naming the relation, since there are none.
* Every declaration's ``predict`` raises ValueError for a description it doesn't have, a window
  that isn't a one-dimensional integer array of ``seq + 1`` tokens, a position outside
  ``range(seq)``, or a ``w`` that isn't a non-negative int; its ``pairs`` and ``value_map`` do for
  a relation it doesn't have, or such a window. Two helpers make those checks for every
  declaration: ``check_window(window, seq)`` returns the window as int64, and
  ``check_prediction_args(description, known, window, position, w, seq)`` checks a ``predict``
  call's arguments against the descriptions ``known`` and returns the window as int64.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

DESCRIPTIONS = ("token(p)", "token(p-1)", "token(p-2)", "token(p-3)", "token(p+1)")
MIN_COUNT = 5


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _positive_int(x: Any, what: str) -> int:
    if not _is_int(x) or x <= 0:
        raise ValueError(f"{what} must be a positive int, got {x!r}")
    return int(x)


def token_descriptions(windows: Any, seq: int) -> dict[str, np.ndarray]:
    seq = _positive_int(seq, "seq")
    arr = np.asarray(windows)
    if arr.ndim != 2 or arr.shape[1] != seq + 1:
        raise ValueError(f"windows must be (B, {seq + 1}), got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"windows must be an integer array, got dtype {arr.dtype}")
    if arr.size and arr.min() < 0:
        raise ValueError("windows must hold non-negative tokens")
    arr = arr.astype(np.int64)
    out: dict[str, np.ndarray] = {"token(p)": arr[:, :seq].copy()}
    for k in (1, 2, 3):
        back = np.full((arr.shape[0], seq), -1, dtype=np.int64)
        back[:, k:] = arr[:, : seq - k]
        out[f"token(p-{k})"] = back
    out["token(p+1)"] = arr[:, 1:].copy()
    return {name: out[name] for name in DESCRIPTIONS}


def values_at(values: Any, positions: Any, *, min_count: int = MIN_COUNT) -> np.ndarray:
    held = np.asarray(values)[:, np.asarray(positions, dtype=np.int64)].reshape(-1).astype(np.int64)
    held = held[held != -1]
    found, counts = np.unique(held, return_counts=True)
    return found[counts >= min_count].astype(np.int64)


def counts_at(values: Any, positions: Any, *, min_count: int = MIN_COUNT) -> bool:
    return len(values_at(values, positions, min_count=min_count)) >= 2


@dataclass(frozen=True)
class Place:
    description: str
    block: int
    cls: str


@dataclass(frozen=True)
class Planted:
    place: Place
    judged: bool


@dataclass(frozen=True)
class Relation:
    name: str
    up: Place
    down: Place
    judged: bool


def _tuple_of(items: Any, kind: type, what: str) -> tuple:
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
        raise ValueError(f"{what} must be a tuple of {kind.__name__}, got {items!r}")
    items = tuple(items)
    for item in items:
        if not isinstance(item, kind):
            raise ValueError(f"{what} must hold only {kind.__name__}, got {item!r}")
    return items


@dataclass(frozen=True, eq=False)
class Declaration:
    run: str
    seq: int
    classes: Mapping[str, Sequence[int]]
    variables: tuple[str, ...]
    planted: tuple[Planted, ...]
    relations: tuple[Relation, ...]
    describe_variables: Callable[[np.ndarray], Mapping[str, np.ndarray]]
    predict: Callable[[str, np.ndarray, int, int], tuple[int, int] | None]
    pairs: Callable[[str, np.ndarray], tuple[tuple[int, int], ...]]
    value_map: Callable[[str, np.ndarray, int], int | None]

    def __post_init__(self) -> None:
        if not isinstance(self.run, str) or not self.run:
            raise ValueError(f"run must be a non-empty string, got {self.run!r}")
        seq = _positive_int(self.seq, "seq")
        if not isinstance(self.classes, Mapping) or not self.classes:
            raise ValueError("classes must be a non-empty mapping")
        classes: dict[str, tuple[int, ...]] = {}
        for name, positions in self.classes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"class names must be non-empty strings, got {name!r}")
            if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
                raise ValueError(f"class {name!r} must hold a sequence of positions, got {positions!r}")
            if not all(_is_int(p) for p in positions):
                raise ValueError(f"class {name!r} must hold ints, got {positions!r}")
            ordered = tuple(sorted(int(p) for p in positions))
            if not ordered:
                raise ValueError(f"class {name!r} is empty")
            if len(set(ordered)) != len(ordered):
                raise ValueError(f"class {name!r} repeats a position")
            if ordered[0] < 0 or ordered[-1] >= seq:
                raise ValueError(f"class {name!r} has a position outside range({seq})")
            classes[name] = ordered
        variables = _tuple_of(self.variables, str, "variables")
        if any(not v for v in variables) or len(set(variables)) != len(variables):
            raise ValueError(f"variables must be distinct non-empty names, got {variables!r}")
        if any(v in DESCRIPTIONS for v in variables):
            raise ValueError(f"a variable is named like one of DESCRIPTIONS: {variables!r}")
        planted = _tuple_of(self.planted, Planted, "planted")
        relations = _tuple_of(self.relations, Relation, "relations")
        known = DESCRIPTIONS + variables

        def check_place(place: Any) -> None:
            if not isinstance(place, Place):
                raise ValueError(f"expected a Place, got {place!r}")
            if place.description not in known:
                raise ValueError(f"place describes {place.description!r}, which this run doesn't have")
            if not _is_int(place.block) or place.block < 0:
                raise ValueError(f"place block must be a non-negative int, got {place.block!r}")
            if place.cls not in classes:
                raise ValueError(f"place class {place.cls!r} isn't declared")

        for item in planted:
            check_place(item.place)
            if not isinstance(item.judged, bool):
                raise ValueError(f"judged must be a bool, got {item.judged!r}")
        for rel in relations:
            if not isinstance(rel.name, str) or not rel.name:
                raise ValueError(f"relation names must be non-empty strings, got {rel.name!r}")
            check_place(rel.up)
            check_place(rel.down)
            if not isinstance(rel.judged, bool):
                raise ValueError(f"judged must be a bool, got {rel.judged!r}")
        if len({rel.name for rel in relations}) != len(relations):
            raise ValueError("relation names must be distinct")
        for name in ("describe_variables", "predict", "pairs", "value_map"):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be callable")
        object.__setattr__(self, "seq", seq)
        object.__setattr__(self, "classes", MappingProxyType(classes))
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "planted", planted)
        object.__setattr__(self, "relations", relations)

    def describe(self, windows: Any) -> dict[str, np.ndarray]:
        out = token_descriptions(windows, self.seq)
        batch = out["token(p)"].shape[0]
        got = self.describe_variables(np.asarray(windows).astype(np.int64))
        if not isinstance(got, Mapping) or set(got) != set(self.variables):
            raise ValueError(f"describe_variables returned {sorted(got) if isinstance(got, Mapping) else got!r}, "
                             f"declared {list(self.variables)}")
        for name in self.variables:
            arr = np.asarray(got[name])
            if arr.shape != (batch, self.seq) or not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"variable {name!r} must be an int array of shape {(batch, self.seq)}, "
                                 f"got {arr.dtype} {arr.shape}")
            out[name] = arr.astype(np.int64)
        return out

    def relation(self, name: str) -> Relation:
        for rel in self.relations:
            if rel.name == name:
                return rel
        raise ValueError(f"run {self.run!r} has no relation {name!r}")


def check_window(window: Any, seq: int) -> np.ndarray:
    arr = np.asarray(window)
    if arr.ndim != 1 or arr.shape[0] != seq + 1:
        raise ValueError(f"window must be a one-dimensional array of {seq + 1} tokens, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"window must be an integer array, got dtype {arr.dtype}")
    return arr.astype(np.int64)


def check_prediction_args(description: Any, known: Sequence[str], window: Any, position: Any, w: Any,
                          seq: int) -> np.ndarray:
    if not isinstance(description, str) or description not in known:
        raise ValueError(f"unknown description {description!r}; this run has {list(known)}")
    arr = check_window(window, seq)
    if not _is_int(position) or not 0 <= position < seq:
        raise ValueError(f"position must be an int in range({seq}), got {position!r}")
    if not _is_int(w) or w < 0:
        raise ValueError(f"w must be a non-negative int, got {w!r}")
    return arr


def reference(seq: int = 256) -> Declaration:
    def describe_variables(windows: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def predict(description: str, window: np.ndarray, position: int, w: int) -> tuple[int, int] | None:
        check_prediction_args(description, DESCRIPTIONS, window, position, w, seq)
        return (int(position), int(w)) if description == "token(p+1)" else None

    def pairs(relation: str, window: np.ndarray) -> tuple[tuple[int, int], ...]:
        raise ValueError(f"the reference run has no relation {relation!r}")

    def value_map(relation: str, window: np.ndarray, v: int) -> int | None:
        raise ValueError(f"the reference run has no relation {relation!r}")

    return Declaration(run="reference", seq=seq, classes={"every": range(seq)}, variables=(), planted=(),
                       relations=(), describe_variables=describe_variables, predict=predict, pairs=pairs,
                       value_map=value_map)
