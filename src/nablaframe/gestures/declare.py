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
from typing import Any

import numpy as np

DESCRIPTIONS = ("token(p)", "token(p-1)", "token(p-2)", "token(p-3)", "token(p+1)")
MIN_COUNT = 5


def token_descriptions(windows: Any, seq: int) -> dict[str, np.ndarray]:
    raise NotImplementedError


def values_at(values: Any, positions: Any, *, min_count: int = MIN_COUNT) -> np.ndarray:
    raise NotImplementedError


def counts_at(values: Any, positions: Any, *, min_count: int = MIN_COUNT) -> bool:
    raise NotImplementedError


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
        raise NotImplementedError

    def describe(self, windows: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def relation(self, name: str) -> Relation:
        raise NotImplementedError


def check_window(window: Any, seq: int) -> np.ndarray:
    raise NotImplementedError


def check_prediction_args(description: Any, known: Sequence[str], window: Any, position: Any, w: Any,
                          seq: int) -> np.ndarray:
    raise NotImplementedError


def reference(seq: int = 256) -> Declaration:
    raise NotImplementedError
