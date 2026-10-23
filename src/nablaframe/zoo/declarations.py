"""The zoo tasks' declarations to the gestures (D29, D49, D50): classes, planted names, predictions and relations.

Spec: docs/specs/phase-6.md, "What each run declares". Each declaration is a
``gestures.declare.Declaration`` built from its task, so the gestures carry no task-specific code.

* ``two_hop(task)``, for a ``TwoHopTask`` with ``facts`` facts, windows of ``task.window_length``
  tokens and ``seq = task.seq``, one less. With ``q = seq - 1``, the ``⇒`` (25 for six facts):

  - ``run`` ``"two_hop"``. Classes: ``"query"`` (``q``), ``"subjects"`` (0, 4, ...,
    ``4 * (facts - 1)``), ``"objects"`` (2, 6, ..., ``4 * (facts - 1) + 2``), ``"every"``
    (``range(seq)``).
  - Variables: ``"b"``, the window's intermediate, ``task.parse(window).b``, at positions
    ``q - 1`` and ``q``, -1 elsewhere.
  - Planted: ``b`` after block 2 at ``query``, judged; ``token(p+1)`` after block 3 at
    ``query``, judged; ``b`` after block 1 at ``query``, recorded.
  - ``predict``: ``b`` at ``q`` moved to ``w`` gives ``(q, the object of w's fact)`` when ``w``
    is a distractor subject, the subject of a fact other than ``a``'s and ``b``'s, and None for
    any other ``w``. ``token(p+1)`` at ``q`` gives ``(q, w)``. Anything else gives None.
  - Relations: ``"hop"``, ``b`` after block 2 at ``query`` to ``token(p+1)`` after block 3 at
    ``query``, judged; ``"carry"``, ``b`` after block 1 at ``query`` to ``b`` after block 2 at
    ``query``, recorded; ``"two back"``, ``token(p)`` after block 0 at ``objects`` to
    ``token(p-2)`` after block 2 at ``subjects``, recorded.
  - ``pairs``: ``((q, q),)`` for the hop and the carry; for two back, ``(p, p + 2)`` for each
    object position ``p`` with ``p + 2`` a subject position, in order.
  - ``value_map``: the hop maps ``v`` to the object of ``v``'s fact in the window, None when
    ``v`` is no fact's subject; the carry and two back map ``v`` to ``v``.

* ``induction(task)``, for an ``InductionTask``, ``seq`` being ``task.seq``:

  - ``run`` ``"induction"``. Classes: ``"mechanism"`` (``task.mechanism_positions``) and
    ``"every"`` (``range(seq)``). No variables.
  - Planted: ``token(p)`` after block 0 at ``every``, judged; ``token(p+1)`` after block 1 at
    ``mechanism``, judged.
  - ``predict``: ``token(p+1)`` at a mechanism position ``p`` gives ``(p, w)``; ``token(p)`` at
    a sole target ``s`` gives ``(s + P - 1, w)``, ``P`` the window's period. Anything else gives
    None.
  - Relations: ``"copy"``, ``token(p)`` after block 0 at ``every`` to ``token(p+1)`` after block
    1 at ``mechanism``, judged. ``pairs``: ``(s, s + P - 1)`` for each sole target ``s``, in
    order. ``value_map``: ``v`` to ``v``.

* ``modular_addition(task)``, for a ``ModularAdditionTask`` (D54, D57), ``seq`` being ``task.seq``:

  - ``run`` ``"modular_addition"``. Classes: ``"answer"`` (2) and ``"every"`` (0 to 2). No
    variables.
  - Planted: ``token(p+1)`` after block 0 at ``answer``, recorded, never judged: Phase 6's names
    don't hold the sum (C11).
  - ``predict``: ``token(p+1)`` at 2 moved to ``w`` gives ``(2, w)``. Anything else gives None.
  - No relations: ``pairs`` and ``value_map`` raise ValueError, naming the relation, as the
    reference run's do.

* ``period(window)``: the smallest ``i >= 1`` with ``window[i] == window[0]``. ValueError when
  the window doesn't repeat its first token.
* ``sole_targets(task, window)``: the sole targets of an induction window, a sorted int64 array:
  the positions ``s`` with ``1 <= s <= P`` and ``s + P - 1`` a mechanism position. There, the
  token at ``s + P - 1`` has occurred once before, at ``s - 1``, so the induction head there copies
  from ``s`` and nowhere else. ValueError for a window of the wrong length or without a period.

Every declaration here checks its arguments as ``gestures.declare`` says every declaration does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gestures.declare import (DESCRIPTIONS, Declaration, Place, Planted, Relation, check_prediction_args,
                                check_window)


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_value(v: Any) -> int:
    if not _is_int(v):
        raise ValueError(f"a value must be an int, got {v!r}")
    return int(v)


def _check_relation(name: Any, known: tuple[str, ...], run: str) -> None:
    if name not in known:
        raise ValueError(f"run {run!r} has no relation {name!r}")


def period(window: Any) -> int:
    arr = np.asarray(window)
    if arr.ndim != 1 or arr.shape[0] == 0 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"window must be a non-empty one-dimensional integer array, got {arr.dtype} {arr.shape}")
    repeats = np.flatnonzero(arr[1:] == arr[0])
    if not repeats.size:
        raise ValueError("the window doesn't repeat its first token")
    return int(repeats[0]) + 1


def sole_targets(task: Any, window: Any) -> np.ndarray:
    arr = np.asarray(window)
    if arr.ndim != 1 or arr.shape[0] != task.window:
        raise ValueError(f"window must be a one-dimensional array of {task.window} tokens, got shape {arr.shape}")
    P = period(arr)
    s = np.arange(1, P + 1, dtype=np.int64)
    return s[np.isin(s + P - 1, np.asarray(task.mechanism_positions))]


def _parse(task: Any, window: np.ndarray) -> Any:
    try:
        return task.parse(window)
    except (KeyError, IndexError) as err:
        raise ValueError(f"not a two-hop window: {err!r}") from err


def two_hop(task: Any) -> Declaration:
    seq, facts = int(task.seq), int(task.facts)
    q = seq - 1
    known = (*DESCRIPTIONS, "b")
    names = ("hop", "carry", "two back")
    subjects = tuple(4 * i for i in range(facts))
    objects = tuple(4 * i + 2 for i in range(facts))

    def describe_variables(windows: np.ndarray) -> dict[str, np.ndarray]:
        b = np.full((len(windows), seq), -1, dtype=np.int64)
        for i, window in enumerate(windows):
            b[i, q - 1 : q + 1] = _parse(task, window).b
        return {"b": b}

    def predict(description: str, window: np.ndarray, position: int, w: int) -> tuple[int, int] | None:
        window = check_prediction_args(description, known, window, position, w, seq)
        if position != q:
            return None
        if description == "token(p+1)":
            return (q, int(w))
        if description == "b":
            parsed = _parse(task, window)
            if w in parsed.facts and w not in (parsed.a, parsed.b):
                return (q, parsed.facts[int(w)])
        return None

    def pairs(relation: str, window: np.ndarray) -> tuple[tuple[int, int], ...]:
        _check_relation(relation, names, "two_hop")
        check_window(window, seq)
        if relation == "two back":
            return tuple((p, p + 2) for p in objects if p + 2 in subjects)
        return ((q, q),)

    def value_map(relation: str, window: np.ndarray, v: int) -> int | None:
        _check_relation(relation, names, "two_hop")
        window = check_window(window, seq)
        v = _check_value(v)
        if relation == "hop":
            return _parse(task, window).facts.get(v)
        return v

    return Declaration(
        run="two_hop", seq=seq,
        classes={"query": (q,), "subjects": subjects, "objects": objects, "every": range(seq)},
        variables=("b",),
        planted=(Planted(Place("b", 2, "query"), True), Planted(Place("token(p+1)", 3, "query"), True),
                 Planted(Place("b", 1, "query"), False)),
        relations=(Relation("hop", Place("b", 2, "query"), Place("token(p+1)", 3, "query"), True),
                   Relation("carry", Place("b", 1, "query"), Place("b", 2, "query"), False),
                   Relation("two back", Place("token(p)", 0, "objects"), Place("token(p-2)", 2, "subjects"), False)),
        describe_variables=describe_variables, predict=predict, pairs=pairs, value_map=value_map)


def induction(task: Any) -> Declaration:
    seq = int(task.seq)
    mechanism = {int(p) for p in task.mechanism_positions}

    def describe_variables(windows: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def predict(description: str, window: np.ndarray, position: int, w: int) -> tuple[int, int] | None:
        window = check_prediction_args(description, DESCRIPTIONS, window, position, w, seq)
        if description == "token(p+1)" and position in mechanism:
            return (int(position), int(w))
        if description == "token(p)":
            if position in sole_targets(task, window):
                return (int(position) + period(window) - 1, int(w))
        return None

    def pairs(relation: str, window: np.ndarray) -> tuple[tuple[int, int], ...]:
        _check_relation(relation, ("copy",), "induction")
        window = check_window(window, seq)
        P = period(window)
        return tuple((int(s), int(s) + P - 1) for s in sole_targets(task, window))

    def value_map(relation: str, window: np.ndarray, v: int) -> int | None:
        _check_relation(relation, ("copy",), "induction")
        check_window(window, seq)
        return _check_value(v)

    return Declaration(
        run="induction", seq=seq, classes={"mechanism": sorted(mechanism), "every": range(seq)}, variables=(),
        planted=(Planted(Place("token(p)", 0, "every"), True), Planted(Place("token(p+1)", 1, "mechanism"), True)),
        relations=(Relation("copy", Place("token(p)", 0, "every"), Place("token(p+1)", 1, "mechanism"), True),),
        describe_variables=describe_variables, predict=predict, pairs=pairs, value_map=value_map)


def modular_addition(task: Any) -> Declaration:
    seq = int(task.seq)
    answer = int(task.mechanism_positions[0])

    def describe_variables(windows: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def predict(description: str, window: np.ndarray, position: int, w: int) -> tuple[int, int] | None:
        check_prediction_args(description, DESCRIPTIONS, window, position, w, seq)
        return (answer, int(w)) if description == "token(p+1)" and position == answer else None

    def pairs(relation: str, window: np.ndarray) -> tuple[tuple[int, int], ...]:
        raise ValueError(f"run 'modular_addition' has no relation {relation!r}")

    def value_map(relation: str, window: np.ndarray, v: int) -> int | None:
        raise ValueError(f"run 'modular_addition' has no relation {relation!r}")

    return Declaration(
        run="modular_addition", seq=seq, classes={"answer": (answer,), "every": range(seq)}, variables=(),
        planted=(Planted(Place("token(p+1)", 0, "answer"), False),), relations=(),
        describe_variables=describe_variables, predict=predict, pairs=pairs, value_map=value_map)
