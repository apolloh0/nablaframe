"""What each run offers crystallize-this: keys, functions and counterfactuals (D56, D57).

Spec: docs/specs/phase-7.md, D56, D57 and "What each run declares". Like ``zoo/declarations.py``,
everything about a task lives here, built from the task object, so crystallize-this carries no
task-specific code (D29). Numpy only.

* ``FORMS = ("table", "additive", "fourier")`` and ``WHERE = ("same", "across")``.
* ``Key(name, n_values, values)``: a frozen dataclass. ``values(windows)`` maps windows of the run's
  task, an integer array (B, seq + 1) of non-negative tokens, to an int64 array (B, seq): the key at
  each position, in ``range(n_values)``, or -1 where it isn't defined. The key at position ``p``
  depends only on the window's tokens at positions 0 to ``p``. ``values`` raises ValueError for
  windows of another shape or dtype, or with a negative token.
* ``Read(description, where)``: a frozen dataclass, something a function reads: a description the
  run's declaration has, and ``where``, ``"same"`` (at the component's own position) or ``"across"``
  (at earlier positions). ValueError for an empty description or another ``where``.
* ``Function(name, block, cls, keys, reads, relation, wrong, counterfactual, forms, judged)``: a
  frozen dataclass.

  - ``name``: a non-empty string. ``block``: the block whose part it replaces, a non-negative int
    (a bool is not an int).
    ``cls``: the position class of the run's declaration whose positions it replaces at.
  - ``keys``: the names of its keys, distinct and in order: one for ``table`` and ``fourier``, two
    or more for ``additive``.
  - ``reads``: a non-empty tuple of ``Read``. ``relation``: the name of the run's declared relation
    for what it reads at its own position, or None.
  - ``wrong``: distinct names of keys, each a wrong function: the same form fitted on that key
    alone. None of them is one of its own ``keys``.
  - ``counterfactual``: the name of one of the run's counterfactuals, or None.
  - ``forms``: the forms it admits, a non-empty tuple of distinct names from ``FORMS``.
  - ``judged``: whether a gate judges it (D57), a bool.

  Construction raises ValueError for a bad field; for ``table`` or ``fourier`` with other than one
  key, or ``additive`` with fewer than two; for a ``relation`` with no ``"same"`` read; and for a
  judged function that reads something at its own position without a ``relation`` to cite (D53).
  The tuples are stored as tuples.
* ``Case(windows, patch, rows, reads, answer)``: a frozen dataclass, one counterfactual's inputs:

  - ``windows``: int32 (B, seq + 1), the windows the model reads, as the counterfactual left them
  - ``patch``: None, or ``(tap, position, sources)``: that tap patched at that one position from
    ``sources``, int32 windows shaped like ``windows``
  - ``rows``: int64 (n, 2), each ``[window, position]`` at which the crystallization is read, one
    per window, in window order
  - ``reads`` and ``answer``: each maps every one of the function's keys, by name, to an int64
    array (n,): the key's value at each row as the component reads it under the counterfactual, and
    as the window's own answer gives it
* ``RunFunctions(run, seq, keys, functions, counterfactuals)``: a frozen dataclass. ``keys`` maps
  names to ``Key``, ``functions`` names to ``Function`` and ``counterfactuals`` names to callables
  ``(windows, rng) -> Case``, each stored read-only, so assigning into it raises TypeError. Each
  mapping's names are its values' own. Construction raises ValueError when a function names a key
  or a counterfactual the run doesn't have.
* ``induction(task)``, for an ``InductionTask``: ``run`` ``"induction"``, ``seq = task.seq``.

  - Keys, each with ``n_values = task.vocab_size``, defined at the mechanism positions only:
    ``"copied"``, ``w[q - P + 1]`` at ``q``, ``P`` being the smallest ``i`` from 1 to ``q`` with
    ``w[i] = w[0]``, the window's period as far as ``q`` has read it, and -1 where there is none;
    ``"current"``, ``w[q]``; ``"previous"``, ``w[q - 1]``. On a task's own windows ``P`` is
    ``declarations.period(w)``.
  - ``"copied"``: block 1, class ``"mechanism"``, keys ``("copied",)``, reads
    ``(Read("token(p)", "across"),)``, no relation, wrong ``("current", "previous")``,
    counterfactual ``"broken"``, forms ``("table",)``, judged.
  - ``"broken"``: a copy of the windows in which, window by window in order, one position ``j``
    gets another token: ``j = int(rng.integers(task.max_period + 1, task.window - 1))``, then the
    token ``int(rng.choice(others))``, ``others`` being every token but the window's at ``j``, in
    ascending order. No patch. Its row is ``[window, j - 1]``. There ``reads["copied"]`` is the
    changed window's copied key and ``answer["copied"]`` its token at ``j``, the new one.

* ``two_hop(task)``, for a ``TwoHopTask``: ``run`` ``"two_hop"``, ``seq = task.seq``, ``q = seq - 1``.

  - Keys, each with ``n_values = len(task.vocab)``, defined at ``q`` only, read from the facts at
    positions 0 to ``4 * facts - 1`` and the query's subject at ``q - 1``: ``"a"``, the query's
    subject; ``"b"``, the object of ``a``'s fact; ``"hop"``, the object of ``b``'s fact, which on
    the task's own windows is the answer ``c``. Where the window has no such fact, -1.
  - ``"hop"``: block 3, class ``"query"``, keys ``("hop",)``, reads ``(Read("b", "same"),
    Read("token(p)", "across"))``, relation ``"hop"``, wrong ``("a", "b")``, counterfactual
    ``"b patched"``, forms ``("table",)``, judged.
  - ``"b patched"``: ``(sources, _, counterparts) = zoo.two_hop.aligned_sources(task, windows,
    rng)``. The windows unchanged; the patch ``("blocks.2.hook_resid_post", q, sources)``; the row
    ``[window, q]``. There ``reads["hop"]`` is the object of the counterpart's fact in the window,
    what block 3 reads once ``b`` is the counterpart, and ``answer["hop"]`` the window's own ``c``.

* ``modular_addition(task)``, for a ``ModularAdditionTask``: ``run`` ``"modular_addition"``,
  ``seq = task.seq``, 3.

  - Keys, each with ``n_values = task.p``, defined at position 2 only, with ``a`` and ``b`` the
    window's tokens at 0 and 1: ``"sum"``, ``(a + b) % p``; ``"difference"``, ``(a - b) % p``;
    ``"shifted"``, ``(a + b + 1) % p``; ``"a"``; ``"b"``.
  - ``"sum"``: block 0, class ``"answer"``, keys ``("sum",)``, reads ``(Read("token(p-2)",
    "across"), Read("token(p-1)", "across"))``, no relation, wrong ``("difference", "shifted",
    "a", "b")``, no counterfactual, forms ``("table", "fourier")``, judged.
  - ``"sum and operands"``: block 0, class ``"answer"``, keys ``("sum", "a", "b")``, reads
    ``(Read("token(p-2)", "same"), Read("token(p-1)", "same"))``: the MLP reads the operands from
    the attention's output at its own position, with no relation to cite, so it is recorded. No
    wrong functions, counterfactual ``"attention patched"``, forms ``("additive",)``, not judged.
  - ``"operands"``: block 0, class ``"answer"``, keys ``("a", "b")``, reads ``(Read("token(p-2)",
    "across"), Read("token(p-1)", "across"))``, no relation, no wrong functions, no counterfactual,
    forms ``("additive",)``, not judged.
  - ``"attention patched"``: with ``B`` windows, ``src = rng.integers(0, B, size=B)``; then, while
    some window's source has the same sum as itself, those windows draw again, ``src[same] =
    rng.integers(0, B, size=int(same.sum()))``, one call per round. The windows unchanged; the patch
    ``("blocks.0.hook_attn_out", 2, windows[src])``; the row ``[window, 2]``. There ``reads`` holds
    the source's ``sum``, ``a`` and ``b``, and ``answer`` the window's own. ValueError when every
    window has the same sum.

* ``for_run(run, task)``: the run's ``RunFunctions`` by name, ``"induction"``, ``"two_hop"`` or
  ``"modular_addition"``; ValueError for another name, or a task of the wrong type. Each function's
  class is one of the run's declaration's classes (``zoo/declarations.py``), each of its reads
  describes something the declaration has, and its relation is one the declaration declares.

Each builder raises ValueError for a task of the wrong type. A counterfactual raises ValueError for
windows of the wrong shape, and leaves the windows it is given unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from ..gestures.declare import DESCRIPTIONS
from . import declarations
from .tasks import InductionTask, ModularAdditionTask, TwoHopTask
from .two_hop import aligned_sources

FORMS = ("table", "additive", "fourier")
WHERE = ("same", "across")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _name(x: Any, what: str) -> str:
    if not isinstance(x, str) or not x:
        raise ValueError(f"{what} must be a non-empty string, got {x!r}")
    return x


def _names(items: Any, what: str, *, distinct: bool = True) -> tuple[str, ...]:
    if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
        raise ValueError(f"{what} must be a tuple of names, got {items!r}")
    items = tuple(_name(item, f"each of {what}") for item in items)
    if distinct and len(set(items)) != len(items):
        raise ValueError(f"{what} must be distinct, got {items!r}")
    return items


@dataclass(frozen=True)
class Key:
    name: str
    n_values: int
    values: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Read:
    description: str
    where: str

    def __post_init__(self) -> None:
        _name(self.description, "description")
        if self.where not in WHERE:
            raise ValueError(f"where must be one of {WHERE}, got {self.where!r}")


@dataclass(frozen=True)
class Function:
    name: str
    block: int
    cls: str
    keys: tuple[str, ...]
    reads: tuple[Read, ...]
    relation: str | None
    wrong: tuple[str, ...]
    counterfactual: str | None
    forms: tuple[str, ...]
    judged: bool

    def __post_init__(self) -> None:
        _name(self.name, "name")
        if not _is_int(self.block) or self.block < 0:
            raise ValueError(f"block must be a non-negative int, got {self.block!r}")
        _name(self.cls, "cls")
        keys = _names(self.keys, "keys")
        if not keys:
            raise ValueError("a function needs at least one key")
        if isinstance(self.reads, (str, bytes, Mapping)) or not isinstance(self.reads, Sequence):
            raise ValueError(f"reads must be a tuple of Read, got {self.reads!r}")
        reads = tuple(self.reads)
        if not reads or not all(isinstance(r, Read) for r in reads):
            raise ValueError(f"reads must be a non-empty tuple of Read, got {reads!r}")
        if self.relation is not None:
            _name(self.relation, "relation")
            if not any(r.where == "same" for r in reads):
                raise ValueError("a relation needs a read at the component's own position")
        wrong = _names(self.wrong, "wrong")
        if set(wrong) & set(keys):
            raise ValueError(f"a wrong function can't be one of the function's own keys: {wrong!r}")
        if self.counterfactual is not None:
            _name(self.counterfactual, "counterfactual")
        forms = _names(self.forms, "forms")
        if not forms or not set(forms) <= set(FORMS):
            raise ValueError(f"forms must be a non-empty selection from {FORMS}, got {forms!r}")
        for form in forms:
            if form == "additive" and len(keys) < 2 or form != "additive" and len(keys) != 1:
                raise ValueError(f"the {form} form can't take {len(keys)} key(s)")
        if not isinstance(self.judged, bool):
            raise ValueError(f"judged must be a bool, got {self.judged!r}")
        if self.judged and self.relation is None and any(r.where == "same" for r in reads):
            raise ValueError("a judged function that reads at its own position needs a relation to cite (D53)")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "wrong", wrong)
        object.__setattr__(self, "forms", forms)


@dataclass(frozen=True)
class Case:
    windows: np.ndarray
    patch: tuple[str, int, np.ndarray] | None
    rows: np.ndarray
    reads: Mapping[str, np.ndarray]
    answer: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RunFunctions:
    run: str
    seq: int
    keys: Mapping[str, Key]
    functions: Mapping[str, Function]
    counterfactuals: Mapping[str, Callable[[np.ndarray, np.random.Generator], Case]]

    def __post_init__(self) -> None:
        _name(self.run, "run")
        if not _is_int(self.seq) or self.seq <= 0:
            raise ValueError(f"seq must be a positive int, got {self.seq!r}")
        for what, mapping, kind in (("keys", self.keys, Key), ("functions", self.functions, Function),
                                    ("counterfactuals", self.counterfactuals, None)):
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{what} must be a mapping, got {mapping!r}")
            for name, item in mapping.items():
                _name(name, f"a name in {what}")
                if kind is None and not callable(item):
                    raise ValueError(f"counterfactual {name!r} must be callable")
                if kind is not None and (not isinstance(item, kind) or item.name != name):
                    raise ValueError(f"{what} must map each name to its own {kind.__name__}, got {name!r}: {item!r}")
        for f in self.functions.values():
            for name in (*f.keys, *f.wrong):
                if name not in self.keys:
                    raise ValueError(f"function {f.name!r} names key {name!r}, which run {self.run!r} doesn't have")
            if f.counterfactual is not None and f.counterfactual not in self.counterfactuals:
                raise ValueError(f"function {f.name!r} names counterfactual {f.counterfactual!r}, "
                                 f"which run {self.run!r} doesn't have")
        for what in ("keys", "functions", "counterfactuals"):
            object.__setattr__(self, what, MappingProxyType(dict(getattr(self, what))))


def _windows(windows: Any, width: int, what: str) -> np.ndarray:
    arr = np.asarray(windows)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{what} need windows of shape (B, {width}), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{what} need an integer array, got dtype {arr.dtype}")
    if arr.size and arr.min() < 0:
        raise ValueError(f"{what} need non-negative tokens")
    return arr.astype(np.int64)


def _key(name: str, n_values: int, seq: int, compute: Callable[[np.ndarray], np.ndarray]) -> Key:
    def values(windows: np.ndarray) -> np.ndarray:
        w = _windows(windows, seq + 1, f"key {name!r}")
        return compute(w)

    return Key(name, n_values, values)


def _check_task(builder: str, task: Any, kind: type) -> None:
    if not isinstance(task, kind):
        raise ValueError(f"{builder} needs a {kind.__name__}, got {type(task).__name__}")


def _rows(n: int, position: np.ndarray | int) -> np.ndarray:
    return np.stack([np.arange(n, dtype=np.int64), np.broadcast_to(np.asarray(position, dtype=np.int64), (n,))], axis=1)


def induction(task: Any) -> RunFunctions:
    _check_task("induction", task, InductionTask)
    seq, L = int(task.seq), int(task.window)
    mech = np.asarray(task.mechanism_positions, dtype=np.int64)

    def copied(w: np.ndarray) -> np.ndarray:
        # P is the first recurrence of w[0] at or before q, so the key reads only tokens 0 to q.
        eq = w[:, 1:] == w[:, :1]
        P = np.argmax(eq, axis=1) + 1
        valid = eq.any(axis=1)[:, None] & (P[:, None] <= mech[None, :])
        at = np.where(valid, mech[None, :] - P[:, None] + 1, 0)
        out = np.full((len(w), seq), -1, dtype=np.int64)
        out[:, mech] = np.where(valid, np.take_along_axis(w, at, axis=1), -1)
        return out

    def shifted(back: int) -> Callable[[np.ndarray], np.ndarray]:
        def compute(w: np.ndarray) -> np.ndarray:
            out = np.full((len(w), seq), -1, dtype=np.int64)
            out[:, mech] = w[:, mech - back]
            return out
        return compute

    keys = {name: _key(name, int(task.vocab_size), seq, compute)
            for name, compute in (("copied", copied), ("current", shifted(0)), ("previous", shifted(1)))}

    def broken(windows: np.ndarray, rng: np.random.Generator) -> Case:
        w = _windows(windows, L, "broken windows").astype(np.int32)
        j = np.empty(len(w), dtype=np.int64)
        for i in range(len(w)):
            j[i] = int(rng.integers(task.max_period + 1, task.window - 1))
            others = np.delete(np.arange(task.vocab_size), w[i, j[i]])
            w[i, j[i]] = int(rng.choice(others))
        rows = _rows(len(w), j - 1)
        at = (rows[:, 0], rows[:, 1])
        return Case(windows=w, patch=None, rows=rows, reads={"copied": keys["copied"].values(w)[at]},
                    answer={"copied": w[rows[:, 0], j].astype(np.int64)})

    function = Function("copied", 1, "mechanism", ("copied",), (Read("token(p)", "across"),), None,
                        ("current", "previous"), "broken", ("table",), True)
    return RunFunctions("induction", seq, keys, {"copied": function}, {"broken": broken})


def two_hop(task: Any) -> RunFunctions:
    _check_task("two_hop", task, TwoHopTask)
    seq, facts = int(task.seq), int(task.facts)
    q, end = seq - 1, 4 * facts

    def object_of(subjects: np.ndarray, objects: np.ndarray, who: np.ndarray) -> np.ndarray:
        match = subjects == who[:, None]
        found = np.take_along_axis(objects, np.argmax(match, axis=1)[:, None], axis=1)[:, 0]
        return np.where(match.any(axis=1), found, -1)

    def chain(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        subjects, objects = w[:, 0:end:4], w[:, 2:end:4]
        a = w[:, q - 1]
        b = object_of(subjects, objects, a)
        return a, b, object_of(subjects, objects, b)

    def at_query(which: int) -> Callable[[np.ndarray], np.ndarray]:
        def compute(w: np.ndarray) -> np.ndarray:
            out = np.full((len(w), seq), -1, dtype=np.int64)
            out[:, q] = chain(w)[which]
            return out
        return compute

    keys = {name: _key(name, len(task.vocab), seq, at_query(which)) for name, which in (("hop", 2), ("a", 0), ("b", 1))}

    def b_patched(windows: np.ndarray, rng: np.random.Generator) -> Case:
        w = _windows(windows, task.window_length, "b patched").astype(np.int32)
        sources, _, counterparts = aligned_sources(task, w, rng)
        reads = np.array([task.parse(window).facts[int(c)] for window, c in zip(w, counterparts)], dtype=np.int64)
        return Case(windows=w, patch=("blocks.2.hook_resid_post", q, sources), rows=_rows(len(w), q),
                    reads={"hop": reads}, answer={"hop": w[:, -1].astype(np.int64)})

    function = Function("hop", 3, "query", ("hop",), (Read("b", "same"), Read("token(p)", "across")), "hop",
                        ("a", "b"), "b patched", ("table",), True)
    return RunFunctions("two_hop", seq, keys, {"hop": function}, {"b patched": b_patched})


def modular_addition(task: Any) -> RunFunctions:
    _check_task("modular_addition", task, ModularAdditionTask)
    seq, p = int(task.seq), int(task.p)
    answer = int(task.mechanism_positions[0])

    def at_answer(formula: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        def compute(w: np.ndarray) -> np.ndarray:
            out = np.full((len(w), seq), -1, dtype=np.int64)
            out[:, answer] = formula(w[:, 0], w[:, 1]) % p
            return out
        return compute

    formulas = {"sum": lambda a, b: a + b, "difference": lambda a, b: a - b, "shifted": lambda a, b: a + b + 1,
                "a": lambda a, b: a, "b": lambda a, b: b}
    keys = {name: _key(name, p, seq, at_answer(formula)) for name, formula in formulas.items()}

    def attention_patched(windows: np.ndarray, rng: np.random.Generator) -> Case:
        w = _windows(windows, task.window, "attention patched").astype(np.int32)
        n = len(w)
        a, b = w[:, 0].astype(np.int64), w[:, 1].astype(np.int64)
        s = (a + b) % p
        if n == 0 or (s == s[0]).all():
            raise ValueError("attention patched needs windows with at least two different sums")
        src = rng.integers(0, n, size=n)
        while (same := s[src] == s).any():
            src[same] = rng.integers(0, n, size=int(same.sum()))
        return Case(windows=w, patch=("blocks.0.hook_attn_out", answer, w[src]), rows=_rows(n, answer),
                    reads={"sum": s[src], "a": a[src], "b": b[src]}, answer={"sum": s, "a": a, "b": b})

    across = (Read("token(p-2)", "across"), Read("token(p-1)", "across"))
    same = (Read("token(p-2)", "same"), Read("token(p-1)", "same"))
    functions = {
        "sum": Function("sum", 0, "answer", ("sum",), across, None, ("difference", "shifted", "a", "b"), None,
                        ("table", "fourier"), True),
        "sum and operands": Function("sum and operands", 0, "answer", ("sum", "a", "b"), same, None, (),
                                     "attention patched", ("additive",), False),
        "operands": Function("operands", 0, "answer", ("a", "b"), across, None, (), None, ("additive",), False),
    }
    return RunFunctions("modular_addition", seq, keys, functions, {"attention patched": attention_patched})


_BUILDERS = {"induction": (induction, declarations.induction), "two_hop": (two_hop, declarations.two_hop),
             "modular_addition": (modular_addition, declarations.modular_addition)}


def for_run(run: str, task: Any) -> RunFunctions:
    if not isinstance(run, str) or run not in _BUILDERS:
        raise ValueError(f"no functions for run {run!r}; known: {list(_BUILDERS)}")
    build, declare = _BUILDERS[run]
    built = build(task)
    declaration = declare(task)
    described = (*DESCRIPTIONS, *declaration.variables)
    for f in built.functions.values():
        if f.cls not in declaration.classes:
            raise ValueError(f"function {f.name!r} is at class {f.cls!r}, which run {run!r} doesn't declare")
        for read in f.reads:
            if read.description not in described:
                raise ValueError(f"function {f.name!r} reads {read.description!r}, which run {run!r} doesn't describe")
        if f.relation is not None:
            declaration.relation(f.relation)
    return built
