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

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

FORMS = ("table", "additive", "fourier")
WHERE = ("same", "across")


@dataclass(frozen=True)
class Key:
    name: str
    n_values: int
    values: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Read:
    description: str
    where: str


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


def induction(task: Any) -> RunFunctions:
    raise NotImplementedError


def two_hop(task: Any) -> RunFunctions:
    raise NotImplementedError


def modular_addition(task: Any) -> RunFunctions:
    raise NotImplementedError


def for_run(run: str, task: Any) -> RunFunctions:
    raise NotImplementedError
