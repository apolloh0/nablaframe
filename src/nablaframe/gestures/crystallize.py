"""crystallize-this (D56, D57): replace a component with an explicit function of what it reads, and measure how far the model's prediction moves.

Spec: docs/specs/phase-7.md, D56 to D58 and C11. Built on ``zoo/functions.py``'s declarations and
``interventions.apply``'s ``set``, with nothing specific to a task (D29). Nothing here caches or
judges: the gates ask for crystallizations, and a record holds what ``crystallize`` returns (D47).

* ``PARTS = ("attention", "mlp", "write")``; ``HOLDS = "holds"``, ``MISSES = "doesn't hold"`` and
  ``INERT = "inert"``; ``KL_HOLDS = 0.25``, ``MOVES = 1.0``, ``ACCURATE = 0.9``, ``ENERGY = 0.75``,
  ``MAX_FREQUENCIES = 8``, ``NAMES_HOLD = 0.7``, ``CHUNK = 256``; and ``NAMES_FALL_SHORT = "Phase 6's
  names don't hold this key: they name its description as planted for fewer than 0.7 of its values.
  The crystallization's own tests stand alone (C11)."``
* ``part_taps(block, part)``: the taps a part writes, ``(f"blocks.{block}.hook_attn_out",)`` for
  ``attention``, ``(f"blocks.{block}.hook_mlp_out",)`` for ``mlp``, and both, attention first, for
  ``write``. ValueError for another part or a block that isn't a non-negative int.
* ``part_edits(block, part, positions, values)``: the edits that put a part to given values, a
  tuple: ``set`` at the part's tap with ``values`` (a content hash) for ``attention`` and ``mlp``;
  for ``write``, ``set`` at ``hook_attn_out`` and then ``zero`` at ``hook_mlp_out``, both at
  ``positions``, so the residual after the block is the residual before it plus the values.
* ``label(kl, ablation)``: ``INERT`` where ``ablation`` is below ``MOVES``; otherwise ``HOLDS``
  where ``kl`` is at most ``KL_HOLDS`` and ``MISSES`` where it is above.
* ``frequencies(embedding, p)``: the fourier form's frequencies, from ``embedding``, the number
  tokens' rows, a float array (p, d), taken in float64. Each frequency ``k`` from 1 to
  ``(p - 1) // 2`` has energy ``‖Σᵥ cos(2πkv/p) Eᵥ‖² + ‖Σᵥ sin(2πkv/p) Eᵥ‖²``, and its share is that
  over every frequency's. Ranked by share, largest first, ties to the lower ``k``, ``K`` is the
  shortest prefix whose shares sum to at least ``ENERGY``, cut at ``MAX_FREQUENCIES``. Returns
  ``{"k": K, "shares": each one's share, "held": their sum}``, ints and floats. ValueError for a
  ``p`` that is even or below 3, or an embedding of another shape.
* ``Fitted``: a form fitted to a component's output, from ``fit``. ``value(keys)`` maps an int64
  array (n, J) of key values, one column per key, to float64 (n, d). ``seen``: for each key, a bool
  array over its values, whether the fitting rows show it. ``frequencies``: ``K`` for ``fourier``,
  None otherwise.
* ``fit(form, keys, Y, n_values, *, k=None)``: the form fitted on rows whose keys are ``keys``, an
  int64 array (n, J) with every value in ``range(n_values[j])``, and whose output is ``Y``, a float
  array (n, d), in float64:

  - ``table``, one key: ``T[v]``, the mean of ``Y`` over the rows whose key is ``v``. A value no row
    shows gets the mean over every row. ``value`` reads ``T`` at the key.
  - ``additive``, two or more keys: the one-hot design, the keys' blocks side by side in order, each
    ``n_values[j]`` columns, fitted by ``numpy.linalg.lstsq(X, Y, rcond=None)`` (minimum norm).
    ``value`` is the design of the given keys times the coefficients, so a value no row shows adds
    nothing.
  - ``fourier``, one key, with ``k`` the frequencies: the design ``[1, cos(2πk₁v/p), sin(2πk₁v/p),
    cos(2πk₂v/p), ...]`` in ``k``'s order, ``p = n_values[0]``, fitted by
    ``numpy.linalg.lstsq(X, Y, rcond=None)``.

  ValueError for another form, a key count the form doesn't take, ``k`` given for a form other than
  ``fourier`` or missing for it, a key outside its range, no rows, or arrays that don't line up.
* ``envelope(rows)``: for checkpoint rows in step order, each with ``"step"``, ``"label"`` and
  ``"accuracy"``: ``{"steps", "runs", "final", "gap", "accurate_from", "before"}``. ``steps``: the
  steps labelled ``HOLDS``; ``runs``: ``[first, last]`` of each maximal run of consecutive rows
  labelled ``HOLDS``, in order; ``final``: the run that ends at the last row, None where the last row
  doesn't hold; ``gap``: the steps whose accuracy is at least ``ACCURATE`` but whose label isn't
  ``HOLDS``; ``accurate_from``: the first step whose accuracy is at least ``ACCURATE``, or None;
  ``before``: the steps labelled ``HOLDS`` before ``accurate_from``, every one of them where it is
  None.
* ``describes(declaration, functions, function, windows)``: for each of the function's keys, in
  order, the first description of ``DESCRIPTIONS`` and then the declaration's variables
  (``declaration.describe(windows)``) that equals the key at every position of the function's class
  in ``windows`` where the key is defined; None where none does or the key is defined nowhere. A
  tuple.
* ``cite(records, *, cls, descriptions, dictionary, site, relation)``: the records a
  crystallization cites, chosen from ``records``, every record of its run (``GestureRecord``\\ s
  sharing one ``request.run``): ``{"names": [...], "edges": [...]}``, each a sorted list of record
  keys. A name record is cited where its request's ``dictionary``, ``site`` and ``class`` are
  ``dictionary``, ``site`` and ``cls`` and its results name one of ``descriptions`` (their
  ``"name"`` isn't None and its ``"description"`` is one of them). An edge record is cited where its
  request's ``relation`` is ``relation``, its ``down`` is ``dictionary`` and its ``down_site``
  ``site``; with ``relation`` None, none is. ValueError for records of more than one run, or for
  something that isn't a ``GestureRecord``.
* ``crystallize(model, params, *, n_layers, functions, declaration, block, part, cls, function,
  form, fit_windows, held_windows, seed, twin=None, steps=None, params_at=None, cited=None)``: D56
  for ``functions.functions[function]``, a plain JSON mapping that ``records.GestureRecord`` takes as
  its results. ``functions`` is the run's ``RunFunctions``; ``fit_windows`` and ``held_windows``
  integer arrays (B, seq + 1) of the run's task, ``seq = declaration.seq``.

  - **Rows:** a window set's rows are its (window, position) pairs at the class's positions where
    every one of the function's keys is defined, window by window and each window's positions in
    the class's order. Its key values are the keys' at those rows, one column per key.
  - **The component's output ``Y``:** the part's tap at each row on the unintervened pass, or for
    ``write`` ``hook_attn_out`` plus ``hook_mlp_out``, summed in float64.
  - **Passes:** every pass reads its windows in chunks of ``CHUNK``, in order. The unintervened one
    is ``model.instrumented`` recording the part's taps. A pass that puts the part to values runs
    ``run_intervened`` with ``part_edits`` at every position of the class: at each row the value
    given, taken to float32, and at every other position of the class the part's own output on the
    unintervened pass, so the part's write is unchanged there. A counterfactual's patch, where it has one,
    comes first in the spec: ``patch`` at its tap and position from its sources.
  - **Measured on the held-out rows,** with ``F = fit(form, ...)`` fitted on the fitting rows'
    output, ``kl`` being ``dictionaries.edges.kl`` and a pass's output at a row being
    ``sets.argmax`` of its logits there:

    - ``"meter"``: ``{"label", "kl", "ablation", "removed", "agreement", "loss_delta"}``. ``kl``
      from the unintervened logits to those of the part put to ``F``'s value; ``ablation`` the same
      with the part put to the mean of ``Y`` over the fitting rows; ``removed``, ``1 - kl /
      ablation``, None where ``ablation`` is 0; ``label``, ``label(kl, ablation)``; ``agreement``, the
      share of rows whose output is the unintervened output; ``loss_delta``, the crystallized loss
      minus the model's.
    - ``"loss"`` and ``"accuracy"``: each ``{"model", "crystallized"}``, the mean cross-entropy of
      the true next token, the window's token after the row's position, and the share of rows whose
      output is that token.
    - ``"r2"``: ``1 - Σ(Y - F)² / Σ(Y - Ȳ)²`` over the held-out rows and every dimension, ``Ȳ`` their
      mean; None where the denominator is 0.
    - ``"values"``: ``{"seen", "unseen"}``: for each key, how many of its values the fitting rows
      show; and how many held-out rows have a key value they don't.
    - ``"frequencies"``: for ``fourier``, ``{"k", "embedding", "coefficients"}``: ``K``; each one's
      share of the embedding's energy at these params, computed as ``frequencies`` computes shares;
      and each one's share of the fitted non-constant coefficients, ``‖Uₖ‖² + ‖Vₖ‖²`` over their sum,
      0 each where they are all 0. None for the other forms.
    - ``"wrong"``: for each of the function's wrong keys, in order, ``{"key", "kl", "agreement"}``:
      the same form fitted on the same rows with that key alone (a ``table`` where the form is
      ``additive``, which takes two keys or more), measured as ``"meter"``'s ``kl``
      and ``agreement``. ValueError where a wrong key isn't defined at one of the rows, or where it
      only relabels the function's keys (D58): on the fitting rows, its value and the function's
      keys' values determine each other, so the form refitted on it is the function itself.
    - ``"counterfactual"``: None where the function declares none. Otherwise its case is
      ``functions.counterfactuals[name](held_windows, numpy.random.default_rng(seed))``. The
      counterfactual pass runs the case's windows with its patch, recording the part's taps. Two
      passes put the part to ``F``'s value at every position of the class where the keys of the
      case's windows are defined, except at the case's rows, where the keys are the case's
      ``reads`` in one and its ``answer`` in the other; elsewhere the part keeps its output on the
      counterfactual pass. ``{"name", "rows", "reads": {"kl", "agreement"}, "answer": {"kl",
      "agreement"}}``: ``kl`` and ``agreement`` at the case's rows, from the counterfactual pass's
      logits and output. ValueError for a case row outside the class's positions.

  - ``"function"``, ``"block"``, ``"part"``, ``"class"``, ``"form"``, ``"keys"``, ``"describes"``
    (``describes`` over the held-out windows), ``"reads"`` (``[[description, where], ...]``),
    ``"relation"``, ``"taps"`` (``part_taps``) and ``"rows"`` (``{"fit", "held"}``, their counts),
    then everything measured above.
  - **The twin:** ``"twin"`` is None without ``twin``. With it, everything measured above, fitted
    and measured on the twin's params, the counterfactual drawn again with a fresh
    ``default_rng(seed)``.
  - **The fourier form's ``K``** is ``frequencies(params.wte[:p], p)["k"]`` at ``params``, and the
    twin and every checkpoint keep it.
  - **Across training:** ``"checkpoints"`` and ``"envelope"`` are None without ``steps``. With
    them, for each step in ``steps`` with ``params_at(step)``, the form fitted again at that step:
    ``{"step", "kl", "ablation", "label", "accuracy"}``, the last the model's accuracy over the
    held-out rows. ``"envelope"`` is ``envelope`` of those rows.
  - **Citations:** ``"cites"`` is None without ``cited``, the records ``cite`` chose. With them,
    ``{"names", "edges"}``. ``names``: for each key whose description isn't None, in order,
    ``{"key", "description", "records", "values", "named", "note"}``: the sorted keys of the cited
    name records naming that description, the key's distinct values over the held-out rows, the
    share of them some cited record names as planted (that description and value, labelled
    ``name.NAMED``), and ``NAMES_FALL_SHORT`` where that share is below ``NAMES_HOLD``, None
    otherwise. ``edges``: for each cited edge record, by key, ``{"key", "relation", "label"}``.

  ValueError for a part not in ``PARTS``; a function the run doesn't declare; a ``block`` or
  ``cls`` that isn't the function's own; a form it doesn't admit; ``functions`` and ``declaration``
  of different runs; windows that aren't integer arrays (B, seq + 1) of non-negative tokens, or
  give no rows; a ``seed`` that isn't a non-negative int; ``steps`` without ``params_at``, or steps
  that aren't non-empty, strictly ascending non-negative ints; and cited records of more than one
  run, or that are neither name records naming one of ``"describes"`` nor edge records of the
  function's relation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..dictionaries.edges import kl
from ..interventions.apply import run_intervened
from ..interventions.spec import InterventionSpec, array_hash, edit
from ..zoo.functions import FORMS
from . import sets
from .declare import DESCRIPTIONS
from .name import NAMED
from .records import GestureRecord

PARTS = ("attention", "mlp", "write")
HOLDS, MISSES, INERT = "holds", "doesn't hold", "inert"
KL_HOLDS, MOVES, ACCURATE, ENERGY, MAX_FREQUENCIES, NAMES_HOLD, CHUNK = 0.25, 1.0, 0.9, 0.75, 8, 0.7, 256
NAMES_FALL_SHORT = ("Phase 6's names don't hold this key: they name its description as planted for fewer than 0.7 "
                    "of its values. The crystallization's own tests stand alone (C11).")

def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _value(recorded: Any) -> np.ndarray:
    return np.asarray(getattr(recorded, "array", recorded))


def part_taps(block: int, part: str) -> tuple[str, ...]:
    if not _is_int(block) or block < 0:
        raise ValueError(f"block must be a non-negative int, got {block!r}")
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    attention, mlp = f"blocks.{int(block)}.hook_attn_out", f"blocks.{int(block)}.hook_mlp_out"
    return {"attention": (attention,), "mlp": (mlp,), "write": (attention, mlp)}[part]


def part_edits(block: int, part: str, positions: Any, values: str) -> tuple[Any, ...]:
    taps = part_taps(block, part)
    if part == "write":
        return edit(taps[0], positions, "set", values=values), edit(taps[1], positions, "zero")
    return (edit(taps[0], positions, "set", values=values),)


def label(kl: float, ablation: float) -> str:
    if ablation < MOVES:
        return INERT
    return HOLDS if kl <= KL_HOLDS else MISSES


def _shares(embedding: Any, p: Any) -> np.ndarray:
    """Every frequency's share of the embedding's non-constant energy, frequency ``k`` at index ``k - 1``."""
    if not _is_int(p) or p < 3 or p % 2 == 0:
        raise ValueError(f"p must be an odd int of at least 3, got {p!r}")
    E = np.asarray(embedding, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] != p:
        raise ValueError(f"the embedding must be the number tokens' rows, shaped ({p}, d), got {E.shape}")
    v = np.arange(int(p))
    energy = []
    for k in range(1, (int(p) - 1) // 2 + 1):
        c, s = np.cos(2 * np.pi * k * v / p) @ E, np.sin(2 * np.pi * k * v / p) @ E
        energy.append(float(c @ c) + float(s @ s))
    energy = np.asarray(energy)
    total = energy.sum()
    if total == 0:
        raise ValueError("the embedding has no non-constant energy to share")
    return energy / total


def frequencies(embedding: Any, p: int) -> dict[str, Any]:
    shares = _shares(embedding, p)
    ranked = sorted(range(len(shares)), key=lambda i: (-shares[i], i))  # largest first, ties to the lower k
    K: list[int] = []
    held = 0.0
    for i in ranked[:MAX_FREQUENCIES]:
        K.append(i + 1)
        held += float(shares[i])
        if held >= ENERGY:
            break
    return {"k": K, "shares": [float(shares[k - 1]) for k in K], "held": float(sum(shares[k - 1] for k in K))}


@dataclass(frozen=True, eq=False)
class Fitted:
    form: str
    n_values: tuple[int, ...]
    weights: np.ndarray
    seen: tuple[np.ndarray, ...]
    frequencies: tuple[int, ...] | None

    def value(self, keys: Any) -> np.ndarray:
        keys = _keys(keys, self.n_values)
        if self.form == "table":
            return self.weights[keys[:, 0]]
        if self.form == "additive":
            offsets = np.cumsum((0,) + self.n_values[:-1])
            return sum(self.weights[offsets[j] + keys[:, j]] for j in range(len(self.n_values)))
        return _fourier_design(keys[:, 0], self.frequencies, self.n_values[0]) @ self.weights


def _keys(keys: Any, n_values: tuple[int, ...]) -> np.ndarray:
    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[1] != len(n_values) or not np.issubdtype(keys.dtype, np.integer):
        raise ValueError(f"keys must be an int array (n, {len(n_values)}), got {keys.dtype} {keys.shape}")
    keys = keys.astype(np.int64)
    for j, n in enumerate(n_values):
        if len(keys) and (keys[:, j].min() < 0 or keys[:, j].max() >= n):
            raise ValueError(f"key {j} has a value outside range({n})")
    return keys


def _fourier_design(v: np.ndarray, k: Sequence[int], p: int) -> np.ndarray:
    columns = [np.ones(len(v))]
    for f in k:
        columns += [np.cos(2 * np.pi * f * v / p), np.sin(2 * np.pi * f * v / p)]
    return np.stack(columns, axis=1)


def fit(form: str, keys: Any, Y: Any, n_values: Sequence[int], *, k: Sequence[int] | None = None) -> Fitted:
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    if isinstance(n_values, (str, bytes)) or not isinstance(n_values, Sequence) or \
            not all(_is_int(n) and n > 0 for n in n_values):
        raise ValueError(f"n_values must be positive ints, one per key, got {n_values!r}")
    n_values = tuple(int(n) for n in n_values)
    keys = _keys(keys, n_values)
    Y = np.asarray(Y)
    if Y.ndim != 2 or not np.issubdtype(Y.dtype, np.floating):
        raise ValueError(f"Y must be a float array (n, d), got {Y.dtype} {Y.shape}")
    Y = Y.astype(np.float64)
    if len(keys) != len(Y) or not len(keys):
        raise ValueError(f"keys and Y must share at least one row, got {len(keys)} and {len(Y)}")
    J = len(n_values)
    if (form == "additive") != (J >= 2) or (form != "additive" and J != 1):
        raise ValueError(f"the {form} form can't take {J} key(s)")
    if (form == "fourier") != (k is not None):
        raise ValueError("k is the fourier form's frequencies: given for it, and only for it")
    seen = []
    for j, n in enumerate(n_values):
        shown = np.zeros(n, dtype=bool)
        shown[keys[:, j]] = True
        seen.append(shown)
    if form == "table":
        T = np.tile(Y.mean(axis=0), (n_values[0], 1))  # a value no row shows gets the mean over every row
        for v in np.flatnonzero(seen[0]):
            T[v] = Y[keys[:, 0] == v].mean(axis=0)
        return Fitted(form, n_values, T, tuple(seen), None)
    if form == "additive":
        X = np.concatenate([np.eye(n)[keys[:, j]] for j, n in enumerate(n_values)], axis=1)
        W = np.linalg.lstsq(X, Y, rcond=None)[0]  # minimum norm: a value no row shows gets 0
        return Fitted(form, n_values, W, tuple(seen), None)
    p = n_values[0]
    k = k.tolist() if isinstance(k, np.ndarray) else k
    if isinstance(k, (str, bytes)) or not isinstance(k, Sequence) or not k or \
            not all(_is_int(f) and 1 <= f <= (p - 1) // 2 for f in k) or len(set(int(f) for f in k)) != len(k):
        raise ValueError(f"k must be distinct frequencies from 1 to {(p - 1) // 2}, got {k!r}")
    k = tuple(int(f) for f in k)
    W = np.linalg.lstsq(_fourier_design(keys[:, 0], k, p), Y, rcond=None)[0]
    return Fitted(form, n_values, W, tuple(seen), k)


def envelope(rows: Sequence[Any]) -> dict[str, Any]:
    steps = [int(r["step"]) for r in rows]
    holds = [r["label"] == HOLDS for r in rows]
    accurate = [r["accuracy"] is not None and r["accuracy"] >= ACCURATE for r in rows]
    runs: list[list[int]] = []
    for i, step in enumerate(steps):
        if holds[i]:
            if i and holds[i - 1]:
                runs[-1][1] = step
            else:
                runs.append([step, step])
    accurate_from = next((step for step, a in zip(steps, accurate) if a), None)
    return {"steps": [s for s, h in zip(steps, holds) if h], "runs": runs,
            "final": runs[-1] if holds and holds[-1] else None,
            "gap": [s for s, h, a in zip(steps, holds, accurate) if a and not h], "accurate_from": accurate_from,
            "before": [s for s, h in zip(steps, holds) if h and (accurate_from is None or s < accurate_from)]}


def _function(functions: Any, function: Any) -> Any:
    if not isinstance(function, str) or function not in functions.functions:
        raise ValueError(f"run {functions.run!r} declares no function {function!r}")
    return functions.functions[function]


def describes(declaration: Any, functions: Any, function: str, windows: Any) -> tuple[str | None, ...]:
    f = _function(functions, function)
    positions = list(declaration.classes[f.cls])
    described = declaration.describe(windows)
    order = list(DESCRIPTIONS) + list(declaration.variables)
    out: list[str | None] = []
    for key in f.keys:
        values = functions.keys[key].values(windows)[:, positions]
        defined = values >= 0
        out.append(None if not defined.any() else next(
            (d for d in order if np.array_equal(described[d][:, positions][defined], values[defined])), None))
    return tuple(out)


def _one_run(records: Any) -> list[GestureRecord]:
    records = list(records)
    for r in records:
        if not isinstance(r, GestureRecord):
            raise ValueError(f"cited records must be GestureRecords, got {type(r).__name__}")
    if len({r.request.run for r in records}) > 1:
        raise ValueError("cited records must all be of one run")
    return records


def _naming(record: GestureRecord) -> Any:
    """The name a name record's results give, or None."""
    return record.results.get("name") if record.request.kind == "name" else None


def cite(records: Any, *, cls: str, descriptions: Sequence[str | None], dictionary: str, site: int,
         relation: str | None) -> dict[str, list[str]]:
    records = _one_run(records)
    wanted = {d for d in descriptions if d is not None}
    names, edges = [], []
    for r in records:
        args = r.request.args
        if r.request.kind == "name" and (args["dictionary"], args["site"], args["class"]) == (dictionary, site, cls):
            named = _naming(r)
            if named is not None and named["description"] in wanted:
                names.append(r.key)
        elif r.request.kind == "edge" and relation is not None and \
                (args["relation"], args["down"], args["down_site"]) == (relation, dictionary, site):
            edges.append(r.key)
    return {"names": sorted(names), "edges": sorted(edges)}


def _windows(windows: Any, seq: int, what: str) -> np.ndarray:
    arr = np.asarray(windows)
    if arr.ndim != 2 or arr.shape[1] != seq + 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{what} must be an integer array (B, {seq + 1}), got {arr.dtype} {arr.shape}")
    if arr.size and arr.min() < 0:
        raise ValueError(f"{what} must hold non-negative tokens")
    return arr.astype(np.int64)


def _rows(functions: Any, names: Sequence[str], positions: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, ...]:
    """The rows, (n, 2) ``[window, position]``, where every key in ``names`` is defined, and their keys (n, J)."""
    at = np.stack([functions.keys[n].values(windows)[:, positions] for n in names], axis=-1)
    w, j = np.nonzero((at >= 0).all(axis=-1))  # window by window, each window's positions in the class's order
    return np.stack([w, positions[j]], axis=1), at[w, j]


class _Passes:
    """A model's passes over windows, ``CHUNK`` windows at a time, each giving its logits and the part's output."""

    def __init__(self, model: Any, n_layers: int, seq: int, block: int, part: str, positions: np.ndarray) -> None:
        self.model, self.n_layers, self.seq, self.block, self.part = model, n_layers, seq, block, part
        self.positions, self.taps = positions, part_taps(block, part)

    def output(self, recorded: dict[str, np.ndarray]) -> np.ndarray:
        """The part's output, float64 (B, seq, d): its tap, or for ``write`` the two summed."""
        return sum(recorded[t].astype(np.float64) for t in self.taps)

    def run(self, params: Any, windows: np.ndarray, values: np.ndarray | None = None,
            patch: tuple[str, int, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(logits, the part's output)``. Unintervened with neither ``values`` nor ``patch``; otherwise through
        ``run_intervened``: the patch first, then the part put to ``values``, (B, the class's positions, d)."""
        tokens = np.ascontiguousarray(windows[:, :self.seq], dtype=np.int32)
        logits, recorded = [], {t: [] for t in self.taps}
        for start in range(0, len(tokens), CHUNK):
            chunk = tokens[start:start + CHUNK]
            if values is None and patch is None:
                out, rec = self.model.instrumented(params, chunk, list(self.taps))
            else:
                store: dict[str, np.ndarray] = {}
                edits: list[Any] = []
                if patch is not None:
                    tap, position, sources = patch
                    source = np.ascontiguousarray(np.asarray(sources)[start:start + CHUNK, :self.seq], dtype=np.int32)
                    store[batch_hash(source)] = source
                    edits.append(edit(tap, [int(position)], "patch", source=batch_hash(source)))
                if values is not None:
                    v = np.ascontiguousarray(values[start:start + CHUNK], dtype=np.float32)
                    store[array_hash(v)] = v
                    edits.extend(part_edits(self.block, self.part, [int(p) for p in self.positions], array_hash(v)))
                out, rec = run_intervened(self.model, params, chunk, InterventionSpec(tuple(edits)),
                                          n_layers=self.n_layers, resolve=store.__getitem__, taps=self.taps)
            logits.append(_value(out))
            for t in self.taps:
                recorded[t].append(_value(rec[t]))
        return np.concatenate(logits), self.output({t: np.concatenate(v) for t, v in recorded.items()})

    def values(self, own: np.ndarray, rows: np.ndarray, at_rows: np.ndarray) -> np.ndarray:
        """The values that put the part to ``at_rows`` at ``rows`` and keep ``own`` at the class's other positions."""
        out = own[:, self.positions].copy()
        out[rows[:, 0], np.searchsorted(self.positions, rows[:, 1])] = at_rows
        return out.astype(np.float32)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = x.max(axis=-1, keepdims=True)
    return x - peak - np.log(np.exp(x - peak).sum(axis=-1, keepdims=True))


def _loss(logits: np.ndarray, target: np.ndarray) -> float:
    return float(-_log_softmax(logits)[np.arange(len(target)), target].mean())


def _agreement(logits: np.ndarray, base: np.ndarray) -> float:
    return float(np.mean(sets.argmax(logits) == sets.argmax(base)))


def crystallize(model: Any, params: Any, *, n_layers: int, functions: Any, declaration: Any, block: int, part: str,
                cls: str, function: str, form: str, fit_windows: Any, held_windows: Any, seed: int, twin: Any = None,
                steps: Sequence[int] | None = None, params_at: Callable[[int], Any] | None = None,
                cited: Sequence[Any] | None = None) -> dict[str, Any]:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    f = _function(functions, function)
    if not _is_int(block) or block != f.block:
        raise ValueError(f"function {function!r} replaces a part of block {f.block}, not {block!r}")
    if cls != f.cls:
        raise ValueError(f"function {function!r} is at class {f.cls!r}, not {cls!r}")
    if form not in f.forms:
        raise ValueError(f"function {function!r} admits the forms {f.forms}, not {form!r}")
    if functions.run != declaration.run:
        raise ValueError(f"the functions are run {functions.run!r}'s and the declaration run {declaration.run!r}'s")
    seq = declaration.seq
    fit_windows = _windows(fit_windows, seq, "fit_windows")
    held_windows = _windows(held_windows, seq, "held_windows")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    if steps is not None:
        if params_at is None:
            raise ValueError("steps need params_at")
        steps = steps.tolist() if isinstance(steps, np.ndarray) else steps
        if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence) or not steps or \
                not all(_is_int(s) and s >= 0 for s in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError(f"steps must be non-empty, strictly ascending non-negative ints, got {steps!r}")

    positions = np.asarray(declaration.classes[cls], dtype=np.int64)
    fit_rows, fit_keys = _rows(functions, f.keys, positions, fit_windows)
    held_rows, held_keys = _rows(functions, f.keys, positions, held_windows)
    if not len(fit_rows) or not len(held_rows):
        raise ValueError(f"the windows give no rows: {len(fit_rows)} to fit and {len(held_rows)} held out")
    n_values = tuple(functions.keys[n].n_values for n in f.keys)
    held_target = held_windows[held_rows[:, 0], held_rows[:, 1] + 1]
    described = describes(declaration, functions, function, held_windows)

    # Each wrong key alone, on the same rows; refused where it is undefined or only relabels the keys (D58).
    wrong_keys = {}
    for name in f.wrong:
        key = functions.keys[name]
        on_fit = key.values(fit_windows)[fit_rows[:, 0], fit_rows[:, 1]]
        on_held = key.values(held_windows)[held_rows[:, 0], held_rows[:, 1]]
        if (on_fit < 0).any() or (on_held < 0).any():
            raise ValueError(f"wrong key {name!r} isn't defined at every row")
        pairs = np.unique(np.concatenate([on_fit[:, None], fit_keys], axis=1), axis=0)
        if len(pairs) == len(np.unique(on_fit)) == len(np.unique(fit_keys, axis=0)):
            raise ValueError(f"wrong key {name!r} only relabels the function's keys on the fitting rows (D58)")
        wrong_keys[name] = (on_fit[:, None], on_held[:, None], (key.n_values,))

    def draw() -> Any:
        if f.counterfactual is None:
            return None
        case = functions.counterfactuals[f.counterfactual](held_windows, np.random.default_rng(seed))
        rows = np.asarray(case.rows, dtype=np.int64)
        if not np.isin(rows[:, 1], positions).all():
            raise ValueError(f"counterfactual {f.counterfactual!r} has a row outside class {cls!r}'s positions")
        return case

    case = draw()
    if cited is not None:
        cited = _one_run(cited)
        for r in cited:
            named = _naming(r)
            is_name = named is not None and named["description"] in {d for d in described if d is not None}
            is_edge = r.request.kind == "edge" and f.relation is not None and r.request.args["relation"] == f.relation
            if not (is_name or is_edge):
                raise ValueError(f"cited record {r.key[:12]}... is neither a name record naming one of {described} "
                                 f"nor an edge record of relation {f.relation!r}")
    K = None
    if form == "fourier":
        K = frequencies(np.asarray(params.wte, dtype=np.float64)[:n_values[0]], n_values[0])["k"]

    passes = _Passes(model, n_layers, seq, block, part, positions)

    def fitted(params_now: Any) -> tuple[Fitted, np.ndarray, np.ndarray, np.ndarray]:
        """The form fitted at ``params_now``, the fitting rows' output, and the held-out logits and output."""
        _, own_fit = passes.run(params_now, fit_windows)
        Y = own_fit[fit_rows[:, 0], fit_rows[:, 1]]
        logits, own = passes.run(params_now, held_windows)
        return fit(form, fit_keys, Y, n_values, k=K), Y, logits, own

    def put(params_now: Any, own: np.ndarray, at_rows: np.ndarray) -> np.ndarray:
        logits, _ = passes.run(params_now, held_windows, passes.values(own, held_rows, at_rows))
        return logits[held_rows[:, 0], held_rows[:, 1]]

    def meter(params_now: Any) -> tuple[Any, ...]:
        F, Y, logits, own = fitted(params_now)
        base = logits[held_rows[:, 0], held_rows[:, 1]]
        crystal = put(params_now, own, F.value(held_keys))
        ablated = put(params_now, own, np.broadcast_to(Y.mean(axis=0), (len(held_rows), Y.shape[1])))
        divergence, ablation = kl(base, crystal), kl(base, ablated)
        loss = {"model": _loss(base, held_target), "crystallized": _loss(crystal, held_target)}
        accuracy = {"model": float(np.mean(sets.argmax(base) == held_target)),
                    "crystallized": float(np.mean(sets.argmax(crystal) == held_target))}
        m = {"label": label(divergence, ablation), "kl": divergence, "ablation": ablation,
             "removed": None if ablation == 0 else 1 - divergence / ablation,
             "agreement": _agreement(crystal, base), "loss_delta": loss["crystallized"] - loss["model"]}
        return m, {"loss": loss, "accuracy": accuracy}, (F, Y, base, own)

    def measured(params_now: Any, case_now: Any) -> dict[str, Any]:
        m, scores, (F, Y, base, own) = meter(params_now)
        Yh, Fh = own[held_rows[:, 0], held_rows[:, 1]], F.value(held_keys)
        spread = float(((Yh - Yh.mean(axis=0)) ** 2).sum())
        out = {"meter": m, **scores, "r2": None if spread == 0 else 1 - float(((Yh - Fh) ** 2).sum()) / spread,
               "values": {"seen": [int(s.sum()) for s in F.seen],
                          "unseen": int((~np.stack([s[held_keys[:, j]] for j, s in enumerate(F.seen)],
                                                   axis=1).all(axis=1)).sum())}}
        out["frequencies"] = None
        if form == "fourier":
            shares = _shares(np.asarray(params_now.wte, dtype=np.float64)[:n_values[0]], n_values[0])
            mass = np.array([float((F.weights[1 + 2 * i] ** 2).sum() + (F.weights[2 + 2 * i] ** 2).sum())
                             for i in range(len(K))])
            out["frequencies"] = {"k": list(K), "embedding": [float(shares[k - 1]) for k in K],
                                  "coefficients": [float(x) for x in (mass / mass.sum() if mass.sum() else mass)]}
        out["wrong"] = []
        for name, (on_fit, on_held, n) in wrong_keys.items():
            W = fit("table" if form == "additive" else form, on_fit, Y, n, k=K if form == "fourier" else None)
            logits = put(params_now, own, W.value(on_held))
            out["wrong"].append({"key": name, "kl": kl(base, logits), "agreement": _agreement(logits, base)})
        out["counterfactual"] = None if case_now is None else counterfactual(params_now, F, case_now)
        return out

    def counterfactual(params_now: Any, F: Fitted, case_now: Any) -> dict[str, Any]:
        windows = np.asarray(case_now.windows).astype(np.int64)
        rows = np.asarray(case_now.rows, dtype=np.int64)
        logits, own = passes.run(params_now, windows, patch=case_now.patch)
        base = logits[rows[:, 0], rows[:, 1]]
        # The class's positions where the case's windows' keys are defined, less the case's rows.
        defined, keys = _rows(functions, f.keys, positions, windows)
        is_case = np.zeros(windows.shape[:1] + (seq,), dtype=bool)
        is_case[rows[:, 0], rows[:, 1]] = True
        rest = ~is_case[defined[:, 0], defined[:, 1]]
        out: dict[str, Any] = {"name": f.counterfactual, "rows": len(rows)}
        for which in ("reads", "answer"):
            at_case = np.stack([np.asarray(getattr(case_now, which)[n], dtype=np.int64) for n in f.keys], axis=1)
            at = np.concatenate([defined[rest], rows])
            values = passes.values(own, at, np.concatenate([F.value(keys[rest]), F.value(at_case)]))
            changed, _ = passes.run(params_now, windows, values, patch=case_now.patch)
            moved = changed[rows[:, 0], rows[:, 1]]
            out[which] = {"kl": kl(base, moved), "agreement": _agreement(moved, base)}
        return out

    results: dict[str, Any] = {
        "function": function, "block": int(block), "part": part, "class": cls, "form": form, "keys": list(f.keys),
        "describes": list(described), "reads": [[r.description, r.where] for r in f.reads], "relation": f.relation,
        "taps": list(part_taps(block, part)), "rows": {"fit": len(fit_rows), "held": len(held_rows)}}
    results.update(measured(params, case))
    results["twin"] = None if twin is None else measured(twin, draw())
    results["checkpoints"] = results["envelope"] = None
    if steps is not None:
        checkpoints = []
        for step in steps:
            m, scores, _ = meter(params_at(step))
            checkpoints.append({"step": int(step), "kl": m["kl"], "ablation": m["ablation"], "label": m["label"],
                                "accuracy": scores["accuracy"]["model"]})
        results["checkpoints"], results["envelope"] = checkpoints, envelope(checkpoints)
    results["cites"] = None if cited is None else _cites(cited, f, described, held_keys)
    return results


def _cites(cited: list[GestureRecord], f: Any, described: Sequence[str | None],
           held_keys: np.ndarray) -> dict[str, Any]:
    names = []
    for j, (key, description) in enumerate(zip(f.keys, described)):
        if description is None:
            continue
        naming_it = [r for r in cited if (n := _naming(r)) is not None and n["description"] == description]
        planted = {int(_naming(r)["value"]) for r in naming_it if _naming(r)["label"] == NAMED}
        values = np.unique(held_keys[:, j]).tolist()
        share = sum(v in planted for v in values) / len(values)
        names.append({"key": key, "description": description, "records": sorted(r.key for r in naming_it),
                      "values": len(values), "named": share, "note": NAMES_FALL_SHORT if share < NAMES_HOLD else None})
    edges = [{"key": r.key, "relation": r.request.args["relation"], "label": r.results["label"]}
             for r in sorted(cited, key=lambda r: r.key) if r.request.kind == "edge"]
    return {"names": names, "edges": edges}
