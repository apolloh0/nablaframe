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

PARTS = ("attention", "mlp", "write")
HOLDS, MISSES, INERT = "holds", "doesn't hold", "inert"
KL_HOLDS, MOVES, ACCURATE, ENERGY, MAX_FREQUENCIES, NAMES_HOLD, CHUNK = 0.25, 1.0, 0.9, 0.75, 8, 0.7, 256
NAMES_FALL_SHORT = ("Phase 6's names don't hold this key: they name its description as planted for fewer than 0.7 "
                    "of its values. The crystallization's own tests stand alone (C11).")


def part_taps(block: int, part: str) -> tuple[str, ...]:
    raise NotImplementedError


def part_edits(block: int, part: str, positions: Any, values: str) -> tuple[Any, ...]:
    raise NotImplementedError


def label(kl: float, ablation: float) -> str:
    raise NotImplementedError


def frequencies(embedding: Any, p: int) -> dict[str, Any]:
    raise NotImplementedError


@dataclass(frozen=True)
class Fitted:
    def value(self, keys: Any) -> np.ndarray:
        raise NotImplementedError


def fit(form: str, keys: Any, Y: Any, n_values: Sequence[int], *, k: Sequence[int] | None = None) -> Fitted:
    raise NotImplementedError


def envelope(rows: Sequence[Any]) -> dict[str, Any]:
    raise NotImplementedError


def describes(declaration: Any, functions: Any, function: str, windows: Any) -> tuple[str | None, ...]:
    raise NotImplementedError


def cite(records: Any, *, cls: str, descriptions: Sequence[str | None], dictionary: str, site: int,
         relation: str | None) -> dict[str, list[str]]:
    raise NotImplementedError


def crystallize(model: Any, params: Any, *, n_layers: int, functions: Any, declaration: Any, block: int, part: str,
                cls: str, function: str, form: str, fit_windows: Any, held_windows: Any, seed: int, twin: Any = None,
                steps: Sequence[int] | None = None, params_at: Callable[[int], Any] | None = None,
                cited: Sequence[Any] | None = None) -> dict[str, Any]:
    raise NotImplementedError
