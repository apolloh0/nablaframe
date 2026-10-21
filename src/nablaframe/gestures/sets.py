"""What the gestures share: a dictionary site read over held-out windows, its F1 tables, the null and steering (D48 to D50).

Spec: docs/specs/phase-6.md, D49 and D50. JAX is imported inside the functions that need it, so this
module imports without it. Nothing here caches, records or judges: ``name.py`` and ``edge.py``
measure, and the gates choose the windows and hold the results to the bars.

* ``SET_SIZE = 4``, ``EXCLUDE = 8``, ``SHARED = 0.9``, ``FLOOR = 0.5`` and ``CANDIDATES = 5``.
* ``read_site(model, params, *, n_layers, declaration, dictionary, site, windows, chunk=256)``: a
  ``SiteReading``, one site of a dictionary read over held-out ``windows``, an integer array
  (B, seq + 1) with ``seq = declaration.seq`` and B at least 1. Its attributes:

  - ``model``, ``params``, ``n_layers``, ``declaration``, ``dictionary`` and ``site``, as given
  - ``windows``, as int64, and ``tokens``, ``windows[:, :seq]`` as int32, what the model reads
  - ``input_tap`` and ``output_tap``, the site's taps; ``block``, the ``l`` of an output tap named
    ``blocks.<l>.…``, None for any other
  - ``width``, the site's ``active_width``
  - ``codes``: float32 (B, seq, width), ``core.encode`` of the input tap's values on the
    unintervened pass, from ``measure.activations``, ``chunk`` windows at a time, keeping latents
    0 to ``width - 1``
  - ``described``: ``declaration.describe(windows)``, its descriptions ordered as
    ``DESCRIPTIONS`` and then the declaration's variables

  ``reading.table(cls)`` is the ``ClassTable`` of ``codes`` and ``described`` at the class's
  positions, built once per class; ValueError for a class the run doesn't declare. ``read_site``
  raises ValueError, naming the problem, for windows that aren't a 2-D integer array of that
  shape with at least one row, a site the dictionary doesn't have, or a ``chunk`` that isn't a
  positive int.
* ``ClassTable(codes, described, positions)``: F1 of every latent against every value of every
  description that counts at one class. ``codes`` is float32 (B, seq, m); ``described`` maps
  descriptions to int64 arrays (B, seq), -1 for nothing; ``positions`` are the class's positions.
  Its rows are the class's (window, position) pairs, window by window and each window's positions in
  the order given. A latent is active on a row where its code is above zero.

  - ``rows``: the number of rows. ``active``: int64 (m,), each latent's active rows. ``shared``:
    the latents active on at least ``SHARED`` of the rows, a sorted tuple of ints.
  - ``descriptions``: the descriptions that count at the class (``declare.counts_at`` at
    ``positions``), in ``described``'s order, a tuple. ``values(d)``: their values,
    ``declare.values_at``, sorted int64.
  - ``f1(d)``: float64 (m, len(values(d))). For latent ``i`` and value ``v``,
    ``2·|i active ∧ d = v| / (|i active| + |d = v|)``, counted over every row, rows where ``d`` is
    -1 among them.
  - ``has(d, v)``: whether ``d`` counts at the class and ``v`` is one of its values.
  - ``candidates(latent)``: a tuple of up to ``CANDIDATES`` ``(d, v, f1)``, the latent's highest F1
    over every counting description's every value, highest first, ties to the earlier description
    and then the lower value; ``d`` a str, ``v`` an int and ``f1`` a float. Empty for a latent active
    on no row.
  - ``ranked(d, v)``: int64 (m,), every latent by its F1 for ``v``, highest first, ties to the lower
    latent. ``set_for(d, v)``: its first ``SET_SIZE``, a tuple of ints.
  - ``name_f1(d, v, latents)``: the highest F1 for ``v`` among ``latents``, a float: the F1 of a
    name read with that set (D49).
  - ``mean_codes(d, v)``: float32 (m,), the mean code over the rows where ``d`` is ``v``, taken in
    float64.

  Every method taking ``d`` and ``v`` raises ValueError unless ``has(d, v)``; ``candidates`` does for
  a latent outside ``range(m)``. ``ClassTable`` raises ValueError for codes that aren't 3-D, a
  description shaped otherwise than (B, seq), or positions that aren't distinct ints in
  ``range(seq)``.
* ``taken(described, d, window)``: the values ruled out as another value ``w`` of ``d`` in held-out
  window ``window`` (an index), a frozenset of ints: for a token description, every token the model
  reads there (``token(p)``'s values), and for a variable, the values it takes there, -1 aside. So a
  token description's ``w`` is a token the window doesn't contain, as the spec says, while a
  variable's may be a token the window holds (two-hop's distractor subjects).
* ``null_sets(table, d, codes, vs, ws, rng)``: ``(removed, added)``, the null's latents for each
  instance, each a list of lists of ints. ``codes`` is (n, m), the unintervened codes at each
  instance's row; ``vs`` and ``ws`` are each instance's value of ``d`` and its other value.
  Instance by instance, in order:

  - removed: from the latents active in that row of ``codes``, less ``table.ranked(d, v)[:EXCLUDE]``
    and ``table.shared``, ascending, ``rng.choice(eligible, size=min(SET_SIZE, len(eligible)),
    replace=False)``, in the order drawn
  - then added: drawn the same way from the latents active on some row of the table, less
    ``table.ranked(d, w)[:EXCLUDE]``, ``table.shared`` and the latents active in that row of
    ``codes``

  Where nothing is eligible, that list is empty and ``rng`` isn't called. The null adds its latents
  at the first amounts of ``w``'s set (D49). ValueError for lists of different lengths.
* ``run_steered(reading, index, positions, remove, add, amounts, *, taps=(), freeze=())``:
  ``(logits, recorded)``, float32 numpy arrays, from ``interventions.apply.run_intervened`` on
  ``reading.tokens[index]``, one sequence per instance. ``index[k]`` and ``positions[k]`` are
  instance ``k``'s window and the position its edit is at; ``remove[k]`` and ``add[k]`` its latents
  to remove and to add, and ``amounts[k]`` one amount per added latent. The spec:

  - one ``dsteer`` edit at ``reading.output_tap``, selecting the instances' distinct positions,
    ascending. Its ``remove`` rows are ``[k, positions[k], latent]`` for each instance in order and
    each of its latents in order; ``add`` likewise; ``amounts`` the amounts in ``add``'s order: int32
    (n, 3) and float32 (n,) arrays, shaped (0, 3) and (0,) when empty
  - then, for each block ``l`` in ``freeze``, ascending, a ``patch`` edit at
    ``blocks.<l>.hook_attn_out`` and then one at ``blocks.<l>.hook_mlp_out``, selecting the
    instances' position, whose source is the batch itself: both outputs held at their unintervened
    values there. With ``freeze``, every instance's position must be the same.

  ``resolve`` holds the dictionary under its ``content_hash``, each array under its ``array_hash``
  and the batch under its ``batch_hash``. ``recorded`` holds ``taps``. With nothing removed or added
  and no ``freeze``, the change is zero, so the pass is the unintervened one (D25). ValueError for
  lists of different lengths, no instances, amounts that don't match their latents, or a ``freeze``
  over more than one position.
* ``argmax(logits)``: the argmax over the last axis, ties to the lower token, int64.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SET_SIZE = 4
EXCLUDE = 8
SHARED = 0.9
FLOOR = 0.5
CANDIDATES = 5


class ClassTable:
    rows: int
    active: np.ndarray
    shared: tuple[int, ...]
    descriptions: tuple[str, ...]

    def __init__(self, codes: Any, described: Mapping[str, Any], positions: Any) -> None:
        raise NotImplementedError

    def has(self, d: Any, v: Any) -> bool:
        raise NotImplementedError

    def values(self, d: str) -> np.ndarray:
        raise NotImplementedError

    def f1(self, d: str) -> np.ndarray:
        raise NotImplementedError

    def candidates(self, latent: Any) -> tuple[tuple[str, int, float], ...]:
        raise NotImplementedError

    def ranked(self, d: str, v: int) -> np.ndarray:
        raise NotImplementedError

    def set_for(self, d: str, v: int) -> tuple[int, ...]:
        raise NotImplementedError

    def name_f1(self, d: str, v: int, latents: Sequence[int]) -> float:
        raise NotImplementedError

    def mean_codes(self, d: str, v: int) -> np.ndarray:
        raise NotImplementedError


class SiteReading:
    model: Any
    params: Any
    n_layers: int
    declaration: Any
    dictionary: Any
    site: int
    windows: np.ndarray
    tokens: np.ndarray
    input_tap: str
    output_tap: str
    block: int | None
    width: int
    codes: np.ndarray
    described: dict[str, np.ndarray]

    def table(self, cls: str) -> ClassTable:
        raise NotImplementedError


def read_site(model: Any, params: Any, *, n_layers: int, declaration: Any, dictionary: Any, site: int, windows: Any,
              chunk: int = 256) -> SiteReading:
    raise NotImplementedError


def taken(described: Mapping[str, np.ndarray], d: str, window: int) -> frozenset[int]:
    raise NotImplementedError


def null_sets(table: ClassTable, d: str, codes: Any, vs: Sequence[int], ws: Sequence[int],
              rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]]]:
    raise NotImplementedError


def run_steered(reading: SiteReading, index: Sequence[int], positions: Sequence[int], remove: Sequence[Sequence[int]],
                add: Sequence[Sequence[int]], amounts: Sequence[Sequence[float]], *, taps: Sequence[str] = (),
                freeze: Sequence[int] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raise NotImplementedError


def argmax(logits: Any) -> np.ndarray:
    raise NotImplementedError
