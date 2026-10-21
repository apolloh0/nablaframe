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

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..dictionaries import measure
from ..dictionaries.core import Dictionary, content_hash, encode
from ..interventions.apply import run_intervened
from ..interventions.spec import InterventionSpec, array_hash, edit
from . import declare

SET_SIZE = 4
EXCLUDE = 8
SHARED = 0.9
FLOOR = 0.5
CANDIDATES = 5

_BLOCK = re.compile(r"blocks\.(0|[1-9][0-9]*)\.")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


class ClassTable:
    rows: int
    active: np.ndarray
    shared: tuple[int, ...]
    descriptions: tuple[str, ...]

    def __init__(self, codes: Any, described: Mapping[str, Any], positions: Any) -> None:
        codes = np.asarray(codes, dtype=np.float32)
        if codes.ndim != 3:
            raise ValueError(f"codes must be (B, seq, m), got shape {codes.shape}")
        B, seq, m = codes.shape
        positions = list(positions)
        if not all(_is_int(p) for p in positions) or len(set(int(p) for p in positions)) != len(positions):
            raise ValueError(f"positions must be distinct ints, got {positions!r}")
        positions = [int(p) for p in positions]
        if any(not 0 <= p < seq for p in positions):
            raise ValueError(f"positions must be in range({seq}), got {positions!r}")
        values: dict[str, np.ndarray] = {}
        for d, arr in described.items():
            arr = np.asarray(arr)
            if arr.shape != (B, seq) or not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"description {d!r} must be an int array shaped {(B, seq)}, got {arr.dtype} "
                                 f"{arr.shape}")
            values[d] = arr.astype(np.int64)
        self._m = m
        self._codes = codes[:, positions].reshape(-1, m)  # rows: window by window, positions in the order given
        self._on = self._codes > 0
        self.rows = len(self._codes)
        self.active = self._on.sum(axis=0).astype(np.int64)
        share = self.active / self.rows if self.rows else np.zeros(m)
        self.shared = tuple(int(i) for i in np.flatnonzero(share >= SHARED))
        self.descriptions = tuple(d for d, arr in values.items() if declare.counts_at(arr, positions))
        self._rows_of = {d: values[d][:, positions].reshape(-1) for d in self.descriptions}
        self._values = {d: declare.values_at(values[d], positions) for d in self.descriptions}
        self._f1: dict[str, np.ndarray] = {}

    def _column(self, d: Any, v: Any) -> int:
        if not self.has(d, v):
            raise ValueError(f"{d!r} = {v!r} isn't a value of a description that counts at this class")
        return int(np.searchsorted(self._values[d], int(v)))

    def has(self, d: Any, v: Any) -> bool:
        if not isinstance(d, str) or d not in self._values or not _is_int(v):
            return False
        return bool(np.isin(int(v), self._values[d]))

    def values(self, d: str) -> np.ndarray:
        if not isinstance(d, str) or d not in self._values:
            raise ValueError(f"{d!r} doesn't count at this class; {list(self.descriptions)} do")
        return self._values[d].copy()

    def f1(self, d: str) -> np.ndarray:
        values = self.values(d)
        if d not in self._f1:
            # Each value's rows, grouped in value order, so each latent's count there is one sum per group.
            held = self._rows_of[d]
            rows = np.flatnonzero(np.isin(held, values))
            rows = rows[np.argsort(held[rows], kind="stable")]
            counts = np.searchsorted(held[rows], values, side="right") - np.searchsorted(held[rows], values)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])  # every value has at least MIN_COUNT rows
            both = np.add.reduceat(self._on[rows].astype(np.int64), starts, axis=0).T  # (m, values)
            self._f1[d] = 2 * both / (self.active[:, None] + counts.astype(np.int64)[None, :])
        return self._f1[d].copy()

    def candidates(self, latent: Any) -> tuple[tuple[str, int, float], ...]:
        if not _is_int(latent) or not 0 <= latent < self._m:
            raise ValueError(f"latent must be an int in range({self._m}), got {latent!r}")
        latent = int(latent)
        if self.active[latent] == 0:
            return ()
        everything = []
        for i, d in enumerate(self.descriptions):
            row = self._f1_of(d)[latent]
            everything += [(-float(f), i, int(v), d) for f, v in zip(row, self._values[d])]
        everything.sort(key=lambda item: item[:3])
        return tuple((d, v, -f) for f, _, v, d in everything[:CANDIDATES])

    def _f1_of(self, d: str) -> np.ndarray:
        if d not in self._f1:
            self.f1(d)
        return self._f1[d]

    def ranked(self, d: str, v: int) -> np.ndarray:
        column = self._column(d, v)
        return np.argsort(-self._f1_of(d)[:, column], kind="stable").astype(np.int64)

    def set_for(self, d: str, v: int) -> tuple[int, ...]:
        return tuple(int(i) for i in self.ranked(d, v)[:SET_SIZE])

    def name_f1(self, d: str, v: int, latents: Sequence[int]) -> float:
        column = self._column(d, v)
        latents = list(latents)
        if not latents or not all(_is_int(i) and 0 <= i < self._m for i in latents):
            raise ValueError(f"latents must be one or more ints in range({self._m}), got {latents!r}")
        return float(self._f1_of(d)[[int(i) for i in latents], column].max())

    def mean_codes(self, d: str, v: int) -> np.ndarray:
        self._column(d, v)
        where = self._rows_of[d] == int(v)
        return self._codes[where].astype(np.float64).mean(axis=0).astype(np.float32)


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

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)
        self._tables: dict[str, ClassTable] = {}

    def table(self, cls: str) -> ClassTable:
        if not isinstance(cls, str) or cls not in self.declaration.classes:
            raise ValueError(f"run {self.declaration.run!r} declares no class {cls!r}; it declares "
                             f"{list(self.declaration.classes)}")
        if cls not in self._tables:
            self._tables[cls] = ClassTable(self.codes, self.described, self.declaration.classes[cls])
        return self._tables[cls]


def read_site(model: Any, params: Any, *, n_layers: int, declaration: Any, dictionary: Any, site: int, windows: Any,
              chunk: int = 256) -> SiteReading:
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    if not isinstance(dictionary, Dictionary):
        raise ValueError(f"dictionary must be a Dictionary, got {type(dictionary).__name__}")
    site = dictionary._site(site)  # ValueError for a site the dictionary doesn't have
    seq = declaration.seq
    arr = np.asarray(windows)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != seq + 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"windows must be an integer array (B, {seq + 1}) with B at least 1, got {arr.dtype} "
                         f"{arr.shape}")
    windows = arr.astype(np.int64)
    tokens = windows[:, :seq].astype(np.int32)
    input_tap, output_tap = dictionary.sites[site]
    match = _BLOCK.match(output_tap)
    width = dictionary.active_width(site)
    parts = []
    for start in range(0, len(tokens), int(chunk)):
        x = measure.activations(model, params, tokens[start:start + int(chunk)], [input_tap], chunk=chunk)[input_tap]
        n = x.shape[0]
        parts.append(encode(dictionary, site, x.reshape(n * seq, -1))[:, :width].reshape(n, seq, width))
    described = declaration.describe(windows)
    order = list(declare.DESCRIPTIONS) + list(declaration.variables)
    return SiteReading(model=model, params=params, n_layers=n_layers, declaration=declaration, dictionary=dictionary,
                       site=site, windows=windows, tokens=tokens, input_tap=input_tap, output_tap=output_tap,
                       block=int(match.group(1)) if match else None, width=width,
                       codes=np.concatenate(parts, axis=0).astype(np.float32),
                       described={d: described[d] for d in order})


def taken(described: Mapping[str, np.ndarray], d: str, window: int) -> frozenset[int]:
    if not isinstance(d, str) or d not in described:
        raise ValueError(f"no description {d!r}; there are {list(described)}")
    source = described["token(p)"] if d in declare.DESCRIPTIONS else described[d]
    if not _is_int(window) or not 0 <= window < len(source):
        raise ValueError(f"window must be an int in range({len(source)}), got {window!r}")
    return frozenset(int(t) for t in np.asarray(source[int(window)]).tolist()) - frozenset({-1})


def _draw(rng: np.random.Generator, eligible: list[int]) -> list[int]:
    if not eligible:
        return []
    return [int(i) for i in rng.choice(eligible, size=min(SET_SIZE, len(eligible)), replace=False)]


def null_sets(table: ClassTable, d: str, codes: Any, vs: Sequence[int], ws: Sequence[int],
              rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]]]:
    codes = np.asarray(codes)
    if codes.ndim != 2 or not len(codes) == len(vs) == len(ws):
        raise ValueError(f"codes (n, m), vs and ws must have one length, got {codes.shape}, {len(vs)} and {len(ws)}")
    shared = set(table.shared)
    pool = set(np.flatnonzero(table.active > 0).tolist())
    removed, added = [], []
    for row, v, w in zip(codes, vs, ws):
        live = set(np.flatnonzero(row > 0).tolist())
        removed.append(_draw(rng, sorted(live - set(table.ranked(d, v)[:EXCLUDE].tolist()) - shared)))
        added.append(_draw(rng, sorted(pool - set(table.ranked(d, w)[:EXCLUDE].tolist()) - shared - live)))
    return removed, added


def _rows(positions: Sequence[int], latents: Sequence[Sequence[int]]) -> np.ndarray:
    rows = [[k, p, int(latent)] for k, (p, chosen) in enumerate(zip(positions, latents)) for latent in chosen]
    return np.array(rows, dtype=np.int32).reshape(-1, 3)


def run_steered(reading: SiteReading, index: Sequence[int], positions: Sequence[int], remove: Sequence[Sequence[int]],
                add: Sequence[Sequence[int]], amounts: Sequence[Sequence[float]], *, taps: Sequence[str] = (),
                freeze: Sequence[int] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(index)
    if not n == len(positions) == len(remove) == len(add) == len(amounts):
        raise ValueError(f"index, positions, remove, add and amounts must have one length, got {len(index)}, "
                         f"{len(positions)}, {len(remove)}, {len(add)} and {len(amounts)}")
    if n == 0:
        raise ValueError("run_steered needs at least one instance")
    batch = len(reading.tokens)
    if not all(_is_int(i) and 0 <= i < batch for i in index):
        raise ValueError(f"every index must be an int in range({batch}), got {list(index)!r}")
    if not all(_is_int(p) for p in positions):
        raise ValueError(f"every position must be an int, got {list(positions)!r}")
    positions = [int(p) for p in positions]
    for k, (latents, values) in enumerate(zip(add, amounts)):
        if len(latents) != len(values):
            raise ValueError(f"instance {k} adds {len(latents)} latents with {len(values)} amounts")
    freeze = sorted(int(l) for l in freeze)
    if freeze and len(set(positions)) != 1:
        raise ValueError(f"freezing blocks needs every instance at one position, got {sorted(set(positions))}")

    tokens = np.ascontiguousarray(reading.tokens[[int(i) for i in index]], dtype=np.int32)
    arrays = {"remove": _rows(positions, remove), "add": _rows(positions, add),
              "amounts": np.array([float(a) for values in amounts for a in values], dtype=np.float32)}
    hashes = {name: array_hash(value) for name, value in arrays.items()}
    store: dict[str, Any] = {content_hash(reading.dictionary): reading.dictionary, batch_hash(tokens): tokens}
    store.update({hashes[name]: value for name, value in arrays.items()})
    edits = [edit(reading.output_tap, sorted(set(positions)), "dsteer", dictionary=content_hash(reading.dictionary),
                  **hashes)]
    for l in freeze:
        for tap in (f"blocks.{l}.hook_attn_out", f"blocks.{l}.hook_mlp_out"):
            edits.append(edit(tap, [positions[0]], "patch", source=batch_hash(tokens)))
    logits, recorded = run_intervened(reading.model, reading.params, tokens, InterventionSpec(tuple(edits)),
                                      n_layers=reading.n_layers, resolve=store.__getitem__, taps=taps)
    return _array(logits), {tap: _array(value) for tap, value in recorded.items()}


def _array(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "array", value), dtype=np.float32)


def argmax(logits: Any) -> np.ndarray:
    return np.argmax(np.asarray(logits), axis=-1).astype(np.int64)  # the first of equal maxima: the lower token
