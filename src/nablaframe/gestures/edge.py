"""assert-this-edge (D50): does one name's set drive another's, and is the edge computed or only carried?

Spec: docs/specs/phase-6.md, D50 and "What each run declares". Built on ``sets.py``, with nothing
specific to a task (D29). Nothing here caches or judges: the gates ask for edges, and a record holds
what ``edge`` returns (D47).

* ``CARRIED = "carried"`` and ``COMPUTED = "computed"``.
* ``EdgeInstance``: a frozen dataclass ``(window, up, down, v, gv, w, gw, read, target)``: the
  held-out window's index; the upstream and downstream positions; the upstream value ``v`` and
  ``g(v)``; the other value ``w`` and ``g(w)``; and ``declaration.predict(dd, window, down, g(w))``'s
  read position and token, both None where it gives none.
* ``draw_edge_instances(up, down, relation, *, limit, rng)``: the instances, from two
  ``sets.SiteReading``s and a relation's name. With ``R = declaration.relation(relation)``, its
  places' descriptions ``du`` and ``dd``, ``Tu = up.table(R.up.cls)`` and
  ``Td = down.table(R.down.cls)``: window by window, each pair ``(pu, pd)`` of
  ``declaration.pairs(relation, window)``, in its order, is a candidate where

  - ``v``, ``du`` at ``pu``, has ``Tu.has(du, v)``
  - ``g(v) = declaration.value_map(relation, window, v)`` isn't None, ``Td.has(dd, g(v))``, and
    ``dd`` is ``g(v)`` at ``pd``
  - the downstream set for ``g(v)`` is active there: its codes in ``down.codes`` at ``pd`` sum
    above zero

  Its options are ``Tu.values(du)`` less the values ``taken`` in its window, ascending, each with a
  ``g(w)`` that isn't None, has ``Td.has(dd, g(w))`` and isn't ``g(v)``. Where
  ``declaration.predict(du, window, pu, w)`` gives a prediction for some option, the options are only
  those. The eligible candidates are those with an option. With more than ``limit``,
  ``sorted(rng.choice(len(eligible), size=limit, replace=False))`` keeps that many; then each kept
  candidate, in order, draws ``w = options[rng.integers(len(options))]``.

  ValueError unless both readings share their model, params and declaration, as the same objects,
  and have equal windows; for a relation the run doesn't declare; for a reading whose ``block`` isn't
  its place's; for a downstream block before the upstream one; for a description that doesn't count
  at its class; or for a ``limit`` that isn't a positive int.
* ``edge(up, down, relation, *, instances, seed)``: D50's measurements, a plain JSON mapping that
  ``records.GestureRecord`` takes as its results. ``rng = numpy.random.default_rng(seed)`` draws the
  instances with ``limit=instances``, then the null with ``sets.null_sets(Tu, du, the upstream codes
  at each instance's upstream position, vs, ws, rng)``. Passes run through ``sets.run_steered`` on
  ``up``, each instance at its upstream position, recording the downstream site's input tap; the
  downstream codes are ``core.encode`` of what that holds at each instance's downstream position,
  keeping latents below ``down.width``. With ``Sv`` and ``Sw`` the upstream sets of ``v`` and ``w``
  (``Tu.set_for``), and ``Dv`` and ``Dw`` the downstream sets of ``g(v)`` and ``g(w)``, a set's mass
  is its codes' sum, taken in float64. Seven passes run in one call, whose sequences are every
  instance for each pass in turn:

  - **unintervened:** nothing removed or added
  - **the swap:** ``Sv`` removed, and ``Sw`` added at ``Tu.mean_codes(du, w)``'s values for those
    latents
  - **the null's swap:** the null's latents removed, and its latents added at the first amounts of
    ``w``'s set
  - **the ablation:** ``Sv`` removed, nothing added; **the null's ablation:** the null's latents
    removed, nothing added
  - **the single swap:** ``Sv[0]`` removed, ``Sw[0]`` added at its amount; **the single ablation:**
    ``Sv[0]`` removed, nothing added

  The mapping:

  - ``"relation"``, ``"judged"`` (``R.judged``), ``"up"`` and ``"down"``, each ``{"description",
    "block", "class", "site", "tap"}`` with the site's output tap, ``"pairs"`` (every pair the
    declaration gave, over every window) and ``"instances"``.
  - ``"follows"``: the share of instances where, after the swap, ``Dw``'s mass exceeds ``Dv``'s.
    ``"null_follows"``: the same after the null's swap.
  - ``"kept"``: the median over instances of ``Dv``'s mass after the ablation over its mass in the
    unintervened pass. ``"null_kept"``: the same after the null's ablation.
  - ``"output"``: None where no instance has a ``read``; otherwise, over those instances,
    ``{"instances", "follows", "null_follows"}``: the shares where the output at ``read``
    (``sets.argmax`` of the logits there) is ``target`` after the swap and after the null's swap.
  - ``"residual_alone"``: None where no instance has ``up == down``; otherwise, over those
    instances, ``{"instances", "blocks", "follows", "follows_unfrozen", "output_follows"}``. The
    swap runs again, one call per position, ascending, over those instances at that position, with
    ``freeze`` the ``blocks``, those after ``R.up.block`` up to and including ``R.down.block``.
    ``follows`` is its share as above, ``follows_unfrozen`` the swap's over the same instances, and
    ``output_follows`` the frozen swap's output share, None where none of them has a ``read``.
  - ``"label"``: ``CARRIED`` where ``residual_alone`` isn't None and its ``follows`` is at least half
    its ``follows_unfrozen``; ``COMPUTED`` otherwise, as every edge across positions is.
  - ``"single"``: ``{"follows", "kept"}`` with each name's best latent in place of its set: after the
    single swap, the share where ``Dw[0]``'s code exceeds ``Dv[0]``'s; and after the single ablation,
    the median of ``Dv[0]``'s code over its unintervened code, over the instances where that is above
    zero, None where it is at none.

  With no instance, ``"instances"`` is 0 and every key from ``"follows"`` on is None. ValueError as
  for ``draw_edge_instances``, or for ``instances`` or ``seed`` that aren't a positive int and a
  non-negative int.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dictionaries import core
from . import sets

CARRIED = "carried"
COMPUTED = "computed"

_MEASURED = ("follows", "null_follows", "kept", "null_kept", "output", "residual_alone", "label", "single")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


@dataclass(frozen=True)
class EdgeInstance:
    window: int
    up: int
    down: int
    v: int
    gv: int
    w: int
    gw: int
    read: int | None
    target: int | None


def _check(up: Any, down: Any, relation: str) -> tuple[Any, sets.ClassTable, sets.ClassTable]:
    if up.model is not down.model or up.params is not down.params or up.declaration is not down.declaration:
        raise ValueError("both readings must share their model, params and declaration, as the same objects")
    if up.windows.shape != down.windows.shape or not np.array_equal(up.windows, down.windows):
        raise ValueError(f"both readings must read the same windows, got {up.windows.shape} and "
                         f"{down.windows.shape} windows that differ")
    R = up.declaration.relation(relation)  # ValueError for a relation the run doesn't declare
    for reading, place, which in ((up, R.up, "upstream"), (down, R.down, "downstream")):
        if reading.block != place.block:
            raise ValueError(f"relation {relation!r} reads its {which} place after block {place.block}, but the "
                             f"{which} reading is at {reading.output_tap!r} (block {reading.block})")
    if R.down.block < R.up.block:
        raise ValueError(f"relation {relation!r} has its downstream block {R.down.block} before its upstream "
                         f"block {R.up.block}")
    Tu, Td = up.table(R.up.cls), down.table(R.down.cls)
    for table, place, which in ((Tu, R.up, "upstream"), (Td, R.down, "downstream")):
        if place.description not in table.descriptions:
            raise ValueError(f"the {which} description {place.description!r} doesn't count at class "
                             f"{place.cls!r}")
    return R, Tu, Td


def _draw(up: Any, down: Any, relation: str, limit: Any,
          rng: np.random.Generator) -> tuple[Any, sets.ClassTable, sets.ClassTable, int, list[EdgeInstance]]:
    if not _is_int(limit) or limit < 1:
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    R, Tu, Td = _check(up, down, relation)
    declaration = up.declaration
    du, dd = R.up.description, R.down.description
    values = [int(u) for u in Tu.values(du)]
    held_up, held_down = up.described[du], down.described[dd]

    pairs = 0
    eligible = []  # (window, pu, pd, v, gv, options), options [(w, g(w)), ...] ascending in w
    for i, window in enumerate(up.windows):
        given = declaration.pairs(relation, window)
        pairs += len(given)
        images: list[tuple[int, int | None]] | None = None  # g of every value not taken here, once per window
        for pu, pd in given:
            pu, pd = int(pu), int(pd)
            v = int(held_up[i, pu])
            if not Tu.has(du, v):
                continue
            gv = declaration.value_map(relation, window, v)
            if gv is None or not Td.has(dd, gv) or held_down[i, pd] != gv:
                continue
            gv = int(gv)
            if not down.codes[i, pd, list(Td.set_for(dd, gv))].astype(np.float64).sum() > 0:
                continue
            if images is None:
                out = sets.taken(up.described, du, i)
                images = [(u, declaration.value_map(relation, window, u)) for u in values if u not in out]
            options = [(u, int(gu)) for u, gu in images if gu is not None and Td.has(dd, gu) and gu != gv]
            # Narrowing to the predicting options never empties a list, so whether a candidate is eligible
            # doesn't depend on predict; it is asked only for the candidates kept.
            if options:
                eligible.append((i, pu, pd, v, gv, options))
    if len(eligible) > limit:
        eligible = [eligible[k] for k in sorted(rng.choice(len(eligible), size=int(limit), replace=False))]
    instances = []
    for i, pu, pd, v, gv, options in eligible:
        window = up.windows[i]
        predicting = [o for o in options if declaration.predict(du, window, pu, o[0]) is not None]
        choices = predicting or options
        w, gw = choices[rng.integers(len(choices))]
        prediction = declaration.predict(dd, window, pd, gw)
        read, target = (None, None) if prediction is None else (int(prediction[0]), int(prediction[1]))
        instances.append(EdgeInstance(int(i), pu, pd, v, gv, int(w), gw, read, target))
    return R, Tu, Td, pairs, instances


def draw_edge_instances(up: Any, down: Any, relation: str, *, limit: int,
                        rng: np.random.Generator) -> list[EdgeInstance]:
    return _draw(up, down, relation, limit, rng)[4]


def _steer(up: Any, down: Any, chosen: Sequence[EdgeInstance], passes: Sequence[tuple[list, list, list]], *,
           freeze: Sequence[int] = ()) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every pass over every instance in one ``run_steered`` call: each pass's logits and downstream codes."""
    n, reps = len(chosen), len(passes)
    logits, recorded = sets.run_steered(
        up, [x.window for x in chosen] * reps, [x.up for x in chosen] * reps,
        [r for p in passes for r in p[0]], [a for p in passes for a in p[1]], [m for p in passes for m in p[2]],
        taps=(down.input_tap,), freeze=freeze)
    held = recorded[down.input_tap][np.arange(n * reps), [x.down for x in chosen] * reps]
    codes = core.encode(down.dictionary, down.site, held)[:, :down.width]
    return [(logits[j * n:(j + 1) * n], codes[j * n:(j + 1) * n]) for j in range(reps)]


def _mass(codes: np.ndarray, latents: np.ndarray) -> np.ndarray:
    return codes.astype(np.float64)[np.arange(len(codes))[:, None], latents].sum(axis=1)


def _share(hits: Any) -> float:
    return float(np.mean(hits))


def _outputs(logits: np.ndarray, chosen: Sequence[EdgeInstance], where: Sequence[int]) -> np.ndarray:
    """Whether the output at each instance's ``read`` is its ``target``, for the instances ``where``."""
    read = [chosen[k].read for k in where]
    target = np.array([chosen[k].target for k in where], dtype=np.int64)
    return sets.argmax(logits[list(where), read]) == target


def _place(reading: Any, place: Any) -> dict[str, Any]:
    return {"description": place.description, "block": int(place.block), "class": place.cls,
            "site": int(reading.site), "tap": reading.output_tap}


def edge(up: Any, down: Any, relation: str, *, instances: int, seed: int) -> dict[str, Any]:
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    rng = np.random.default_rng(int(seed))
    R, Tu, Td, pairs, chosen = _draw(up, down, relation, int(instances), rng)
    du, dd = R.up.description, R.down.description
    results: dict[str, Any] = {"relation": relation, "judged": bool(R.judged), "up": _place(up, R.up),
                               "down": _place(down, R.down), "pairs": int(pairs), "instances": len(chosen)}
    if not chosen:
        results.update({key: None for key in _MEASURED})
        return results

    n = len(chosen)
    index, ups = [x.window for x in chosen], [x.up for x in chosen]
    removed, added = sets.null_sets(Tu, du, up.codes[index, ups], [x.v for x in chosen], [x.w for x in chosen], rng)
    Sv = [list(Tu.set_for(du, x.v)) for x in chosen]
    Sw = [list(Tu.set_for(du, x.w)) for x in chosen]
    amounts = [[float(a) for a in Tu.mean_codes(du, x.w)[s]] for x, s in zip(chosen, Sw)]
    Dv = np.array([Td.set_for(dd, x.gv) for x in chosen], dtype=np.int64)
    Dw = np.array([Td.set_for(dd, x.gw) for x in chosen], dtype=np.int64)
    none: list[list[Any]] = [[] for _ in range(n)]
    first = lambda lists: [s[:1] for s in lists]  # noqa: E731
    swap = (Sv, Sw, amounts)
    (clean, clean_codes), (swapped, swapped_codes), (null, null_codes), (_, ablated_codes), \
        (_, null_ablated_codes), (_, single_codes), (_, single_ablated_codes) = _steer(up, down, chosen, [
            (none, none, none),
            swap,
            (removed, added, [a[:len(s)] for a, s in zip(amounts, added)]),
            (Sv, none, none),
            (removed, none, none),
            (first(Sv), first(Sw), first(amounts)),
            (first(Sv), none, none)])

    follows = _mass(swapped_codes, Dw) > _mass(swapped_codes, Dv)
    clean_mass = _mass(clean_codes, Dv)
    results["follows"] = _share(follows)
    results["null_follows"] = _share(_mass(null_codes, Dw) > _mass(null_codes, Dv))
    results["kept"] = float(np.median(_mass(ablated_codes, Dv) / clean_mass))
    results["null_kept"] = float(np.median(_mass(null_ablated_codes, Dv) / clean_mass))

    reading = [k for k, x in enumerate(chosen) if x.read is not None]
    results["output"] = None if not reading else {
        "instances": len(reading), "follows": _share(_outputs(swapped, chosen, reading)),
        "null_follows": _share(_outputs(null, chosen, reading))}

    same = [k for k, x in enumerate(chosen) if x.up == x.down]
    if not same:
        results["residual_alone"] = None
    else:
        blocks = list(range(R.up.block + 1, R.down.block + 1))
        frozen_follows, frozen_outputs = [], []
        for position in sorted({chosen[k].up for k in same}):
            group = [k for k in same if chosen[k].up == position]
            here = [chosen[k] for k in group]
            [(logits, codes)] = _steer(up, down, here, [tuple([s[k] for k in group] for s in swap)], freeze=blocks)
            frozen_follows += (_mass(codes, Dw[group]) > _mass(codes, Dv[group])).tolist()
            with_read = [j for j, x in enumerate(here) if x.read is not None]
            if with_read:
                frozen_outputs += _outputs(logits, here, with_read).tolist()
        results["residual_alone"] = {"instances": len(same), "blocks": blocks, "follows": _share(frozen_follows),
                                     "follows_unfrozen": _share(follows[same]),
                                     "output_follows": _share(frozen_outputs) if frozen_outputs else None}
    ra = results["residual_alone"]
    results["label"] = CARRIED if ra is not None and ra["follows"] >= 0.5 * ra["follows_unfrozen"] else COMPUTED

    rows = np.arange(n)
    best_v, best_w = Dv[:, 0], Dw[:, 0]
    base = clean_codes[rows, best_v].astype(np.float64)
    live = base > 0
    results["single"] = {
        "follows": _share(single_codes[rows, best_w].astype(np.float64) > single_codes[rows, best_v]),
        "kept": float(np.median(single_ablated_codes[rows, best_v][live].astype(np.float64) / base[live]))
        if live.any() else None}
    return results
