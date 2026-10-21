"""generalize-this (D52): where a name or an edge holds across held-out slices, dictionary seeds and the training run.

Spec: docs/specs/phase-6.md, D52. Built on ``sets.py``, ``name.py`` and ``edge.py``, with nothing
specific to a task (D29): the task's own readings at each checkpoint, such as a residual patch, come
from the caller through ``extras``. Nothing here caches or judges: the gates ask for generalize-this,
and a record holds what ``generalize_name`` or ``generalize_edge`` returns (D47).

* ``SLICES = 8``, ``HOLDS = 0.5`` and ``PROBE_READS = 0.9``.
* ``description_f1(table, sets_from, d)``: a description's F1 read again with given sets: the mean,
  over ``d``'s values at ``table`` that ``sets_from`` also has, of
  ``table.name_f1(d, v, sets_from.set_for(d, v))``, a float; None where there are none, or where
  ``d`` doesn't count at ``table``.
* ``steer_description(reading, cls, d, *, instances, rng, windows=None, sets_from=None)``: D49's
  steering for a description over all its values, each instance with its own value, or None where no
  instance is eligible. ``table = reading.table(cls)`` gives the amounts (``mean_codes``) and the
  null's shared and active latents; ``sets_from``, a ``ClassTable`` (``table`` by default), gives
  the sets and the null's top-8 exclusions (``ranked``). A value is usable where both tables have
  it.

  - Candidates: every (window, position), over ``windows`` (held-out window indices, ascending;
    every window by default) and each window's positions in the class's order, where ``d``'s value
    ``v`` is usable. Options: the usable values less those ``sets.taken`` in the window, ascending;
    predicting options: those for which ``declaration.predict(d, window, position, w)`` gives a
    prediction.
  - As in ``name.draw_instances``: ``causal`` is True where some candidate has a predicting option,
    and then the eligible candidates are those with one and draw from them; otherwise those with an
    option, drawing from all their options. More than ``instances`` eligible are cut with
    ``sorted(rng.choice(len(eligible), size=instances, replace=False))``; then each kept candidate,
    in order, draws ``w = choices[rng.integers(len(choices))]``.
  - The null: ``sets.null_sets`` with a table whose ``ranked`` is ``sets_from``'s and whose
    ``shared`` and ``active`` are ``table``'s, the codes at each instance's row, its ``v`` and ``w``,
    and ``rng``.
  - Three passes in one ``sets.run_steered`` call, every instance at its position for each pass in
    turn: unintervened; swapped, ``sets_from.set_for(d, v)`` removed and ``sets_from.set_for(d, w)``
    added at ``table.mean_codes(d, w)``'s values for those latents; the null, its latents removed and
    its added latents at the first amounts of ``w``'s set.

  Causal: ``{"causal": True, "instances", "follows", "null_follows"}``, the shares of instances whose
  output at the predicted position (``sets.argmax``) is the predicted token after the swap and after
  the null. Otherwise ``{"causal": False, "instances", "kl", "null_kl"}``, the mean
  ``dictionaries.edges.kl`` of the unintervened logits and the swapped, and the null's, at each
  instance's position. ValueError for a class the run doesn't declare, ``d`` not counting at it, or
  ``instances`` not a positive int.
* ``holds(f1, causal, follows)``: whether a name holds at a checkpoint: ``f1`` at least ``HOLDS``,
  and, where its steering is causal, ``follows`` at least ``HOLDS`` too; where the run predicts
  nothing for it (``causal`` False), ``f1`` alone decides; with no steering instance (``causal``
  None), or no ``f1``, it doesn't hold.
* ``envelope(rows)``: for checkpoint rows in step order, each with ``"step"``, ``"holds"`` and
  ``"probe"``: ``{"steps", "runs", "final", "gap", "probe_reads_from"}``. ``steps``: the steps whose
  row holds; ``runs``: ``[first, last]`` of each maximal run of consecutive rows that hold, in order;
  ``final``: the run that ends at the last row, None where the last row doesn't hold; ``gap``: the
  steps whose probe is at least ``PROBE_READS`` but whose row doesn't hold; ``probe_reads_from``: the
  first step whose probe is at least ``PROBE_READS``, or None. A probe of None reads nothing.
* ``generalize_name(reading, cls, d, *, instances, seed, steps, params_at, seed_readings=None,
  extras=None)``: D52 for the description ``d`` read over all its values, as a plain JSON mapping
  that ``records.GestureRecord`` takes as its results. ``reading`` is a ``sets.SiteReading`` at the
  run's final checkpoint, ``T = reading.table(cls)``; ``rng = numpy.random.default_rng(seed)`` is used
  by every ``steer_description`` below, in the order given.

  - ``"description"``, ``"class"``, ``"site"``, ``"tap"`` (the site's output tap) and ``"values"``
    (``d``'s values at ``T``).
  - ``"slices"``: the held-out windows split by ``numpy.array_split(range(B), SLICES)``. For each
    slice, ``{"windows", "f1", "follows", "null_follows"}``: its window count;
    ``description_f1`` of the ``ClassTable`` of the slice's codes and descriptions at the class's
    positions, with ``T``'s sets; and ``follows`` and ``null_follows`` from ``steer_description`` on
    ``reading`` over the slice's windows, None where it isn't causal or finds no instance. ``"slices"`` is
    ``{"rows", "f1", "follows"}``, the last two ``[least, greatest]`` over the rows that have one, or
    None.
  - ``"seeds"``: for each seed in ``seed_readings`` (a mapping from a dictionary's seed to a
    ``SiteReading`` of it at the same site, model, params and windows), ascending, ``{"seed", "f1",
    "follows", "null_follows", "cosine"}``. The name is derived again at that dictionary, never matched
    by latent id: ``f1`` is ``description_f1`` of its own table with its own sets, and the steering is
    ``steer_description`` on it, its ``follows`` None where it isn't causal or finds no instance.
    ``cosine`` is the mean, over the values both tables have, of the mean over that seed's set of each
    latent's highest cosine between decoder rows (``W_dec[unit(site)]``) and ``reading``'s set's; None
    where there are none. Where ``d`` doesn't count at the seed's class, all three are None and no
    steering is drawn.
  - ``"checkpoints"``: for each step in ``steps``, ascending, with ``params = params_at(step)`` and
    the reading ``sets.read_site`` gives at those params with ``reading``'s dictionary, site and
    windows (the final dictionaries, D44): ``{"step", "f1", "causal", "follows", "null_follows",
    "probe", "accuracy", "extras", "holds"}``.

    - ``f1``: ``description_f1`` of that reading's table with ``T``'s sets
    - the steering: ``steer_description`` on that reading with ``sets_from=T``, its ``causal``,
      ``follows`` and ``null_follows``, the last two None where it isn't causal; all three None where
      it finds no instance, or where ``d`` doesn't count at that reading's class, and then none is
      drawn
    - ``probe``: ``measure.probe_accuracy`` of the site's input tap values at the class's rows where
      ``d`` isn't -1, those rows' values, and the largest value plus one; None for fewer than two rows
    - ``accuracy``: ``measure.readout_loss``'s accuracy over the windows at the class's positions,
      from the unintervened logits
    - ``extras``: ``extras(step, params)`` as given, a mapping of plain JSON values; None without
      ``extras``
    - ``holds``: ``holds(f1, causal, follows)``
  - ``"envelope"``: ``envelope`` of the checkpoint rows.

  ValueError for ``d`` not counting at the class; ``steps`` that aren't non-empty, strictly
  ascending non-negative ints; a seed reading that doesn't share ``reading``'s model, params and
  declaration as the same objects, its windows and its site's taps; or ``instances`` or ``seed`` that
  aren't a positive int and a non-negative int.
* ``generalize_edge(up, down, relation, *, instances, seed, steps, params_at, seed_readings=None)``:
  D52 for an edge, a plain JSON mapping. Each reading below is measured with
  ``edge.edge(..., instances=instances, seed=seed)``, and its row holds ``"instances"``,
  ``"follows"``, ``"null_follows"``, ``"kept"``, ``"null_kept"`` and ``"label"`` from it.

  - ``"relation"``.
  - ``"slices"``: for each slice as above, ``sets.read_site`` of both sites over the slice's windows,
    with ``up``'s model and params; the row adds ``"windows"``.
  - ``"seeds"``: for each seed in ``seed_readings`` (a mapping from a seed to its ``(up, down)``
    readings), ascending; the row adds ``"seed"``.
  - ``"checkpoints"``: for each step in ``steps``, ascending, both sites read at ``params_at(step)``
    with the final dictionaries; the row adds ``"step"``.

  ValueError as ``edge.edge`` raises it, for ``steps`` as above, or for a seed's readings that don't
  share ``up``'s model, params, declaration and windows and their sites' taps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..dictionaries import measure
from ..dictionaries.edges import kl
from . import edge as edges
from . import sets

SLICES = 8
HOLDS = 0.5
PROBE_READS = 0.9

_EDGE_KEYS = ("instances", "follows", "null_follows", "kept", "null_kept", "label")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


class _NullTable:
    """What ``sets.null_sets`` reads: ``ranked`` from the table the sets come from, ``shared`` and ``active`` from
    the table being steered."""

    def __init__(self, table: sets.ClassTable, sets_from: sets.ClassTable) -> None:
        self.shared = table.shared
        self.active = table.active
        self.ranked = sets_from.ranked


def description_f1(table: Any, sets_from: Any, d: str) -> float | None:
    if not isinstance(d, str) or d not in table.descriptions:
        return None
    shared = [int(v) for v in table.values(d) if sets_from.has(d, int(v))]
    if not shared:
        return None
    return float(np.mean([table.name_f1(d, v, sets_from.set_for(d, v)) for v in shared]))


def steer_description(reading: Any, cls: str, d: str, *, instances: int, rng: np.random.Generator,
                      windows: Sequence[int] | None = None, sets_from: Any = None) -> dict[str, Any] | None:
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    table = reading.table(cls)  # ValueError for a class the run doesn't declare
    if not isinstance(d, str) or d not in table.descriptions:
        raise ValueError(f"{d!r} doesn't count at class {cls!r}; {list(table.descriptions)} do")
    sets_from = table if sets_from is None else sets_from
    batch = len(reading.windows)
    windows = list(range(batch)) if windows is None else list(windows)
    if not all(_is_int(i) and 0 <= i < batch for i in windows) or any(a >= b for a, b in zip(windows, windows[1:])):
        raise ValueError(f"windows must be ascending distinct ints in range({batch}), got {windows!r}")
    declaration = reading.declaration
    usable = [int(u) for u in table.values(d) if sets_from.has(d, int(u))]
    is_usable = set(usable)
    held = reading.described[d]

    def predictions(i: int, p: int, options: list[int], *, first: bool) -> dict[int, tuple[int, int]]:
        found = {}
        for u in options:
            prediction = declaration.predict(d, reading.windows[i], p, u)
            if prediction is not None:
                found[u] = (int(prediction[0]), int(prediction[1]))
                if first:
                    break
        return found

    candidates = []  # (window, position, v, options, whether some option predicts)
    for i in windows:
        i = int(i)
        out = None
        for p in declaration.classes[cls]:
            v = int(held[i, p])
            if v not in is_usable:
                continue
            if out is None:
                out = sets.taken(reading.described, d, i)
                options = [u for u in usable if u not in out]
            candidates.append((i, int(p), v, options, bool(options) and bool(predictions(i, p, options, first=True))))
    causal = any(c[4] for c in candidates)
    eligible = [c for c in candidates if (c[4] if causal else c[3])]
    if not eligible:
        return None
    if len(eligible) > instances:
        eligible = [eligible[k] for k in sorted(rng.choice(len(eligible), size=int(instances), replace=False))]
    chosen = []  # (window, position, v, w, read, target)
    for i, p, v, options, _ in eligible:
        if causal:
            predicted = predictions(i, p, options, first=False)
            choices = [u for u in options if u in predicted]
        else:
            choices = options
        w = int(choices[rng.integers(len(choices))])
        read, target = predicted[w] if causal else (None, None)
        chosen.append((i, p, v, w, read, target))

    n = len(chosen)
    index = [x[0] for x in chosen]
    positions = [x[1] for x in chosen]
    removed, added = sets.null_sets(_NullTable(table, sets_from), d, reading.codes[index, positions],
                                    [x[2] for x in chosen], [x[3] for x in chosen], rng)
    Sv = [list(sets_from.set_for(d, x[2])) for x in chosen]
    Sw = [list(sets_from.set_for(d, x[3])) for x in chosen]
    amounts = [[float(a) for a in table.mean_codes(d, x[3])[s]] for x, s in zip(chosen, Sw)]
    none: list[list[Any]] = [[] for _ in range(n)]
    logits, _ = sets.run_steered(reading, index * 3, positions * 3, none + Sv + removed, none + Sw + added,
                                 none + amounts + [a[:len(s)] for a, s in zip(amounts, added)])
    clean, swapped, null = (logits[j * n:(j + 1) * n] for j in range(3))
    k = np.arange(n)
    if causal:
        read = [x[4] for x in chosen]
        target = np.array([x[5] for x in chosen], dtype=np.int64)
        return {"causal": True, "instances": n,
                "follows": float(np.mean(sets.argmax(swapped[k, read]) == target)),
                "null_follows": float(np.mean(sets.argmax(null[k, read]) == target))}
    return {"causal": False, "instances": n, "kl": float(kl(clean[k, positions], swapped[k, positions])),
            "null_kl": float(kl(clean[k, positions], null[k, positions]))}


def holds(f1: float | None, causal: bool | None, follows: float | None) -> bool:
    if f1 is None or causal is None or not f1 >= HOLDS:
        return False
    if causal:
        return follows is not None and follows >= HOLDS
    return True


def envelope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = [int(r["step"]) for r in rows]
    held = [bool(r["holds"]) for r in rows]
    reads = [r["probe"] is not None and r["probe"] >= PROBE_READS for r in rows]
    runs: list[list[int]] = []
    for j, (step, h) in enumerate(zip(steps, held)):
        if not h:
            continue
        if j > 0 and held[j - 1]:
            runs[-1][1] = step
        else:
            runs.append([step, step])
    return {"steps": [s for s, h in zip(steps, held) if h], "runs": runs,
            "final": list(runs[-1]) if held and held[-1] else None,
            "gap": [s for s, h, r in zip(steps, held, reads) if r and not h],
            "probe_reads_from": next((s for s, r in zip(steps, reads) if r), None)}


def _check_steps(steps: Any) -> list[int]:
    steps = list(steps)
    if not steps or not all(_is_int(s) and s >= 0 for s in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError(f"steps must be non-empty, strictly ascending non-negative ints, got {steps!r}")
    return [int(s) for s in steps]


def _check_counts(instances: Any, seed: Any) -> None:
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")


def _check_alike(reading: Any, other: Any, what: str) -> None:
    if (other.model is not reading.model or other.params is not reading.params
            or other.declaration is not reading.declaration):
        raise ValueError(f"{what} must share the model, params and declaration, as the same objects")
    if other.windows.shape != reading.windows.shape or not np.array_equal(other.windows, reading.windows):
        raise ValueError(f"{what} must read the same windows")
    if (other.input_tap, other.output_tap) != (reading.input_tap, reading.output_tap):
        raise ValueError(f"{what} reads {other.input_tap!r} to {other.output_tap!r}, not "
                         f"{reading.input_tap!r} to {reading.output_tap!r}")


def _parts(reading: Any) -> list[np.ndarray]:
    return np.array_split(np.arange(len(reading.windows)), SLICES)


def _span(values: list[float | None]) -> list[float] | None:
    have = [v for v in values if v is not None]
    return [min(have), max(have)] if have else None


def _steered(steering: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if steering is None or not steering["causal"]:
        return None, None
    return steering["follows"], steering["null_follows"]


def _decoders(reading: Any) -> np.ndarray:
    dictionary = reading.dictionary
    return np.asarray(dictionary.arrays["W_dec"][dictionary.unit(reading.site)], dtype=np.float64)


def _cosine(reading: Any, T: sets.ClassTable, other: Any, Ts: sets.ClassTable, d: str) -> float | None:
    both = [int(v) for v in Ts.values(d) if T.has(d, int(v))] if d in Ts.descriptions else []
    if not both:
        return None
    mine, theirs = _decoders(reading), _decoders(other)
    unit = lambda rows: rows / np.linalg.norm(rows, axis=1, keepdims=True)  # noqa: E731
    means = []
    for v in both:
        cos = unit(theirs[list(Ts.set_for(d, v))]) @ unit(mine[list(T.set_for(d, v))]).T
        means.append(float(cos.max(axis=1).mean()))
    return float(np.mean(means))


def _checkpoint(reading: Any, T: sets.ClassTable, cls: str, d: str, step: int, params: Any, *, instances: int,
                rng: np.random.Generator, extras: Callable[[int, Any], Mapping[str, Any]] | None) -> dict[str, Any]:
    at = sets.read_site(reading.model, params, n_layers=reading.n_layers, declaration=reading.declaration,
                        dictionary=reading.dictionary, site=reading.site, windows=reading.windows)
    table = at.table(cls)
    causal = follows = null_follows = None
    if d in table.descriptions:
        steering = steer_description(at, cls, d, instances=instances, rng=rng, sets_from=T)
        if steering is not None:
            causal = steering["causal"]
            follows, null_follows = _steered(steering)

    positions = list(reading.declaration.classes[cls])
    logits, taps = [], []
    for start in range(0, len(at.tokens), 256):
        out, recorded = reading.model.instrumented(params, at.tokens[start:start + 256], [at.input_tap])
        logits.append(np.asarray(getattr(out, "array", out), dtype=np.float32))
        value = recorded[at.input_tap]
        taps.append(np.asarray(getattr(value, "array", value), dtype=np.float32))
    logits, x = np.concatenate(logits, axis=0), np.concatenate(taps, axis=0)
    held = at.described[d][:, positions].reshape(-1)
    rows = x[:, positions].reshape(len(held), -1)[held != -1]
    values = held[held != -1]
    probe = measure.probe_accuracy(rows, values, int(values.max()) + 1) if len(values) >= 2 else None
    f1 = description_f1(table, T, d)
    return {"step": step, "f1": f1, "causal": causal, "follows": follows, "null_follows": null_follows,
            "probe": probe, "accuracy": measure.readout_loss(logits, at.windows, positions)[1],
            "extras": None if extras is None else dict(extras(step, params)), "holds": holds(f1, causal, follows)}


def generalize_name(reading: Any, cls: str, d: str, *, instances: int, seed: int, steps: Sequence[int],
                    params_at: Callable[[int], Any], seed_readings: Mapping[int, Any] | None = None,
                    extras: Callable[[int, Any], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    _check_counts(instances, seed)
    T = reading.table(cls)  # ValueError for a class the run doesn't declare
    if not isinstance(d, str) or d not in T.descriptions:
        raise ValueError(f"{d!r} doesn't count at class {cls!r}; {list(T.descriptions)} do")
    steps = _check_steps(steps)
    seed_readings = dict(seed_readings or {})
    for s, other in seed_readings.items():
        _check_alike(reading, other, f"the reading of seed {s!r}")
    instances = int(instances)
    rng = np.random.default_rng(int(seed))
    results: dict[str, Any] = {"description": d, "class": cls, "site": int(reading.site), "tap": reading.output_tap,
                               "values": len(T.values(d))}

    positions = reading.declaration.classes[cls]
    rows = []
    for part in _parts(reading):
        Ts = sets.ClassTable(reading.codes[part], {k: v[part] for k, v in reading.described.items()}, positions)
        follows, null_follows = _steered(steer_description(reading, cls, d, instances=instances, rng=rng,
                                                           windows=part.tolist()))
        rows.append({"windows": len(part), "f1": description_f1(Ts, T, d), "follows": follows,
                     "null_follows": null_follows})
    results["slices"] = {"rows": rows, "f1": _span([x["f1"] for x in rows]),
                         "follows": _span([x["follows"] for x in rows])}

    results["seeds"] = []
    for s in sorted(seed_readings):
        other = seed_readings[s]
        Ts = other.table(cls)
        f1 = follows = null_follows = None
        if d in Ts.descriptions:
            f1 = description_f1(Ts, Ts, d)
            follows, null_follows = _steered(steer_description(other, cls, d, instances=instances, rng=rng))
        results["seeds"].append({"seed": int(s), "f1": f1, "follows": follows, "null_follows": null_follows,
                                 "cosine": _cosine(reading, T, other, Ts, d)})

    results["checkpoints"] = [_checkpoint(reading, T, cls, d, step, params_at(step), instances=instances, rng=rng,
                                          extras=extras) for step in steps]
    results["envelope"] = envelope(results["checkpoints"])
    return results


def generalize_edge(up: Any, down: Any, relation: str, *, instances: int, seed: int, steps: Sequence[int],
                    params_at: Callable[[int], Any], seed_readings: Mapping[int, Any] | None = None) -> dict[str, Any]:
    _check_counts(instances, seed)
    edges._check(up, down, relation)  # ValueError as edge.edge raises it
    steps = _check_steps(steps)
    seed_readings = dict(seed_readings or {})
    for s, pair in seed_readings.items():
        seed_up, seed_down = pair
        _check_alike(up, seed_up, f"the upstream reading of seed {s!r}")
        _check_alike(down, seed_down, f"the downstream reading of seed {s!r}")

    def measured(a: Any, b: Any) -> dict[str, Any]:
        result = edges.edge(a, b, relation, instances=int(instances), seed=int(seed))
        return {k: result[k] for k in _EDGE_KEYS}

    def read(params: Any, reading: Any, windows: np.ndarray) -> Any:
        return sets.read_site(up.model, params, n_layers=reading.n_layers, declaration=reading.declaration,
                              dictionary=reading.dictionary, site=reading.site, windows=windows)

    results: dict[str, Any] = {"relation": relation}
    results["slices"] = [{"windows": len(part), **measured(read(up.params, up, up.windows[part]),
                                                           read(up.params, down, up.windows[part]))}
                         for part in _parts(up)]
    results["seeds"] = [{"seed": int(s), **measured(*seed_readings[s])} for s in sorted(seed_readings)]
    results["checkpoints"] = []
    for step in steps:
        params = params_at(step)
        results["checkpoints"].append({"step": step, **measured(read(params, up, up.windows),
                                                                read(params, down, up.windows))})
    return results
