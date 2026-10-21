"""name-this-direction (D49): what a latent stands for, named only from declared descriptions and held with a set of four.

Spec: docs/specs/phase-6.md, D49 and "What each run declares". Built on ``sets.py``, with nothing
specific to a task (D29). Nothing here caches or judges: the gates ask for names, and a record holds
what ``name`` returns (D47).

* ``NAMED = "named"``, ``WEAK = "weak"``, ``EXAMPLES = 8``, ``TOP_TOKENS = 5`` and
  ``NO_PREDICTION = "No causal test applies: the run declares no prediction for this description
  at these instances."``
* ``latent_effects(dictionary, site, params, latents)``: float64 (len(latents), vocab), each
  latent's direct effect on the logits, as the features view computes it (D45):
  ``e = decode_delta(dictionary, site, one-hot)`` taken to float64, then
  ``((e - mean(e)) * ln_f.weight) @ wte.T``.
* ``ends(effect, value)``: ``{"top", "bottom", "rank_from_top", "rank_from_bottom"}`` for a float64
  ``effect`` over the vocabulary. ``top`` holds the ``TOP_TOKENS`` largest, largest first, and
  ``bottom`` the smallest, smallest first, each ``[token, effect]``, ties to the lower token.
  ``rank_from_top`` is ``value``'s place, from 0, with every token ordered largest first, ties to the
  lower token; ``rank_from_bottom`` the same with every token ordered smallest first. ValueError for
  a ``value`` outside the vocabulary.
* ``coverage(reading, latents, value)``: for each description in ``reading.described``, in order,
  the share of the latents' summed codes over every held-out position (every window, every position)
  that falls where the description is ``value``, summed in float64; every share None where the
  latents never fire.
* ``Instance``: a frozen dataclass ``(window, position, w, read, target)``: the held-out window's
  index, the position the edit is at, the other value ``w``, and the position the output is read at
  and the token the run predicts there when ``d`` is moved to ``w``, both None where no causal test
  applies.
* ``draw_instances(reading, cls, d, v, *, limit, rng)``: ``(causal, instances)``. Every
  (window, position) pair where ``d`` is ``v``, window by window and each window's positions in the
  class's order, is a candidate. Its options are ``table.values(d)`` less the values ``taken`` in its
  window, ascending; its predicting options are those for which ``declaration.predict(d, window,
  position, w)`` gives ``(read, target)``.

  - ``causal`` is True where some candidate has a predicting option: the eligible candidates are
    those with one, and each draws from its predicting options. Otherwise the eligible candidates
    are those with an option, and each draws from its options.
  - With more than ``limit`` eligible, ``sorted(rng.choice(len(eligible), size=limit,
    replace=False))`` keeps that many. Then each kept candidate, in order, draws
    ``w = choices[rng.integers(len(choices))]``.

  ValueError unless ``reading.table(cls).has(d, v)``, or for a ``limit`` that isn't a positive int.
* ``steer(reading, cls, d, v, *, instances, rng)``: D49's step 6 for the name ``d = v``, or None
  where ``draw_instances`` finds no instance. Its instances are drawn with ``limit=instances``, then
  the null with ``sets.null_sets(table, d, the codes at each instance's row, vs, ws, rng)``. Four
  passes run in one ``sets.run_steered`` call, whose sequences are every instance, each at its
  position, for each pass in turn:

  - **unintervened:** nothing removed or added
  - **swapped:** ``table.set_for(d, v)`` removed, and ``table.set_for(d, w)`` added at
    ``table.mean_codes(d, w)``'s values for those latents
  - **null:** the null's latents removed, and its added latents at the first amounts of ``w``'s set
  - **ablated:** ``v``'s set removed, nothing added

  Causal: ``{"causal": True, "instances", "follows", "unchanged", "ablated_changed",
  "null_follows", "null_unchanged"}``. A pass's output is ``sets.argmax`` of its logits at the
  instance's ``read``. ``follows`` is the share of instances where the swapped output is ``target``,
  ``unchanged`` where it is the unintervened output; ``ablated_changed`` the share where the ablated
  output isn't; ``null_follows`` and ``null_unchanged`` as for the swap.

  Otherwise: ``{"causal": False, "note": NO_PREDICTION, "instances", "kl", "null_kl", "examples"}``.
  ``kl`` is ``dictionaries.edges.kl`` of the unintervened and the swapped logits at each instance's
  position, a mean over instances; ``null_kl`` the same for the null. ``examples``: for the first
  ``EXAMPLES`` instances, ``{"window", "position", "w", "unsteered", "steered"}``, the last two the
  ``TOP_TOKENS`` tokens with the largest logits there, largest first, ties to the lower token.
* ``name(reading, cls, latent, *, instances, seed)``: D49's six steps for ``latent``, a plain JSON
  mapping that ``records.GestureRecord`` takes as its results. ``rng =
  numpy.random.default_rng(seed)``; ``table = reading.table(cls)``.

  - ``"class"``, ``"latent"``, ``"windows"`` (the held-out windows' count) and ``"rows"``
    (``table.rows``).
  - ``"candidates"``: ``[{"description", "value", "f1"}, ...]``, ``table.candidates(latent)``.
  - ``"name"``: None where there is no candidate, the latent firing on no row of the class, and then
    the mapping holds nothing more. Otherwise the top candidate, ``{"description", "value", "f1",
    "label"}``, labelled ``NAMED`` where its F1 is at least ``sets.FLOOR`` and ``WEAK`` below it.
    Every step below runs for both, with ``d`` and ``v`` the name's description and value.
  - ``"set"``: ``{"latents", "f1", "name_f1", "pointed_rank", "in_set"}``: ``table.set_for(d, v)``,
    each one's F1 for ``v``, the highest of those, ``latent``'s place in ``table.ranked(d, v)`` from
    0, and whether that place is below ``sets.SET_SIZE``.
  - ``"coverage"``: ``{"latent": coverage(reading, [latent], v), "set": coverage(reading, the set,
    v)}``.
  - ``"effect"``: ``{"set", "latent", "vocabulary"}``. ``set`` is ``ends`` of the set's effect, the
    sum over the set of ``table.mean_codes(d, v)[l]`` times ``latent_effects[l]``: weighted by its
    mean codes where the name holds. ``latent`` is ``{"latent": latent}`` with ``ends`` of the
    latent's own effect, labelled as that latent's, never as the name's. ``vocabulary`` is the
    vocabulary's size.
  - ``"steering"``: ``steer(reading, cls, d, v, instances=instances, rng=rng)``.

  ValueError for a latent outside ``range(reading.width)``, a class the run doesn't declare or at
  which no description counts, or ``instances`` or ``seed`` that aren't a positive int and a
  non-negative int.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dictionaries import core
from ..dictionaries.edges import kl
from . import sets

NAMED = "named"
WEAK = "weak"
EXAMPLES = 8
TOP_TOKENS = 5
NO_PREDICTION = ("No causal test applies: the run declares no prediction for this description at these "
                 "instances.")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


@dataclass(frozen=True)
class Instance:
    window: int
    position: int
    w: int
    read: int | None
    target: int | None


def latent_effects(dictionary: Any, site: int, params: Any, latents: Any) -> np.ndarray:
    latents = list(latents)
    width = dictionary.active_width(site)
    if not all(_is_int(i) and 0 <= i < width for i in latents):
        raise ValueError(f"latents must be ints in range({width}), the site's active width, got {latents!r}")
    one_hot = np.zeros((len(latents), dictionary.settings.width), dtype=np.float32)
    one_hot[np.arange(len(latents)), [int(i) for i in latents]] = 1.0
    e = core.decode_delta(dictionary, site, one_hot).astype(np.float64)
    e -= e.mean(axis=1, keepdims=True)
    gain = np.asarray(params.ln_f.weight, dtype=np.float64)
    wte = np.asarray(params.wte, dtype=np.float64)
    return (e * gain) @ wte.T


def ends(effect: Any, value: Any) -> dict[str, Any]:
    effect = np.asarray(effect, dtype=np.float64)
    if not _is_int(value) or not 0 <= value < len(effect):
        raise ValueError(f"value must be a token in range({len(effect)}), got {value!r}")
    top = np.argsort(-effect, kind="stable")  # ties to the lower token
    bottom = np.argsort(effect, kind="stable")
    return {"top": [[int(t), float(effect[t])] for t in top[:TOP_TOKENS]],
            "bottom": [[int(t), float(effect[t])] for t in bottom[:TOP_TOKENS]],
            "rank_from_top": int(np.flatnonzero(top == value)[0]),
            "rank_from_bottom": int(np.flatnonzero(bottom == value)[0])}


def coverage(reading: Any, latents: Any, value: int) -> dict[str, float | None]:
    latents = list(latents)
    if not latents or not all(_is_int(i) and 0 <= i < reading.width for i in latents):
        raise ValueError(f"latents must be one or more ints in range({reading.width}), got {latents!r}")
    mass = reading.codes[:, :, [int(i) for i in latents]].astype(np.float64).sum(axis=-1)
    total = mass.sum()
    if not total > 0:
        return {d: None for d in reading.described}
    return {d: float(mass[arr == value].sum() / total) for d, arr in reading.described.items()}


def draw_instances(reading: Any, cls: str, d: str, v: int, *, limit: int,
                   rng: np.random.Generator) -> tuple[bool, list[Instance]]:
    if not _is_int(limit) or limit < 1:
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    table = reading.table(cls)
    if not table.has(d, v):
        raise ValueError(f"{d!r} = {v!r} isn't a value of a description that counts at class {cls!r}")
    declaration = reading.declaration
    values = [int(u) for u in table.values(d)]
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

    candidates = []  # (window, position, options, whether some option predicts)
    for i in range(len(reading.windows)):
        out = sets.taken(reading.described, d, i)
        options = [u for u in values if u not in out]
        for p in declaration.classes[cls]:
            if held[i, p] == v:
                candidates.append((i, p, options, bool(options) and bool(predictions(i, p, options, first=True))))
    causal = any(predicts for *_, predicts in candidates)
    eligible = [c for c in candidates if (c[3] if causal else c[2])]
    if len(eligible) > limit:
        eligible = [eligible[k] for k in sorted(rng.choice(len(eligible), size=int(limit), replace=False))]
    instances = []
    for i, p, options, _ in eligible:
        if causal:
            predicted = predictions(i, p, options, first=False)
            choices = [u for u in options if u in predicted]
        else:
            choices = options
        w = int(choices[rng.integers(len(choices))])
        read, target = predicted[w] if causal else (None, None)
        instances.append(Instance(int(i), int(p), w, read, target))
    return causal, instances


def _share(hits: np.ndarray) -> float:
    return float(np.mean(hits))


def _top_tokens(logits: np.ndarray) -> list[int]:
    return [int(t) for t in np.argsort(-logits, kind="stable")[:TOP_TOKENS]]


def steer(reading: Any, cls: str, d: str, v: int, *, instances: int, rng: np.random.Generator) -> dict[str, Any] | None:
    causal, chosen = draw_instances(reading, cls, d, v, limit=instances, rng=rng)
    if not chosen:
        return None
    table = reading.table(cls)
    n = len(chosen)
    index = [x.window for x in chosen]
    positions = [x.position for x in chosen]
    removed, added = sets.null_sets(table, d, reading.codes[index, positions], [v] * n, [x.w for x in chosen], rng)
    Sv = list(table.set_for(d, v))
    Sw = [list(table.set_for(d, x.w)) for x in chosen]
    amounts = [[float(a) for a in table.mean_codes(d, x.w)[s]] for x, s in zip(chosen, Sw)]
    none: list[list[Any]] = [[] for _ in range(n)]
    logits, _ = sets.run_steered(
        reading, index * 4, positions * 4,
        none + [Sv] * n + removed + [Sv] * n,
        none + Sw + added + none,
        none + amounts + [a[:len(x)] for a, x in zip(amounts, added)] + none)
    clean, swapped, null, ablated = (logits[j * n:(j + 1) * n] for j in range(4))
    k = np.arange(n)
    if causal:
        read = [x.read for x in chosen]
        target = np.array([x.target for x in chosen], dtype=np.int64)
        out = {name: sets.argmax(passed[k, read]) for name, passed in
               (("clean", clean), ("swapped", swapped), ("null", null), ("ablated", ablated))}
        return {"causal": True, "instances": n,
                "follows": _share(out["swapped"] == target), "unchanged": _share(out["swapped"] == out["clean"]),
                "ablated_changed": _share(out["ablated"] != out["clean"]),
                "null_follows": _share(out["null"] == target), "null_unchanged": _share(out["null"] == out["clean"])}
    return {"causal": False, "note": NO_PREDICTION, "instances": n,
            "kl": float(kl(clean[k, positions], swapped[k, positions])),
            "null_kl": float(kl(clean[k, positions], null[k, positions])),
            "examples": [{"window": x.window, "position": x.position, "w": x.w,
                          "unsteered": _top_tokens(clean[j, x.position]), "steered": _top_tokens(swapped[j, x.position])}
                         for j, x in enumerate(chosen[:EXAMPLES])]}


def name(reading: Any, cls: str, latent: int, *, instances: int, seed: int) -> dict[str, Any]:
    if not _is_int(latent) or not 0 <= latent < reading.width:
        raise ValueError(f"latent must be an int in range({reading.width}), got {latent!r}")
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    table = reading.table(cls)  # ValueError for a class the run doesn't declare
    if not table.descriptions:
        raise ValueError(f"no description counts at class {cls!r}")
    latent = int(latent)
    rng = np.random.default_rng(int(seed))
    results: dict[str, Any] = {"class": cls, "latent": latent, "windows": int(len(reading.windows)),
                               "rows": int(table.rows),
                               "candidates": [{"description": d, "value": v, "f1": f}
                                              for d, v, f in table.candidates(latent)]}
    if not results["candidates"]:
        results["name"] = None
        return results
    top = results["candidates"][0]
    d, v = top["description"], top["value"]
    results["name"] = {**top, "label": NAMED if top["f1"] >= sets.FLOOR else WEAK}

    chosen = list(table.set_for(d, v))
    column = list(table.values(d)).index(v)
    ranked = table.ranked(d, v)
    rank = int(np.flatnonzero(ranked == latent)[0])
    results["set"] = {"latents": chosen, "f1": [float(table.f1(d)[l, column]) for l in chosen],
                      "name_f1": table.name_f1(d, v, chosen), "pointed_rank": rank, "in_set": rank < sets.SET_SIZE}
    results["coverage"] = {"latent": coverage(reading, [latent], v), "set": coverage(reading, chosen, v)}

    effects = latent_effects(reading.dictionary, reading.site, reading.params, chosen + [latent])
    weights = table.mean_codes(d, v)[chosen].astype(np.float64)
    results["effect"] = {"set": ends(weights @ effects[:len(chosen)], v),
                         "latent": {"latent": latent, **ends(effects[len(chosen)], v)},
                         "vocabulary": int(np.shape(reading.params.wte)[0])}
    results["steering"] = steer(reading, cls, d, v, instances=instances, rng=rng)
    return results
