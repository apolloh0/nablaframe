"""Shared by the Phase 6 gates. Read with docs/specs/phase-6.md, "What each run declares", "Discrimination tests on the zoo", "The reference run" and "Gates".

May load ``gates/phase5/_common.py``, and through it the earlier phases' ``_common.py``, by path,
but never changes them. Reports go to ``reports/phase6/``, working files to ``runs/phase6/``, and
each run's gesture records to ``runs/phase6/gestures/<run>/`` (D47), ``<run>`` being
``induction``, ``two_hop`` or ``reference``. Every gate runs on the Mac only, refuses a dirty tree
unless ``--allow-dirty``, refuses to overwrite its report unless ``--overwrite``, prints a
progress line at least every 30 seconds during anything longer than a minute, and writes one
schema-2 report. Each gate refuses to start unless the reports it builds on exist, pass
``validate_report`` and say ``pass``: G4.1's; G5.2's and G5.3's, whose dictionaries the zoo gates
read, or G5.5's for the reference run's; and, for G6.2 to G6.6, G6.1's. No gate trains a
dictionary: each is the Phase 5 TopK SAE (or its twin's, or a robustness seed's) from the Phase 5
cache, under the key the Phase 5 gates computed, and a missing one stops the gate.

Every gesture reads the Phase 5 held-out windows (``zoo_windows(task, name, "held_out")``, or the
reference run's), each site through ``sets.read_site`` with ``chunk=cli.READ_CHUNK``. Every record
a gate writes goes through ``cli.record``.

The settings are the spec's, which are feasibility round 3's:

* ``INSTANCES = {"name": 512, "edge": 512, "open": 128}``: instances per name (one value of a
  description, D49), per edge, and per open-this-up target.
* ``SEEDS = {"name": 31, "edge": 32, "open": 35, "generalize": 40, "reference": 70}``: the seeds
  that draw them.
* ``REFERENCE = {"latents": 256, "values": 32, "per_value": 8, "open": 32}``: G6.6's latents named
  per block, ``token(p+1)`` values steered per block, instances per value, and open-this-up
  instances per block.
* ``IDENTITY_WINDOWS = 64`` and ``RECORD_INSTANCES = 32``: G6.1's windows for ``dsteer``'s
  identities, and instances for the records it builds twice.
* ``BARS``: ``{"named_as_planted": 0.7, "follows": 0.8, "null_follows": 0.05, "twin_named": 0.1,
  "twin_follows": 0.05, "edge_follows": 0.8, "edge_null_follows": 0.1, "output_follows": 0.8,
  "kept": 0.4, "null_kept": 0.8, "residual_alone": 0.1, "twin_excess": 0.1, "reading_ratio":
  1.0, "within": 0.8, "planted_top": 0.6, "random_within": 0.1, "twin_planted_top": 0.1,
  "slice_f1": 0.5, "slice_follows": 0.8, "seed_f1": 0.5, "seed_follows": 0.8, "start_offset":
  2}``.
* ``OPEN = {"two_hop": {"carry": 16, "hop": 64}, "induction": {"copy": 64}}``: G6.4's judged
  targets, each opened along a relation the run declares: its downstream place is the target, its
  upstream place the planted set. Each maps to its bar on the median reading. ``TWIN_OPEN =
  (("induction", "copy"),)``: the targets judged on the twin as well.
* ``GENERALIZE = {"two_hop": ("b", 2, "query"), "induction": ("token(p+1)", 1, "mechanism")}``:
  G6.5's judged names, each a description, a block and a class.
* ``IDENTITIES = ("empty", "zero codes", "zero amounts")``: D48's three identities, in order.

Judgements, each returning plain Python floats, ints, bools and None, ready for a report:

* ``least(values)`` and ``greatest(values)``: the least and greatest of numbers, as a float; None
  where there are none, or where any is None, so a judged reading that wasn't made fails its bar.
* ``ratio(reading, bar)``: ``reading / bar``; None for a reading of None.
* ``twin_excess(follows, null_follows)``: ``follows - null_follows``; None where either is None.
* ``pool(steerings)``: name-this-direction's steering pooled over a description's values:
  ``{"values", "instances", "follows", "unchanged", "ablated_changed", "null_follows",
  "null_unchanged"}`` over the steerings that are causal (``name.steer``'s; None and non-causal
  ones are left out). ``values`` counts them and ``instances`` sums theirs; each share is their
  mean weighted by instances, the share over every instance at once. With none, ``values`` and
  ``instances`` are 0 and every share None.
* ``planted_rank(order, position, latents, width)``: the least index in ``order`` (an int array
  of parts ``position * width + latent``, as ``open.order`` gives them) of any part at
  ``position`` whose latent is in ``latents``. ValueError where there is none.
* ``envelope_checks(envelope, steps)``: G6.5's three checks of ``generalize.envelope``'s result
  over checkpoint ``steps``, ascending: ``{"step0_outside", "one_final_run", "start_offset"}``.
  ``step0_outside``: step 0 is one of ``steps`` and not one of the envelope's. ``one_final_run``:
  the envelope is exactly one run, and it is ``final``. ``start_offset``: how many checkpoints
  apart ``final``'s first step and ``probe_reads_from`` are, by their places in ``steps``; None
  where either is None.

Measurements several gates share:

* ``recovery(table, d, v)``: ``{"best_f1", "largest_code"}``, the value ``d = v`` pointed at its
  best latent by F1 (``table.ranked(d, v)[0]``) and at its largest-code latent (the largest
  ``table.mean_codes(d, v)`` outside ``table.shared``, ties to the lower latent). Each is
  ``{"latent", "name", "named", "as_planted", "in_set"}``: ``name`` the latent's top candidate,
  ``{"description", "value", "f1"}``, or None where it has none; ``named``, that its F1 is at
  least ``sets.FLOOR``; ``as_planted``, that it is named, and named ``d = v``; ``in_set``, that the
  latent is in ``table.set_for(d, v)``. ValueError unless ``table.has(d, v)``.
* ``planted_name(reading, cls, d, *, instances, seed)``: G6.2's reading of a planted name over
  every value ``v`` of ``d`` at the class (``table.values(d)``, ascending), on one model. Each value:
  ``{"value", "best_f1", "largest_code", "steering", "effect"}``: ``recovery``; ``name.steer(reading,
  cls, d, v, instances=instances, rng=numpy.random.default_rng(seed))``, a fresh generator for each
  value, so that each equals the steering of a name record pointed at a latent named ``d = v`` with
  that seed; and ``{"rank_from_top", "rank_from_bottom"}`` of ``name.ends`` of the set's effect,
  weighted by its mean codes as ``name.name`` weights it. The mapping: ``"description"``,
  ``"class"``, ``"block"``, ``"tap"`` (the output tap), ``"values"`` (their count); ``"named"`` and
  ``"named_as_planted"``, the shares of values whose best latent is; ``"largest_code"``,
  ``{"named", "named_as_planted", "in_set"}``, the shares for the largest-code latents;
  ``"steering"``, ``pool`` of the values' steerings; ``"effect"``, ``{"last", "first",
  "mean_rank_from_top", "vocabulary"}``, the shares of values the set's effect ranks last and
  first, and the mean rank from the top; ``"coverage"``, for each description in
  ``reading.described``, the mean over values of ``name.coverage(reading, the set, v)``'s share,
  over the values where it isn't None, or None; and ``"by_value"``, the values' entries.
  ValueError where ``d`` doesn't count at the class.
* ``OpenCandidate(window, position, latents, planted_position, planted_latents)``: a frozen
  dataclass, one open-this-up target: a held-out window's index, the position read, and the
  downstream latents whose summed pre-activation is opened, as ``open.Target`` holds them; and the
  upstream position and latents of its planted set, both None where nothing is planted.
* ``relation_candidates(up, down, relation)``: G6.4's candidates for a target read along a
  relation. With ``R = declaration.relation(relation)``, its descriptions ``du`` and ``dd``,
  ``Tu = up.table(R.up.cls)`` and ``Td = down.table(R.down.cls)``: window by window, each pair
  ``(pu, pd)`` of ``declaration.pairs(relation, window)``, in order, where ``v``, ``du`` at
  ``pu``, has ``Tu.has(du, v)``, ``u``, ``dd`` at ``pd``, has ``Td.has(dd, u)``, and
  ``Td.set_for(dd, u)``'s codes in ``down.codes`` at ``pd`` sum above zero, in float64, gives
  ``OpenCandidate(window, pd, Td.set_for(dd, u), pu, Tu.set_for(du, v))``. ValueError unless both
  readings share their model, params and declaration as the same objects, and equal windows;
  for a relation the run doesn't declare; for a reading whose ``block`` isn't its place's; or for
  a description that doesn't count at its class.
* ``open_targets(up, down, candidates, *, instances, seed, chunk=cli.OPEN_CHUNK)``: D51's
  measurements over targets that change from one instance to the next, which no single
  open-this-up request covers (G6.4, G6.6). ``rng = numpy.random.default_rng(seed)``.

  - More than ``instances`` candidates are cut to ``instances`` with
    ``sorted(rng.choice(len(candidates), size=instances, replace=False))``. Then each kept
    instance, in order, draws its random order, ``open.order`` of its codes in ``up.codes`` up to
    and including its position, with ``"random"`` and ``rng``, as ``open_up`` draws it.
  - ``open.attribute(up, down, targets, chunk=chunk)``, the targets ``open.Target(window,
    position, latents)``. Instances whose ``|full − none|`` is below ``open.NO_SPAN`` are left out
    of everything below, and counted.
  - Each measured instance's parts in each of ``open.ORDERS``, and its error and reading at each
    ``k`` in ``open.PARTS``, as ``open_up`` defines them, from one ``open.evaluate`` call over
    every measured instance, then every order, then every ``k``, in that order.
  - Where the candidates hold planted sets, each measured instance's planted rank:
    ``planted_rank`` of its absolute order at its planted position and latents.

  Returns ``{"candidates", "instances", "no_span", "measured", "windows", "orders", "planted"}``:
  the first five and ``"orders"`` as ``open_up``'s; ``"planted"``, ``{"ranks", "rank_median",
  "top"}``, each measured instance's rank, their median and the share below ``open.TOP_PARTS``,
  the last two None with no measured instance; or None where no candidate holds a planted set.
  ValueError for candidates of which some hold a planted set and some don't, for ``instances`` or
  ``seed`` that aren't a positive int and a non-negative int, or as ``open.attribute`` raises it.
* ``reference_sample(table, n, rng)``: G6.6's latents: those active on some row of the table,
  ascending, as ints; more than ``n`` are cut to ``sorted(rng.choice(those, size=n,
  replace=False))``, ``rng`` being untouched otherwise.
* ``reference_values(table, d, n, rng)``: G6.6's steered values: the values of ``d`` at the table,
  ascending, as ints, whose best latent's F1 (the greatest in its column of ``table.f1(d)``) is
  at least ``sets.FLOOR``; more than ``n`` are cut as ``reference_sample`` cuts. ValueError where
  ``d`` doesn't count at the table.
* ``identity_edits(codes)``: D48's three identities as ``dsteer`` arrays, ``{identity: (remove,
  add, amounts)}`` in ``IDENTITIES`` order, for codes (B, seq, width) at a site. ``"empty"``:
  nothing removed or added. ``"zero codes"``: for each sequence ``k`` and position ``p``, in that
  order, the row ``[k, p, l]`` removed, ``l`` the lowest latent whose code is zero there (none
  where every code is above zero). ``"zero amounts"``: for each ``k`` and ``p``, ``[k, p, l]``
  added at 0.0, ``l`` the latent with the largest code there, ties to the lower. Rows are int32
  (n, 3) and amounts float32 (n,), shaped (0, 3) and (0,) when empty.
* ``dsteer_identities(model, params, *, n_layers, dictionary, tokens, site=0)``: whether each of
  ``identity_edits``'s identities, a ``dsteer`` edit at the site's output tap at every position,
  gives logits through ``run_intervened`` bit for bit equal to the pass with an empty spec, as a
  list of bools in ``IDENTITIES`` order. The codes are ``core.encode`` of the site's input tap on
  ``tokens``, through ``measure.activations``, keeping the site's active width.
* ``requests(run, steps, dictionaries)``: G6.1's 64 requests on one run. ``dictionaries`` is two
  distinct content hashes ``(a, b)``; ``steps`` the run's checkpoints, ascending. Eight bases, in
  order: a name at ``a``, site 0, class ``"query"``, latent 0; the same at ``b``; an edge
  ``"hop"`` from ``a``, site 0, to ``b``, site 0; open-this-up of ``b``'s latent 0 at site 0,
  class ``"query"``, into ``a``, site 0; the same of ``b``'s name ``"b" = 0``; generalize-this of
  ``"b"`` at ``a``, site 0, class ``"query"``; the same at ``b``; generalize-this of the edge
  ``"hop"`` from ``a`` to ``b``. Each base at ``steps[0]`` and ``steps[-1]``, then with 32 and 512
  instances, then with seed 0 and 1, nested in that order.
* ``request_hashes(requests)``: ``{"checked", "distinct", "stable"}``: how many; whether their
  ``request_hash``es are all different; and whether each equals the hash of its own JSON form read
  back through ``json.dumps`` and ``request_from_json``, and of the same request with its
  arguments given in the reverse order.
"""

# Beyond the contract above, the Phase 6 gates add here what several of them share: the start checks on
# earlier reports, the paths, the models and their twins at their final checkpoints, the Phase 5 TopK SAEs
# from the cache, each run's ``cli.Context``, and each site's reading. Nothing here imports JAX at module
# level, so every gate script imports in any environment.

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.dictionaries import core, measure
from nablaframe.gestures import cli, records, sets
from nablaframe.gestures import name as naming
from nablaframe.gestures import open as opening
from nablaframe.interventions.apply import run_intervened
from nablaframe.interventions.spec import InterventionSpec, array_hash, edit

INSTANCES = {"name": 512, "edge": 512, "open": 128}
SEEDS = {"name": 31, "edge": 32, "open": 35, "generalize": 40, "reference": 70}
REFERENCE = {"latents": 256, "values": 32, "per_value": 8, "open": 32}
IDENTITY_WINDOWS, RECORD_INSTANCES = 64, 32
BARS = {"named_as_planted": 0.7, "follows": 0.8, "null_follows": 0.05, "twin_named": 0.1, "twin_follows": 0.05,
        "edge_follows": 0.8, "edge_null_follows": 0.1, "output_follows": 0.8, "kept": 0.4, "null_kept": 0.8,
        "residual_alone": 0.1, "twin_excess": 0.1, "reading_ratio": 1.0, "within": 0.8, "planted_top": 0.6,
        "random_within": 0.1, "twin_planted_top": 0.1, "slice_f1": 0.5, "slice_follows": 0.8, "seed_f1": 0.5,
        "seed_follows": 0.8, "start_offset": 2}
OPEN = {"two_hop": {"carry": 16, "hop": 64}, "induction": {"copy": 64}}
TWIN_OPEN = (("induction", "copy"),)
GENERALIZE = {"two_hop": ("b", 2, "query"), "induction": ("token(p+1)", 1, "mechanism")}
IDENTITIES = ("empty", "zero codes", "zero amounts")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase6"
RUN_DIR = REPO_ROOT / "runs" / "phase6"
RECORD_DIR = RUN_DIR / "gestures"


def _load_phase5_common() -> Any:
    """gates/phase5/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase5_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase5"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p5 = _load_phase5_common()
p4, p3, p2, p1 = p5.p4, p5.p3, p5.p2, p5.p1
ticking = p5.ticking


# ---------------------------------------------------------------------------
# Judgements
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def least(values: Sequence[Any]) -> float | None:
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return float(min(values))


def greatest(values: Sequence[Any]) -> float | None:
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return float(max(values))


def ratio(reading: Any, bar: float) -> float | None:
    return None if reading is None else float(reading) / float(bar)


def twin_excess(follows: Any, null_follows: Any) -> float | None:
    if follows is None or null_follows is None:
        return None
    return float(follows) - float(null_follows)


_SHARES = ("follows", "unchanged", "ablated_changed", "null_follows", "null_unchanged")


def pool(steerings: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    causal = [s for s in steerings if s is not None and s.get("causal") is True]
    counts = [int(s["instances"]) for s in causal]
    total = sum(counts)
    pooled: dict[str, Any] = {"values": len(causal), "instances": total}
    for share in _SHARES:
        pooled[share] = float(sum(float(s[share]) * n for s, n in zip(causal, counts)) / total) if total else None
    return pooled


def planted_rank(order: Any, position: int, latents: Sequence[int], width: int) -> int:
    order = np.asarray(order, dtype=np.int64)
    at = (order // int(width) == int(position)) & np.isin(order % int(width), [int(i) for i in latents])
    hits = np.flatnonzero(at)
    if not len(hits):
        raise ValueError(f"no part at position {position} with a latent in {list(latents)!r} is in the order")
    return int(hits[0])


def envelope_checks(envelope: Mapping[str, Any], steps: Sequence[int]) -> dict[str, Any]:
    steps = [int(s) for s in steps]
    held = [int(s) for s in envelope["steps"]]
    runs, final, probe = envelope["runs"], envelope["final"], envelope["probe_reads_from"]
    one = final is not None and len(runs) == 1 and [int(s) for s in runs[0]] == [int(s) for s in final]
    offset = None
    if final is not None and probe is not None:
        offset = abs(steps.index(int(final[0])) - steps.index(int(probe)))
    return {"step0_outside": 0 in steps and 0 not in held, "one_final_run": bool(one), "start_offset": offset}


# ---------------------------------------------------------------------------
# Measurements several gates share
# ---------------------------------------------------------------------------


def _pointed(table: Any, d: str, v: int, latent: int) -> dict[str, Any]:
    """A value pointed at one latent: its top candidate, whether that names it as planted, and whether it's in the
    value's set."""
    found = table.candidates(latent)
    name = None if not found else {"description": found[0][0], "value": int(found[0][1]), "f1": float(found[0][2])}
    named = name is not None and name["f1"] >= sets.FLOOR
    return {"latent": int(latent), "name": name, "named": bool(named),
            "as_planted": bool(named and name["description"] == d and name["value"] == int(v)),
            "in_set": int(latent) in table.set_for(d, v)}


def recovery(table: Any, d: str, v: int) -> dict[str, Any]:
    if not table.has(d, v):
        raise ValueError(f"{d!r} = {v!r} isn't a value of a description that counts at this class")
    means = table.mean_codes(d, v).astype(np.float64)
    outside = np.setdiff1d(np.arange(len(means)), np.asarray(table.shared, dtype=np.int64))
    if not len(outside):
        raise ValueError("every latent is shared, so no latent has a largest code outside the shared ones")
    largest = int(outside[np.argmax(means[outside])])  # the first of equal maxima: the lower latent
    return {"best_f1": _pointed(table, d, v, int(table.ranked(d, v)[0])),
            "largest_code": _pointed(table, d, v, largest)}


def _share(flags: Sequence[Any]) -> float:
    return float(np.mean([bool(f) for f in flags]))


def planted_name(reading: Any, cls: str, d: str, *, instances: int, seed: int) -> dict[str, Any]:
    table = reading.table(cls)  # ValueError for a class the run doesn't declare
    if d not in table.descriptions:
        raise ValueError(f"{d!r} doesn't count at class {cls!r}; {list(table.descriptions)} do")
    by_value = []
    for v in (int(u) for u in table.values(d)):
        chosen = list(table.set_for(d, v))
        effects = naming.latent_effects(reading.dictionary, reading.site, reading.params, chosen)
        ends = naming.ends(table.mean_codes(d, v)[chosen].astype(np.float64) @ effects, v)
        by_value.append({"value": v, **recovery(table, d, v),
                         "steering": naming.steer(reading, cls, d, v, instances=instances,
                                                  rng=np.random.default_rng(seed)),
                         "effect": {"rank_from_top": ends["rank_from_top"],
                                    "rank_from_bottom": ends["rank_from_bottom"]}})
    shares: dict[str, list[float]] = {name: [] for name in reading.described}
    for entry in by_value:
        for name, share in naming.coverage(reading, table.set_for(d, entry["value"]), entry["value"]).items():
            if share is not None:
                shares[name].append(share)
    largest = [e["largest_code"] for e in by_value]
    ranks = [e["effect"] for e in by_value]
    return {"description": d, "class": cls, "block": reading.block, "tap": reading.output_tap,
            "values": len(by_value),
            "named": _share([e["best_f1"]["named"] for e in by_value]),
            "named_as_planted": _share([e["best_f1"]["as_planted"] for e in by_value]),
            "largest_code": {"named": _share([x["named"] for x in largest]),
                             "named_as_planted": _share([x["as_planted"] for x in largest]),
                             "in_set": _share([x["in_set"] for x in largest])},
            "steering": pool([e["steering"] for e in by_value]),
            "effect": {"last": _share([r["rank_from_bottom"] == 0 for r in ranks]),
                       "first": _share([r["rank_from_top"] == 0 for r in ranks]),
                       "mean_rank_from_top": float(np.mean([r["rank_from_top"] for r in ranks])),
                       "vocabulary": int(np.shape(reading.params.wte)[0])},
            "coverage": {name: float(np.mean(found)) if found else None for name, found in shares.items()},
            "by_value": by_value}


@dataclass(frozen=True)
class OpenCandidate:
    window: int
    position: int
    latents: tuple[int, ...]
    planted_position: int | None
    planted_latents: tuple[int, ...] | None


def _same_source(up: Any, down: Any) -> None:
    if up.model is not down.model or up.params is not down.params or up.declaration is not down.declaration:
        raise ValueError("both readings must share their model, params and declaration, as the same objects")
    if up.windows.shape != down.windows.shape or not np.array_equal(up.windows, down.windows):
        raise ValueError("both readings must read equal windows")


def relation_candidates(up: Any, down: Any, relation: str) -> list[OpenCandidate]:
    _same_source(up, down)
    declaration = up.declaration
    R = declaration.relation(relation)  # ValueError for a relation the run doesn't declare
    for reading, place, which in ((up, R.up, "upstream"), (down, R.down, "downstream")):
        if reading.block != place.block:
            raise ValueError(f"relation {relation!r} reads its {which} place after block {place.block}, but the "
                             f"{which} reading is after block {reading.block}")
    du, dd = R.up.description, R.down.description
    Tu, Td = up.table(R.up.cls), down.table(R.down.cls)
    for d, table, cls in ((du, Tu, R.up.cls), (dd, Td, R.down.cls)):
        if d not in table.descriptions:
            raise ValueError(f"{d!r} doesn't count at class {cls!r}")
    up_sets: dict[int, tuple[int, ...]] = {}
    down_sets: dict[int, tuple[int, ...]] = {}
    held_up, held_down = up.described[du], down.described[dd]
    found = []
    for i in range(len(up.windows)):
        for pu, pd in declaration.pairs(relation, up.windows[i]):
            pu, pd = int(pu), int(pd)
            v, u = int(held_up[i, pu]), int(held_down[i, pd])
            if not Tu.has(du, v) or not Td.has(dd, u):
                continue
            if u not in down_sets:
                down_sets[u] = Td.set_for(dd, u)
            latents = down_sets[u]
            if not down.codes[i, pd, list(latents)].astype(np.float64).sum() > 0:
                continue
            if v not in up_sets:
                up_sets[v] = Tu.set_for(du, v)
            found.append(OpenCandidate(int(i), pd, latents, pu, up_sets[v]))
    return found


def _empty_orders() -> dict[str, Any]:
    return {kind: {"error_median": None, "readings": [], "reading_median": None, "within": None}
            for kind in opening.ORDERS}


def _reading_median(readings: Sequence[int | None]) -> float | None:
    """The median reading, a reading of None counting as larger than any ``k``, and None where it is one."""
    median = float(np.median([np.inf if r is None else r for r in readings]))
    return None if np.isinf(median) else median


def open_targets(up: Any, down: Any, candidates: Sequence[OpenCandidate], *, instances: int, seed: int,
                 chunk: int | None = None) -> dict[str, Any]:
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    chunk = cli.OPEN_CHUNK if chunk is None else chunk
    candidates = list(candidates)
    holds = {(x.planted_position is not None, x.planted_latents is not None) for x in candidates}
    if holds - {(True, True), (False, False)} or len(holds) > 1:
        raise ValueError("either every candidate holds a planted set, its position and latents, or none does")
    planted = holds == {(True, True)}

    rng = np.random.default_rng(int(seed))
    chosen = candidates
    if len(candidates) > instances:
        chosen = [candidates[k] for k in sorted(rng.choice(len(candidates), size=int(instances), replace=False))]
    shuffles = [opening.order(up.codes[x.window, :x.position + 1], up.codes[x.window, :x.position + 1], "random", rng)
                for x in chosen]
    results: dict[str, Any] = {"candidates": len(candidates), "instances": len(chosen), "no_span": 0, "measured": 0,
                               "windows": [], "orders": _empty_orders(),
                               "planted": {"ranks": [], "rank_median": None, "top": None} if planted else None}
    if not chosen:
        return results

    targets = [opening.Target(x.window, x.position, tuple(x.latents)) for x in chosen]
    A = opening.attribute(up, down, targets, chunk=chunk)
    span = np.abs(A.full - A.none)
    measured = [k for k in range(len(chosen)) if not span[k] < opening.NO_SPAN]
    results["no_span"] = len(chosen) - len(measured)
    results["measured"] = len(measured)
    if not measured:
        return results
    results["windows"] = [[chosen[k].window, chosen[k].position] for k in measured]

    ranked = {k: [opening.order(A.attributions[k, :chosen[k].position + 1], A.codes[k, :chosen[k].position + 1], kind)
                  if kind != "random" else shuffles[k] for kind in opening.ORDERS] for k in measured}
    # Every kept-parts reading in one evaluate call: each measured instance, then each order, then each k.
    rows = [(k, o, size) for k in measured for o in range(len(opening.ORDERS)) for size in opening.PARTS]
    kept = np.zeros((len(rows),) + A.codes.shape[1:], dtype=np.float32)
    for r, (k, o, size) in enumerate(rows):
        parts = ranked[k][o][:size]
        kept[r].reshape(-1)[parts] = A.codes[k].reshape(-1)[parts]
    pre = opening.evaluate(up, down, [targets[k] for k, _, _ in rows], kept, chunk=chunk)
    full = np.array([A.full[k] for k, _, _ in rows])
    errors = (np.abs(pre - full) / np.array([span[k] for k, _, _ in rows])).reshape(
        len(measured), len(opening.ORDERS), len(opening.PARTS))
    for o, kind in enumerate(opening.ORDERS):
        readings = [next((size for size, e in zip(opening.PARTS, errors[m, o]) if e <= opening.ERROR), None)
                    for m in range(len(measured))]
        results["orders"][kind] = {
            "error_median": [float(np.median(errors[:, o, s])) for s in range(len(opening.PARTS))],
            "readings": readings, "reading_median": _reading_median(readings),
            "within": float(np.mean([r is not None for r in readings]))}
    if planted:
        ranks = [planted_rank(ranked[k][0], chosen[k].planted_position, chosen[k].planted_latents, up.width)
                 for k in measured]
        results["planted"] = {"ranks": ranks, "rank_median": float(np.median(ranks)),
                              "top": float(np.mean([r < opening.TOP_PARTS for r in ranks]))}
    return results


def _cut(found: list[int], n: int, rng: np.random.Generator) -> list[int]:
    if len(found) <= n:
        return found
    return sorted(int(x) for x in rng.choice(np.asarray(found, dtype=np.int64), size=n, replace=False))


def reference_sample(table: Any, n: int, rng: np.random.Generator) -> list[int]:
    return _cut([int(i) for i in np.flatnonzero(np.asarray(table.active) > 0)], int(n), rng)


def reference_values(table: Any, d: str, n: int, rng: np.random.Generator) -> list[int]:
    if d not in table.descriptions:
        raise ValueError(f"{d!r} doesn't count at this table; {list(table.descriptions)} do")
    best = table.f1(d).max(axis=0)
    return _cut([int(v) for v, f in zip(table.values(d), best) if f >= sets.FLOOR], int(n), rng)


def _edit_rows(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int32).reshape(-1, 3)


def identity_edits(codes: Any) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    codes = np.asarray(codes)
    if codes.ndim != 3:
        raise ValueError(f"codes must be shaped (B, seq, width), got {codes.shape}")
    none = _edit_rows([])
    zero_codes, largest = [], []
    for k in range(codes.shape[0]):
        for p in range(codes.shape[1]):
            zeros = np.flatnonzero(codes[k, p] == 0)
            if len(zeros):
                zero_codes.append([k, p, int(zeros[0])])
            largest.append([k, p, int(np.argmax(codes[k, p]))])  # the first of equal maxima: the lower latent
    return {"empty": (none, none.copy(), np.zeros(0, dtype=np.float32)),
            "zero codes": (_edit_rows(zero_codes), none.copy(), np.zeros(0, dtype=np.float32)),
            "zero amounts": (none.copy(), _edit_rows(largest), np.zeros(len(largest), dtype=np.float32))}


def dsteer_identities(model: Any, params: Any, *, n_layers: int, dictionary: Any, tokens: Any,
                      site: int = 0) -> list[bool]:
    site = dictionary._site(site)
    input_tap, output_tap = dictionary.sites[site]
    tokens = np.ascontiguousarray(tokens, dtype=np.int32)
    batch, seq = tokens.shape
    width = dictionary.active_width(site)
    x = measure.activations(model, params, tokens, [input_tap])[input_tap]
    codes = core.encode(dictionary, site, x.reshape(batch * seq, -1))[:, :width].reshape(batch, seq, width)

    def logits(spec: Any, store: dict[str, Any]) -> np.ndarray:
        out, _ = run_intervened(model, params, tokens, spec, n_layers=n_layers, resolve=store.__getitem__)
        return np.asarray(getattr(out, "array", out))

    plain = logits(InterventionSpec(()), {})
    exact = []
    for remove, add, amounts in identity_edits(codes).values():
        arrays = {"remove": remove, "add": add, "amounts": amounts}
        hashes = {name: array_hash(value) for name, value in arrays.items()}
        store: dict[str, Any] = {core.content_hash(dictionary): dictionary}
        store.update({hashes[name]: value for name, value in arrays.items()})
        spec = InterventionSpec((edit(output_tap, "all", "dsteer", dictionary=core.content_hash(dictionary),
                                      **hashes),))
        got = logits(spec, store)
        exact.append(bool(got.dtype == plain.dtype and got.shape == plain.shape and np.array_equal(got, plain)))
    return exact


def requests(run: str, steps: Sequence[int], dictionaries: Sequence[str]) -> list[Any]:
    a, b = dictionaries
    if a == b:
        raise ValueError("requests needs two distinct dictionaries")
    edge = {"relation": "hop", "up": a, "up_site": 0, "down": b, "down_site": 0}
    opened = {"dictionary": b, "site": 0, "class": "query", "upstream": a, "upstream_site": 0}
    bases = [("name", {"dictionary": a, "site": 0, "class": "query", "latent": 0}),
             ("name", {"dictionary": b, "site": 0, "class": "query", "latent": 0}),
             ("edge", edge),
             ("open", {**opened, "latent": 0}),
             ("open", {**opened, "description": "b", "value": 0}),
             ("generalize", {"dictionary": a, "site": 0, "class": "query", "description": "b"}),
             ("generalize", {"dictionary": b, "site": 0, "class": "query", "description": "b"}),
             ("generalize", dict(edge))]
    steps = list(steps)
    return [records.GestureRequest(kind, run, step, dict(args), instances, seed)
            for kind, args in bases for step in (steps[0], steps[-1]) for instances in (32, 512) for seed in (0, 1)]


def request_hashes(reqs: Sequence[Any]) -> dict[str, Any]:
    reqs = list(reqs)
    hashes = [records.request_hash(r) for r in reqs]
    stable = True
    for r, h in zip(reqs, hashes):
        read_back = records.request_from_json(json.loads(json.dumps(records.request_json(r))))
        reversed_args = records.GestureRequest(r.kind, r.run, r.step, dict(reversed(list(r.args.items()))),
                                               r.instances, r.seed)
        stable = stable and records.request_hash(read_back) == h and records.request_hash(reversed_args) == h
    return {"checked": len(reqs), "distinct": len(set(hashes)) == len(hashes), "stable": bool(stable)}


# ---------------------------------------------------------------------------
# Beyond the contract: what the Phase 6 gates share
# ---------------------------------------------------------------------------
#
# * The start checks on earlier reports (``start``), the report paths, and Phase 6's working folders
#   (``fresh_dir``, under RUN_DIR).
# * ``zoo_model``: Phase 5's reading of one zoo run at its final checkpoint, with its twin, and the run's
#   declaration; ``reference_model``: Phase 5's reading of G1.4's run, with the reference declaration.
# * ``topk``: a Phase 5 TopK SAE from the cache under the key Phase 5 computed, a missing one stopping the
#   gate; ``context``: a zoo run's ``cli.Context``; ``read``: one site read over the held-out windows.
#
# Paths are read from this module's globals, and those of ``p5`` and the earlier phases, when a function
# runs, never bound at import, so everything here can be pointed at a scratch folder.

ZOO = ("induction", "two_hop")
VERSIONS = p5.VERSIONS
REPORTS = {
    "G5.2": ("phase5", "g52_induction.json", "the induction run's dictionaries come from G5.2"),
    "G5.3": ("phase5", "g53_two_hop.json", "the two-hop run's dictionaries come from G5.3"),
    "G5.5": ("phase5", "g55_reference.json", "the reference run's dictionaries come from G5.5"),
    "G6.1": ("phase6", "g61_determinism.json",
             "no gesture is judged before dsteer, the requests and the records pass their determinism checks"),
}
TRAINED_BY = {"induction": "G5.2", "two_hop": "G5.3", "reference": "G5.5"}


def say(line: str) -> None:
    print(line, flush=True)


def report_relative(name: str) -> str:
    """``reports/phase6/<name>``, as Phase 1's ``start`` takes it."""
    return str((Path(REPORT_DIR) / name).relative_to(p1.REPO_ROOT))


def require(gate_id: str) -> dict[str, Any]:
    """The report a Phase 6 gate builds on, once it exists, passes ``validate_report`` and says ``pass``."""
    if gate_id == "G4.1":
        return p4.require_g41()
    phase, name, why = REPORTS[gate_id]
    folder = p5.REPORT_DIR if phase == "phase5" else REPORT_DIR
    return p5._require(gate_id, Path(folder) / name, why=why)


def start(args: Any, report_name: str, needs: Sequence[str]) -> tuple[Path, dict[str, Any]]:
    """Phase 1's start checks, then every report in ``needs``: ``(the report's path, G4.1's report)``."""
    report = p1.start(args, report_relative(report_name))
    reports = {gate_id: require(gate_id) for gate_id in needs}
    return report, reports.get("G4.1")


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    import shutil

    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 6 working folders live under {RUN_DIR}")
    if Path(path).exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)
    return Path(path)


def zoo_model(name: str, g41: dict[str, Any]) -> Any:
    """Phase 5's ``zoo_model``, G4.1's run at its final checkpoint with its twin, plus ``declaration``, the run's."""
    m = p5.zoo_model(name, g41)
    m.declaration = cli.declaration_for(name, m.config)
    return m


def reference_model() -> Any:
    """Phase 5's ``reference_model``, G1.4's run at its final checkpoint, plus the reference ``declaration``."""
    m = p5.reference_model()
    m.declaration = cli.declaration_for("reference", m.config)
    return m


def topk(m: Any, layer: int, *, twin: bool = False, seed: int = p5.DICT_SEED) -> Any:
    """The Phase 5 TopK SAE after block ``layer`` of model ``m`` (or its twin, or a robustness seed's), from the cache.

    Its key is the one the Phase 5 gates computed (``p5.dictionary_spec``). A dictionary the cache doesn't
    hold stops the gate: no Phase 6 gate trains one.
    """
    _, _, key = p5.dictionary_spec(m, "topk", layer, twin=twin, seed=seed)
    who = "twin" if twin else ("trained" if seed == p5.DICT_SEED else f"seed {seed}")
    label = f"{who} TopK SAE after block {layer}"

    def missing() -> Any:
        raise SystemExit(f"{m.name}: the Phase 5 cache holds no {label}, key {key}, in"
                         f" {p1.rel(Path(p5.DICTIONARY_DIR) / key)}. Run {TRAINED_BY[m.name]} first; no Phase 6"
                         " gate trains a dictionary. No report written.")

    dictionary, _ = p5.cached(p5.DICTIONARY_DIR, key, missing)
    say(f"{m.name}: {label} from the cache, key {key[:12]}, content {core.content_hash(dictionary)[:12]}")
    return dictionary


def held(dictionaries: Sequence[Any]) -> Any:
    """A context's ``dictionary``: each of ``dictionaries`` by its content hash, ValueError for any other."""
    by_hash = {core.content_hash(d): d for d in dictionaries}

    def dictionary(digest: str) -> Any:
        if digest not in by_hash:
            raise ValueError(f"the context holds no dictionary {digest}")
        return by_hash[digest]

    return dictionary


def context(m: Any, dictionaries: Sequence[Any], seeds: Mapping[int, Mapping[str, str]] | None = None) -> Any:
    """The zoo run's ``cli.Context``: its held-out windows, every checkpoint, ``dictionaries`` and ``seeds``.

    ``params_at`` is the run's own loader, so the final checkpoint's params are ``m.params``, the object
    every reading the gate makes shares.
    """
    run = m.run
    return cli.Context(run=m.run_id, declaration=m.declaration, model=m.model, n_layers=m.n_layers,
                       steps=tuple(int(s) for s in run.steps), params_at=run.params_for, state_at=run.state_sha256,
                       windows=m.held_out, dictionary=held(dictionaries),
                       seeds={int(s): dict(c) for s, c in (seeds or {}).items()})


def read(m: Any, params: Any, dictionary: Any, label: str, site: int = 0) -> Any:
    """``sets.read_site`` of one site over the held-out windows, at ``cli.READ_CHUNK``, inside ``ticking``."""
    with ticking(label, write=say):
        return sets.read_site(m.model, params, n_layers=m.n_layers, declaration=m.declaration, dictionary=dictionary,
                              site=site, windows=m.held_out, chunk=cli.READ_CHUNK)


def place_name(run: str, description: str, block: int, cls: str) -> str:
    """A place as the reports name it, ``"<run> <description> after block <l> at <class>"``."""
    return f"{run} {description} after block {block} at {cls}"


def results_of(record: Any) -> dict[str, Any]:
    """A record's results as plain JSON values."""
    return records.record_json(record)["results"]


def number(value: Any) -> bool:
    """A plain int or float, not a bool or None: what a bar can be held to."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def show(value: Any) -> str:
    return "null" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
