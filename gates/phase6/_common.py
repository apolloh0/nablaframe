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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def least(values: Sequence[Any]) -> float | None:
    raise NotImplementedError


def greatest(values: Sequence[Any]) -> float | None:
    raise NotImplementedError


def ratio(reading: Any, bar: float) -> float | None:
    raise NotImplementedError


def twin_excess(follows: Any, null_follows: Any) -> float | None:
    raise NotImplementedError


def pool(steerings: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    raise NotImplementedError


def planted_rank(order: Any, position: int, latents: Sequence[int], width: int) -> int:
    raise NotImplementedError


def envelope_checks(envelope: Mapping[str, Any], steps: Sequence[int]) -> dict[str, Any]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Measurements several gates share
# ---------------------------------------------------------------------------


def recovery(table: Any, d: str, v: int) -> dict[str, Any]:
    raise NotImplementedError


def planted_name(reading: Any, cls: str, d: str, *, instances: int, seed: int) -> dict[str, Any]:
    raise NotImplementedError


@dataclass(frozen=True)
class OpenCandidate:
    window: int
    position: int
    latents: tuple[int, ...]
    planted_position: int | None
    planted_latents: tuple[int, ...] | None


def relation_candidates(up: Any, down: Any, relation: str) -> list[OpenCandidate]:
    raise NotImplementedError


def open_targets(up: Any, down: Any, candidates: Sequence[OpenCandidate], *, instances: int, seed: int,
                 chunk: int | None = None) -> dict[str, Any]:
    raise NotImplementedError


def reference_sample(table: Any, n: int, rng: np.random.Generator) -> list[int]:
    raise NotImplementedError


def reference_values(table: Any, d: str, n: int, rng: np.random.Generator) -> list[int]:
    raise NotImplementedError


def identity_edits(codes: Any) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    raise NotImplementedError


def dsteer_identities(model: Any, params: Any, *, n_layers: int, dictionary: Any, tokens: Any,
                      site: int = 0) -> list[bool]:
    raise NotImplementedError


def requests(run: str, steps: Sequence[int], dictionaries: Sequence[str]) -> list[Any]:
    raise NotImplementedError


def request_hashes(reqs: Sequence[Any]) -> dict[str, Any]:
    raise NotImplementedError
