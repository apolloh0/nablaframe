"""G6.7: every gesture record built twice, the gestures panel, and the badges (D38, D47).

Spec: docs/specs/phase-6.md, "Records, the viewer and badges" and "Gates";
docs/deltas/phase-6-delta-5.md; helpers in ``_common.py``. Report:
``reports/phase6/g67_records_panel.json``. Run in nf-jax, on the Mac, after G6.1 to G6.6 have run.

Refuses to start unless G4.1's, G5.2's, G5.3's, G5.5's, G5.6's and G6.1's reports pass, and G4.2 to
G4.4's, G5.1's, G5.4's and G6.2 to G6.6's exist and pass ``validate_report``, whatever their status:
a failed judged gate shows on its badge, never hidden by a missing one.

**Records.** On each run at its final checkpoint, with the Phase 5 TopK SAEs from the cache, every
request below goes through ``cli.record`` with the run's context. A zoo run's context is
``_common.context`` holding the TopK SAE of every block, with the seed 1 and 2 TopK SAEs of its
``GENERALIZE`` block as its seeds, as G6.5 held them. The reference run's holds its run id, every
checkpoint of G1.4's run with its params and state hash, its held-out windows and the TopK SAE of
every block, with no seeds. Requests are made in the order listed; one equal to an earlier one is
built once.

* ``two_hop`` and ``induction``, each place read once with ``_common.read``:

  1. **Names:** for each place G6.2 reads (a planted name's description, block and class; two-hop's
     ``b`` after block 1 included), for each value ``v`` of the description at the class,
     ascending, a name request pointed at ``v``'s best latent by F1, ``table.ranked(d, v)[0]``, on
     the block's TopK SAE at site 0, with ``INSTANCES["name"]`` and ``SEEDS["name"]``. Where that
     latent is named ``d = v``, its steering is G6.2's for ``v``.
  2. **The readout:** on the probe windows (``p4.probe`` of the task, the tokens the model reads),
     at each of those places' blocks, the latents active at each window's last position, largest
     code first, ties to the lower latent, up to 8 per window: what the features view lists there.
     The codes are ``core.encode`` of the site's input tap, from ``measure.activations`` at the
     final params. A name request for each, at the place's class, with ``INSTANCES["name"]`` and
     ``SEEDS["name"]``.
  3. **Edges:** for each relation the run declares, G6.3's request: ``{"relation", "up",
     "up_site": 0, "down", "down_site": 0}`` with the TopK SAEs at its places' blocks,
     ``INSTANCES["edge"]`` and ``SEEDS["edge"]``.
  4. **Open-this-up:** for each of the run's targets in ``OPEN``, read along its relation, its
     downstream place the target and its upstream block the one opened into: for each value ``u``
     of the downstream description at its class, ascending, ``{"dictionary", "site": 0, "class",
     "upstream", "upstream_site": 0, "description", "value": u}``, with ``INSTANCES["open"]`` and
     ``SEEDS["open"]``.
  5. **Generalize-this:** the run's name in ``GENERALIZE``, G6.5's request: ``{"dictionary",
     "site": 0, "class", "description"}``, ``INSTANCES["name"]`` and ``SEEDS["generalize"]``.

* ``reference``, at each block ``l``, with ``T`` its reading's table at ``every`` and ``rng =
  numpy.random.default_rng([SEEDS["reference"], l])``, drawing as G6.6 draws:

  1. **Names:** each latent of ``reference_sample(T, REFERENCE["latents"], rng)``, at ``every``,
     with ``REFERENCE["per_value"]`` instances and ``SEEDS["reference"]``.
  2. **Steered values:** each value ``v`` of ``reference_values(T, "token(p+1)",
     REFERENCE["values"], rng)``, the same ``rng`` drawing after the sample: a name request pointed
     at ``v``'s best latent by F1, at ``every``, with the names' instances and seed.
  3. **Open-this-up,** at blocks 1 and above: for each such ``v``, the set of ``"token(p+1)" = v``
     at ``every``, opened into block ``l - 1``'s TopK SAE, with ``REFERENCE["open"]`` instances and
     ``SEEDS["reference"]``.

**Two builds.** Before anything is built, every request's key is found, ``records.record_key`` of
the request, ``cli.inputs`` and ``cli.code``. A run's record folder, ``RECORD_DIR / run``, holding a
record whose key isn't one of them stops the gate, naming it. The first build goes into that folder,
through the record cache, so the records G6.3 and G6.5 wrote are kept as they are; the second into
``RUN_DIR / "scratch" / "gestures" / run``, which starts empty. An existing second folder, or an
existing badges file, is replaced only with ``--overwrite``. Each folder then gets
``views.gestures.write_panel``. The two are compared file by file: every file in either folder, by
name, with its bytes.

**Badges.** ``runs/phase6/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1
to G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, then gesture_badges of G6.1 to G6.5, as
one mapping>, "reports": {<gate id>: {"git_commit", "timestamp_utc", "sha256"}}}`` for each of those
thirteen reports, the last being the report file's SHA-256. Built a second time into
``runs/phase6/scratch/badges.json`` and compared byte for byte. One file holds every model-level
badge, so the viewer reads it with ``?badges=`` and finds every phase's subjects.

Measurements: REQUIRED_MEASUREMENTS["G6.7"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``,
  ``dictionary_badges.SUBJECTS`` and ``gesture_badges.SUBJECTS`` without a badge.
* ``records``: per run, ``{"name", "edge", "open", "generalize"}``, how many records of each kind
  its first folder holds.
* ``builds_identical``: the two badges files hold the same bytes, and each run's two record folders
  hold the same files with the same bytes.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.
* ``reports_match``: each edge record G6.3's report names and each generalize record G6.5's names,
  by key, is in its run's first folder, and each reference block's steered values are G6.6's
  (``blocks[l]["steering"]["values"]``). So the records the panel shows are the ones the gates
  judged, and the reference's are the ones G6.6 recorded.

Recorded beside them: ``builds``, each part's detail (files compared and files differing);
``matches``, every key and value list compared; ``run_ids``; ``steps``; ``requests``, per run and
kind, how many requests were made before equal ones were merged; ``intermediate``, two-hop's ``b``
after block 2: its generalize record's key and its results' ``envelope``, and for each value ``v``
ascending, ``{"value", "latent", "key", "meter"}``, the name record of step 1 and its meter from the
panel; and ``addresses``.

Pass needs ``builds_identical``, ``external_urls == 0``, ``subjects_missing == 0`` and
``reports_match``. Every long step prints a progress line at least every 30 seconds, saying how many
records of how many it has built. At the end it prints the viewer's address for each run: G5.6's,
from its report's ``addresses``, with ``badges=`` pointing at ``../runs/phase6/badges.json`` and
``&gestures=../runs/phase6/gestures/<run>/`` added before the fragment.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase6/g67_records_panel.py``. Takes --overwrite and --allow-dirty. --overwrite also lets it replace the
#   second build's folders and both badges files; the first build's folders are never cleared.
# * Start checks, before anything is written: the reports, as above (the passing ones through ``_common.require``,
#   G5.6's through Phase 5's ``_require``); ``model_level_badges``, ``dictionary_badges`` and ``gesture_badges`` of
#   the reports, whose ValueError is a setup error; an address in G5.6's report for every run; then the models,
#   every dictionary from the Phase 5 cache (``_common.topk``), every request and its key, and every first folder
#   checked for a record G6.7 doesn't ask for. Each fails with exit 1 and no report.
# * Runs in the order ``two_hop``, ``induction``, ``reference``. Each place is read once to make the requests,
#   and the readings are dropped before anything is built; every record is then built by ``cli.record``, which
#   reads its own sites.
# * The readout's codes: ``measure.activations`` of the probe tokens at the site's input and output taps with the
#   run's ``CHUNK``, as ``features_payload`` takes them, then one ``core.encode`` call over every probe position.
# * Requests are merged by request hash, keeping the first; each is labelled by the step that made it first.
# * The reference context's ``params_at`` returns the final checkpoint's params, read once, for the final step,
#   and loads any other checkpoint when asked; ``state_at`` reads each checkpoint's sidecar.
# * ``external_urls``: G4.5's ``offline``, the script loaded by path with Phase 4's ``_common``, as G5.6 loads it.
# * ``builds``: ``{"badges": {...}, <run>: {...}}``, each ``{"identical", "files_compared", "files_differing"}``, a
#   run's also naming its two folders. ``matches``: ``{"edges", "generalize", "reference"}``, a list each: the
#   place and key G6.3 or G6.5 names, with whether it is in the run's first folder; and per block, G6.6's values
#   and G6.7's.

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p4, p5 = _common.p1, _common.p4, _common.p5

GATE_ID = "G6.7"
REPORT_NAME = "g67_records_panel.json"
REPORT_FILES = {
    "G4.1": ("phase4", "g41_zoo_training.json"), "G4.2": ("phase4", "g42_induction.json"),
    "G4.3": ("phase4", "g43_two_hop.json"), "G4.4": ("phase4", "g44_lenses.json"),
    "G5.1": ("phase5", "g51_determinism.json"), "G5.2": ("phase5", "g52_induction.json"),
    "G5.3": ("phase5", "g53_two_hop.json"), "G5.4": ("phase5", "g54_edges.json"),
    "G5.5": ("phase5", "g55_reference.json"), "G5.6": ("phase5", "g56_features_badges.json"),
    "G6.1": ("phase6", "g61_determinism.json"), "G6.2": ("phase6", "g62_names.json"),
    "G6.3": ("phase6", "g63_edges.json"), "G6.4": ("phase6", "g64_open.json"),
    "G6.5": ("phase6", "g65_generalize.json"), "G6.6": ("phase6", "g66_reference.json"),
}
PASSING = ("G4.1", "G5.2", "G5.3", "G5.5", "G5.6", "G6.1")
BADGE_GATES = ("G4.1", "G4.2", "G4.3", "G4.4", "G5.1", "G5.2", "G5.3", "G5.4", "G6.1", "G6.2", "G6.3", "G6.4", "G6.5")
RUNS = ("two_hop", "induction", "reference")
KINDS = ("name", "edge", "open", "generalize")
READOUT_LATENTS = 8
BADGES_NAME = "badges.json"
REFERENCE_DESCRIPTION, REFERENCE_CLASS = "token(p+1)", "every"
INTERMEDIATE = ("two_hop", "b", 2, "query")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    reports = read_reports()
    made_badges(reports)
    g56_addresses(reports["G5.6"])
    badge_files = (Path(_common.RUN_DIR) / BADGES_NAME, Path(_common.RUN_DIR) / "scratch" / BADGES_NAME)
    folders = {run: (Path(_common.RECORD_DIR) / run, Path(_common.RUN_DIR) / "scratch" / "gestures" / run)
               for run in RUNS}
    existing = [path for path in badge_files if path.exists()]
    existing += [second for _, second in folders.values() if second.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{p1.rel(existing[0])} exists. Pass --overwrite to start it over.")

    runs = {"two_hop": zoo_run("two_hop", reports["G4.1"]), "induction": zoo_run("induction", reports["G4.1"]),
            "reference": reference_run()}
    for run in runs.values():
        plan(run)
    for run in runs.values():
        refuse_strays(run, folders[run.name][0])

    builds, counts = {}, {}
    for name in RUNS:
        builds[name] = build_twice(runs[name], *folders[name], overwrite=args.overwrite)
        counts[name] = held_by_kind(folders[name][0])
    matched = match_reports(runs, folders, reports)
    badges = badges_twice(badge_files)
    builds = {"badges": badges.pop("build"), **builds}
    measurements = {
        **badges,
        "records": counts,
        "builds_identical": all(part["identical"] is True for part in builds.values()),
        "builds": builds,
        **external_urls(),
        **matched,
        "run_ids": {name: runs[name].m.run_id for name in RUNS},
        "steps": {name: runs[name].m.step for name in RUNS},
        "requests": {name: runs[name].requested for name in RUNS},
        "intermediate": intermediate(runs["two_hop"], folders["two_hop"][0]),
        "addresses": addresses(reports["G5.6"], badge_files[0], folders),
        "record_dirs": {name: p1.rel(folders[name][0]) for name in RUNS},
        "settings": {"instances": _common.INSTANCES, "seeds": _common.SEEDS, "reference": _common.REFERENCE,
                     "readout_latents": READOUT_LATENTS},
        "versions": p1.phase0.package_versions(_common.VERSIONS),
    }
    code = p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)
    print(f"{GATE_ID}: serve the repo root with `python3 -m http.server`, then open:", flush=True)
    for name, address in measurements["addresses"].items():
        print(f"  {name}: {address}", flush=True)
    return code


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.7 rule holds."""
    ok = (m["builds_identical"] is True and m["external_urls"] == 0 and m["subjects_missing"] == 0
          and m["reports_match"] is True)
    return "pass" if ok else "fail"


# ---------------------------------------------------------------------------
# Start checks: the reports
# ---------------------------------------------------------------------------


def report_path(gate_id: str) -> Path:
    phase, name = REPORT_FILES[gate_id]
    folder = {"phase4": p4.REPORT_DIR, "phase5": p5.REPORT_DIR, "phase6": _common.REPORT_DIR}[phase]
    return Path(folder) / name


def read_reports() -> dict[str, dict[str, Any]]:
    """Every report G6.7 reads, each from its file and checked. SystemExit otherwise.

    G4.1, G5.2, G5.3, G5.5, G5.6 and G6.1 must pass; the rest need only exist and be valid, whatever
    their status: a failed judged gate shows on its badge, never hidden by a missing one.
    """
    from nablaframe.reports import validate_report

    reports = {}
    for gate_id in REPORT_FILES:
        if gate_id in PASSING:
            if gate_id == "G5.6":
                reports[gate_id] = p5._require(gate_id, report_path(gate_id),
                                               why="G6.7 opens the viewer at the addresses G5.6 built")
            else:
                reports[gate_id] = _common.require(gate_id)
            continue
        path = report_path(gate_id)
        rel = p1.rel(path)
        if not path.exists():
            raise SystemExit(f"{rel} is missing. Run {gate_id} first. No report written.")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{rel} can't be read ({exc}). No report written.") from exc
        problems = validate_report(report) if isinstance(report, dict) else ["not a JSON object"]
        if problems:
            raise SystemExit(f"{rel} is not a valid report: {'; '.join(problems)}. No report written.")
        if report["gate_id"] != gate_id:
            raise SystemExit(f"{rel} is a {report['gate_id']} report, not {gate_id}'s. No report written.")
        reports[gate_id] = report
    return reports


def made_badges(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every phase's badges of the reports, as one mapping. SystemExit, a setup error, when any refuses them."""
    from nablaframe.views.badges import model_level_badges
    from nablaframe.views.dictionary_badges import dictionary_badges
    from nablaframe.views.gesture_badges import gesture_badges

    try:
        return {**model_level_badges(reports), **dictionary_badges(reports), **gesture_badges(reports)}
    except ValueError as exc:
        raise SystemExit(f"The reports don't make badges: {exc}. No report written.") from exc


def g56_addresses(g56: dict[str, Any]) -> dict[str, str]:
    """G5.6's address for every run. SystemExit, a setup error, when one is missing."""
    found = g56["measurements"].get("addresses")
    missing = [run for run in RUNS if not isinstance(found, dict) or not isinstance(found.get(run), str)]
    if missing:
        raise SystemExit(f"G5.6's report has no address for {', '.join(missing)}. Run G5.6 again. No report written.")
    return found


# ---------------------------------------------------------------------------
# The runs and their requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Planned:
    """One request, the step of the docstring that made it, and what it asks, for the progress lines."""

    step: str
    what: str
    request: Any


def zoo_run(name: str, g41: dict[str, Any]) -> SimpleNamespace:
    """A zoo run at its final checkpoint, its TopK SAEs, and its context: every block's, and the seeds as G6.5's."""
    m = _common.zoo_model(name, g41)
    _common.say(f"{GATE_ID}: {name} run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows")
    dictionaries = [_common.topk(m, l) for l in range(m.n_layers)]
    _, block, _ = _common.GENERALIZE[name]
    seeds = {s: _common.topk(m, block, seed=s) for s in p5.ROBUST_SEEDS}
    content = _common.core.content_hash
    ctx = _common.context(m, [*dictionaries, *seeds.values()],
                          {s: {content(dictionaries[block]): content(d)} for s, d in seeds.items()})
    return SimpleNamespace(name=name, m=m, dictionaries=dictionaries, ctx=ctx)


def reference_run() -> SimpleNamespace:
    """G1.4's run at its final checkpoint, its TopK SAEs, and its context: every checkpoint, no seeds."""
    from nablaframe.gestures import cli

    m = _common.reference_model()
    _common.say(f"{GATE_ID}: reference run {m.run_id}, final step {m.step:,}; {len(m.held_out)} held-out windows")
    dictionaries = [_common.topk(m, l) for l in range(m.n_layers)]
    ref = m.ref

    def params_at(step: int) -> Any:
        return m.params if step == m.step else ref.params_for(step)

    def state_at(step: int) -> str:
        sidecar = p1.trace_checkpoint(ref.manifest, step).with_suffix(".json")
        return json.loads(sidecar.read_text(encoding="utf-8"))["state_sha256"]

    ctx = cli.Context(run=m.run_id, declaration=m.declaration, model=m.model, n_layers=m.n_layers,
                      steps=tuple(int(s) for s in ref.steps), params_at=params_at, state_at=state_at,
                      windows=m.held_out, dictionary=_common.held(dictionaries), seeds={})
    return SimpleNamespace(name="reference", m=m, dictionaries=dictionaries, ctx=ctx)


def plan(run: Any) -> None:
    """Every request of the run, in the docstring's order, then merged, each with its key.

    Sets ``run.planned`` (merged, in order), ``run.requested`` (per kind, before merging), ``run.keys`` (by
    request hash) and, for the matches and the intermediate, ``run.values`` or ``run.intermediate``.
    """
    from nablaframe.gestures import cli, records

    made = plan_zoo(run) if run.name != "reference" else plan_reference(run)
    run.requested = {kind: sum(1 for p in made if p.request.kind == kind) for kind in KINDS}
    merged: dict[str, Planned] = {}
    for p in made:
        merged.setdefault(records.request_hash(p.request), p)
    run.planned = list(merged.values())
    with _common.ticking(f"{GATE_ID}: {run.name}: the keys of {len(run.planned)} requests", write=_common.say):
        run.keys = {h: records.record_key(p.request, cli.inputs(p.request, run.ctx), cli.code(p.request.kind))
                    for h, p in merged.items()}
    _common.say(f"{GATE_ID}: {run.name}: {len(made)} requests, {len(run.planned)} once equal ones are merged ("
                + ", ".join(f"{kind} {run.requested[kind]}" for kind in KINDS) + " before merging)")


def _request(run: Any, kind: str, args: dict[str, Any], instances: int, seed: int) -> Any:
    from nablaframe.gestures.records import GestureRequest

    return GestureRequest(kind, run.m.run_id, run.m.step, args, instances, seed)


def plan_zoo(run: Any) -> list[Planned]:
    m, content = run.m, _common.core.content_hash
    hashes = [content(d) for d in run.dictionaries]
    readings: dict[int, Any] = {}

    def reading(block: int) -> Any:
        if block not in readings:
            readings[block] = _common.read(m, m.params, run.dictionaries[block],
                                           f"{GATE_ID}: {run.name}: reading block {block}")
        return readings[block]

    def name(block: int, cls: str, latent: int) -> dict[str, Any]:
        return {"dictionary": hashes[block], "site": 0, "class": cls, "latent": int(latent)}

    made: list[Planned] = []
    run.intermediate = []
    places = [planted.place for planted in m.declaration.planted]
    for place in places:  # 1. names: every value pointed at its best latent by F1
        where = _common.place_name(run.name, place.description, place.block, place.cls)
        table = reading(place.block).table(place.cls)
        for v in (int(u) for u in table.values(place.description)):
            latent = int(table.ranked(place.description, v)[0])
            request = _request(run, "name", name(place.block, place.cls, latent), _common.INSTANCES["name"],
                               _common.SEEDS["name"])
            made.append(Planned("names", f"{where}, value {v}, latent {latent}", request))
            if (run.name, place.description, place.block, place.cls) == INTERMEDIATE:
                run.intermediate.append((v, latent, request))
    for place in places:  # 2. the readout: the latents the features view lists at each probe window's last position
        where = _common.place_name(run.name, place.description, place.block, place.cls)
        for window, latents in enumerate(readout(m, run.dictionaries[place.block])):
            for latent in latents:
                request = _request(run, "name", name(place.block, place.cls, latent), _common.INSTANCES["name"],
                                   _common.SEEDS["name"])
                made.append(Planned("readout", f"{where}, probe window {window}, latent {latent}", request))
    for R in m.declaration.relations:  # 3. edges, as G6.3 asks for them
        args = {"relation": R.name, "up": hashes[R.up.block], "up_site": 0, "down": hashes[R.down.block],
                "down_site": 0}
        made.append(Planned("edges", f"{run.name} {R.name}",
                            _request(run, "edge", args, _common.INSTANCES["edge"], _common.SEEDS["edge"])))
    for relation in _common.OPEN.get(run.name, {}):  # 4. open-this-up: every value of each judged target
        R = m.declaration.relation(relation)
        table = reading(R.down.block).table(R.down.cls)
        for u in (int(x) for x in table.values(R.down.description)):
            args = {"dictionary": hashes[R.down.block], "site": 0, "class": R.down.cls, "upstream": hashes[R.up.block],
                    "upstream_site": 0, "description": R.down.description, "value": u}
            made.append(Planned("open", f"{run.name} {relation}, value {u}",
                                _request(run, "open", args, _common.INSTANCES["open"], _common.SEEDS["open"])))
    description, block, cls = _common.GENERALIZE[run.name]  # 5. generalize-this, G6.5's request
    args = {"dictionary": hashes[block], "site": 0, "class": cls, "description": description}
    run.generalize = _request(run, "generalize", args, _common.INSTANCES["name"], _common.SEEDS["generalize"])
    made.append(Planned("generalize", _common.place_name(run.name, description, block, cls), run.generalize))
    readings.clear()
    return made


def readout(m: Any, dictionary: Any) -> list[list[int]]:
    """Per probe window, the latents active at its last position, largest code first, ties to the lower latent,
    up to ``READOUT_LATENTS``: what the features view lists there."""
    from nablaframe.dictionaries import core, measure

    probe = p4.tokens(m.task, p4.probe(m.task))
    input_tap, output_tap = dictionary.sites[0]
    width = dictionary.active_width(0)
    seen = measure.activations(m.model, m.params, probe, [input_tap, output_tap], chunk=m.chunk)
    x = seen[input_tap].reshape(-1, dictionary.d_in)
    codes = np.asarray(core.encode(dictionary, 0, x))[:, :width].reshape(probe.shape[0], probe.shape[1], width)
    out = []
    for row in codes[:, -1]:
        active = np.flatnonzero(row > 0)
        ranked = active[np.lexsort((active, -row[active]))]  # largest code first, ties to the lower latent
        out.append([int(latent) for latent in ranked[:READOUT_LATENTS]])
    return out


def plan_reference(run: Any) -> list[Planned]:
    m, content = run.m, _common.core.content_hash
    hashes = [content(d) for d in run.dictionaries]
    d, cls = REFERENCE_DESCRIPTION, REFERENCE_CLASS
    per_value, seed = _common.REFERENCE["per_value"], _common.SEEDS["reference"]
    made: list[Planned] = []
    run.values = {}
    for l, dictionary in enumerate(run.dictionaries):
        reading = _common.read(m, m.params, dictionary, f"{GATE_ID}: reference: reading block {l}")
        table = reading.table(cls)
        rng = np.random.default_rng([seed, l])
        sample = _common.reference_sample(table, _common.REFERENCE["latents"], rng)
        values = _common.reference_values(table, d, _common.REFERENCE["values"], rng)
        run.values[l] = values
        for latent in sample:
            args = {"dictionary": hashes[l], "site": 0, "class": cls, "latent": int(latent)}
            made.append(Planned("reference names", f"block {l}, latent {latent}",
                                _request(run, "name", args, per_value, seed)))
        for v in values:
            latent = int(table.ranked(d, v)[0])
            args = {"dictionary": hashes[l], "site": 0, "class": cls, "latent": latent}
            made.append(Planned("steered values", f"block {l}, {d} = {v}, latent {latent}",
                                _request(run, "name", args, per_value, seed)))
        if l >= 1:
            for v in values:
                args = {"dictionary": hashes[l], "site": 0, "class": cls, "upstream": hashes[l - 1],
                        "upstream_site": 0, "description": d, "value": v}
                made.append(Planned("reference open", f"block {l} into block {l - 1}, {d} = {v}",
                                    _request(run, "open", args, _common.REFERENCE["open"], seed)))
        _common.say(f"{GATE_ID}: reference block {l}: {len(sample)} latents sampled, {len(values)} values steered")
        del reading, table
    return made


# ---------------------------------------------------------------------------
# Two builds
# ---------------------------------------------------------------------------


def _record_files(folder: Path) -> dict[str, str]:
    """``{key: what names it}`` for every record the folder holds: its gesture files and its index's entries."""
    held = {}
    for path in sorted(Path(folder).glob("gesture-*.json")):
        held[path.name[len("gesture-"):-len(".json")]] = path.name
    index = Path(folder) / "index.json"
    if index.is_file():
        try:
            entries = json.loads(index.read_text(encoding="utf-8")).get("records", [])
        except (OSError, ValueError, AttributeError) as exc:
            raise SystemExit(f"{p1.rel(index)} can't be read ({exc}). No report written.") from exc
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                held.setdefault(str(entry.get("key")), f"{index.name}'s entry {entry.get('key')}")
    return held


def refuse_strays(run: Any, first: Path) -> None:
    """Stop the gate when the run's record folder holds a record whose key isn't one of G6.7's requests'."""
    keys = set(run.keys.values())
    strays = {key: where for key, where in _record_files(first).items() if key not in keys}
    if strays:
        key, where = next(iter(sorted(strays.items())))
        raise SystemExit(f"{p1.rel(first)} holds {len(strays)} record(s) G6.7 doesn't ask for, the first {where},"
                         f" key {key}. Its key changed, or it was built by other code; move the folder aside and run"
                         " G6.3, G6.5 and G6.7 again. No report written.")
    held = set(_record_files(first)) if first.exists() else set()
    _common.say(f"{GATE_ID}: {run.name}: {p1.rel(first)} holds {len(held)} of the {len(keys)} records already")


def build(run: Any, folder: Path, label: str) -> None:
    """Every planned request of the run through ``cli.record`` into ``folder``, with a progress line per record."""
    from nablaframe.gestures import cli, records

    held = set(_record_files(folder)) if folder.exists() else set()
    total, done = len(run.planned), [0]

    def tick(line: str) -> None:
        _common.say(f"{line}; {done[0]} of {total} records built")

    began = time.perf_counter()
    with _common.ticking(f"{GATE_ID}: {run.name}, {label}", write=tick):
        for p in run.planned:
            key = run.keys[records.request_hash(p.request)]
            start = time.perf_counter()
            record = cli.record(folder, p.request, run.ctx)
            if record.key != key:
                raise RuntimeError(f"{run.name}: the record of {p.what} has key {record.key}, not {key}")
            done[0] += 1
            how = "in the folder already" if key in held else f"built in {time.perf_counter() - start:.1f} s"
            _common.say(f"{GATE_ID}: {run.name}, {label}: {done[0]} of {total} records, {p.step}: {p.what}, key"
                        f" {key[:12]}, {how}")
    _common.say(f"{GATE_ID}: {run.name}, {label}: {total} records in {p1._duration(time.perf_counter() - began)}")


def build_twice(run: Any, first: Path, second: Path, *, overwrite: bool) -> dict[str, Any]:
    """The first build through the record cache in ``first``, the second from nothing in ``second``; each folder's
    panel written; the two compared file by file."""
    from nablaframe.views.gestures import write_panel

    build(run, first, "first build")
    _common.fresh_dir(second, overwrite=overwrite)
    build(run, second, "second build")
    for folder in (first, second):
        write_panel(folder)
    comparison = p4.p3.compare_trees(first, second)
    _common.say(f"{GATE_ID}: {run.name}: the two builds are {'byte-identical' if comparison['identical'] else 'different'}"
                f" over {comparison['files_compared']:,} files")
    return {**comparison, "first_build_dir": p1.rel(first), "second_build_dir": p1.rel(second)}


def held_by_kind(folder: Path) -> dict[str, int]:
    """How many records of each kind the folder's index lists."""
    entries = json.loads((Path(folder) / "index.json").read_text(encoding="utf-8"))["records"]
    return {kind: sum(1 for e in entries if e["kind"] == kind) for kind in KINDS}


# ---------------------------------------------------------------------------
# The records shown are the ones judged
# ---------------------------------------------------------------------------


def match_reports(runs: dict[str, Any], folders: dict[str, tuple[Path, Path]],
                  reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``reports_match`` and ``matches``: G6.3's edge records and G6.5's generalize records, by key, in each run's
    first folder, and G6.6's steered values, block by block, the reference requests'."""
    held = {name: set(_record_files(folders[name][0])) for name in RUNS}
    matches: dict[str, list[dict[str, Any]]] = {"edges": [], "generalize": [], "reference": []}
    for part, gate_id, mapping in (("edges", "G6.3", "edges"), ("generalize", "G6.5", "names")):
        for place, entry in sorted((reports[gate_id]["measurements"].get(mapping) or {}).items()):
            run, key = place.split(" ", 1)[0], (entry or {}).get("record")
            found = run in held and key in held[run]
            matches[part].append({"place": place, "run": run, "key": key, "in_folder": bool(found)})
    blocks = reports["G6.6"]["measurements"].get("blocks") or {}
    ours = runs["reference"].values
    for l in sorted({*map(int, blocks), *ours}):
        recorded = ((blocks.get(str(l)) or {}).get("steering") or {}).get("values")
        mine = ours.get(l)
        matches["reference"].append({"block": l, "g66": recorded, "g67": mine,
                                     "equal": recorded is not None and mine is not None and list(recorded) == mine})
    ok = (bool(matches["edges"]) and bool(matches["generalize"]) and bool(matches["reference"])
          and all(m["in_folder"] for part in ("edges", "generalize") for m in matches[part])
          and all(m["equal"] for m in matches["reference"]))
    for part, entries in matches.items():
        good = sum(bool(m.get("in_folder", m.get("equal"))) for m in entries)
        _common.say(f"{GATE_ID}: {part}: {good} of {len(entries)} match")
    _common.say(f"{GATE_ID}: the records shown are the ones judged: {ok}")
    return {"reports_match": ok, "matches": matches}


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def badges_document() -> tuple[dict[str, Any], bytes]:
    """The badges file's contents, from the report files as they are now: ``(badges, canonical JSON bytes)``."""
    from nablaframe.trace.schema import canonical_json, sha256_hex

    reports, cited = {}, {}
    for gate_id in BADGE_GATES:
        data = report_path(gate_id).read_bytes()
        reports[gate_id] = json.loads(data.decode("utf-8"))
        cited[gate_id] = {"git_commit": reports[gate_id]["git_commit"],
                          "timestamp_utc": reports[gate_id]["timestamp_utc"], "sha256": sha256_hex(data)}
    made = made_badges(reports)
    return made, canonical_json({"badges": made, "reports": cited}).encode("ascii")


def subjects() -> tuple[str, ...]:
    from nablaframe.views import badges, dictionary_badges, gesture_badges

    return tuple(badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS) + tuple(gesture_badges.SUBJECTS)


def badges_twice(files: tuple[Path, Path]) -> dict[str, Any]:
    """The badges, built twice, into ``runs/phase6/badges.json`` and the scratch folder, compared byte for byte."""
    from nablaframe.trace.schema import sha256_hex

    made = None
    for path in files:
        made, data = badges_document()
        p1.write_atomic(path, data)
    first, second = (path.read_bytes() for path in files)
    identical = first == second
    missing = [subject for subject in subjects() if subject not in made]
    _common.say(f"{GATE_ID}: {len(made)} badges written to {p1.rel(files[0])}; the second build is"
                f" {'byte-identical' if identical else 'different'}; {len(missing)} subjects without a badge")
    for subject in subjects():
        if subject in made:
            _common.say(f"{GATE_ID}:   {subject}: {made[subject]['status']}, {len(made[subject]['facts'])} facts")
    return {"badges": made, "subjects_missing": len(missing), "subjects_without_badge": missing,
            "badges_file": p1.rel(files[0]), "badges_sha256": sha256_hex(first),
            "build": {"identical": identical, "files_compared": 1, "files_differing": [] if identical else [BADGES_NAME]}}


# ---------------------------------------------------------------------------
# The viewer, the intermediate and the addresses
# ---------------------------------------------------------------------------


def load_gate(path: Path, common: Any) -> Any:
    """An earlier phase's gate script, loaded by path under a name of its own, its ``_common`` being ``common``."""
    name = f"nablaframe_gates_{path.parent.name}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    saved_common, saved_path = sys.modules.get("_common"), list(sys.path)
    sys.modules["_common"] = common
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    finally:
        if saved_common is None:
            sys.modules.pop("_common", None)
        else:
            sys.modules["_common"] = saved_common
        sys.path[:] = saved_path
    return module


def external_urls() -> dict[str, Any]:
    """External references in the viewer's files, counted as G4.5 counts them."""
    g45 = load_gate(Path(__file__).resolve().parents[1] / "phase4" / "g45_formation_viewer.py", p4)
    return g45.offline(Path(p1.REPO_ROOT) / "viewer")


def intermediate(run: Any, first: Path) -> dict[str, Any]:
    """Two-hop's ``b`` after block 2: its generalize record's key and envelope, and each value's name record and meter."""
    from nablaframe.gestures import records
    from nablaframe.views.gestures import PANEL

    run_name, description, block, cls = INTERMEDIATE
    panel = json.loads((first / PANEL).read_text(encoding="utf-8"))
    meters = {entry["key"]: entry["meter"] for entry in panel["records"]}
    key = run.keys[records.request_hash(run.generalize)]
    envelope = _common.results_of(records.read_record(first, key))["envelope"]
    values = []
    for v, latent, request in run.intermediate:
        name_key = run.keys[records.request_hash(request)]
        values.append({"value": v, "latent": latent, "key": name_key, "meter": meters.get(name_key)})
    return {"place": _common.place_name(run_name, description, block, cls), "generalize": {"key": key,
            "envelope": envelope}, "values": values}


def addresses(g56: dict[str, Any], badges_file: Path, folders: dict[str, tuple[Path, Path]]) -> dict[str, str]:
    """G5.6's address for each run, its ``badges=`` pointing at Phase 6's badges file and ``gestures=`` at the run's
    first record folder, both relative to viewer/, added before the fragment."""
    viewer = Path(p1.REPO_ROOT) / "viewer"
    given = g56_addresses(g56)

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, viewer)).as_posix()

    out = {}
    for run in RUNS:
        query, mark, fragment = given[run].partition("#")
        badges = f"badges={relative(badges_file)}"
        query, found = re.subn(r"(?<=[?&])badges=[^&]*", lambda _: badges, query)
        if not found:
            query += ("&" if "?" in query else "?") + badges
        query += f"&gestures={relative(folders[run][0])}/"
        out[run] = query + mark + fragment
    return out


if __name__ == "__main__":
    raise SystemExit(main())
