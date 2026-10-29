"""G7.5: modular addition's views, every Phase 7 record built twice, the gestures panels, and the badges (D38, D47, D53, D59).

Spec: docs/specs/phase-7.md, "Records, the viewer and badges" and "Gates";
docs/deltas/phase-7-delta-4.md; helpers in ``_common.py``. Report:
``reports/phase7/g75_records_views.json``. Run in nf-jax, on the Mac, after G7.1 to G7.4 pass and
G7.2 has run again with its ``meters``.

Refuses to start unless G4.1's, G5.6's, G6.7's and G7.1 to G7.4's reports pass, and G4.2 to G4.4's,
G5.1 to G5.4's and G6.1 to G6.5's exist and pass ``validate_report``, whatever their status: a
failed judged gate shows on its badge, never hidden by a missing one. G7.2's report must hold
``meters``; one without them is a setup error that says to run G7.2 again.

**Records.** Each run's requests, made in the order listed; one equal to an earlier one is built
once.

* ``modular_addition``, at G7.1's final checkpoint, its site read once with Phase 6's ``read`` on
  G7.2's TopK SAE (``_common.dictionary``, from the cache):

  1. **Names:** for each of ``NAMES`` at ``answer``, for each of its values ``v``, ascending, a name
     request pointed at ``v``'s best latent by F1, ``table.ranked(d, v)[0]``, at site 0, with Phase
     6's ``INSTANCES["name"]`` and ``SEEDS["name"]``: G7.2's records.
  2. **The readout:** on the probe windows (``probe(task)``, the tokens the model reads), the
     latents active at position 2, ``answer``, largest code first, ties to the lower latent, up to 8
     per window: what the features view lists there. The codes are ``core.encode`` of the site's
     input tap, from ``measure.activations`` at the final params with ``CHUNK``. A name request for
     each, at ``answer``, with the names' instances and seed.
  3. **Crystallizations:** for each of ``JUDGED`` and then ``RECORDED`` on the run, in order, the
     request of the record G7.4's report names, ``crystallizations[<name>]["key"]``, read from the
     run's records folder. A record missing there, or whose request isn't that crystallization's
     (its block, part, class, function and form, the run's final step, ``instances`` the held-out
     windows' count and ``SEED``), is a setup error.

* ``induction`` and ``two_hop``, at G4.1's final checkpoint: their crystallizations, as in step 3.

The name records' context holds the run's checkpoints, its held-out pairs and G7.2's dictionary, as
G7.2's did. Each crystallization's is the run's context as G7.4 built it (``_common.context`` with
every checkpoint and ``cited``), ``cited`` reading, for modular addition, the folder the build
writes into, whose name records are built before any crystallization, and for induction and two-hop,
Phase 6's records, ``runs/phase6/gestures/<run>/``. Every record goes through ``cli.record``.

**Two builds.** Before anything is built, every request's key is found, ``records.record_key`` of the
request, ``cli.inputs`` and ``cli.code``. A run's records folder, ``record_dir(run)``, holding a
record whose key isn't one of them stops the gate, naming it. The first build goes into that
folder, through the record cache, so the records G7.2 and G7.4 wrote are kept as they are; the
second into ``RUN_DIR / "scratch" / "gestures" / run``, which starts empty. Each folder then gets
``views.gestures.write_panel``. The two are compared file by file: every file in either folder, by
name, with its bytes. Phase 6's records and panels stay as they are (D59).

**Views.** Modular addition's views, built as G4.5 and G5.6 build a zoo run's: ``build_views`` over
G7.1's trace store, which must hold exactly G7.1's checkpoints, each with its logits, traced on the
probe; ``params_for`` its checkpoints; ``probe`` what the model reads of ``probe(task)``; ``vocab``
the task's; ``next_token`` the probe windows' last tokens; ``curves`` one curve, ``"answer
position's loss"`` in ``"nats"``: at each checkpoint, the mean over the probe of ``-log
softmax(logits[b, 2])[window[b, 3]]`` from the traced logits, in float64, null where it isn't
finite; and ``features``, one ``features_payload`` of G7.2's TopK SAE at site 0 with the final
params and step, that probe, ``held_out`` the tokens the model reads of the held-out pairs,
``meters`` ``{"gate": "G7.2", "path": ["meters"], "values": <that entry of G7.2's report>}``,
``variables`` each held-out pair's ``a``, ``b`` and ``c``, its tokens 0, 1 and 3, and ``chunk``
``CHUNK``. The first build goes to ``runs/phase7/views/modular_addition/`` and the second to
``runs/phase7/scratch/views/modular_addition/``; both start empty, and each computes its payloads
afresh. Compared file by file.

**Badges.** ``runs/phase7/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1 to
G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, then gesture_badges of G6.1 to G6.5, then
crystallize_badges of G7.2 to G7.4, as one mapping>, "reports": {<gate id>: {"git_commit",
"timestamp_utc", "sha256"}}}`` for each of those sixteen reports, the last being the report file's
SHA-256. Built a second time into ``runs/phase7/scratch/badges.json`` and compared byte for byte.
One file holds every phase's badges, with D53's correction to the open-this-up badge, so the viewer
reads it with ``?badges=`` for every run (D59).

An existing second records folder, view folder or badges file, or an existing first view folder,
is replaced only with ``--overwrite``; a first records folder is never cleared.

Measurements: REQUIRED_MEASUREMENTS["G7.5"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``,
  ``dictionary_badges.SUBJECTS``, ``gesture_badges.SUBJECTS`` and ``crystallize_badges.SUBJECTS``
  without a badge.
* ``records``: per run, ``{"name", "crystallize"}``, how many records of each kind its first folder
  holds.
* ``builds_identical``: the two badges files hold the same bytes, the two view folders the same
  files with the same bytes, and each run's two records folders the same files with the same bytes.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.
* ``reports_match``: each crystallization's record, as built here, has the key G7.4's report names,
  and each name record G7.2's report names (``record_keys``) is in modular addition's first folder.
  So the records the panel shows are the ones G7.4 measured and G7.2 recorded.

Recorded beside them: ``builds``, each part's detail (files compared and files differing);
``matches``, every key compared; ``run_ids``; ``steps``; ``requests``, per run and step, how many
requests were made before equal ones were merged; ``readout``, per probe window, the latents of step
2; ``clock``, modular addition's fourier crystallization from its first folder: its key, and its
results' ``frequencies``, ``envelope`` and ``cites``; ``features``, the features payload's file and
dictionary; and ``addresses``.

Pass needs ``builds_identical``, ``external_urls == 0``, ``subjects_missing == 0`` and
``reports_match``. Every long step prints a progress line at least every 30 seconds, saying how many
records of how many it has built, and each crystallization a line per checkpoint, as G7.4 does. At
the end it prints the viewer's address for each run:

* modular addition: ``http://localhost:8000/viewer/?views=../runs/phase7/views/modular_addition/
  &trace=../runs/phase7/modular_addition/store/&badges=../runs/phase7/badges.json
  &gestures=../runs/phase7/gestures/modular_addition/#step=<final>&seq=0&pos=2
  &features=<dictionary>-0``, as one line
* each of G6.7's runs: G6.7's address, from its report's ``addresses``, with ``badges=`` pointing at
  ``../runs/phase7/badges.json``, and for induction and two-hop ``,../runs/phase7/gestures/<run>/``
  added after its Phase 6 records folder, so the viewer reads both (``viewer/app.js``)
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase7/g75_records_views.py``. Takes --overwrite and --allow-dirty. --overwrite also lets it replace the
#   second build's records folders, both view folders and both badges files; the first records folders are never
#   cleared.
# * Start checks, before anything is written: the reports, as above (the passing ones through ``_common.require``,
#   G5.6's through Phase 5's ``_require``); G7.2's ``meters``; every phase's badges of the reports, whose
#   ValueError is a setup error; G6.7's addresses; then the runs, G7.1's trace store, G7.2's dictionary from the
#   cache, every request and its key, and every first records folder checked for a record G7.5 doesn't ask for.
#   Each fails with exit 1 and no report.
# * Runs in the order ``modular_addition``, ``induction``, ``two_hop``. Modular addition's site is read once to
#   make the name requests and dropped before anything is built; every record is then built by ``cli.record``,
#   which reads its own sites.
# * The readout's codes: ``measure.activations`` of the probe tokens at the site's input and output taps with
#   ``CHUNK``, as ``features_payload`` takes them, then one ``core.encode`` call over every probe position.
# * Requests are merged by request hash, keeping the first; each is labelled by the step that made it first.
# * Contexts: modular addition's name records use ``_common.context`` with G7.2's dictionary, as G7.2's did; each
#   crystallization uses ``_common.context`` with ``cited`` the folder named above, its ``params_at`` wrapped in
#   ``_common.announced`` so each checkpoint it loads prints a line, as G7.4's do. A crystallization the record
#   cache already holds is read, so it prints no checkpoint lines.
# * ``external_urls``: G4.5's ``offline``, the script loaded by path with Phase 4's ``_common``, as G6.7 loads it;
#   its ``curve`` makes the timeline's curve.
# * ``builds``: ``{"badges": {...}, "views": {...}, <run>: {...}}``, each ``{"identical", "files_compared",
#   "files_differing"}``, the views' and each run's also naming their two folders. ``matches``:
#   ``{"crystallizations", "names"}``: each crystallization's name, run, the key G7.4's report names and the key
#   built here, with whether they are equal; and each latent of G7.2's ``record_keys`` with its key and whether
#   modular addition's first folder holds it.
# * ``requests``: modular addition's ``{"names", "readout", "crystallizations"}``, the zoo runs'
#   ``{"crystallizations"}``.

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p4, p5, p6 = _common.p1, _common.p4, _common.p5, _common.p6

GATE_ID = "G7.5"
REPORT_NAME = "g75_records_views.json"
REPORT_FILES = {
    "G4.1": ("phase4", "g41_zoo_training.json"), "G4.2": ("phase4", "g42_induction.json"),
    "G4.3": ("phase4", "g43_two_hop.json"), "G4.4": ("phase4", "g44_lenses.json"),
    "G5.1": ("phase5", "g51_determinism.json"), "G5.2": ("phase5", "g52_induction.json"),
    "G5.3": ("phase5", "g53_two_hop.json"), "G5.4": ("phase5", "g54_edges.json"),
    "G5.6": ("phase5", "g56_features_badges.json"),
    "G6.1": ("phase6", "g61_determinism.json"), "G6.2": ("phase6", "g62_names.json"),
    "G6.3": ("phase6", "g63_edges.json"), "G6.4": ("phase6", "g64_open.json"),
    "G6.5": ("phase6", "g65_generalize.json"), "G6.7": ("phase6", "g67_records_panel.json"),
    "G7.1": ("phase7", "g71_specimen.json"), "G7.2": ("phase7", "g72_names.json"),
    "G7.3": ("phase7", "g73_determinism.json"), "G7.4": ("phase7", "g74_crystallizations.json"),
}
PASSING = ("G4.1", "G5.6", "G6.7", "G7.1", "G7.2", "G7.3", "G7.4")
# The passing reports _common.require doesn't read, each with why G7.5 needs it to pass.
WHY = {"G5.6": "G5.6 built the zoo runs' views, which G6.7's addresses open",
       "G6.7": "G7.5 opens the viewer at the addresses G6.7 built",
       "G7.4": "G7.5 shows the crystallizations G7.4 measured, and builds them again from its records"}
BADGE_GATES = ("G4.1", "G4.2", "G4.3", "G4.4", "G5.1", "G5.2", "G5.3", "G5.4", "G6.1", "G6.2", "G6.3", "G6.4", "G6.5",
               "G7.2", "G7.3", "G7.4")
RUNS = (_common.RUN_NAME, "induction", "two_hop")
G67_RUNS = ("two_hop", "induction", "reference")
KINDS = ("name", "crystallize")
CLASS, ANSWER = "answer", 2
READOUT_LATENTS = 8
BADGES_NAME = "badges.json"
VIEWS_NAME = "views"
VIEWER_ADDRESS = "http://localhost:8000/viewer/"
ANSWER_CURVE = ("answer position's loss", "nats")
METERS = ("fvu", "twin_fvu")
VARIABLES = (("a", 0), ("b", 1), ("c", 3))


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    reports = read_reports()
    meters = g72_meters(reports["G7.2"])
    made_badges(reports)
    g67_addresses(reports["G6.7"])
    run_dir = Path(_common.RUN_DIR)
    badge_files = (run_dir / BADGES_NAME, run_dir / "scratch" / BADGES_NAME)
    view_folders = (run_dir / VIEWS_NAME / _common.RUN_NAME, run_dir / "scratch" / VIEWS_NAME / _common.RUN_NAME)
    folders = {run: (_common.record_dir(run), run_dir / "scratch" / "gestures" / run) for run in RUNS}
    existing = [path for path in (*badge_files, *view_folders) if path.exists()]
    existing += [second for _, second in folders.values() if second.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{p1.rel(existing[0])} exists. Pass --overwrite to start it over.")

    g45 = load_gate(Path(__file__).resolve().parents[1] / "phase4" / "g45_formation_viewer.py", p4)
    runs = {_common.RUN_NAME: modadd_run(reports["G7.1"])}
    runs |= {name: zoo_run(name, reports["G4.1"]) for name in RUNS[1:]}
    reader = open_reader(runs[_common.RUN_NAME])
    for name in RUNS:
        plan(runs[name], reports["G7.4"])
    for name in RUNS:
        refuse_strays(runs[name], folders[name][0])

    builds, counts = {}, {}
    for name in RUNS:
        builds[name] = build_twice(runs[name], *folders[name], overwrite=args.overwrite)
        counts[name] = held_by_kind(folders[name][0])
    matched = match_reports(runs, folders, reports)
    viewed = views_twice(runs[_common.RUN_NAME], reader, view_folders, meters, g45, overwrite=args.overwrite)
    badges = badges_twice(badge_files)
    builds = {"badges": badges.pop("build"), "views": viewed.pop("build"), **builds}
    measurements = {
        **badges,
        "records": counts,
        "builds_identical": all(part["identical"] is True for part in builds.values()),
        "builds": builds,
        **external_urls(g45),
        **matched,
        "run_ids": {name: runs[name].m.run_id for name in RUNS},
        "steps": {name: runs[name].m.step for name in RUNS},
        "requests": {name: runs[name].requested for name in RUNS},
        "readout": runs[_common.RUN_NAME].readout,
        "clock": clock(runs[_common.RUN_NAME], folders[_common.RUN_NAME][0]),
        **viewed,
        "addresses": addresses(runs[_common.RUN_NAME], reports["G6.7"], view_folders[0], badge_files[0], folders),
        "record_dirs": {name: p1.rel(folders[name][0]) for name in RUNS},
        "settings": {"instances": p6.INSTANCES["name"], "seed": p6.SEEDS["name"], "crystallize_seed": _common.SEED,
                     "readout_latents": READOUT_LATENTS, "chunk": _common.CHUNK},
        "versions": p1.phase0.package_versions(_common.VERSIONS),
    }
    code = p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)
    print(f"{GATE_ID}: serve the repo root with `python3 -m http.server`, then open:", flush=True)
    for name, address in measurements["addresses"].items():
        print(f"  {name}: {address}", flush=True)
    return code


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G7.5 rule holds."""
    ok = (m["builds_identical"] is True and m["external_urls"] == 0 and m["subjects_missing"] == 0
          and m["reports_match"] is True)
    return "pass" if ok else "fail"


# ---------------------------------------------------------------------------
# Start checks: the reports
# ---------------------------------------------------------------------------


def report_path(gate_id: str) -> Path:
    phase, name = REPORT_FILES[gate_id]
    folder = {"phase4": p4.REPORT_DIR, "phase5": p5.REPORT_DIR, "phase6": p6.REPORT_DIR,
              "phase7": _common.REPORT_DIR}[phase]
    return Path(folder) / name


def read_reports() -> dict[str, dict[str, Any]]:
    """Every report G7.5 reads, each from its file and checked. SystemExit otherwise.

    G4.1, G5.6, G6.7 and G7.1 to G7.4 must pass; the rest need only exist and be valid, whatever their status: a
    failed judged gate shows on its badge, never hidden by a missing one.
    """
    from nablaframe.reports import validate_report

    reports = {}
    for gate_id in REPORT_FILES:
        if gate_id in PASSING:
            if gate_id in WHY:
                reports[gate_id] = p5._require(gate_id, report_path(gate_id), why=WHY[gate_id])
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


def g72_meters(g72: dict[str, Any]) -> dict[str, Any]:
    """G7.2's ``meters``, which the features payload cites. SystemExit, a setup error, for a report without them."""
    meters = g72["measurements"].get("meters")
    if not isinstance(meters, dict) or any(name not in meters for name in METERS):
        raise SystemExit(f"{p1.rel(report_path('G7.2'))} holds no meters {{{', '.join(METERS)}}}: it was written"
                         " before G7.2 recorded them. Run G7.2 again with --overwrite; its dictionaries and records"
                         " come from the cache. No report written.")
    return meters


def made_badges(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Every phase's badges of the reports, as one mapping. SystemExit, a setup error, when any refuses them."""
    from nablaframe.views.badges import model_level_badges
    from nablaframe.views.crystallize_badges import crystallize_badges
    from nablaframe.views.dictionary_badges import dictionary_badges
    from nablaframe.views.gesture_badges import gesture_badges

    try:
        return {**model_level_badges(reports), **dictionary_badges(reports), **gesture_badges(reports),
                **crystallize_badges(reports)}
    except ValueError as exc:
        raise SystemExit(f"The reports don't make badges: {exc}. No report written.") from exc


def g67_addresses(g67: dict[str, Any]) -> dict[str, str]:
    """G6.7's address for each of its runs, each zoo run's naming its records folder. SystemExit otherwise."""
    found = g67["measurements"].get("addresses")
    missing = [run for run in G67_RUNS if not isinstance(found, dict) or not isinstance(found.get(run), str)]
    if missing:
        raise SystemExit(f"G6.7's report has no address for {', '.join(missing)}. Run G6.7 again. No report written.")
    for run in RUNS[1:]:
        if not re.search(r"[?&]gestures=", found[run].partition("#")[0]):
            raise SystemExit(f"G6.7's address for {run} names no records folder. Run G6.7 again. No report written.")
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


def modadd_run(g71: dict[str, Any]) -> SimpleNamespace:
    """G7.1's run at its final checkpoint, with G7.2's TopK SAE from the cache and its name records' context."""
    from nablaframe.dictionaries import core

    m = _common.modadd_model(g71)
    _common.say(f"{GATE_ID}: {m.name} run {m.run_id}, final step {m.step:,}, {len(m.run.steps)} checkpoints;"
                f" {len(m.fit):,} training and {len(m.held_out):,} held-out pairs")
    dictionary, _ = _common.dictionary(m)
    return SimpleNamespace(name=m.name, m=m, dictionary=dictionary, content=core.content_hash(dictionary),
                           names_ctx=_common.context(m, dictionaries=[dictionary]), cited=None)


def zoo_run(name: str, g41: dict[str, Any]) -> SimpleNamespace:
    """G4.1's run at its final checkpoint; its crystallizations cite Phase 6's records, which stay as they are."""
    m = _common.zoo_model(name, g41)
    cited = Path(p6.RECORD_DIR) / name
    _common.say(f"{GATE_ID}: {name} run {m.run_id}, final step {m.step:,}, {len(m.run.steps)} checkpoints;"
                f" {len(m.fit):,} fitting and {len(m.held_out):,} held-out windows; citing from {p1.rel(cited)}")
    return SimpleNamespace(name=name, m=m, dictionary=None, content=None, names_ctx=None, cited=cited)


def crystal_ctx(run: Any, folder: Path) -> Any:
    """The context of the run's crystallizations, as G7.4 built it: every checkpoint, and ``cited`` reading
    modular addition's records from ``folder``, the one its build writes into, or a zoo run's from Phase 6's."""
    return _common.context(run.m, cited=Path(folder) if run.cited is None else run.cited)


def context_for(run: Any, request: Any, folder: Path) -> Any:
    return crystal_ctx(run, folder) if request.kind == "crystallize" else run.names_ctx


def plan(run: Any, g74: dict[str, Any]) -> None:
    """Every request of the run, in the docstring's order, then merged, each with its key.

    Sets ``run.planned`` (merged, in order), ``run.requested`` (per step, before merging), ``run.keys`` (by request
    hash), ``run.crystallizations`` (``{name: request}``) and, for modular addition, ``run.readout``.
    """
    from nablaframe.gestures import cli, records

    made = []
    if run.name == _common.RUN_NAME:
        made += plan_names(run) + plan_readout(run)
    made += plan_crystallizations(run, g74)
    steps = ("names", "readout", "crystallizations") if run.name == _common.RUN_NAME else ("crystallizations",)
    run.requested = {step: sum(1 for p in made if p.step == step) for step in steps}
    merged: dict[str, Planned] = {}
    for p in made:
        merged.setdefault(records.request_hash(p.request), p)
    run.planned = list(merged.values())
    first = _common.record_dir(run.name)
    with _common.ticking(f"{GATE_ID}: {run.name}: the keys of {len(run.planned)} requests", write=_common.say):
        run.keys = {h: records.record_key(p.request, cli.inputs(p.request, context_for(run, p.request, first)),
                                          cli.code(p.request.kind))
                    for h, p in merged.items()}
    _common.say(f"{GATE_ID}: {run.name}: {len(made)} requests, {len(run.planned)} once equal ones are merged ("
                + ", ".join(f"{step} {n}" for step, n in run.requested.items()) + " before merging)")


def _name_request(run: Any, latent: int) -> Any:
    from nablaframe.gestures.records import GestureRequest

    m = run.m
    return GestureRequest("name", m.run_id, m.step, {"dictionary": run.content, "site": 0, "class": CLASS,
                                                      "latent": int(latent)},
                          p6.INSTANCES["name"], p6.SEEDS["name"])


def plan_names(run: Any) -> list[Planned]:
    """1. For each of ``NAMES`` at ``answer``, each value pointed at its best latent by F1: G7.2's records."""
    m = run.m
    reading = p6.read(m, m.params, run.dictionary, f"{GATE_ID}: {run.name}: reading the site after block 0")
    table = reading.table(CLASS)
    made = []
    for d in _common.NAMES:
        for v in sorted(int(u) for u in table.values(d)):
            latent = int(table.ranked(d, v)[0])
            made.append(Planned("names", f"{d} at {CLASS}, value {v}, latent {latent}", _name_request(run, latent)))
    del reading, table
    return made


def readout(m: Any, dictionary: Any) -> list[list[int]]:
    """Per probe window, the latents active at ``answer``, largest code first, ties to the lower latent, up to
    ``READOUT_LATENTS``: what the features view lists there."""
    from nablaframe.dictionaries import core, measure

    probe = _common.tokens(_common.probe(m.task))
    input_tap, output_tap = dictionary.sites[0]
    width = dictionary.active_width(0)
    seen = measure.activations(m.model, m.params, probe, [input_tap, output_tap], chunk=_common.CHUNK)
    x = seen[input_tap].reshape(-1, dictionary.d_in)
    codes = np.asarray(core.encode(dictionary, 0, x))[:, :width].reshape(probe.shape[0], probe.shape[1], width)
    out = []
    for row in codes[:, ANSWER]:
        active = np.flatnonzero(row > 0)
        ranked = active[np.lexsort((active, -row[active]))]  # largest code first, ties to the lower latent
        out.append([int(latent) for latent in ranked[:READOUT_LATENTS]])
    return out


def plan_readout(run: Any) -> list[Planned]:
    """2. A name request for each latent the features view lists at ``answer`` in each probe window."""
    run.readout = readout(run.m, run.dictionary)
    made = [Planned("readout", f"probe window {window}, latent {latent}", _name_request(run, latent))
            for window, latents in enumerate(run.readout) for latent in latents]
    distinct = len({latent for latents in run.readout for latent in latents})
    _common.say(f"{GATE_ID}: {run.name}: the readout lists {len(made)} latents over {len(run.readout)} probe windows,"
                f" {distinct} distinct")
    return made


def plan_crystallizations(run: Any, g74: dict[str, Any]) -> list[Planned]:
    """3. Each of the run's crystallizations: the request of the record G7.4's report names, once it is that
    crystallization's. SystemExit, a setup error, otherwise."""
    from nablaframe.gestures import records

    m, folder = run.m, _common.record_dir(run.name)
    named = g74["measurements"].get("crystallizations") or {}
    made = []
    run.crystallizations = {}
    for entry in (e for e in _common.JUDGED + _common.RECORDED if e[0] == run.name):
        name = _common.crystallization_name(entry)
        key = (named.get(name) or {}).get("key") if isinstance(named, dict) else None
        if not isinstance(key, str):
            raise SystemExit(f"G7.4's report names no record for {name}. Run G7.4 again. No report written.")
        try:
            record = records.read_record(folder, key)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{name}: G7.4's record {key} can't be read from {p1.rel(folder)} ({exc}). Run G7.4"
                             " again. No report written.") from exc
        request = record.request
        _, block, part, cls, function, form = entry
        wanted = {"block": int(block), "part": part, "class": cls, "function": function, "form": form}
        found = {field: request.args.get(field) for field in wanted} if request.kind == "crystallize" else None
        if (found != wanted or request.run != m.run_id or request.step != m.step
                or request.instances != len(m.held_out) or request.seed != _common.SEED):
            raise SystemExit(f"{name}: the record {key} in {p1.rel(folder)} isn't that crystallization's request, at"
                             f" run {m.run_id}, step {m.step:,}, {len(m.held_out):,} instances and seed"
                             f" {_common.SEED}. Run G7.4 again. No report written.")
        run.crystallizations[name] = request
        made.append(Planned("crystallizations", f"{name}{'' if entry in _common.JUDGED else ', recorded'}, citing"
                                                f" {len(request.args['cites'])} records", request))
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
    """Stop the gate when the run's records folder holds a record whose key isn't one of G7.5's requests'."""
    keys = set(run.keys.values())
    strays = {key: where for key, where in _record_files(first).items() if key not in keys}
    if strays:
        key, where = next(iter(sorted(strays.items())))
        raise SystemExit(f"{p1.rel(first)} holds {len(strays)} record(s) G7.5 doesn't ask for, the first {where},"
                         f" key {key}. Its key changed, or it was built by other code; move the folder aside and run"
                         " G7.2, G7.4 and G7.5 again. No report written.")
    held = set(_record_files(first)) if first.exists() else set()
    _common.say(f"{GATE_ID}: {run.name}: {p1.rel(first)} holds {len(held)} of the {len(keys)} records already")


def build(run: Any, folder: Path, label: str) -> None:
    """Every planned request of the run through ``cli.record`` into ``folder``, with a progress line per record.

    The name records come first, so a crystallization of modular addition finds the records it cites in the folder.
    """
    from nablaframe.gestures import cli, records

    held = set(_record_files(folder)) if folder.exists() else set()
    total, done = len(run.planned), [0]

    def tick(line: str) -> None:
        _common.say(f"{line}; {done[0]} of {total} records built")

    began = time.perf_counter()
    with _common.ticking(f"{GATE_ID}: {run.name}, {label}", write=tick):
        for p in run.planned:
            key = run.keys[records.request_hash(p.request)]
            ctx = context_for(run, p.request, folder)
            if p.request.kind == "crystallize":
                what = f"{GATE_ID}: {run.name}, {label}: {p.what.split(',')[0]}"
                ctx = replace(ctx, params_at=_common.announced(ctx.params_at, what, ctx.steps))
            start = time.perf_counter()
            record = cli.record(folder, p.request, ctx)
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
    comparison = _common.p3.compare_trees(first, second)
    _common.say(f"{GATE_ID}: {run.name}: the two builds are {'byte-identical' if comparison['identical'] else 'different'}"
                f" over {comparison['files_compared']:,} files")
    return {**comparison, "first_build_dir": p1.rel(first), "second_build_dir": p1.rel(second)}


def held_by_kind(folder: Path) -> dict[str, int]:
    """How many records of each kind the folder's index lists."""
    entries = json.loads((Path(folder) / "index.json").read_text(encoding="utf-8"))["records"]
    return {kind: sum(1 for e in entries if e["kind"] == kind) for kind in KINDS}


# ---------------------------------------------------------------------------
# The records shown are the ones G7.4 measured and G7.2 recorded
# ---------------------------------------------------------------------------


def match_reports(runs: dict[str, Any], folders: dict[str, tuple[Path, Path]],
                  reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``reports_match`` and ``matches``: each crystallization's key, as built here, G7.4's; and each name record
    G7.2's report names in modular addition's first folder."""
    from nablaframe.gestures import records

    named = reports["G7.4"]["measurements"]["crystallizations"]
    matches: dict[str, list[dict[str, Any]]] = {"crystallizations": [], "names": []}
    for entry in _common.JUDGED + _common.RECORDED:
        run, name = runs[entry[0]], _common.crystallization_name(entry)
        built = run.keys[records.request_hash(run.crystallizations[name])]
        g74 = named[name]["key"]
        matches["crystallizations"].append({"name": name, "run": run.name, "key": g74, "built": built,
                                            "equal": built == g74})
    held = set(_record_files(folders[_common.RUN_NAME][0]))
    recorded = reports["G7.2"]["measurements"].get("record_keys") or {}
    for latent in sorted(recorded, key=lambda k: (not str(k).isdigit(), int(k) if str(k).isdigit() else 0, str(k))):
        key = recorded[latent]
        matches["names"].append({"latent": latent, "key": key, "in_folder": key in held})
    ok = (bool(matches["crystallizations"]) and bool(matches["names"])
          and all(m["equal"] for m in matches["crystallizations"]) and all(m["in_folder"] for m in matches["names"]))
    for part, flag in (("crystallizations", "equal"), ("names", "in_folder")):
        good = sum(bool(m[flag]) for m in matches[part])
        _common.say(f"{GATE_ID}: {part}: {good} of {len(matches[part])} match")
    _common.say(f"{GATE_ID}: the records shown are the ones G7.4 measured and G7.2 recorded: {ok}")
    return {"reports_match": ok, "matches": matches}


def clock(run: Any, first: Path) -> dict[str, Any]:
    """Modular addition's fourier crystallization from its first folder: its key, frequencies, envelope and cites."""
    from nablaframe.gestures import records

    (entry,) = [e for e in _common.JUDGED if e[0] == _common.RUN_NAME and e[5] == "fourier"]
    name = _common.crystallization_name(entry)
    key = run.keys[records.request_hash(run.crystallizations[name])]
    results = p6.results_of(records.read_record(first, key))
    return {"name": name, "key": key, "frequencies": results["frequencies"], "envelope": results["envelope"],
            "cites": results["cites"]}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def probe_tokens(m: Any) -> np.ndarray:
    """The probe as G7.1's trace recorded it: what the model reads of ``probe(task)``."""
    return _common.tokens(_common.probe(m.task))


def open_reader(run: Any) -> Any:
    """A ``TraceReader`` on G7.1's store, once it holds exactly G7.1's checkpoints, each with its logits, on the probe."""
    from nablaframe.data.corpus import batch_hash
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.readouts import TraceReader

    m = run.m
    root = Path(m.run.store)
    if not (root / "events.jsonl").exists():
        raise SystemExit(f"{p1.rel(root)} holds no trace store. Run G7.1 first. No report written.")
    reader = TraceReader(root)
    missing = [step for step in m.run.steps if step not in reader.steps()]
    if missing or tuple(reader.steps()) != tuple(m.run.steps):
        raise SystemExit(f"{p1.rel(root)} doesn't hold exactly G7.1's checkpoints (missing {missing}). Run G7.1"
                         " again. No report written.")
    expected = batch_hash(probe_tokens(m))
    for step in m.run.steps:
        try:
            reader.event(step, LOGITS)
        except KeyError:
            raise SystemExit(f"{p1.rel(root)} has no logits at step {step}. Run G7.1 again. No report written.")
        if reader.batch_hash(step) != expected:
            raise SystemExit(f"step {step} in {p1.rel(root)} was traced on a different batch than the probe."
                             " No report written.")
    return reader


def answer_loss(m: Any, reader: Any) -> list[float]:
    """At each checkpoint, the mean over the probe of ``-log softmax(logits[b, 2])[window[b, 3]]``, in float64."""
    targets = _common.probe(m.task)[:, ANSWER + 1].astype(np.intp)
    losses = []
    progress = p1.Progress(f"{GATE_ID}: the answer position's loss, checkpoints", len(m.run.steps),
                           write=_common.say)
    for step in m.run.steps:
        logits = reader.tensor(step, "logits").astype(np.float64)[:, ANSWER]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        losses.append(float(-np.take_along_axis(log_probs, targets[:, None], axis=-1).mean()))
        progress.advance()
    _common.say(f"{GATE_ID}: the answer position's loss goes from {losses[0]:.4f} at step {m.run.steps[0]:,} to"
                f" {losses[-1]:.4f} at step {m.run.steps[-1]:,}")
    return losses


def features(run: Any, meters: dict[str, Any], label: str) -> dict[str, Any]:
    """The features payload of G7.2's TopK SAE at site 0, computed afresh, citing G7.2's meters."""
    from nablaframe.views.features import features_payload

    m = run.m
    held = np.asarray(m.held_out)
    variables = {name: held[:, column].astype(np.int64) for name, column in VARIABLES}
    with _common.ticking(f"{GATE_ID}: {label}: the features payload", write=_common.say):
        payload = features_payload(m.model, m.params, run.dictionary, 0, probe_tokens(m), m.held_tokens, step=m.step,
                                   meters={"gate": "G7.2", "path": ["meters"], "values": meters},
                                   variables=variables, chunk=_common.CHUNK)
    _common.say(f"{GATE_ID}: {label}: the features payload of {run.content[:12]}, site 0")
    return payload


def build_views_once(run: Any, out: Path, label: str, curve: dict[str, Any], meters: dict[str, Any]) -> list[Path]:
    """One build of modular addition's views into ``out``, as G4.5 and G5.6 build a zoo run's."""
    from nablaframe.views.build import build_views
    from nablaframe.views.readouts import TraceReader

    m = run.m
    payload = features(run, meters, label)
    reader = TraceReader(m.run.store)
    progress = p1.Progress(f"{GATE_ID}: views, {label}, steps", len(reader.steps()), write=_common.say)
    with _common.ticking(f"{GATE_ID}: views, {label}", write=_common.say):
        paths = build_views(reader, out, params_for=m.run.params_for, probe=probe_tokens(m), n_layers=m.n_layers,
                            run_id=m.run_id, vocab=m.task.vocab,
                            next_token=np.asarray(_common.probe(m.task)[:, -1], dtype=np.int64), curves=[curve],
                            progress=lambda done, total: progress.advance(), features=[payload])
    _common.say(f"{GATE_ID}: views, {label}: {len(paths):,} files written to {p1.rel(out)}")
    return paths


def views_twice(run: Any, reader: Any, folders: tuple[Path, Path], meters: dict[str, Any], g45: Any, *,
                overwrite: bool) -> dict[str, Any]:
    """Modular addition's views built twice, each into an empty folder, compared file by file."""
    m = run.m
    curve = g45.curve(ANSWER_CURVE, list(m.run.steps), answer_loss(m, reader))
    first, second = folders
    written = []
    for label, out in (("first build", first), ("second build", second)):
        _common.fresh_dir(out, overwrite=overwrite)
        written.append(build_views_once(run, out, label, curve, meters))
    comparison = _common.p3.compare_trees(first, second)
    _common.say(f"{GATE_ID}: views: the two builds are {'byte-identical' if comparison['identical'] else 'different'}"
                f" over {comparison['files_compared']:,} files")
    return {"build": {**comparison, "files_written": len(written[0]), "views_dir": p1.rel(first),
                      "second_build_dir": p1.rel(second)},
            "features": {"file": f"features-{run.content}-site-0.json", "dictionary": run.content},
            "curves": [curve["name"]], "answer_loss_by_step": [point["value"] for point in curve["points"]]}


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
    from nablaframe.views import badges, crystallize_badges, dictionary_badges, gesture_badges

    return (tuple(badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS) + tuple(gesture_badges.SUBJECTS)
            + tuple(crystallize_badges.SUBJECTS))


def badges_twice(files: tuple[Path, Path]) -> dict[str, Any]:
    """The badges, built twice, into ``runs/phase7/badges.json`` and the scratch folder, compared byte for byte."""
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
# The viewer and the addresses
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


def external_urls(g45: Any) -> dict[str, Any]:
    """External references in the viewer's files, counted as G4.5 counts them."""
    return g45.offline(Path(p1.REPO_ROOT) / "viewer")


def addresses(run: Any, g67: dict[str, Any], views: Path, badges_file: Path,
              folders: dict[str, tuple[Path, Path]]) -> dict[str, str]:
    """Modular addition's address, then G6.7's for each of its runs, with ``badges=`` pointing at Phase 7's badges
    file and each zoo run's ``gestures=`` naming its Phase 7 records folder after its Phase 6 one; every path
    relative to viewer/."""
    viewer = Path(p1.REPO_ROOT) / "viewer"

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, viewer)).as_posix()

    m = run.m
    out = {_common.RUN_NAME: f"{VIEWER_ADDRESS}?views={relative(views)}/&trace={relative(m.run.store)}/"
                             f"&badges={relative(badges_file)}&gestures={relative(folders[_common.RUN_NAME][0])}/"
                             f"#step={m.step}&seq=0&pos={ANSWER}&features={run.content}-0"}
    given = g67_addresses(g67)
    for name in G67_RUNS:
        query, mark, fragment = given[name].partition("#")
        badges = f"badges={relative(badges_file)}"
        query, found = re.subn(r"(?<=[?&])badges=[^&]*", lambda _: badges, query)
        if not found:
            query += ("&" if "?" in query else "?") + badges
        if name in folders:
            added = f",{relative(folders[name][0])}/"
            query = re.sub(r"(?<=[?&])gestures=[^&]*", lambda found: found.group(0) + added, query, count=1)
        out[name] = query + mark + fragment
    return out


if __name__ == "__main__":
    raise SystemExit(main())
