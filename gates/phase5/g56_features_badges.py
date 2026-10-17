"""G5.6: the features view, the dictionary badges, and every run's views rebuilt (D38, D39, D44, D45).

Spec: docs/specs/phase-5.md, D44, D45 and "Gates"; docs/deltas/phase-5-delta-5.md; helpers in
``_common.py``. Report: ``reports/phase5/g56_features_badges.json``. Run in nf-jax, on the Mac,
after G5.1 to G5.5 have run.

Refuses to start unless G4.1's and G5.1's reports pass, and G4.2 to G4.4's and G5.2 to G5.5's exist
and pass ``validate_report``, whatever their status: a failed judged gate shows on its badge, never
hidden by a missing one. It reads G4.1's two zoo runs, as G4.6 does, and the reference run, G1.4's
checkpoints and G1.6's trace, as G2.6 does.

* **Badges:** ``runs/phase5/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1
  to G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, as one mapping>, "reports": {<gate id>:
  {"git_commit", "timestamp_utc", "sha256"}}}`` for each of those eight reports, the last being the
  report file's SHA-256. Built a second time into ``runs/phase5/scratch/badges.json`` and compared
  byte for byte. One file holds every model-level badge, so the viewer reads it with ``?badges=``
  and finds both the Phase 4 subjects and the Phase 5 ones.
* **Shipped dictionaries:** on each run, the dictionaries its gate trained on the trained model at
  ``DICT_SEED``:

  - ``reference`` (G5.5): ``topk`` at every block, the ``staircase``, a ``transcoder`` at every block
  - ``induction`` (G5.2): ``topk`` at blocks 0 and 1, the ``staircase``, a ``transcoder`` at blocks 0
    and 1
  - ``two_hop`` (G5.3): ``topk`` at every block, the ``staircase``, a ``transcoder`` at every block

  Each comes through ``cached(DICTIONARY_DIR, key, make)`` under the key its gate trained it with:
  ``dictionary_key`` of the run's final state hash, ``sites``, ``settings(run, kind, d_model)`` and
  the tokens the model reads of the training windows, ``p4.tokens(task, zoo_windows(task, name,
  "train"))`` for a zoo run and ``reference_windows(train, validation, seq, "train")[:, :seq]`` for
  the reference run. ``make`` never trains: if it is called, the dictionary isn't in the cache, and
  the gate exits 1 naming the gate to run, with no report.
* **Features:** for each shipped dictionary, in the order above, and each of its sites in order,
  ``features_payload`` with the run's final parameters and step; ``probe``, the probe the run's
  views use; ``held_out``, the tokens the model reads of the run's held-out windows
  (``p4.tokens(task, zoo_windows(task, name, "held_out"))``, or ``reference_windows(...,
  "held_out")[:, :seq]``); ``meters``, ``{"gate": <the run's gate>, "path": ["meters", "<kind>
  <output tap>"], "values": <that entry of its report>}``; and for two-hop, ``variables``: each
  held-out window's ``a``, ``b`` and ``c`` from ``task.parse``. A missing meters entry is a setup
  error.
* **Views:** each run's views built with ``build_views`` as its gate built them, plus ``features``,
  the run's payloads: the reference run's as G2.6 builds them (its vocabulary, G1.4's
  validation-loss curve and, when G2.9's report passes, the J-lens), and each zoo run's as G4.5
  does (the task's vocabulary, ``next_token`` and ``curves``), using those gates' own functions,
  loaded by path. Their D14 badges now end with D39's sentence. The first build goes to
  ``runs/phase5/views/<run>/`` and the second to ``runs/phase5/scratch/views/<run>/``; both start
  empty, and each computes its payloads afresh. Compared file by file.

Measurements: REQUIRED_MEASUREMENTS["G5.6"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``
  and ``dictionary_badges.SUBJECTS`` without a badge.
* ``dictionaries``: per run, a list in the order above of ``{"kind", "sites", "key",
  "content_hash", "meters_gate", "files"}``, ``files`` naming its features payloads.
  ``dictionaries_missing``: the shipped dictionaries whose kind has no badge.
* ``features``: per run, the number of features payloads.
* ``builds_identical``: the two badges files hold the same bytes, and each run's two view folders
  hold the same files, by name, with the same bytes. ``builds`` gives each part's detail: files
  compared and files differing.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.

Recorded beside them: ``views_dirs``, ``run_ids`` and ``steps`` (each run's final step), and
``query_latents``: from the two-hop payload for the ``topk`` at G5.3's ``best_swap_block``, for
each probe window, its ``b`` from ``task.parse`` and, at position 25, each active latent with its
code and the ``b`` of each of its top contexts' windows, read from the payload. So the report shows
whether the latents active at ``⇒`` are the intermediate's.

Pass needs ``builds_identical``, ``external_urls == 0``, ``dictionaries_missing == 0`` and
``subjects_missing == 0``. At the end it prints the viewer's address for each run, with
``?views=``, ``?trace=`` and ``?badges=`` set, and for each zoo run ``?interventions=`` G4.6's
folder.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g56_features_badges.py``. Takes --overwrite and --allow-dirty. --overwrite also
#   lets it replace both badges files and every build folder.
# * Start checks, before anything is written: the reports, as above; the runs and stores, through
#   the checks G2.6 and G4.6 make; ``model_level_badges`` and ``dictionary_badges`` of the reports,
#   whose ValueError is a setup error; G5.3's ``best_swap_block`` a block of the two-hop model;
#   every shipped dictionary in the cache, and every meters entry in its report. Each fails with
#   exit 1 and no report.
# * Every long step prints a progress line at least every 30 seconds.
# * The earlier gates' functions: G2.6's (and through it G2.10's) and G4.5's scripts are loaded by
#   path, each under a module name of its own, with ``_common`` pointing at its own phase's module
#   while it loads, so each keeps reading its own phase's paths and settings.
# * Runs in the order the docstring lists them: ``reference``, ``induction``, ``two_hop``. Each
#   dictionary is loaded once; each build computes its features payloads afresh, with ``chunk``
#   the run's ``CHUNK``. The zoo runs' curves come from G4.5's ``formation``, ``answer_loss`` and
#   ``curve`` once per run, as G4.5 computes them once for its two builds.
# * ``dictionaries[run][i]["sites"]`` is the dictionary's ``[input tap, output tap]`` pairs, and
#   ``files`` its payload files in site order. ``subjects_without_badge`` and
#   ``dictionaries_without_badge`` list what the two counts count.
# * ``builds``: ``{"badges": {...}, <run>: {...}}``, each ``{"identical", "files_compared",
#   "files_differing"}``, a run's also naming its two folders and the files the first build wrote.
# * ``query_latents``: ``{"block", "position", "file", "windows": [{"window", "b", "latents":
#   [{"latent", "code", "context_b": [...]}]}]}``, the latents in the payload's order (largest code
#   first), ``context_b`` in its contexts' order.
# * ``addresses``: ``{<run>: <the address printed for it>}``. Each opens at the run's final step
#   (``#step=<step>``), where the features payloads are built; two-hop's also opens H5's view,
#   ``&seq=0&pos=25&features=<content hash of the topk at best_swap_block>-0`` (apolloh0, in the thread).

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p2, p4 = _common.p1, _common.p2, _common.p4

GATE_ID = "G5.6"
REPORT_NAME = "g56_features_badges.json"
PHASE4_REPORTS = {"G4.1": "g41_zoo_training.json", "G4.2": "g42_induction.json", "G4.3": "g43_two_hop.json",
                  "G4.4": "g44_lenses.json"}
PHASE5_REPORTS = {"G5.1": "g51_determinism.json", "G5.2": "g52_induction.json", "G5.3": "g53_two_hop.json",
                  "G5.4": "g54_edges.json", "G5.5": "g55_reference.json"}
BADGE_GATES = ("G4.1", "G4.2", "G4.3", "G4.4", "G5.1", "G5.2", "G5.3", "G5.4")
RUNS = ("reference", "induction", "two_hop")
METERS_GATE = {"reference": "G5.5", "induction": "G5.2", "two_hop": "G5.3"}
BADGES_NAME = "badges.json"
VIEWS_NAME = "views"
INTERVENTIONS_NAME = "interventions"  # G4.6's compare payloads, in each zoo run's folder
VIEWER_ADDRESS = "http://localhost:8000/viewer/"
VARIABLES = ("a", "b", "c")


def shipped(name: str, n_layers: int) -> list[tuple[str, int | None]]:
    """The dictionaries a run ships, ``(kind, layer)`` in the docstring's order."""
    blocks = range(n_layers) if name in ("reference", "two_hop") else (0, 1)
    return [("topk", l) for l in blocks] + [("staircase", None)] + [("transcoder", l) for l in blocks]


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    _common.require_g51()
    reports = read_reports()
    made_badges(reports)
    gates = earlier_gates()
    runs = {"reference": reference_run(gates)}
    runs |= {name: zoo_run(name, g41, gates) for name in _common.ZOO}
    best = best_swap_block(reports["G5.3"], runs["two_hop"])
    for name in RUNS:
        runs[name].dictionaries = find_dictionaries(runs[name], reports[METERS_GATE[name]])
    badge_files = (Path(_common.RUN_DIR) / BADGES_NAME, Path(_common.RUN_DIR) / "scratch" / BADGES_NAME)
    folders = {name: (Path(_common.RUN_DIR) / VIEWS_NAME / name, Path(_common.RUN_DIR) / "scratch" / VIEWS_NAME / name)
               for name in RUNS}
    existing = [path for path in badge_files if path.exists()]
    existing += [folder for pair in folders.values() for folder in pair if folder.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{p1.rel(existing[0])} exists. Pass --overwrite to start it over.")

    measurements = badges(badge_files)
    built = views(runs, folders, gates, overwrite=args.overwrite)
    measurements |= {**built, "builds": {**measurements["builds"], **built["builds"]}}
    measurements["builds_identical"] = all(part["identical"] for part in measurements["builds"].values())
    measurements |= without_badge(runs, measurements["badges"])
    measurements["query_latents"] = query_latents(runs["two_hop"], folders["two_hop"][0], best)
    measurements |= gates.g45.offline(Path(p1.REPO_ROOT) / "viewer")
    measurements["run_ids"] = {name: runs[name].m.run_id for name in RUNS}
    measurements["steps"] = {name: runs[name].m.step for name in RUNS}
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    measurements["addresses"] = addresses(runs, folders, badge_files[0], best)
    code = p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)
    print(f"{GATE_ID}: serve the repo root with `python3 -m http.server`, then open:", flush=True)
    for name, address in measurements["addresses"].items():
        print(f"  {name}: {address}", flush=True)
    return code


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G5.6 rule holds."""
    ok = (m["builds_identical"] is True and m["external_urls"] == 0 and m["dictionaries_missing"] == 0
          and m["subjects_missing"] == 0)
    return "pass" if ok else "fail"


# ---------------------------------------------------------------------------
# Start checks: the reports, the earlier gates' functions and the runs
# ---------------------------------------------------------------------------


def report_path(gate_id: str) -> Path:
    if gate_id in PHASE4_REPORTS:
        return Path(p4.REPORT_DIR) / PHASE4_REPORTS[gate_id]
    return Path(_common.REPORT_DIR) / PHASE5_REPORTS[gate_id]


def read_reports() -> dict[str, dict[str, Any]]:
    """G4.1 to G4.4's and G5.1 to G5.5's reports, each read from its file and checked. SystemExit otherwise.

    Whatever their status: a failed judged gate shows on its badge, never hidden by a missing one.
    """
    from nablaframe.reports import validate_report

    reports = {}
    for gate_id in (*PHASE4_REPORTS, *PHASE5_REPORTS):
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
    """``model_level_badges`` of G4.1 to G4.4, then ``dictionary_badges`` of G4.1 and G5.1 to G5.4, as one mapping.

    Raises SystemExit, a setup error, when either refuses the reports.
    """
    from nablaframe.views.badges import model_level_badges
    from nablaframe.views.dictionary_badges import dictionary_badges

    try:
        return {**model_level_badges(reports), **dictionary_badges(reports)}
    except ValueError as exc:
        raise SystemExit(f"The reports don't make badges: {exc}. No report written.") from exc


def load_gate(path: Path, common: Any) -> Any:
    """An earlier phase's gate script, loaded by path under a name of its own, its ``_common`` being ``common``.

    Each gate script imports ``_common`` from its own folder; this one already has a module by that
    name, so while the script loads, ``_common`` is its own phase's module and its folder is on the
    path, for the sibling scripts it imports. Both are put back afterwards.
    """
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


def earlier_gates() -> SimpleNamespace:
    """G2.6's and G4.5's scripts, and G2.10's, which G2.6 loaded, as this gate uses their functions."""
    gates = Path(__file__).resolve().parents[1]
    g26 = load_gate(gates / "phase2" / "g26_viewer.py", p2)
    g45 = load_gate(gates / "phase4" / "g45_formation_viewer.py", p4)
    return SimpleNamespace(g26=g26, g210=g26.g210, g45=g45)


def reference_run(gates: Any) -> SimpleNamespace:
    """G1.4's run at its final checkpoint, G1.6's store and what G2.6 builds its views with, after G2.6's checks."""
    m = _common.reference_model()
    ref = m.ref
    probe = p1.probe(p1.load_split("validation"), ref.config.seq)
    store = p2.g16_store()
    p2.open_reader(store, ref.steps, probe)
    curve = p1.phase0.load_report(p1.REPO_ROOT / gates.g26.G14_REPORT)["measurements"].get("val_loss_curve")
    if not curve:
        raise SystemExit(f"{gates.g26.G14_REPORT} has no val_loss_curve. No report written.")
    problem = gates.g210.fit_report_problem()
    if problem is None:
        jacobians_for = gates.g210.jacobians_for(ref, p1.load_split("train"))
        print(f"{GATE_ID}: G2.9's report passes, so the reference run's builds include the J-lens, with"
              f" {len(jacobians_for.keys)} J's from {p1.rel(jacobians_for.root)}", flush=True)
    else:
        jacobians_for = None
        print(f"{GATE_ID}: the reference run's builds go without the J-lens, as G2.6's do: {problem}", flush=True)
    return SimpleNamespace(name="reference", m=m, store=store, probe=probe, vocab=p1.load_tokenizer().vocab,
                           val_loss_curve=curve, jacobians_for=jacobians_for, variables=None)


def zoo_run(name: str, g41: dict[str, Any], gates: Any) -> SimpleNamespace:
    """G4.1's run of ``name`` at its final checkpoint, with its store, after G4.6's and G4.5's checks."""
    m = _common.zoo_model(name, g41)
    gates.g45.open_reader(m.run)
    variables = None
    if name == "two_hop":
        parsed = [m.task.parse(window) for window in m.held_out]
        variables = {v: np.array([getattr(p, v) for p in parsed], dtype=np.int64) for v in VARIABLES}
    return SimpleNamespace(name=name, m=m, store=m.run.store, probe=gates.g45.probe(m.run),
                           next_token=gates.g45.next_token(m.run), variables=variables)


def best_swap_block(g53: dict[str, Any], two_hop: Any) -> int:
    """G5.3's ``best_swap_block``, once it is a block of the two-hop model."""
    best = g53["measurements"].get("best_swap_block")
    n_layers = two_hop.m.n_layers
    if isinstance(best, bool) or not isinstance(best, int) or not 0 <= best < n_layers:
        raise SystemExit(f"G5.3's best_swap_block {best!r} is not a block of the two-hop model, 0 to {n_layers - 1}."
                         " No report written.")
    return best


def find_dictionaries(run: Any, meters_report: dict[str, Any]) -> list[SimpleNamespace]:
    """Each shipped dictionary of ``run``, from the cache under the key its gate trained it with, and its meters.

    Nothing trains here: a dictionary the cache doesn't hold, or a site its gate's report has no
    meters for, is a setup error.
    """
    from nablaframe.dictionaries.core import content_hash

    m, gate = run.m, METERS_GATE[run.name]
    meters = meters_report["measurements"].get("meters")
    out = []
    for kind, layer in shipped(run.name, m.n_layers):
        sites, _, key = _common.dictionary_spec(m, kind, layer)
        label = f"{run.name}'s {_common.dictionary_label(kind, m.n_layers, layer)}"

        def refuse(label: str = label, key: str = key) -> Any:
            raise SystemExit(f"{label} is not in the cache {p1.rel(_common.DICTIONARY_DIR)} under key {key}. Run"
                             f" {gate} first: G5.6 trains nothing. No report written.")

        try:
            made, _ = _common.cached(_common.DICTIONARY_DIR, key, refuse)
        except ValueError as exc:
            raise SystemExit(f"{label}: {exc}. No report written.") from exc
        names = _common.site_names(kind, made.sites)
        for site in names:
            if not isinstance(meters, dict) or not isinstance(meters.get(site), dict):
                raise SystemExit(f"{gate}'s report has no meters for {run.name}'s {site}. No report written.")
        print(f"{GATE_ID}: {label} from the cache, key {key[:12]}", flush=True)
        out.append(SimpleNamespace(kind=kind, layer=layer, dictionary=made, sites=[list(s) for s in sites], key=key,
                                   content_hash=content_hash(made), site_names=names, gate=gate,
                                   meters={site: meters[site] for site in names}))
    return out


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def badges_document() -> tuple[dict[str, Any], bytes]:
    """The badges file's contents, from the report files as they are now: ``(badges, canonical JSON bytes)``."""
    from nablaframe.trace.schema import canonical_json, sha256_hex
    from nablaframe.views.badges import model_level_badges
    from nablaframe.views.dictionary_badges import dictionary_badges

    reports, cited = {}, {}
    for gate_id in BADGE_GATES:
        data = report_path(gate_id).read_bytes()
        reports[gate_id] = json.loads(data.decode("utf-8"))
        cited[gate_id] = {"git_commit": reports[gate_id]["git_commit"],
                          "timestamp_utc": reports[gate_id]["timestamp_utc"], "sha256": sha256_hex(data)}
    made = {**model_level_badges(reports), **dictionary_badges(reports)}
    return made, canonical_json({"badges": made, "reports": cited}).encode("ascii")


def subjects() -> tuple[str, ...]:
    from nablaframe.views import badges as phase4_badges
    from nablaframe.views import dictionary_badges

    return tuple(phase4_badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS)


def badges(files: tuple[Path, Path]) -> dict[str, Any]:
    """The badges, built twice, into ``runs/phase5/badges.json`` and the scratch folder, compared byte for byte."""
    made = None
    for path in files:
        made, data = badges_document()
        p1.write_atomic(path, data)
    first, second = (path.read_bytes() for path in files)
    identical = first == second
    missing = [subject for subject in subjects() if subject not in made]
    print(f"{GATE_ID}: {len(made)} badges written to {p1.rel(files[0])}; the second build is"
          f" {'byte-identical' if identical else 'different'}; {len(missing)} subjects without a badge", flush=True)
    for subject in subjects():
        if subject in made:
            print(f"{GATE_ID}:   {subject}: {made[subject]['status']}, {len(made[subject]['facts'])} facts", flush=True)
    return {
        "badges": made,
        "subjects_missing": len(missing),
        "subjects_without_badge": missing,
        "badges_file": p1.rel(files[0]),
        "badges_sha256": _sha256(first),
        "builds": {"badges": {"identical": identical, "files_compared": 1,
                              "files_differing": [] if identical else [BADGES_NAME]}},
    }


def without_badge(runs: dict[str, Any], made: dict[str, Any]) -> dict[str, Any]:
    """The shipped dictionaries whose kind has no badge."""
    missing = [f"{name} {_common.dictionary_label(d.kind, runs[name].m.n_layers, d.layer)}"
               for name in RUNS for d in runs[name].dictionaries if d.kind not in made]
    return {"dictionaries_missing": len(missing), "dictionaries_without_badge": missing}


# ---------------------------------------------------------------------------
# Features and views
# ---------------------------------------------------------------------------


def feature_file(d: Any, site: int) -> str:
    """A features payload's file name, as ``build_views`` writes it."""
    return f"features-{d.content_hash}-site-{site}.json"


def features(run: Any, label: str) -> list[dict[str, Any]]:
    """The run's features payloads, one per shipped dictionary and site, in order, computed afresh."""
    from nablaframe.views.features import features_payload

    m = run.m
    payloads = []
    total = sum(len(d.dictionary.sites) for d in run.dictionaries)
    with _common.ticking(f"{GATE_ID}: {run.name}, {label}: features payloads", write=print):
        for d in run.dictionaries:
            for site, name in enumerate(d.site_names):
                meters = {"gate": d.gate, "path": ["meters", name], "values": d.meters[name]}
                payloads.append(features_payload(m.model, m.params, d.dictionary, site, run.probe, m.held_tokens,
                                                 step=m.step, meters=meters, variables=run.variables, chunk=m.chunk))
                print(f"{GATE_ID}: {run.name}, {label}: features payload {len(payloads)}/{total}, {name}", flush=True)
    return payloads


def curves(run: Any, gates: Any) -> list[dict[str, Any]] | None:
    """A zoo run's timeline curves, from its trace, as G4.5 computes them; None for the reference run."""
    from nablaframe.views.readouts import TraceReader

    g45 = gates.g45
    if run.name == "reference":
        return None
    reader = TraceReader(run.store)
    with _common.ticking(f"{GATE_ID}: {run.name}'s curves", write=print):
        if run.name == "induction":
            f = g45.formation(run.m.run, reader)
            return [g45.curve(g45.LOSS_CURVE, f["steps"], f["mechanism_loss_by_step"]),
                    g45.curve(g45.ATTENTION_CURVE, f["steps"], f["best_attention_by_step"])]
        a = g45.answer_loss(run.m.run, reader)
        return [g45.curve(g45.ANSWER_CURVE, a["answer_steps"], a["answer_loss_by_step"])]


def build(run: Any, out: Path, label: str, run_curves: list[dict[str, Any]] | None) -> list[Path]:
    """One build of the run's views into ``out``, as its gate built them, plus its features payloads."""
    from nablaframe.views.build import build_views
    from nablaframe.views.readouts import TraceReader

    payloads = features(run, label)
    reader = TraceReader(run.store)
    progress = p1.Progress(f"views, {run.name}, {label}, steps", len(reader.steps()))
    with _common.ticking(f"{GATE_ID}: {run.name}, {label}: views", write=print):
        if run.name == "reference":
            ref = run.m.ref
            paths = build_views(reader, out, params_for=ref.params_for, probe=run.probe, n_layers=ref.config.n_layers,
                                run_id=ref.manifest.run_id, vocab=run.vocab, val_loss_curve=run.val_loss_curve,
                                progress=lambda done, total: progress.advance(), jacobians_for=run.jacobians_for,
                                features=payloads)
        else:
            zoo = run.m.run
            paths = build_views(reader, out, params_for=zoo.params_for, probe=run.probe, n_layers=zoo.config.n_layers,
                                run_id=zoo.run_id, vocab=zoo.task.vocab, next_token=run.next_token, curves=run_curves,
                                progress=lambda done, total: progress.advance(), features=payloads)
    print(f"{GATE_ID}: {run.name}, {label}: {len(paths):,} files written to {p1.rel(out)}", flush=True)
    return paths


def views(runs: dict[str, Any], folders: dict[str, tuple[Path, Path]], gates: Any, *,
          overwrite: bool) -> dict[str, Any]:
    """Each run's views built twice, with its features payloads, compared file by file."""
    builds, dictionaries, counts = {}, {}, {}
    for name in RUNS:
        run = runs[name]
        first, second = folders[name]
        run_curves = curves(run, gates)
        written = []
        for label, out in (("first build", first), ("second build", second)):
            _common.fresh_dir(out, overwrite=overwrite)
            written.append(build(run, out, label, run_curves))
        comparison = _common.p3.compare_trees(first, second)
        print(f"{GATE_ID}: {name}: the two builds are {'byte-identical' if comparison['identical'] else 'different'}"
              f" over {comparison['files_compared']:,} files", flush=True)
        builds[name] = {**comparison, "files_written": len(written[0]), "views_dir": p1.rel(first),
                        "second_build_dir": p1.rel(second)}
        dictionaries[name] = [{"kind": d.kind, "sites": d.sites, "key": d.key, "content_hash": d.content_hash,
                               "meters_gate": d.gate,
                               "files": [feature_file(d, site) for site in range(len(d.dictionary.sites))]}
                              for d in run.dictionaries]
        counts[name] = sum(len(d["files"]) for d in dictionaries[name])
    return {"builds": builds, "dictionaries": dictionaries, "features": counts,
            "views_dirs": {name: p1.rel(folders[name][0]) for name in RUNS}}


def query_latents(run: Any, folder: Path, best: int) -> dict[str, Any] | None:
    """At ``⇒``, the latents active in the two-hop TopK SAE at the best swap block, with their contexts' ``b``.

    For each probe window, its ``b`` from ``task.parse``, and each latent active at position 25,
    with its code and the ``b`` of each of its top contexts' windows, read from the first build's
    payload. None when the payload is missing.
    """
    (d,) = [d for d in run.dictionaries if d.kind == "topk" and d.layer == best]
    name = feature_file(d, 0)
    path = Path(folder) / name
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    position = _common.READOUT["two_hop"][0]
    context_b = payload["variables"]["b"]
    windows = []
    for i, window in enumerate(p4.probe(run.m.task)):
        latents, codes = payload["probe"]["latents"][i][position], payload["probe"]["values"][i][position]
        windows.append({"window": i, "b": int(run.m.task.parse(window).b),
                        "latents": [{"latent": latent, "code": code,
                                     "context_b": [context_b[w] for w, _, _ in payload["contexts"][latent]]}
                                    for latent, code in zip(latents, codes)]})
    print(f"{GATE_ID}: two-hop TopK SAE after block {best}, at position {position}: "
          + "; ".join(f"window {w['window']} (b {w['b']}): {len(w['latents'])} active" for w in windows), flush=True)
    return {"block": best, "position": position, "file": name, "windows": windows}


# ---------------------------------------------------------------------------


def addresses(runs: dict[str, Any], folders: dict[str, tuple[Path, Path]], badges_file: Path,
              best: int) -> dict[str, str]:
    """The viewer's address for each run, every path relative to viewer/, opening at the run's final step.

    Two-hop's opens on probe window 0 at ``⇒`` with the TopK SAE at ``best`` chosen.
    """
    viewer = Path(p1.REPO_ROOT) / "viewer"

    def relative(path: Path, folder: bool = True) -> str:
        text = Path(os.path.relpath(path, viewer)).as_posix()
        return f"{text}/" if folder else text

    out = {}
    for name in RUNS:
        run = runs[name]
        query = f"?views={relative(folders[name][0])}&trace={relative(run.store)}"
        if name != "reference":
            query += f"&interventions={relative(run.m.run.folder / INTERVENTIONS_NAME)}"
        query += f"&badges={relative(badges_file, folder=False)}"
        fragment = f"#step={run.m.step}"
        if name == "two_hop":
            (d,) = [d for d in run.dictionaries if d.kind == "topk" and d.layer == best]
            fragment += f"&seq=0&pos={_common.READOUT['two_hop'][0]}&features={d.content_hash}-0"
        out[name] = f"{VIEWER_ADDRESS}{query}{fragment}"
    return out


def _sha256(data: bytes) -> str:
    from nablaframe.trace.schema import sha256_hex

    return sha256_hex(data)


if __name__ == "__main__":
    raise SystemExit(main())
