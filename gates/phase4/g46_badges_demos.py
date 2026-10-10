"""G4.6: model-level badges and the zoo's demos for the viewer (D34, D38).

Spec: docs/specs/phase-4.md, "Gates"; docs/deltas/phase-4-delta-4.md. Report:
``reports/phase4/g46_badges_demos.json``. Run in nf-jax, on the Mac, after G4.1 to G4.4 have run.

Refuses to start unless G4.1's report passes and G4.2 to G4.4's reports exist and pass
``validate_report``, whatever their status: a failed judged gate is shown on its badge, never
hidden by a missing one.

* **Badges:** ``model_level_badges`` of the G4.1 to G4.4 reports, written as canonical JSON to
  ``runs/phase4/badges.json``: ``{"badges": <the badges>, "reports": {<gate id>:
  {"git_commit", "timestamp_utc", "sha256"}}}``, the last being each report file's SHA-256.
  Built a second time into ``runs/phase4/scratch/badges.json`` and compared byte for byte.
* **Demos:** compare payloads written by ``interventions.cli.main`` for each zoo model's final
  checkpoint, on what the model reads of ``probe(task)``, with ``--taps`` every block's
  ``hook_resid_post``, into ``runs/phase4/<name>/interventions/``, which starts empty:

  - induction: G4.2's ``induction_heads`` zeroed together through ``blocks.1.attn.hook_z``, one
    ``zero`` edit with ``head`` per head, in one spec
  - two-hop: two patches at G4.3's ``best_swap`` site and block, of
    ``blocks.<block>.hook_resid_post``: one from the aligned sources and one from their
    controls. Both come from ``aligned_sources(task, eval_windows(task),
    numpy.random.default_rng([SOURCE_SEED, 0]))``, rows 0 to 7, since the probe is the first
    eight evaluation windows, so they are G4.3's own sources for those windows.

  Every object a spec names is written to the command line's ``--objects`` folder. All of it
  is built a second time into ``runs/phase4/scratch/interventions/<name>/`` and compared file by
  file.

Measurements: REQUIRED_MEASUREMENTS["G4.6"]. ``badges`` is the badges mapping;
``subjects_missing`` counts ``SUBJECTS`` without a badge; ``builds_identical`` holds when the
badges and every compare folder match their second builds; ``payloads`` counts each model's
compare payloads. Recorded beside them, ``demo``: for the two-hop source patch on the probe's
eight windows, each window's ``c``, ``d`` and ``d′`` and its top-1 answer at position 25 before
and after the patch, read from the payload's ``top_tokens``. Pass needs ``builds_identical``
and ``subjects_missing == 0``.

At the end it prints the viewer's address for each zoo run, with ``?views=``, ``?trace=``,
``?interventions=`` and ``?badges=`` set.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g46_badges_demos.py``. Takes --overwrite and --allow-dirty. --overwrite also
#   lets it replace both badges files and every folder below.
# * Start checks, before anything is written: ``require_g41``; G4.2 to G4.4's reports each read,
#   ``validate_report``-ed and carrying their own gate id; ``zoo_run``'s checks on both runs;
#   ``model_level_badges`` of the four reports, whose ValueError is a setup error; G4.3's
#   ``best_swap`` naming one of ``TWO_HOP_SITES`` and a two-hop block. Each fails with exit 1
#   and no report.
# * Badges: each build reads the four report files afresh. ``sha256`` is the file's bytes';
#   ``git_commit`` and ``timestamp_utc`` are the report's own. Written as ``canonical_json``,
#   ASCII, no trailing newline, as ``export_view`` writes a payload.
# * Demos: the command line's code is ``nablaframe.interventions.cli.main``, called in this
#   process, so every pass goes through ``run_intervened``. Its inputs are written once per
#   model to ``runs/phase4/scratch/g46_inputs/<name>/``: the model's config, the probe's first
#   ``seq`` tokens as ``tokens.npy``, ``objects/`` and one JSON per spec. Its checkpoints are
#   G4.1's, at the final step. Each build folder starts empty, and the same runs, in the same
#   order, fill each. The induction spec is the ``zero`` edits in ``induction_heads``' order; the
#   two-hop specs are the source patch, then the control patch, each at every position of the
#   site, in order.
# * ``builds_identical``: both badges files hold the same bytes, every command-line run exited 0,
#   each build holds its model's payloads, and each model's two folders hold the same files,
#   ``results/`` included, by relative path, with equal bytes. ``builds`` gives each part's
#   detail: files compared, files differing, and any run that exited 1 with its error line.
# * ``payloads``: the payload files in each model's ``runs/phase4/<name>/interventions/``.
#   ``examples`` names each spec's model, label, spec hash, file and taps.
# * ``demo``: ``c`` and ``d′`` are the target's and the source's last tokens, ``d`` the object
#   of the counterpart's fact in the target, as ``patch_outcomes`` counts a swap; ``d_prime``
#   holds ``d′``. ``before`` and ``after`` are the top token at position 25 in the payload's
#   unintervened and intervened ``top_tokens``. ``swapped`` and ``transplanted`` count the
#   windows whose ``after`` is ``d`` and ``d′``. Null when the payload is missing.
# * The viewer's addresses assume ``python3 -m http.server`` from the repo root, on its default
#   port; each path is relative to viewer/, as the page requires.

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.6"
REPORT_NAME = "g46_badges_demos.json"
SOURCE_REPORTS = {"G4.1": _common.G41_REPORT, "G4.2": "g42_induction.json", "G4.3": "g43_two_hop.json",
                  "G4.4": "g44_lenses.json"}
BADGES_NAME = "badges.json"
INTERVENTIONS_NAME = "interventions"
INPUTS_NAME = "g46_inputs"
ZERO_TAP = "blocks.1.attn.hook_z"
ANSWER_POSITION = 25
VIEWER_ADDRESS = "http://localhost:8000/viewer/"
PAYLOAD = re.compile(r"intervention-[0-9a-f]{64}-step-[0-9]+\.json")
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    from nablaframe.views.badges import model_level_badges

    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    reports = read_reports()
    runs = {name: _common.zoo_run(name, g41) for name in _common.NAMES}
    try:
        model_level_badges(reports)
    except ValueError as exc:
        raise SystemExit(f"The Phase 4 reports don't make badges: {exc}. No report written.") from exc
    best_swap(reports["G4.3"], runs["two_hop"])
    badge_files = (Path(_common.RUN_DIR) / BADGES_NAME, _common.scratch_dir() / BADGES_NAME)
    folders = {name: (run.folder / INTERVENTIONS_NAME, _common.scratch_dir() / INTERVENTIONS_NAME / name,
                      _common.scratch_dir() / INPUTS_NAME / name) for name, run in runs.items()}
    existing = [path for path in badge_files if path.exists()]
    existing += [folder for trio in folders.values() for folder in trio if folder.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{p1.rel(existing[0])} exists. Pass --overwrite to start it over.")

    measurements = badges(badge_files)
    demo = demos(runs, reports, folders, overwrite=args.overwrite)
    measurements |= {**demo, "builds": {**measurements["builds"], **demo["builds"]}}
    measurements["builds_identical"] = (measurements["builds"]["badges"]["identical"]
                                        and all(measurements["builds"][name]["identical"] for name in runs))
    measurements["run_ids"] = {name: run.run_id for name, run in runs.items()}
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    code = p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)
    print_addresses(runs, badge_files[0])
    return code


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G4.6 rule holds."""
    return "pass" if m["builds_identical"] is True and m["subjects_missing"] == 0 else "fail"


def report_path(gate_id: str) -> Path:
    return Path(_common.REPORT_DIR) / SOURCE_REPORTS[gate_id]


def read_reports() -> dict[str, dict[str, Any]]:
    """G4.1 to G4.4's reports, each read from its file and checked. Raises SystemExit otherwise.

    G4.2 to G4.4 count whatever their status: a failed judged gate shows on its badge.
    """
    from nablaframe.reports import validate_report

    reports = {}
    for gate_id in SOURCE_REPORTS:
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


def best_swap(g43: dict[str, Any], run: Any) -> tuple[str, int]:
    """G4.3's ``best_swap`` site and block, once they name a site and a block of the two-hop model."""
    best = g43["measurements"].get("best_swap")
    site = best.get("site") if isinstance(best, dict) else None
    block = best.get("block") if isinstance(best, dict) else None
    if (site not in _common.TWO_HOP_SITES or isinstance(block, bool) or not isinstance(block, int)
            or not 0 <= block < run.config.n_layers):
        raise SystemExit(f"G4.3's best_swap {best!r} doesn't name one of {tuple(_common.TWO_HOP_SITES)} and a block"
                         f" from 0 to {run.config.n_layers - 1}. No report written.")
    return site, block


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def badges_document() -> tuple[dict[str, Any], bytes]:
    """The badges file's contents, from the report files as they are now: ``(badges, canonical JSON bytes)``."""
    from nablaframe.trace.schema import canonical_json, sha256_hex
    from nablaframe.views.badges import model_level_badges

    reports, cited = {}, {}
    for gate_id in SOURCE_REPORTS:
        data = report_path(gate_id).read_bytes()
        reports[gate_id] = json.loads(data.decode("utf-8"))
        cited[gate_id] = {"git_commit": reports[gate_id]["git_commit"],
                          "timestamp_utc": reports[gate_id]["timestamp_utc"], "sha256": sha256_hex(data)}
    made = model_level_badges(reports)
    return made, canonical_json({"badges": made, "reports": cited}).encode("ascii")


def badges(files: tuple[Path, Path]) -> dict[str, Any]:
    """The badges, built twice, into ``runs/phase4/badges.json`` and the scratch folder, compared byte for byte."""
    from nablaframe.views.badges import SUBJECTS

    made = None
    for path in files:
        made, data = badges_document()
        p1.write_atomic(path, data)
    first, second = (path.read_bytes() for path in files)
    identical = first == second
    missing = [subject for subject in SUBJECTS if subject not in made]
    print(f"{GATE_ID}: {len(made)} badges written to {p1.rel(files[0])}; the second build is"
          f" {'byte-identical' if identical else 'different'}; {len(missing)} subjects without a badge", flush=True)
    for subject in SUBJECTS:
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


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_specs(runs: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each model's demo: the tokens, objects and ``(label, spec)`` pairs the command line runs, and for two-hop
    the probe windows, their sources and counterparts, and the site and block."""
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.zoo.two_hop import aligned_sources

    out = {}
    induction = runs["induction"]
    heads = [int(head) for head in reports["G4.2"]["measurements"]["induction_heads"]]
    zero = InterventionSpec(tuple(edit(ZERO_TAP, "all", "zero", head=head) for head in heads))
    out["induction"] = SimpleNamespace(tokens=_common.tokens(induction.task, _common.probe(induction.task)),
                                       objects=_common.p3.Objects(), specs=[("induction heads zeroed", zero)])

    two_hop = runs["two_hop"]
    task = two_hop.task
    site, block = best_swap(reports["G4.3"], two_hop)
    windows = _common.eval_windows(task)
    sources, controls, counterparts = aligned_sources(task, windows, np.random.default_rng([_common.SOURCE_SEED, 0]))
    n = _common.PROBE_WINDOWS
    probe = _common.probe(task)
    if not np.array_equal(windows[:n], probe):
        raise SystemExit("the probe is not the first evaluation windows. No report written.")
    objects = _common.p3.Objects()
    tap, positions = f"blocks.{block}.hook_resid_post", list(_common.TWO_HOP_SITES[site])
    specs = []
    for label, rows in (("source patch", sources), ("control patch", controls)):
        digest = objects.batch(_common.tokens(task, rows[:n]))
        specs.append((label, InterventionSpec((edit(tap, positions, "patch", source=digest),))))
    out["two_hop"] = SimpleNamespace(tokens=_common.tokens(task, probe), objects=objects, specs=specs, windows=probe,
                                     sources=sources[:n], counterparts=counterparts[:n], site=site, block=block)
    return out


def demos(runs: dict[str, Any], reports: dict[str, dict[str, Any]], folders: dict[str, tuple[Path, Path, Path]], *,
          overwrite: bool) -> dict[str, Any]:
    """Each zoo model's compare payloads, built twice by the command line, compared file by file."""
    from nablaframe.interventions.spec import spec_hash

    chosen = demo_specs(runs, reports)
    total = 2 * sum(len(demo.specs) for demo in chosen.values())
    progress = p1.Progress("command-line runs", total)
    builds, payloads, examples, demo_record = {}, {}, [], None
    for name, run in runs.items():
        demo = chosen[name]
        first, second, inputs_dir = folders[name]
        _common.fresh_dir(inputs_dir, overwrite=overwrite)
        inputs = _common.p3.cli_inputs(inputs_dir, run.config, demo.tokens, demo.objects)
        taps = [f"blocks.{layer}.hook_resid_post" for layer in range(run.config.n_layers)]
        paths = [_common.p3.write_spec(inputs_dir / "specs" / f"{spec_hash(spec)}.json", spec)
                 for _, spec in demo.specs]
        failed = []
        for label, out in (("first build", first), ("second build", second)):
            _common.fresh_dir(out, overwrite=overwrite)
            for (spec_label, spec), path in zip(demo.specs, paths):
                code, error = _common.p3.run_cli(inputs, run.folder / "checkpoints", run.final, path, out, taps=taps)
                if code != 0:
                    failed.append({"build": label, "spec": spec_label, "spec_hash": spec_hash(spec), "error": error})
                    print(f"{GATE_ID}: {name}, {label}: {spec_label} failed: {error}", flush=True)
                progress.advance()
        compared = _common.p3.compare_trees(first, second)
        expected = len(demo.specs)
        found = (_payloads(first), _payloads(second))
        identical = (compared["identical"] and not failed and len(found[0]) == expected
                     and len(found[1]) == expected)
        print(f"{GATE_ID}: {name}: {len(found[0])} payloads in {p1.rel(first)}; the two builds are"
              f" {'byte-identical' if identical else 'different'} over {compared['files_compared']} files", flush=True)
        builds[name] = {"identical": identical, "files_compared": compared["files_compared"],
                        "files_differing": compared["files_differing"], "runs_failed": failed,
                        "interventions_dir": p1.rel(first), "second_build_dir": p1.rel(second)}
        payloads[name] = len(found[0])
        examples += [{"model": name, "spec": label, "spec_hash": spec_hash(spec),
                      "file": f"intervention-{spec_hash(spec)}-step-{run.final}.json", "taps": taps}
                     for label, spec in demo.specs]
        if name == "two_hop":
            source_file = f"intervention-{spec_hash(demo.specs[0][1])}-step-{run.final}.json"
            demo_record = two_hop_demo(demo, first / source_file)
    return {"builds": builds, "payloads": payloads, "examples": examples, "demo": demo_record,
            "step": {name: run.final for name, run in runs.items()}}


def two_hop_demo(demo: Any, payload_path: Path) -> dict[str, Any] | None:
    """The source patch on the probe: each window's ``c``, ``d``, ``d′``, and its top-1 answer before and after."""
    from nablaframe.zoo.two_hop import _distractor_object

    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    before = [row[ANSWER_POSITION][0] for row in payload["top_tokens"]["unintervened"]]
    after = [row[ANSWER_POSITION][0] for row in payload["top_tokens"]["intervened"]]
    windows = []
    for b, window in enumerate(demo.windows):
        windows.append({"c": int(window[-1]), "d": _distractor_object(window, int(demo.counterparts[b])),
                        "d_prime": int(demo.sources[b][-1]), "before": int(before[b]), "after": int(after[b])})
    swapped = sum(w["after"] == w["d"] for w in windows)
    transplanted = sum(w["after"] == w["d_prime"] for w in windows)
    print(f"{GATE_ID}: two-hop demo, the source patch at {demo.site} after block {demo.block}: the answer becomes d"
          f" in {swapped} of {len(windows)} probe windows and d' in {transplanted}", flush=True)
    return {"site": demo.site, "block": demo.block, "position": ANSWER_POSITION, "file": payload_path.name,
            "windows": windows, "swapped": swapped, "transplanted": transplanted}


# ---------------------------------------------------------------------------


def print_addresses(runs: dict[str, Any], badges_file: Path) -> None:
    """The viewer's address for each zoo run, every path relative to viewer/."""
    viewer = Path(p1.REPO_ROOT) / "viewer"

    def relative(path: Path, folder: bool = True) -> str:
        text = Path(os.path.relpath(path, viewer)).as_posix()
        return f"{text}/" if folder else text

    print(f"{GATE_ID}: serve the repo root with `python3 -m http.server`, then open:", flush=True)
    for name, run in runs.items():
        query = (f"?views={relative(run.folder / 'views')}&trace={relative(run.store)}"
                 f"&interventions={relative(run.folder / INTERVENTIONS_NAME)}"
                 f"&badges={relative(badges_file, folder=False)}")
        print(f"  {name}: {VIEWER_ADDRESS}{query}", flush=True)


def _payloads(folder: Path) -> list[str]:
    return sorted(path.name for path in Path(folder).glob("*.json") if PAYLOAD.fullmatch(path.name))


def _sha256(data: bytes) -> str:
    from nablaframe.trace.schema import sha256_hex

    return sha256_hex(data)


if __name__ == "__main__":
    raise SystemExit(main())
