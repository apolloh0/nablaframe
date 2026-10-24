"""G7.3: ``set`` keeps its identities on every zoo model, crystallize-this requests hash apart, and each form's record rebuilds byte for byte (D47, D55, D56).

Spec: docs/specs/phase-7.md, D55, D56 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase7/g73_determinism.json``. Run in nf-jax, on the Mac, after G4.1, G5.2, G5.3, G7.1
and G7.2 pass.

* **Identities:** for the induction, two-hop and modular-addition models at their final
  checkpoints, ``set_identities`` on the tokens the model reads of the first
  ``IDENTITY_WINDOWS`` held-out windows: Phase 5's for the zoo runs, the held-out pairs for
  modular addition.
  ``identities_by_run``: ``{run: {tap: bool}}``; ``identities_checked`` and ``identities_exact``,
  counts over all three.
* **Request hashes:** ``request_hashes(requests(run id, its checkpoint steps, two record keys))``,
  the keys being any two of G7.2's name records: ``request_hashes_checked``,
  ``request_hashes_distinct`` and ``request_hashes_stable``.
* **Records:** on the modular-addition run's ``cli.Context`` with two checkpoints, step 0 and
  the final one: its training pairs as fitting windows, its held-out pairs, its twin, its
  functions, and ``cited`` reading ``runs/phase7/gestures/modular_addition/``. Two checkpoints
  keep nine builds affordable; G7.4 reads every one. One request per form, at the final
  checkpoint, with ``SEED``: the write as a table by the sum and the MLP additive in the sum and
  operands, each citing G7.2's records as G7.4 cites them; and the write in the fourier form,
  citing nothing. Each is built twice with
  ``cli.build``, never through a cache, and once more by the command line's code: ``cli.main``,
  called in this process, with ``--run modular_addition`` and files written under
  ``runs/phase7/g73_determinism/``: the config, both windows' ``.npy`` files, the request, a
  ``checkpoints`` folder holding copies of those two checkpoints with their sidecars, and a fresh
  ``--out``; ``--records`` is the run's record folder. ``records_by_form``: ``{form: [the two
  builds' canonical JSON is identical, the command line's file is that and a newline]}``.
  ``records_identical``: all three forms checked, every value true.

Measurements: REQUIRED_MEASUREMENTS["G7.3"]. Pass needs at least one identity checked and every
one exact, the request hashes distinct and stable, and ``records_identical``. The working folder,
``runs/phase7/g73_determinism/``, is started fresh, and ``--overwrite`` lets it replace one.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g73_determinism.py``.
#   Takes --overwrite and --allow-dirty. It reads G4.1's runs and G7.1's after ``require`` of G4.1, G5.2,
#   G5.3, G7.1 and G7.2, with ``zoo_run``'s and ``modadd_run``'s checks. G7.2's dictionary comes from the cache
#   (``_common.dictionary``). It writes nothing outside its report but its working folder.
# * ``identities_by_run`` is keyed by run name. The request hashes use the first two keys, in sorted order, of
#   the name records in G7.2's records folder.
# * The records: one ``cli.Context`` of the specimen with its two checkpoints. ``cites`` is chosen as G7.4
#   chooses it (``_common.citations``), from every record in the run's records folder, the dictionary being
#   G7.2's. The two in-process builds and the command line's share nothing but their inputs: each
#   ``cli.build`` runs every pass again, and ``cli.main`` loads the checkpoints and the windows from their files.
#   The command line writes to ``records/``, which starts empty, so no record comes from a cache.
#   ``records_by_form``: ``{form: [builds identical, command line identical]}``, and ``record_keys``:
#   ``{form: key}`` beside it.

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G7.3"
REPORT_NAME = "g73_determinism.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G7.1", "G7.2")
WORK_NAME = "g73_determinism"
FORMS = {"table": ("modular_addition", 0, "write", "answer", "sum", "table"),
         "fourier": ("modular_addition", 0, "write", "answer", "sum", "fourier"),
         "additive": ("modular_addition", 0, "mlp", "answer", "sum and operands", "additive")}
CITING = ("table", "additive")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, reports = _common.start(args, REPORT_NAME, NEEDS)
    work = _common.fresh_dir(Path(_common.RUN_DIR) / WORK_NAME, overwrite=args.overwrite)
    measurements = measure(reports["G4.1"], reports["G7.1"], work)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G7.3 rule holds."""
    n = _common.number
    ok = (n(m["identities_checked"]) and m["identities_checked"] >= 1
          and n(m["identities_exact"]) and m["identities_exact"] == m["identities_checked"]
          and m["request_hashes_distinct"] is True and m["request_hashes_stable"] is True
          and m["records_identical"] is True)
    return "pass" if ok else "fail"


def measure(g41: dict[str, Any], g71: dict[str, Any], work: Path) -> dict[str, Any]:
    models = {name: _common.zoo_model(name, g41) for name in _common.p6.ZOO}
    models[_common.RUN_NAME] = _common.modadd_model(g71)
    m = models[_common.RUN_NAME]

    by_run = identities(models)
    flags = [flag for taps in by_run.values() for flag in taps.values()]

    folder = _common.record_dir(_common.RUN_NAME)
    held = _common.folder_records(folder)
    names = sorted(r.key for r in held if r.request.kind == "name")
    if len(names) < 2:
        raise SystemExit(f"{p1.rel(folder)} holds {len(names)} name records; the requests cite two. Run G7.2 first."
                         " No report written.")
    reqs = _common.requests(m.run_id, m.run.steps, names[:2])
    request_hashes = _common.request_hashes(reqs)
    _common.say(f"{GATE_ID}: {request_hashes['checked']} requests, hashes distinct {request_hashes['distinct']},"
                f" stable {request_hashes['stable']}")

    by_form, keys, cites = rebuilt_records(m, held, work)
    out = {
        "identities_checked": len(flags),
        "identities_exact": sum(bool(flag) for flag in flags),
        "identities_by_run": by_run,
        "identity_windows": _common.IDENTITY_WINDOWS,
        "request_hashes_checked": request_hashes["checked"],
        "request_hashes_distinct": request_hashes["distinct"],
        "request_hashes_stable": request_hashes["stable"],
        "request_cites": names[:2],
        "records_identical": len(by_form) == len(FORMS) and all(all(v) for v in by_form.values()),
        "records_by_form": by_form,
        "record_keys": keys,
        "record_cites": cites,
        "record_steps": [m.run.steps[0], m.step],
        "seed": _common.SEED,
        "run_ids": {name: model.run_id for name, model in models.items()},
        "steps": {name: model.step for name, model in models.items()},
        "held_out_sha256": {name: _batch_hash(model.held_out) for name, model in models.items()},
    }
    _common.say(f"{GATE_ID}: identities exact {out['identities_exact']} of {out['identities_checked']}; records"
                f" identical {out['records_identical']}")
    return out


def identities(models: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """D55's identities at every residual-shaped tap of each model, on the first held-out windows."""
    by_run = {}
    for name, m in models.items():
        tokens = m.held_tokens[:_common.IDENTITY_WINDOWS]
        with _common.ticking(f"{GATE_ID}: set's identities on {name}", write=_common.say):
            by_run[name] = _common.set_identities(m.model, m.params, tokens, n_layers=m.n_layers)
        exact = sum(by_run[name].values())
        _common.say(f"{GATE_ID}: {name}: {exact} of {len(by_run[name])} taps exact"
                    + ("" if exact == len(by_run[name]) else ": not " + ", ".join(
                        tap for tap, ok in by_run[name].items() if not ok)))
    return by_run


def rebuilt_records(m: Any, held: list[Any], work: Path) -> tuple[dict[str, list[bool]], dict[str, str],
                                                                    dict[str, list[str]]]:
    """Each form's record built twice with ``cli.build`` and once by ``cli.main``, compared byte for byte."""
    from nablaframe.dictionaries import core
    from nablaframe.gestures import cli, records
    from nablaframe.trace.schema import canonical_json

    dictionary, _ = _common.dictionary(m)
    content = core.content_hash(dictionary)
    first, final = m.run.steps[0], m.step
    folder = _common.record_dir(_common.RUN_NAME)
    ctx = _common.context(m, steps=(first, final), cited=folder)

    files = {"config": work / "config.json", "windows": work / "held_out.npy", "fit": work / "fit_windows.npy",
             "checkpoints": work / "checkpoints"}
    files["config"].write_text(json.dumps(asdict(m.config), sort_keys=True) + "\n", encoding="utf-8")
    np.save(files["windows"], m.held_out, allow_pickle=False)
    np.save(files["fit"], m.fit, allow_pickle=False)
    files["checkpoints"].mkdir()
    for step in (first, final):
        source = _common.p4.checkpoint(m.run.folder / "checkpoints", m.run_id, step)
        for path in (source, source.with_suffix(".json")):
            shutil.copy2(path, files["checkpoints"] / path.name)
    out = work / "records"
    out.mkdir()

    by_form, keys, cites = {}, {}, {}
    for form, entry in FORMS.items():
        cites[form] = _common.citations(m, held, entry[4], content) if form in CITING else []
        request = _common.crystallize_request(m, entry, cites[form])
        label = f"{GATE_ID}: the {form} record ({_common.crystallization_name(entry)})"
        built = []
        for attempt in (1, 2):
            run_ctx = _replace(ctx, params_at=_common.announced(ctx.params_at, f"{label}, build {attempt}", ctx.steps))
            with _common.ticking(f"{label}, build {attempt}", write=_common.say):
                built.append(cli.build(request, run_ctx))
        texts = [canonical_json(records.record_json(r)) for r in built]
        path = work / f"request-{form}.json"
        path.write_text(json.dumps(records.request_json(request), sort_keys=True) + "\n", encoding="utf-8")
        argv = ["--run", _common.RUN_NAME, "--config", str(files["config"]), "--checkpoints",
                str(files["checkpoints"]), "--windows", str(files["windows"]), "--fit-windows", str(files["fit"]),
                "--request", str(path), "--out", str(out), "--records", str(folder)]
        with _common.ticking(f"{label}, the command line's", write=_common.say):
            code = cli.main(argv)
        written = out / f"gesture-{built[0].key}.json"
        by_command = code == 0 and written.is_file() and written.read_bytes() == (texts[0] + "\n").encode("ascii")
        by_form[form] = [texts[0] == texts[1], bool(by_command)]
        keys[form] = built[0].key
        _common.say(f"{label}, key {built[0].key[:12]}, citing {len(cites[form])}: builds identical"
                    f" {by_form[form][0]}, the command line's identical {by_form[form][1]}")
    return by_form, keys, cites


def _replace(ctx: Any, **changes: Any) -> Any:
    from dataclasses import replace

    return replace(ctx, **changes)


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
