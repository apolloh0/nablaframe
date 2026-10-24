"""G7.4: every judged and recorded crystallization, trained and twin, with its counterfactual and its envelope (D56 to D58).

Spec: docs/specs/phase-7.md, D56 to D58, "What each run declares" and "Discrimination tests on the
zoo"; helpers in ``_common.py``. Report: ``reports/phase7/g74_crystallizations.json``. Run in
nf-jax, on the Mac, after G4.1, G5.2, G5.3, G6.3, G6.7 and G7.1 to G7.3 pass.

For each of ``JUDGED`` and then ``RECORDED``, in order, a crystallize-this record through
``cli.record`` into ``runs/phase7/gestures/<run>/``. The request: the run's id, its final
checkpoint, the crystallization's block, part, class, function and form, ``instances`` the
held-out windows' count, ``SEED``, and ``cites`` as below. The run's ``cli.Context``:

* **induction and two-hop:** G4.1's checkpoints, every one; Phase 5's training and held-out
  windows (``zoo_windows``) as the fitting and held-out windows; the twin,
  ``init_params(config, seed=1)``; ``functions_for(run)``; and ``cited`` reading Phase 6's
  records, ``runs/phase6/gestures/<run>/``.
* **modular addition:** G7.1's checkpoints, every one; its training and held-out pairs; its twin
  and functions; and ``cited`` reading ``runs/phase7/gestures/modular_addition/``.

``cites``: ``crystallize.cite`` of every record in the folder ``cited`` reads, by its
``index.json``, with the crystallization's class, ``describes`` of its run's held-out windows,
the dictionary at its part's output site, and its function's relation. The dictionary is the run's Phase 5 TopK SAE at the block's
``hook_resid_post`` for induction and two-hop, and G7.2's for modular addition, each at site 0.

Each record's results are summarized in ``crystallizations``, ``{"<run> <part> <function>
<form>": {"key", "judged", "meter", "wrong", "counterfactual", "twin", "envelope", "cites"}}``,
those being the record's, ``twin`` holding the twin's meter and wrong functions. The judged ones'
``crystal_checks`` go through ``judged``: its seven values are the report's measurements of the
same names.

Measurements: REQUIRED_MEASUREMENTS["G7.4"]. Pass needs every bar. Records already in a folder
under their keys are read, not built again, as ``cli.record`` does.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase7/g74_crystallizations.py``. Takes --overwrite and --allow-dirty. It reads G4.1's runs and G7.1's
#   after ``require`` of G4.1, G5.2, G5.3, G6.3, G6.7, G7.1, G7.2 and G7.3, with ``zoo_run``'s and
#   ``modadd_run``'s checks. Every dictionary comes from a cache: the zoo runs' from Phase 5's
#   (``p6.topk``), the specimen's from G7.2's (``_common.dictionary``). It writes nothing but its records and
#   its report.
# * Each run is read once, when its first crystallization comes up, with every record of the folder its
#   crystallizations cite (``_common.folder_records``); a folder without an ``index.json`` stops the gate.
# * Each crystallization's context loads its checkpoints through ``_common.announced``, which prints a line as
#   each is loaded, and the whole build runs inside ``ticking``. A record already in the folder is read, so it
#   prints no checkpoint lines.
# * ``crystallizations[name]["cites"]`` is the record's ``results["cites"]``: for each key the names it cites,
#   with their share of values named as planted and the note where they fall short, and the edges with their
#   labels. ``checks`` holds each judged crystallization's ``crystal_checks``.

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1, p6 = _common.p1, _common.p6

GATE_ID = "G7.4"
REPORT_NAME = "g74_crystallizations.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G6.3", "G6.7", "G7.1", "G7.2", "G7.3")
SUMMARY = ("meter", "wrong", "counterfactual", "envelope", "cites")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, reports = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(reports["G4.1"], reports["G7.1"])
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G7.4 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["kl_max"]) and m["kl_max"] <= bars["kl"]
          and n(m["ablation_min"]) and m["ablation_min"] >= bars["ablation"]
          and n(m["wrong_min"]) and m["wrong_min"] >= bars["wrong"]
          and n(m["reads_agreement_min"]) and m["reads_agreement_min"] >= bars["reads_agreement"]
          and n(m["answer_agreement_max"]) and m["answer_agreement_max"] <= bars["answer_agreement"]
          and n(m["twin_within_max"]) and m["twin_within_max"] <= bars["twin_within"]
          and m["envelope_ok"] is True)
    return "pass" if ok else "fail"


class Runs:
    """Each run's model, context, cited records and the dictionary at a block's output, built when first needed."""

    def __init__(self, g41: dict[str, Any], g71: dict[str, Any]) -> None:
        self.g41, self.g71 = g41, g71
        self.models: dict[str, Any] = {}
        self.contexts: dict[str, Any] = {}
        self.cited: dict[str, list[Any]] = {}
        self.dictionaries: dict[tuple[str, int], str] = {}

    def get(self, run: str) -> tuple[Any, Any, list[Any]]:
        if run not in self.models:
            if run == _common.RUN_NAME:
                m, folder = _common.modadd_model(self.g71), _common.record_dir(run)
            else:
                m, folder = _common.zoo_model(run, self.g41), Path(p6.RECORD_DIR) / run
            _common.say(f"{GATE_ID}: {run} run {m.run_id}, final step {m.step:,}, {len(m.run.steps)} checkpoints;"
                        f" {len(m.fit):,} fitting and {len(m.held_out):,} held-out windows; citing from"
                        f" {p1.rel(folder)}")
            self.cited[run] = _common.folder_records(folder)
            self.contexts[run] = _common.context(m, cited=folder)
            self.models[run] = m
        return self.models[run], self.contexts[run], self.cited[run]

    def dictionary(self, run: str, block: int) -> str:
        """The content hash of the dictionary at the block's ``hook_resid_post``: Phase 5's TopK SAE, or G7.2's."""
        from nablaframe.dictionaries import core

        if (run, block) not in self.dictionaries:
            m = self.models[run]
            made = _common.dictionary(m)[0] if run == _common.RUN_NAME else p6.topk(m, block)
            self.dictionaries[(run, block)] = core.content_hash(made)
        return self.dictionaries[(run, block)]


def measure(g41: dict[str, Any], g71: dict[str, Any]) -> dict[str, Any]:
    runs = Runs(g41, g71)
    crystallizations: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for entry in _common.JUDGED + _common.RECORDED:
        name = _common.crystallization_name(entry)
        judged = entry in _common.JUDGED
        results, key = crystallization(runs, entry, judged)
        crystallizations[name] = {"key": key, "judged": judged, **{field: results[field] for field in SUMMARY},
                                  "twin": {"meter": results["twin"]["meter"], "wrong": results["twin"]["wrong"]}}
        if judged:
            checks[name] = _common.crystal_checks(results)
        say_results(name, results)

    out = {
        "crystallizations": crystallizations,
        **_common.judged(list(checks.values())),
        "checks": checks,
        "bars": _common.BARS,
        "seed": _common.SEED,
        "run_ids": {run: m.run_id for run, m in runs.models.items()},
        "steps": {run: m.step for run, m in runs.models.items()},
        "checkpoints": {run: list(m.run.steps) for run, m in runs.models.items()},
        "dictionaries": {f"{run} block {block}": digest for (run, block), digest in runs.dictionaries.items()},
    }
    _common.say(f"{GATE_ID}: KL at most {_common.show(out['kl_max'])}, ablation at least"
                f" {_common.show(out['ablation_min'])}, wrong functions at least {_common.show(out['wrong_min'])};"
                f" counterfactual agreement, keyed by what it reads at least"
                f" {_common.show(out['reads_agreement_min'])},"
                f" by the answer at most {_common.show(out['answer_agreement_max'])}; the twins within"
                f" {_common.show(out['twin_within_max'])}; envelopes {out['envelope_ok']}")
    return out


def crystallization(runs: Runs, entry: tuple[Any, ...], judged: bool) -> tuple[dict[str, Any], str]:
    """One crystallization's record, through ``cli.record`` into the run's Phase 7 records folder."""
    from nablaframe.gestures import cli

    run, block = entry[0], entry[1]
    m, ctx, cited = runs.get(run)
    cites = _common.citations(m, cited, entry[4], runs.dictionary(run, block))
    request = _common.crystallize_request(m, entry, cites)
    label = f"{GATE_ID}: {_common.crystallization_name(entry)}{'' if judged else ', recorded'}"
    _common.say(f"{label}: citing {len(cites)} records, {len(m.run.steps)} checkpoints")
    announced = replace(ctx, params_at=_common.announced(ctx.params_at, label, ctx.steps))
    with _common.ticking(label, write=_common.say):
        record = cli.record(_common.record_dir(run), request, announced)
    return _common.p6.results_of(record), record.key


def say_results(name: str, results: dict[str, Any]) -> None:
    meter, twin, envelope = results["meter"], results["twin"]["meter"], results["envelope"]
    wrong = ", ".join(f"{w['key']} {w['kl']:.4f}" for w in results["wrong"]) or "none"
    cf = results["counterfactual"]
    shown = "none" if cf is None else (f"reads {cf['reads']['agreement']:.3f}, answer {cf['answer']['agreement']:.3f}")
    _common.say(f"{GATE_ID}: {name}: {meter['label']}, KL {meter['kl']:.4f} against the ablation's"
                f" {meter['ablation']:.4f}, agreement {meter['agreement']:.3f}, loss delta {meter['loss_delta']:.4f};"
                f" wrong functions {wrong}; counterfactual {shown}; twin KL {twin['kl']:.4f}; envelope runs"
                f" {envelope['runs']}, gap {envelope['gap']}")


if __name__ == "__main__":
    raise SystemExit(main())
