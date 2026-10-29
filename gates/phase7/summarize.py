"""The Phase 7 summary: reports/phase7/summary.md, generated from the five Phase 7 reports.

Same rules as gates/phase6/summarize.py: never written by hand, a missing or invalid report says
so rather than being skipped, and every number comes from a report. Its H7 evidence:

* G7.1: the specimen's behaviour beside its bars: held-out, training, control and twin accuracy,
  and chance; the clock: the fewest frequencies and their energy beside the twin's, the fourier
  form's frequencies and the logits' R²; the recompute; the trace's events; and the first step at
  which training and held-out accuracy each reach 0.99
* G7.2: the dictionary's FVU beside the twin's; for each description at ``answer``, trained beside
  twin: its values, the shares named and named as planted, and steering's follows beside the
  null's; and the records written
* G7.3: the identities exact of those checked, by run; the request hashes checked, and whether they
  were distinct and stable; and for each form, whether its record built twice, and through the
  command line, was identical
* G7.4: each crystallization, judged or recorded: its label; the KL beside the ablation's, with the
  share removed; agreement and the loss delta; each wrong function's KL; the counterfactual, keyed
  by what the component reads and by the answer; the twin's label, KL and wrong functions; the
  envelope's runs, its gap and its checkpoints before accuracy reaches 0.9; and its citations: for
  each key, the records cited, the share of values named as planted and its note; then the judged
  bounds beside the bars
* G7.5: the badges, each with its status and facts, as Phase 6's summary lists them; whether the
  builds matched; each run's records by kind; whether the reports matched; and the clock: modular
  addition's fourier crystallization, each frequency with its share of the embedding's energy and
  of the fitted coefficients, and its envelope

The verdict covers G7.1, G7.3, G7.4 and G7.5. G7.2 always passes, so its numbers are listed without
a verdict of their own.
"""

# summarize() raises ValueError, naming every problem, when a report is unreadable, invalid
# (validate_report, which includes the status rules), not a Phase 7 report, or when the inputs
# are not exactly one report per gate. main() prints the problems, writes nothing and returns 1.
#
# A pure function of the report files: no timestamps of its own, rows in gate order. Reports
# written from a dirty tree, or from different commits or lockfiles, are named as such. It never
# states a result that is not in a report. apolloh0 rules H7.
#
# Reports are canonical JSON, whose keys come back sorted, so the summary puts them back in the
# spec's order: G7.4's crystallizations as D57 lists them, judged then recorded (``_common.JUDGED``
# and ``RECORDED``); G7.2's descriptions as ``NAMES``; G7.3's forms as table, fourier, additive;
# runs as modular addition, induction, two-hop; checkpoints by step. A rate, a share or a fraction
# is shown as a percentage; a KL in nats, an R², an FVU or a ratio to three decimals; a step, a count
# or a frequency as a number. Badges follow views/badges.py's SUBJECTS, then
# views/dictionary_badges.py's, views/gesture_badges.py's and views/crystallize_badges.py's, each
# fact in the badge's own order, its value shown by its unit.
#
# Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
# gates/phase7/summarize.py``.

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nablaframe.reports import PHASE7_GATE_IDS, validate_report
from nablaframe.views import badges as phase4_badges
from nablaframe.views import crystallize_badges, dictionary_badges, gesture_badges

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "phase7"
SUMMARY_PATH = REPORTS_DIR / "summary.md"

GATE_NAMES = {
    "G7.1": "The modular-addition specimen",
    "G7.2": "Its dictionary and names",
    "G7.3": "Determinism and the operation",
    "G7.4": "Crystallizations",
    "G7.5": "Records, views and badges",
}
# The spec's "On failure" column, docs/specs/phase-7.md, Gates.
ON_FAILURE = {
    "G7.1": "diagnose training",
    "G7.2": "not applicable",
    "G7.3": "diagnose the operation or the record",
    "G7.4": "diagnose the gesture",
    "G7.5": "diagnose the build",
}
RECORDED_GATES = ("G7.2",)
VERDICT_GATES = tuple(gate_id for gate_id in PHASE7_GATE_IDS if gate_id not in RECORDED_GATES)
STATUS_TEXT = {"pass": "pass", "conditional_pass": "conditional pass", "fail": "fail"}
SUBJECTS = (tuple(phase4_badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS) + tuple(gesture_badges.SUBJECTS)
            + tuple(crystallize_badges.SUBJECTS))
RUN_NAMES = {"modular_addition": "Modular addition", "induction": "Induction", "two_hop": "Two-hop"}
# D57's crystallizations, judged then recorded, as gates/phase7/_common.py's JUDGED and RECORDED name them.
CRYSTALLIZATIONS = ("induction attention copied table", "two_hop write hop table", "modular_addition write sum table",
                    "modular_addition write sum fourier", "modular_addition mlp sum and operands additive",
                    "modular_addition attention operands additive", "two_hop attention hop table")
DESCRIPTIONS = ("token(p+1)", "token(p-2)", "token(p-1)")
FORMS = ("table", "fourier", "additive")
KINDS = ("name", "crystallize")
WHO = ("trained", "twin")
_RUNS = ("modular_addition", "induction", "two_hop")


def summarize(report_paths: list[Path]) -> str:
    reports = _load(report_paths)
    m = {gate_id: report["measurements"] for gate_id, report in reports.items()}
    status = {gate_id: report["status"] for gate_id, report in reports.items()}

    lines = [
        "# Phase 7 summary",
        "",
        "Generated by `gates/phase7/summarize.py` from the five Phase 7 gate reports. Do not edit by hand;"
        " regenerate with `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/summarize.py`.",
        "",
        "## Gates",
        "",
        "| Gate | Status | Key measurements |",
        "| --- | --- | --- |",
    ]
    for gate_id in PHASE7_GATE_IDS:
        shown = "always passes" if gate_id in RECORDED_GATES else STATUS_TEXT[status[gate_id]]
        lines.append(f"| {gate_id} {GATE_NAMES[gate_id]} | {shown} | {_cell(_key(gate_id, m[gate_id]))} |")

    lines += [
        "",
        "## Provenance",
        "",
        "| Gate | Commit | Dirty tree | Env | Chip | Threads | XLA_FLAGS | Lockfile SHA-256 | Report time (UTC) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for gate_id in PHASE7_GATE_IDS:
        r = reports[gate_id]
        flags = f"`{_cell(r['xla_flags'])}`" if r["xla_flags"] else "none"
        lines.append(
            f"| {gate_id} | `{r['git_commit'][:12]}` | {_yes(r['git_dirty'])} | {r['env_name']} | {_cell(r['chip'])}"
            f" | {r['thread_count']} | {flags} | `{r['env_lock_sha256'][:12]}` | {r['timestamp_utc']} |"
        )
    lines += ["", *_consistency(reports)]

    lines += [
        "",
        "## H7 evidence",
        "",
        "H7 asks two questions. Does crystallize-this replace each judged component with its explicit function at a"
        " small KL, while every wrong function fails, the twin shows nothing, and on counterfactual inputs the"
        " crystallization follows what the component reads, not the answer? And does modular addition's block write at"
        " `=` crystallize into a few frequencies of `(a + b) mod p`, with its loss delta, its envelope across training"
        " and its gap shown, and with the record saying why Phase 6's names fall short?",
        "",
        _verdict(status),
        "",
        "### The modular-addition specimen (G7.1)",
        "",
        *_specimen(m["G7.1"]),
        "",
        "### Its dictionary and names (G7.2)",
        "",
        *_names(m["G7.2"]),
        "",
        "### Determinism and the operation (G7.3)",
        "",
        *_determinism(m["G7.3"]),
        "",
        "### Crystallizations (G7.4)",
        "",
        *_crystallizations(m["G7.4"]),
        "",
        "### Records, views and badges (G7.5)",
        "",
        *_records(m["G7.5"]),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    paths = sorted(args.reports_dir.glob("g7*.json"))
    try:
        text = summarize(paths)
    except ValueError as exc:
        print(f"No summary written.\n{exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Summary written to {args.out}")
    return 0


# ---------------------------------------------------------------------------


def _load(report_paths: list[Path]) -> dict[str, Mapping[str, Any]]:
    problems: list[str] = []
    by_gate: dict[str, list[tuple[Path, Mapping[str, Any]]]] = {}
    for path in report_paths:
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path}: cannot read JSON ({exc})")
            continue
        if not isinstance(report, Mapping):
            problems.append(f"{path}: not a JSON object")
            continue
        invalid = validate_report(report)
        if invalid:
            problems += [f"{path}: {problem}" for problem in invalid]
            continue
        if report["gate_id"] not in PHASE7_GATE_IDS:
            problems.append(f"{path}: {report['gate_id']} is not a Phase 7 gate")
            continue
        by_gate.setdefault(report["gate_id"], []).append((Path(path), report))
    for gate_id in PHASE7_GATE_IDS:
        found = by_gate.get(gate_id, [])
        if not found:
            problems.append(f"no report for {gate_id}")
        elif len(found) > 1:
            problems.append(f"{len(found)} reports for {gate_id}: {', '.join(str(p) for p, _ in found)}")
    if problems:
        raise ValueError("cannot summarize:\n" + "\n".join(f"  - {p}" for p in problems))
    return {gate_id: by_gate[gate_id][0][1] for gate_id in PHASE7_GATE_IDS}


def _verdict(status: dict[str, str]) -> str:
    not_passed = [gate_id for gate_id in VERDICT_GATES if status[gate_id] != "pass"]
    if not not_passed:
        return ("The gates support H7: G7.1, G7.3, G7.4 and G7.5 all pass. G7.2 records, not judges; its numbers are"
                " below, beside the crystallizations that cite them. Modular addition's fourier crystallization, its"
                " frequencies and its envelope are at the end, with each badge's facts. A crystallization is demonstrated"
                " over its envelope, never proven. Your own session in the viewer is the rest of the evidence.")
    items = "\n".join(f"- {gate_id} {GATE_NAMES[gate_id]}: {STATUS_TEXT[status[gate_id]]}. The spec's next step:"
                       f" {ON_FAILURE[gate_id]}." for gate_id in not_passed)
    return f"The gates do **not** yet support H7. These gates did not pass:\n\n{items}"


def _key(gate_id: str, m: Mapping[str, Any]) -> str:
    if gate_id == "G7.1":
        return (f"held-out accuracy {_pct(m['held_accuracy'])}; twin {_pct(m['twin_accuracy'], 2)} and position 0"
                f" {_pct(m['control_accuracy'], 2)}, chance {_pct(m['chance'], 2)}; {_num(m['frequencies'], ',')}"
                f" frequencies hold {_pct(m['energy'])} of the embedding's energy (twin {_num(m['twin_frequencies'], ',')});"
                f" the logits' R² {_three(m['logit_r2'])}; recompute identical: {_yes(m['recompute_identical'])}")
    if gate_id == "G7.2":
        sum_name = _map(_map(_map(m["names"]).get("token(p+1)")).get("trained"))
        return (f"recorded, not judged; FVU {_three(m['fvu'])} (twin {_three(m['twin_fvu'])}); the sum named as planted"
                f" for {_pct(sum_name.get('named_as_planted'))} of its values; {_num(m['records'], ',')} name records")
    if gate_id == "G7.3":
        return (f"{_num(m['identities_exact'], ',')} of {_num(m['identities_checked'], ',')} identities bit for bit;"
                f" {_num(m['request_hashes_checked'], ',')} request hashes, distinct: {_yes(m['request_hashes_distinct'])},"
                f" stable: {_yes(m['request_hashes_stable'])}; records rebuilt identically: {_yes(m['records_identical'])}")
    if gate_id == "G7.4":
        return (f"KL at most {_nats(m['kl_max'])}, the ablation's at least {_nats(m['ablation_min'])}; wrong functions at"
                f" least {_nats(m['wrong_min'])}; counterfactual agreement keyed by what it reads at least"
                f" {_pct(m['reads_agreement_min'])}, by the answer at most {_pct(m['answer_agreement_max'])}; the twins"
                f" within {_nats(m['twin_within_max'])}; envelopes: {_yes(m['envelope_ok'])}")
    records = _map(m["records"])
    counts = "; ".join(f"{RUN_NAMES.get(run, _cell(run))} "
                       + ", ".join(f"{_num(_map(records[run]).get(kind), ',')} {kind}" for kind in KINDS)
                       for run in _runs(records))
    return (f"{_num(m['subjects_missing'], ',')} subjects without a badge; builds identical: {_yes(m['builds_identical'])};"
            f" {_num(m['external_urls'], ',')} external URLs; records shown are the ones G7.4 measured and G7.2"
            f" recorded: {_yes(m['reports_match'])}; records: {counts or 'none recorded'}")


def _consistency(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    commits = sorted({r["git_commit"] for r in reports.values()})
    locks = sorted({r["env_lock_sha256"] for r in reports.values()})
    lines.append(f"All five reports come from commit `{commits[0]}`." if len(commits) == 1
                 else f"The reports come from {len(commits)} different commits.")
    lines.append(f"All five reports come from lockfile `{locks[0]}`." if len(locks) == 1
                 else f"The reports come from {len(locks)} different lockfiles.")
    dirty = [gate_id for gate_id in PHASE7_GATE_IDS if reports[gate_id]["git_dirty"]]
    if dirty:
        lines.append(f"Written from a tree with uncommitted changes: {', '.join(dirty)}. Those reports can't be"
                     " reproduced from their commit alone.")
    return [" ".join(lines)]


def _bound(name: str, value: Any, bars: Mapping[str, Any], bar: str, *, at_least: bool, shown: Any = None) -> str:
    """One judged bound beside its bar from the report's own ``bars``: ``<name> <value>, bar at least <bar>: met yes``."""
    shown = shown or _pct
    limit = bars.get(bar)
    met = _at_least(value, limit) if at_least else _at_most(value, limit)
    return f"{name} {shown(value)}, bar {'at least' if at_least else 'at most'} {shown(limit)}: met {met}."


# ---------------------------------------------------------------------------
# G7.1
# ---------------------------------------------------------------------------


def _specimen(m: Mapping[str, Any]) -> list[str]:
    bars, clock, trace = _map(m.get("bars")), _map(m.get("clock")), _map(m.get("trace"))
    chance = m["chance"]
    ratio = bars.get("control_ratio")
    twice = ratio * chance if _is_number(ratio) and _is_number(chance) else None
    behaviour = _map(m.get("behaviour"))
    lines = [f"`ModularAdditionTask()` at its defaults, trained from step 0 on {_num(behaviour.get('training_pairs'), ',')}"
             f" training pairs and measured on {_num(behaviour.get('held_out_pairs'), ',')} held-out pairs, from logits"
             " alone at the final checkpoint. Run"
             f" `{_cell(str(m.get('run_id', ''))[:12])}`.", "",
             "- " + _bound("Held-out accuracy at position 2", m["held_accuracy"], bars, "held_accuracy", at_least=True),
             f"- Training accuracy at position 2 {_pct(m['train_accuracy'])}, recorded.",
             f"- Position 0, which can't predict `b` from `a`, {_pct(m['control_accuracy'], 2)}, bar at most"
             f" {_three(ratio)} times chance, {_pct(chance, 2)}: met {_at_most(m['control_accuracy'], twice)}.",
             f"- The random-weights twin at position 2 {_pct(m['twin_accuracy'], 2)}, bar at most {_three(ratio)} times"
             f" chance: met {_at_most(m['twin_accuracy'], twice)}.",
             "", "The clock, on the final checkpoint's weights and logits, never a view:", "",
             "- " + _bound("The fewest frequencies holding 0.75 of the embedding's non-constant energy",
                           m["frequencies"], bars, "frequencies", at_least=False, shown=lambda v: _num(v, ","))
             + f" They hold {_pct(m['energy'])}; the twin needs {_num(m['twin_frequencies'], ',')}"
             f" ({_pct(clock.get('twin_energy'))}).",
             f"- The fourier form's frequencies: {', '.join(_num(k, ',') for k in _list(clock.get('fourier_k'))) or 'none'},"
             f" holding {_pct(clock.get('fourier_held'))}.",
             "- " + _bound("The logits against those frequencies' cosines, R² on held-out pairs,", m["logit_r2"], bars,
                           "logit_r2", at_least=True, shown=_three),
             "", f"Retrained from step 0, the final state hash is identical: {_yes(m['recompute_identical'])}"
             f" (`{_cell(str(m.get('final_state_sha256', ''))[:12])}` and `{_cell(str(m.get('recompute_state_sha256', ''))[:12])}`)."
             f" Checkpoints written: {_num(m['checkpoints_written'], ',')} of {_num(m['checkpoints_scheduled'], ',')}. The"
             f" trace recorded {_num(m['events_recorded'], ',')} of {_num(m['events_expected'], ',')} events on a probe of"
             f" {_num(_list(trace.get('probe_shape'))[0] if _list(trace.get('probe_shape')) else None, ',')} held-out pairs"
             + (f"; it stopped at the {_cell(trace.get('error_stage'))} stage: {_cell(trace.get('error'))}."
                if trace.get("error") else ".")]
    first = _map(m.get("first_at_099"))
    reached = [f"{what} accuracy at step {_num(first.get(key), ',')}" if first.get(key) is not None
               else f"{what} accuracy at no checkpoint" for key, what in (("train", "training"), ("held", "held-out"))]
    lines += ["", f"The first checkpoint reaching 0.99: {reached[0]}, {reached[1]}."]
    return lines


# ---------------------------------------------------------------------------
# G7.2
# ---------------------------------------------------------------------------


def _names(m: Mapping[str, Any]) -> list[str]:
    names = _map(m["names"])
    lines = [f"G7.2's TopK SAE after block 0, recorded, not judged (C11): FVU {_three(m['fvu'])} on the held-out pairs,"
             f" the twin's {_three(m['twin_fvu'])}. Phase 6's name-this-direction at `{_cell(m.get('class', 'answer'))}`,"
             f" each value steered with {_num(m.get('instances'), ',')} instances.", "",
             "| Description | Model | Values | Named | Named as planted | Follows | Null follows |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in [d for d in DESCRIPTIONS if d in names] + sorted(d for d in names if d not in DESCRIPTIONS):
        entry = _map(names[d])
        for who in WHO:
            e = _map(entry.get(who))
            s = _map(e.get("steering"))
            lines.append(f"| {f'`{_cell(d)}`' if who == 'trained' else ''} | {who} | {_num(e.get('values'), ',')}"
                         f" | {_pct(e.get('named'))} | {_pct(e.get('named_as_planted'))} | {_pct(s.get('follows'))}"
                         f" | {_pct(s.get('null_follows'))} |")
    if not names:
        lines.append("| none recorded | | | | | | |")
    lines += ["", f"Name records written or found: {_num(m['records'], ',')}, one per distinct best latent."]
    return lines


# ---------------------------------------------------------------------------
# G7.3
# ---------------------------------------------------------------------------


def _determinism(m: Mapping[str, Any]) -> list[str]:
    by_run = _map(m["identities_by_run"])
    lines = [f"`set`'s identities, on {_num(m.get('identity_windows'), ',')} held-out windows of each zoo model, at every"
             f" residual-shaped tap but `hook_z`: {_num(m['identities_exact'], ',')} of {_num(m['identities_checked'], ',')}"
             " bit for bit.", "", "| Run | Taps exact | Taps checked | Not exact |", "| --- | --- | --- | --- |"]
    for run in ["induction", "two_hop", "modular_addition"] + sorted(r for r in by_run
                                                                     if r not in ("induction", "two_hop", "modular_addition")):
        if run not in by_run:
            continue
        taps = _map(by_run[run])
        wrong = [tap for tap, exact in sorted(taps.items()) if exact is not True]
        lines.append(f"| {RUN_NAMES.get(run, _cell(run))} | {sum(1 for v in taps.values() if v is True)} | {len(taps)}"
                     f" | {', '.join(f'`{_cell(t)}`' for t in wrong) or 'none'} |")
    if not by_run:
        lines.append("| none recorded | | | |")
    lines += ["", f"Request hashes: {_num(m['request_hashes_checked'], ',')} checked, all distinct:"
              f" {_yes(m['request_hashes_distinct'])}, stable through their JSON form and with their arguments reversed:"
              f" {_yes(m['request_hashes_stable'])}.", "",
              "Each form's record, built twice from one request, and once more by the command line:", "",
              "| Form | Built twice, identical | The command line's, identical |", "| --- | --- | --- |"]
    by_form = _map(m["records_by_form"])
    for form in [f for f in FORMS if f in by_form] + sorted(f for f in by_form if f not in FORMS):
        pair = _list(by_form[form]) + [None, None]
        lines.append(f"| {_cell(form)} | {_yes(pair[0])} | {_yes(pair[1])} |")
    if not by_form:
        lines.append("| no forms recorded | | |")
    return lines


# ---------------------------------------------------------------------------
# G7.4
# ---------------------------------------------------------------------------


def _crystallizations(m: Mapping[str, Any]) -> list[str]:
    found, bars = _map(m["crystallizations"]), _map(m.get("bars"))
    lines = ["Each crystallization at its run's final checkpoint, read on every held-out window, trained beside its"
             " random-weights twin. The KL from the model's prediction is the meter, beside a mean ablation's; the loss"
             " delta is recorded, never judged. The envelope is where it holds, at every checkpoint with the form fitted"
             " again; the gap is where the model's accuracy is at least 0.9 but it doesn't hold."]
    for name in _crystal_order(found):
        c = _map(found[name])
        meter, twin = _map(c.get("meter")), _map(c.get("twin"))
        twin_meter = _map(twin.get("meter"))
        judged = c.get("judged")
        lines += ["", f"**{_cell(name)}**, {'judged' if judged else 'recorded'}, record"
                  f" `{_cell(str(c.get('key', ''))[:12])}`: **{_cell(meter.get('label', 'not measured'))}**. KL"
                  f" {_nats(meter.get('kl'))} against the ablation's {_nats(meter.get('ablation'))}, removing"
                  f" {_pct(meter.get('removed'))} of it; agreement {_pct(meter.get('agreement'))}; loss delta"
                  f" {_num(meter.get('loss_delta'), '+.4f')} nats.", ""]
        lines += ["| Wrong function | KL | Agreement | Twin's KL | Twin's agreement |", "| --- | --- | --- | --- | --- |"]
        twin_wrong = {_map(w).get("key"): _map(w) for w in _list(twin.get("wrong"))}
        wrong = [_map(w) for w in _list(c.get("wrong"))]
        for w in wrong:
            t = twin_wrong.get(w.get("key"), {})
            lines.append(f"| `{_cell(w.get('key'))}` | {_nats(w.get('kl'))} | {_pct(w.get('agreement'))}"
                         f" | {_nats(t.get('kl'))} | {_pct(t.get('agreement'))} |")
        if not wrong:
            lines.append("| none declared | | | | |")
        cf = c.get("counterfactual")
        if isinstance(cf, Mapping):
            reads, answer = _map(cf.get("reads")), _map(cf.get("answer"))
            lines += ["", f"The counterfactual, {_cell(cf.get('name', ''))}, over {_num(cf.get('rows'), ',')} rows: keyed by"
                      f" what the component reads, KL {_nats(reads.get('kl'))} and agreement {_pct(reads.get('agreement'))};"
                      f" keyed by the answer, KL {_nats(answer.get('kl'))} and agreement {_pct(answer.get('agreement'))}."]
        else:
            lines += ["", "No counterfactual is declared."]
        lines += ["", f"The twin: {_cell(twin_meter.get('label', 'not measured'))}, KL {_nats(twin_meter.get('kl'))} against"
                  f" the ablation's {_nats(twin_meter.get('ablation'))}."]
        envelope = _map(c.get("envelope"))
        runs = ", ".join(_span(run) for run in _list(envelope.get("runs"))) or "no checkpoint"
        gap = ", ".join(_num(step, ",") for step in _list(envelope.get("gap"))) or "none"
        before = ", ".join(_num(step, ",") for step in _list(envelope.get("before"))) or "none"
        lines += ["", f"The envelope holds {runs}; its run to the final checkpoint {_span(envelope.get('final'))}. The"
                  f" model's accuracy reaches 0.9 {_reached(envelope.get('accurate_from'))}. The gap: {gap}. Holding"
                  f" before accuracy reaches 0.9: {before}."]
        lines += ["", *_cites(c.get("cites"))]
    if not found:
        lines += ["", "No crystallizations recorded."]
    lines += ["", "Over the judged crystallizations:", "",
              "- " + _bound("KL, most", m["kl_max"], bars, "kl", at_least=False, shown=_nats),
              "- " + _bound("The ablation's KL, least", m["ablation_min"], bars, "ablation", at_least=True, shown=_nats),
              "- " + _bound("Each wrong function's KL, least", m["wrong_min"], bars, "wrong", at_least=True, shown=_nats),
              "- " + _bound("The counterfactual keyed by what it reads, agreement, least", m["reads_agreement_min"], bars,
                            "reads_agreement", at_least=True),
              "- " + _bound("The counterfactual keyed by the answer, agreement, most", m["answer_agreement_max"], bars,
                            "answer_agreement", at_least=False),
              "- " + _bound("On the twins, every wrong function's KL from the right one's, most", m["twin_within_max"],
                            bars, "twin_within", at_least=False, shown=_nats),
              f"- Each holds at the final checkpoint, with step 0 outside: {_yes(m['envelope_ok'])}."]
    return lines


def _cites(cites: Any) -> list[str]:
    if not isinstance(cites, Mapping):
        return ["It cites nothing."]
    names, edges = [_map(n) for n in _list(cites.get("names"))], [_map(e) for e in _list(cites.get("edges"))]
    lines = ["| Key | Description | Name records cited | Values | Named as planted | Note |",
             "| --- | --- | --- | --- | --- | --- |"]
    for n in names:
        lines.append(f"| `{_cell(n.get('key'))}` | `{_cell(n.get('description'))}` | {len(_list(n.get('records')))}"
                     f" | {_num(n.get('values'), ',')} | {_pct(n.get('named'))} | {_cell(n.get('note') or 'none')} |")
    if not names:
        lines.append("| no names cited | | | | | |")
    if edges:
        lines += ["", "Edges cited: " + "; ".join(f"`{_cell(e.get('relation'))}`, {_cell(e.get('label'))}, record"
                                                   f" `{_cell(str(e.get('key', ''))[:12])}`" for e in edges) + "."]
    return lines


def _crystal_order(found: Mapping[str, Any]) -> list[str]:
    return [name for name in CRYSTALLIZATIONS if name in found] + sorted(n for n in found if n not in CRYSTALLIZATIONS)


# ---------------------------------------------------------------------------
# G7.5
# ---------------------------------------------------------------------------


def _records(m: Mapping[str, Any]) -> list[str]:
    builds = _map(m.get("builds"))
    lines = [f"Modular addition's views and every Phase 7 record built twice, the gestures panel written into each"
             f" records folder, and the badges file built twice, byte-identical: {_yes(m['builds_identical'])}. External"
             f" URLs in the viewer's files: {_num(m['external_urls'], ',')}. Subjects without a badge:"
             f" {_num(m['subjects_missing'], ',')}.", ""]
    if builds:
        lines += ["| Built | Identical | Files compared | Files differing |", "| --- | --- | --- | --- |"]
        parts = [p for p in ("badges", "views") if p in builds] + [r for r in _runs(builds) if r not in ("badges", "views")]
        for part in parts:
            b = _map(builds[part])
            differing = _list(b.get("files_differing"))
            shown = {"badges": "Badges", "views": "Modular addition's views"}.get(part)
            shown = shown or f"{RUN_NAMES.get(part, _cell(part))}'s records"
            lines.append(f"| {shown} | {_yes(b.get('identical'))} | {_num(b.get('files_compared'), ',')}"
                         f" | {', '.join(_cell(f) for f in differing) or 'none'} |")
        lines.append("")
    records, requests = _map(m["records"]), _map(m.get("requests"))
    lines += ["Each run's Phase 7 records by kind, as its first folder holds them, with the requests made before equal ones"
              " were merged.", "", "| Run | Name | Crystallize | Requests |", "| --- | --- | --- | --- |"]
    for run in _runs(records):
        held, asked = _map(records[run]), _map(requests.get(run))
        total = sum(v for v in asked.values() if _is_number(v)) if asked else None
        lines.append(f"| {RUN_NAMES.get(run, _cell(run))} | " + " | ".join(_num(held.get(k), ",") for k in KINDS)
                     + f" | {_num(total, ',')} |")
    if not records:
        lines.append("| none recorded | | | |")
    lines += ["", *_matches(m), "", *_clock(m.get("clock")), "", *_badges(m["badges"])]
    return lines


def _matches(m: Mapping[str, Any]) -> list[str]:
    matches = _map(m.get("matches"))
    lines = [f"The records the panel shows are the ones G7.4 measured and G7.2 recorded: {_yes(m['reports_match'])}.", "",
             "| Crystallization | G7.4's record | Built here | Equal |", "| --- | --- | --- | --- |"]
    crystals = [_map(e) for e in _list(matches.get("crystallizations"))]
    order = {name: i for i, name in enumerate(CRYSTALLIZATIONS)}
    for entry in sorted(crystals, key=lambda e: (order.get(e.get("name"), len(order)), str(e.get("name")))):
        lines.append(f"| {_cell(entry.get('name', ''))} | `{_cell(str(entry.get('key', ''))[:12])}`"
                     f" | `{_cell(str(entry.get('built', ''))[:12])}` | {_yes(entry.get('equal'))} |")
    if not crystals:
        lines.append("| none recorded | | | |")
    names = [_map(e) for e in _list(matches.get("names"))]
    missing = [e for e in names if e.get("in_folder") is not True]
    lines += ["", f"G7.2's name records: {len(names) - len(missing)} of {len(names)} in modular addition's first folder"
              + (". Missing: " + ", ".join(f"latent {_cell(e.get('latent'))} `{_cell(str(e.get('key', ''))[:12])}`"
                                          for e in missing) + "." if missing else ".")]
    return lines


def _clock(found: Any) -> list[str]:
    if not isinstance(found, Mapping):
        return ["Modular addition's fourier crystallization: not recorded."]
    frequencies, envelope = _map(found.get("frequencies")), _map(found.get("envelope"))
    k, energy, coefficients = (_list(frequencies.get(f)) for f in ("k", "embedding", "coefficients"))
    lines = [f"The clock: {_cell(found.get('name', 'modular addition write sum fourier'))}, record"
             f" `{_cell(str(found.get('key', ''))[:12])}`, from modular addition's first folder.", "",
             "| Frequency | Share of the embedding's energy | Share of the fitted coefficients |", "| --- | --- | --- |"]
    for i, f in enumerate(k):
        lines.append(f"| {_num(f, ',')} | {_pct(energy[i] if i < len(energy) else None)}"
                     f" | {_pct(coefficients[i] if i < len(coefficients) else None)} |")
    if not k:
        lines.append("| none recorded | | |")
    runs = ", ".join(_span(run) for run in _list(envelope.get("runs"))) or "no checkpoint"
    gap = ", ".join(_num(step, ",") for step in _list(envelope.get("gap"))) or "none"
    before = ", ".join(_num(step, ",") for step in _list(envelope.get("before"))) or "none"
    lines += ["", f"Its envelope holds {runs}; its run to the final checkpoint {_span(envelope.get('final'))}. Held-out"
              f" accuracy reaches 0.9 {_reached(envelope.get('accurate_from'))}; the checkpoints where it holds before"
              f" that: {before}. The gap: {gap}."]
    names = [_map(n) for n in _list(_map(found.get("cites")).get("names"))]
    for n in names:
        lines += ["", f"It cites {len(_list(n.get('records')))} name records for `{_cell(n.get('description'))}`, its key"
                  f" `{_cell(n.get('key'))}`, naming it as planted for {_pct(n.get('named'))} of"
                  f" {_num(n.get('values'), ',')} values. {_cell(n.get('note') or '')}".rstrip()]
    return lines


def _badges(badges: Any) -> list[str]:
    badges = _map(badges)
    lines = [
        "Each badge, as G7.5 generated it from G4.1 to G4.4's, G5.1 to G5.4's, G6.1 to G6.5's and G7.2 to G7.4's reports:"
        " Phase 4's model-level badges, then each shipped dictionary kind's and `dpatch`'s, then each gesture's and"
        " `dsteer`'s, then crystallize-this's and `set`'s. Every value is the cited report's own; a rank is 0-based, 0"
        " being the top token.",
        "",
        "| Subject | Status | Fact | Value | Gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    notes = []
    for subject in list(SUBJECTS) + sorted(s for s in badges if s not in SUBJECTS):
        badge = badges.get(subject)
        if not isinstance(badge, Mapping):
            lines.append(f"| {_cell(subject)} | no badge | | | |")
            continue
        status = _cell(badge.get("status", "not stated"))
        facts = [fact for fact in badge.get("facts") or [] if isinstance(fact, Mapping)]
        if not facts:
            lines.append(f"| {_cell(subject)} | {status} | none | | |")
        for i, fact in enumerate(facts):
            shown = (f"| {_cell(subject)} | {status} |" if i == 0 else "| | |")
            lines.append(f"{shown} {_cell(fact.get('label', ''))} | {_by_unit(fact.get('value'), fact.get('unit'))}"
                         f" | {_cell(fact.get('gate', ''))} |")
        notes += [f"- {_cell(subject)}: {_cell(note)}" for note in badge.get("notes") or []]
    if notes:
        lines += ["", "Notes on the badges:", "", *notes]
    return lines


def _by_unit(value: Any, unit: Any) -> str:
    if value is None:
        return "none"
    if unit == "fraction":
        return _pct(value)
    if unit in ("fraction removed", "score", "ratio"):
        return _three(value)
    if unit == "nats":
        return _nats(value)
    if unit == "rank":
        return _num(value, "g")
    if unit in ("step", "tokens", "edges", "count"):
        return _num(value, ",")
    if unit == "logit":
        return _num(value, "+.3f")
    if unit == "parts":
        return _parts(value)
    if unit == "span":
        return _span(value)
    if unit == "flag":
        return _yes(value)
    if unit == "label":
        return _cell(value)
    return _num(value, "g") if _is_number(value) else _cell(value)


# ---------------------------------------------------------------------------


def _runs(mapping: Any) -> list[str]:
    """Run names as modular addition, induction and two-hop; anything else after, sorted."""
    return sorted((str(k) for k in mapping), key=lambda k: (_RUNS.index(k) if k in _RUNS else len(_RUNS), k))


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _at_least(value: Any, bound: Any) -> str:
    return "not measured" if not (_is_number(value) and _is_number(bound)) else _yes(value >= bound)


def _at_most(value: Any, bound: Any) -> str:
    return "not measured" if not (_is_number(value) and _is_number(bound)) else _yes(value <= bound)


def _three(value: Any) -> str:
    return "not measured" if not _is_number(value) else f"{value:.3f}"


def _nats(value: Any) -> str:
    return "not measured" if not _is_number(value) else f"{value:.3f} nats"


def _reached(step: Any) -> str:
    return "at no checkpoint" if step is None else f"from step {_num(step, ',')}"


def _parts(value: Any) -> str:
    """A number of upstream parts: an integer, or to one decimal where a median falls between two."""
    if not _is_number(value):
        return "not measured" if value is None else _cell(value)
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _span(value: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"steps {_num(value[0], ',')} to {_num(value[1], ',')}"
    return "none" if value is None else _cell(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _num(value: Any, style: str) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return _yes(value)
    if not _is_number(value):
        return _cell(value)
    if style == ",":
        return f"{value:,.0f}" if isinstance(value, float) else f"{value:,}"
    return format(value, style)


def _pct(value: Any, digits: int = 1) -> str:
    return "not measured" if not _is_number(value) else f"{100 * value:.{digits}f}%"


def _yes(value: Any) -> str:
    return "not measured" if value is None else ("yes" if value else "no")


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
