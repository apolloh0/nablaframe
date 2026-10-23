"""The Phase 6 summary: reports/phase6/summary.md, generated from the seven Phase 6 reports.

Same rules as gates/phase5/summarize.py: never written by hand, a missing or invalid report says
so rather than being skipped, and every number comes from a report. Its H6 evidence:

* G6.1: the identities exact of those checked, by site; the request hashes checked, and whether
  they were distinct and stable; and for each kind, whether its record built twice, and through the
  command line, was identical
* G6.2: each place, judged or recorded, trained beside twin: its values, the shares named and named
  as planted, steering's follows beside the null's with unchanged and ablated, the largest-code
  latents' shares, the set's effect (the shares of values ranked first and last, and the mean rank
  from the top), and coverage at the name's own description
* G6.3: each edge, trained beside twin: follows and kept beside their nulls, the output, through
  the residual alone, the label, and the single latents; the judged edges' bounds beside the bars
* G6.4: each target, trained beside twin: the reading beside its bar, the share within 128 parts,
  random order's share, and the planted set's median rank and share in the top 8
* G6.5: each judged name: the slices' F1 and follows ranges, each seed's F1, follows and cosine,
  the envelope, where the probe reads from, the start offset and step 0; then a table of every
  checkpoint's F1, follows, the null's follows, the probe, the patch, the accuracy and whether the
  name holds
* G6.6: each block of the reference run: the latents named, by description, with their coverage;
  the values named and steered, and the pooled steering; open-this-up's share within 128; and the
  seconds
* G6.7: the badges, each with its status and facts, as Phase 5's summary lists them; whether the
  builds matched; each run's records by kind; whether the reports matched; and the two-hop
  intermediate: its generalize record's envelope, and each value's name record with its label, its
  name and the rank of the named value from the bottom of its set's effect

The verdict covers G6.1 to G6.5 and G6.7. G6.6 always passes, so its numbers are listed without a
verdict of their own.
"""

# summarize() raises ValueError, naming every problem, when a report is unreadable, invalid
# (validate_report, which includes the status rules), not a Phase 6 report, or when the inputs
# are not exactly one report per gate. main() prints the problems, writes nothing and returns 1.
#
# A pure function of the report files: no timestamps of its own, rows in gate order. Reports
# written from a dirty tree, or from different commits or lockfiles, are named as such. It never
# states a result that is not in a report. apolloh0 rules H6.
#
# Reports are canonical JSON, whose keys come back sorted, so the summary puts them back in the
# spec's order: runs as two-hop then induction, places by block, checkpoints by step, blocks by
# number. A rate, a share or a fraction is shown as a percentage; an F1, a ratio or a cosine to three
# decimals; a reading in parts, a rank, a step, a count or a time as a number. Badges follow
# views/badges.py's SUBJECTS, then views/dictionary_badges.py's, then views/gesture_badges.py's, each
# fact in the badge's own order, its value shown by its unit.
#
# Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
# gates/phase6/summarize.py``.

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nablaframe.reports import PHASE6_GATE_IDS, validate_report
from nablaframe.views import badges as phase4_badges
from nablaframe.views import dictionary_badges, gesture_badges

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "phase6"
SUMMARY_PATH = REPORTS_DIR / "summary.md"

GATE_NAMES = {
    "G6.1": "Determinism and the operation",
    "G6.2": "Names",
    "G6.3": "Edges",
    "G6.4": "Open-this-up",
    "G6.5": "Generalize-this",
    "G6.6": "Reference run",
    "G6.7": "Records, panel and badges",
}
# The spec's "On failure" column, docs/specs/phase-6.md, Gates.
ON_FAILURE = {
    "G6.1": "diagnose the operation or the record",
    "G6.2": "diagnose the gesture",
    "G6.3": "diagnose the gesture",
    "G6.4": "diagnose the gesture",
    "G6.5": "diagnose the gesture",
    "G6.6": "not applicable",
    "G6.7": "diagnose the build",
}
RECORDED_GATES = ("G6.6",)
VERDICT_GATES = tuple(gate_id for gate_id in PHASE6_GATE_IDS if gate_id not in RECORDED_GATES)
STATUS_TEXT = {"pass": "pass", "conditional_pass": "conditional pass", "fail": "fail"}
SUBJECTS = tuple(phase4_badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS) + tuple(gesture_badges.SUBJECTS)
RUN_NAMES = {"two_hop": "Two-hop", "induction": "Induction", "reference": "Reference"}
KINDS = ("name", "edge", "open", "generalize")
ORDERS = ("absolute", "signed", "random")
WHO = ("trained", "twin")
# The order runs come back in: two-hop, then induction, then the reference run.
_RUNS = ("two_hop", "induction", "reference")
_AFTER_BLOCK = re.compile(r" after block (\d+)")
_BLOCK = re.compile(r"blocks\.(\d+)\.")


def summarize(report_paths: list[Path]) -> str:
    reports = _load(report_paths)
    m = {gate_id: report["measurements"] for gate_id, report in reports.items()}
    status = {gate_id: report["status"] for gate_id, report in reports.items()}

    lines = [
        "# Phase 6 summary",
        "",
        "Generated by `gates/phase6/summarize.py` from the seven Phase 6 gate reports. Do not edit by hand;"
        " regenerate with `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/summarize.py`.",
        "",
        "## Gates",
        "",
        "| Gate | Status | Key measurements |",
        "| --- | --- | --- |",
    ]
    for gate_id in PHASE6_GATE_IDS:
        shown = "always passes" if gate_id in RECORDED_GATES else STATUS_TEXT[status[gate_id]]
        lines.append(f"| {gate_id} {GATE_NAMES[gate_id]} | {shown} | {_cell(_key(gate_id, m[gate_id]))} |")

    lines += [
        "",
        "## Provenance",
        "",
        "| Gate | Commit | Dirty tree | Env | Chip | Threads | XLA_FLAGS | Lockfile SHA-256 | Report time (UTC) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for gate_id in PHASE6_GATE_IDS:
        r = reports[gate_id]
        flags = f"`{_cell(r['xla_flags'])}`" if r["xla_flags"] else "none"
        lines.append(
            f"| {gate_id} | `{r['git_commit'][:12]}` | {_yes(r['git_dirty'])} | {r['env_name']} | {_cell(r['chip'])}"
            f" | {r['thread_count']} | {flags} | `{r['env_lock_sha256'][:12]}` | {r['timestamp_utc']} |"
        )
    lines += ["", *_consistency(reports)]

    lines += [
        "",
        "## H6 evidence",
        "",
        "H6 asks two questions. Does every gesture recover the zoo's planted structure, come up empty on the"
        " random-weights twin, and render with its meter? And does the two-hop intermediate's name show its set, the"
        " set's effect with `b` at the bottom, and its envelope across training, where the name appears at the"
        " transition the probe finds?",
        "",
        _verdict(status),
        "",
        "### Determinism and the operation (G6.1)",
        "",
        *_determinism(m["G6.1"]),
        "",
        "### Names (G6.2)",
        "",
        *_names(m["G6.2"]),
        "",
        "### Edges (G6.3)",
        "",
        *_edges(m["G6.3"]),
        "",
        "### Open-this-up (G6.4)",
        "",
        *_open(m["G6.4"]),
        "",
        "### Generalize-this (G6.5)",
        "",
        *_generalize(m["G6.5"]),
        "",
        "### Reference run (G6.6)",
        "",
        *_reference(m["G6.6"]),
        "",
        "### Records, panel and badges (G6.7)",
        "",
        *_records(m["G6.7"]),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    paths = sorted(args.reports_dir.glob("g6*.json"))
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
        if report["gate_id"] not in PHASE6_GATE_IDS:
            problems.append(f"{path}: {report['gate_id']} is not a Phase 6 gate")
            continue
        by_gate.setdefault(report["gate_id"], []).append((Path(path), report))
    for gate_id in PHASE6_GATE_IDS:
        found = by_gate.get(gate_id, [])
        if not found:
            problems.append(f"no report for {gate_id}")
        elif len(found) > 1:
            problems.append(f"{len(found)} reports for {gate_id}: {', '.join(str(p) for p, _ in found)}")
    if problems:
        raise ValueError("cannot summarize:\n" + "\n".join(f"  - {p}" for p in problems))
    return {gate_id: by_gate[gate_id][0][1] for gate_id in PHASE6_GATE_IDS}


def _verdict(status: dict[str, str]) -> str:
    not_passed = [gate_id for gate_id in VERDICT_GATES if status[gate_id] != "pass"]
    if not not_passed:
        return ("The gates support H6: G6.1, G6.2, G6.3, G6.4, G6.5 and G6.7 all pass. G6.6 records, not judges; its"
                " numbers are below. The two-hop intermediate's name records and its envelope are at the end, with each"
                " badge's facts. Your own session with the gestures panel is the rest of the evidence.")
    items = "\n".join(f"- {gate_id} {GATE_NAMES[gate_id]}: {STATUS_TEXT[status[gate_id]]}. The spec's next step:"
                       f" {ON_FAILURE[gate_id]}." for gate_id in not_passed)
    return f"The gates do **not** yet support H6. These gates did not pass:\n\n{items}"


def _key(gate_id: str, m: Mapping[str, Any]) -> str:
    if gate_id == "G6.1":
        return (f"{_num(m['identities_exact'], ',')} of {_num(m['identities_checked'], ',')} identities bit for bit;"
                f" {_num(m['request_hashes_checked'], ',')} request hashes, distinct: {_yes(m['request_hashes_distinct'])},"
                f" stable: {_yes(m['request_hashes_stable'])}; records rebuilt identically: {_yes(m['records_identical'])}")
    if gate_id == "G6.2":
        return (f"named as planted at least {_pct(m['named_as_planted_min'])}; follows at least {_pct(m['follows_min'])},"
                f" null at most {_pct(m['null_follows_max'])}; on the twins, named at most {_pct(m['twin_named_max'])},"
                f" follows at most {_pct(m['twin_follows_max'])}")
    if gate_id == "G6.3":
        return (f"follows at least {_pct(m['follows_min'])}, null at most {_pct(m['null_follows_max'])}; output at least"
                f" {_pct(m['output_follows_min'])}; kept at most {_pct(m['kept_max'])}, null at least"
                f" {_pct(m['null_kept_min'])}; through the residual alone at most {_pct(m['residual_alone_max'])};"
                f" computed: {_yes(m['labels_computed'])}; the twins' excess at most {_pct(m['twin_excess_max'])}")
    if gate_id == "G6.4":
        return (f"reading over its bar at most {_three(m['reading_ratio_max'])}; within 128 parts at least"
                f" {_pct(m['within_min'])}; planted in the top 8 at least {_pct(m['planted_top_min'])}; random order"
                f" within 128 at most {_pct(m['random_within_max'])}; the induction twin's planted set in the top 8 at"
                f" most {_pct(m['twin_planted_top_max'])}")
    if gate_id == "G6.5":
        return (f"slices' F1 at least {_three(m['slice_f1_min'])}, follows at least {_pct(m['slice_follows_min'])};"
                f" seeds' F1 at least {_three(m['seed_f1_min'])}, follows at least {_pct(m['seed_follows_min'])}; step 0"
                f" outside: {_yes(m['step0_outside'])}; one run to the end: {_yes(m['one_final_run'])}; the envelope"
                f" starts at most {_num(m['start_offset_max'], ',')} checkpoint(s) from the probe's first reading")
    if gate_id == "G6.6":
        blocks = _map(m["blocks"])
        return f"recorded, not judged; {len(blocks)} blocks at step {_num(m.get('step'), ',')}"
    records = _map(m["records"])
    counts = "; ".join(f"{RUN_NAMES.get(run, _cell(run))} {_num(sum(v for v in _map(records[run]).values() if _is_number(v)), ',')}"
                       for run in _runs(records))
    return (f"{_num(m['subjects_missing'], ',')} subjects without a badge; builds identical: {_yes(m['builds_identical'])};"
            f" {_num(m['external_urls'], ',')} external URLs; records shown are the ones judged:"
            f" {_yes(m['reports_match'])}; records: {counts or 'none recorded'}")


def _consistency(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    commits = sorted({r["git_commit"] for r in reports.values()})
    locks = sorted({r["env_lock_sha256"] for r in reports.values()})
    lines.append(f"All seven reports come from commit `{commits[0]}`." if len(commits) == 1
                 else f"The reports come from {len(commits)} different commits.")
    lines.append(f"All seven reports come from lockfile `{locks[0]}`." if len(locks) == 1
                 else f"The reports come from {len(locks)} different lockfiles.")
    dirty = [gate_id for gate_id in PHASE6_GATE_IDS if reports[gate_id]["git_dirty"]]
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
# G6.1
# ---------------------------------------------------------------------------


def _determinism(m: Mapping[str, Any]) -> list[str]:
    identities = [str(i) for i in _list(m.get("identities"))] or ["empty", "zero codes", "zero amounts"]
    by_site = _map(m["identities_by_site"])
    lines = [
        f"`dsteer`'s identities, on {_num(m.get('identity_windows'), ',')} held-out windows at every TopK site of both"
        f" zoo models: {_num(m['identities_exact'], ',')} of {_num(m['identities_checked'], ',')} bit for bit.",
        "",
        "| Site | " + " | ".join(_cell(i) for i in identities) + " |",
        "| --- |" + " --- |" * len(identities),
    ]
    for site in sorted(by_site, key=_site_order):
        flags = _list(by_site[site]) + [None] * len(identities)
        lines.append(f"| {_cell(site)} | " + " | ".join(_exact(flag) for flag in flags[:len(identities)]) + " |")
    if not by_site:
        lines.append("| no sites recorded |" + " |" * len(identities))
    lines += ["", f"Request hashes: {_num(m['request_hashes_checked'], ',')} checked, all distinct:"
              f" {_yes(m['request_hashes_distinct'])}, stable through their JSON form and with their arguments reversed:"
              f" {_yes(m['request_hashes_stable'])}.", "",
              f"Each kind's record, built twice from one request with {_num(m.get('record_instances'), ',')} instances,"
              " and once more by the command line:", "",
              "| Kind | Built twice, identical | The command line's, identical |", "| --- | --- | --- |"]
    by_kind = _map(m["records_by_kind"])
    for kind in [k for k in KINDS if k in by_kind] + sorted(k for k in by_kind if k not in KINDS):
        pair = _list(by_kind[kind]) + [None, None]
        lines.append(f"| {_cell(kind)} | {_yes(pair[0])} | {_yes(pair[1])} |")
    if not by_kind:
        lines.append("| no kinds recorded | | |")
    return lines


def _exact(value: Any) -> str:
    return "not measured" if value is None else ("exact" if value else "**not exact**")


# ---------------------------------------------------------------------------
# G6.2
# ---------------------------------------------------------------------------


def _names(m: Mapping[str, Any]) -> list[str]:
    names, bars = _map(m["names"]), _map(m.get("bars"))
    lines = [f"Every planted name, each value pointed at its best latent by F1, steered with"
             f" {_num(m.get('instances'), ',')} instances per value, on the trained model and its random-weights twin."
             " Steering's shares pool every value's instances; the effect counts the values whose set ranks them first"
             " and last of the vocabulary; coverage is the set's share of activation mass where the name's own"
             " description holds its value, a mean over values.", "",
             "| Place | Judged | Model | Values | Named | Named as planted | Follows | Null follows | Unchanged | Null unchanged"
             " | Ablated changes | Largest code: named as planted | Largest code: in its set | Effect: first | Effect: last"
             " | Mean rank from the top | Coverage |",
             "| --- |" + " --- |" * 16]
    for place in sorted(names, key=_place_order):
        entry = _map(names[place])
        for who in WHO:
            e = _map(entry.get(who))
            s, largest, effect = _map(e.get("steering")), _map(e.get("largest_code")), _map(e.get("effect"))
            coverage = _map(e.get("coverage")).get(e.get("description"))
            vocab = effect.get("vocabulary")
            mean_rank = _num(effect.get("mean_rank_from_top"), ",.1f") + (f" of {_num(vocab, ',')}" if _is_number(vocab)
                                                                          else "")
            lines.append(f"| {_cell(place) if who == 'trained' else ''} | {_yes(entry.get('judged')) if who == 'trained' else ''}"
                         f" | {who} | {_num(e.get('values'), ',')} | {_pct(e.get('named'))} | {_pct(e.get('named_as_planted'))}"
                         f" | {_pct(s.get('follows'))} | {_pct(s.get('null_follows'))} | {_pct(s.get('unchanged'))}"
                         f" | {_pct(s.get('null_unchanged'))} | {_pct(s.get('ablated_changed'))}"
                         f" | {_pct(largest.get('named_as_planted'))} | {_pct(largest.get('in_set'))}"
                         f" | {_pct(effect.get('first'))} | {_pct(effect.get('last'))} | {mean_rank} | {_pct(coverage)} |")
    if not names:
        lines.append("| none recorded |" + " |" * 16)
    lines += ["", "Over the judged names:", "",
              "- " + _bound("Named as planted, least", m["named_as_planted_min"], bars, "named_as_planted", at_least=True),
              "- " + _bound("Follows, least", m["follows_min"], bars, "follows", at_least=True),
              "- " + _bound("The null follows, most", m["null_follows_max"], bars, "null_follows", at_least=False),
              "- " + _bound("On the twins, values named at all, most", m["twin_named_max"], bars, "twin_named",
                            at_least=False),
              "- " + _bound("On the twins, follows, most", m["twin_follows_max"], bars, "twin_follows", at_least=False)]
    return lines


# ---------------------------------------------------------------------------
# G6.3
# ---------------------------------------------------------------------------


def _edges(m: Mapping[str, Any]) -> list[str]:
    edges, bars = _map(m["edges"]), _map(m.get("bars"))
    lines = [f"Every relation the runs declare, {_num(m.get('instances'), ',')} instances each, on the trained model and"
             " its twin. Kept is the median share of the downstream set's mass left with the upstream set removed. An"
             " edge at one position is labelled carried where, with every block between frozen, the swap still follows"
             " at least half as often.", "",
             "| Edge | Judged | Model | Follows | Null follows | Kept | Null kept | Output follows | Output null"
             " | Residual alone follows | Label | One latent: follows | One latent: kept |",
             "| --- |" + " --- |" * 12]
    for place in sorted(edges, key=lambda key: _relation_order(key, edges[key])):
        entry = _map(edges[place])
        for who in WHO:
            e = _map(entry.get(who))
            output, alone, single = _map(e.get("output")), e.get("residual_alone"), _map(e.get("single"))
            alone_shown = "not applicable: across positions" if alone is None else _pct(_map(alone).get("follows"))
            lines.append(f"| {_cell(place) if who == 'trained' else ''} | {_yes(entry.get('judged')) if who == 'trained' else ''}"
                         f" | {who} | {_pct(e.get('follows'))} | {_pct(e.get('null_follows'))} | {_pct(e.get('kept'))}"
                         f" | {_pct(e.get('null_kept'))} | {_pct(output.get('follows')) if output else 'no prediction'}"
                         f" | {_pct(output.get('null_follows')) if output else ''} | {alone_shown}"
                         f" | {_cell(e.get('label', 'not measured'))} | {_pct(single.get('follows'))} | {_pct(single.get('kept'))} |")
    if not edges:
        lines.append("| none recorded |" + " |" * 12)
    lines += ["", "Over the judged edges:", "",
              "- " + _bound("Follows, least", m["follows_min"], bars, "edge_follows", at_least=True),
              "- " + _bound("The null follows, most", m["null_follows_max"], bars, "edge_null_follows", at_least=False),
              "- " + _bound("The output follows, least", m["output_follows_min"], bars, "output_follows", at_least=True),
              "- " + _bound("Kept, most", m["kept_max"], bars, "kept", at_least=False),
              "- " + _bound("The null kept, least", m["null_kept_min"], bars, "null_kept", at_least=True),
              "- " + _bound("Through the residual alone, follows, most", m["residual_alone_max"], bars, "residual_alone",
                            at_least=False),
              f"- Every judged edge labelled computed: {_yes(m['labels_computed'])}.",
              "- " + _bound("On the twins, follows less the null's, most", m["twin_excess_max"], bars, "twin_excess",
                            at_least=False)]
    return lines


# ---------------------------------------------------------------------------
# G6.4
# ---------------------------------------------------------------------------


def _open(m: Mapping[str, Any]) -> list[str]:
    targets, bars = _map(m["targets"]), _map(m.get("bars"))
    lines = [f"Every judged target, {_num(m.get('instances'), ',')} instances each, opened into the block its planted"
             " set is read after. The reading is the median number of upstream parts, in absolute order, that bring the"
             " target's pre-activation within 0.1 of its full value; random order keeps active parts in a random order."
             " The planted rank is where the planted set's first part falls in absolute order, from 0.", "",
             "| Target | Opened into | Model | Measured | Reading | Bar | Within 128 parts | Signed: within 128"
             " | Random: within 128 | Planted: median rank | Planted: in the top 8 |",
             "| --- |" + " --- |" * 10]
    for place in sorted(targets, key=lambda key: _target_order(key, targets[key])):
        entry = _map(targets[place])
        target, planted = _map(entry.get("target")), _map(entry.get("planted"))
        for who in WHO:
            e = _map(entry.get(who))
            orders = _map(e.get("orders"))
            absolute = _map(orders.get("absolute"))
            p = e.get("planted")
            reading = absolute.get("reading_median")
            first = who == "trained"
            label = (f"{_cell(place)}: `{_cell(target.get('description'))}` after block"
                     f" {_num(target.get('block'), ',')}" if first else "")
            into = f"block {_num(planted.get('block'), ',')}" if first else ""
            twin_note = " (judged)" if who == "twin" and entry.get("twin_judged") else ""
            lines.append(f"| {label} | {into} | {who}{twin_note} | {_num(e.get('measured'), ',')} of {_num(e.get('instances'), ',')}"
                         f" | {'not within 128' if reading is None and e.get('measured') else _parts(reading)}"
                         f" | {_parts(entry.get('bar')) if first else ''} | {_pct(absolute.get('within'))}"
                         f" | {_pct(_map(orders.get('signed')).get('within'))} | {_pct(_map(orders.get('random')).get('within'))}"
                         f" | {_parts(_map(p).get('rank_median')) if p else 'none planted'} | {_pct(_map(p).get('top')) if p else ''} |")
    if not targets:
        lines.append("| none recorded |" + " |" * 10)
    lines += ["", "Over the targets:", "",
              "- " + _bound("The trained reading over its bar, most", m["reading_ratio_max"], bars, "reading_ratio",
                            at_least=False, shown=_three),
              "- " + _bound("Within 128 parts, least", m["within_min"], bars, "within", at_least=True),
              "- " + _bound("The planted set in the top 8, least", m["planted_top_min"], bars, "planted_top",
                            at_least=True),
              "- " + _bound("In random order, within 128 parts, most", m["random_within_max"], bars, "random_within",
                            at_least=False),
              "- " + _bound("On the twin, the planted set in the top 8, most", m["twin_planted_top_max"], bars,
                            "twin_planted_top", at_least=False)]
    return lines


# ---------------------------------------------------------------------------
# G6.5
# ---------------------------------------------------------------------------


def _generalize(m: Mapping[str, Any]) -> list[str]:
    names, bars = _map(m["names"]), _map(m.get("bars"))
    lines = [f"Each judged name across 8 held-out slices, dictionary seeds 1 and 2 (the name derived again by its"
             f" description and value) and every checkpoint, read through the final dictionaries, with"
             f" {_num(m.get('instances'), ',')} instances. The name holds where its F1 and follows are both at least 0.5;"
             " the envelope is where it holds."]
    if not names:
        lines += ["", "No names recorded."]
    for place in sorted(names, key=_place_order):
        entry = _map(names[place])
        r, checks = _map(entry.get("results")), _map(entry.get("checks"))
        two_hop = place.startswith("two_hop ")
        slices, envelope = _map(r.get("slices")), _map(r.get("envelope"))
        lines += ["", f"**{_cell(place)}**, {_num(r.get('values'), ',')} values, record `{_cell(str(entry.get('record', ''))[:12])}`.",
                  "", f"Held-out slices: F1 {_range(slices.get('f1'), _three)}, follows {_range(slices.get('follows'), _pct)}.",
                  "", "| Seed | F1 | Follows | Null follows | Decoder cosine with seed 0's set |", "| --- | --- | --- | --- | --- |"]
        seeds = [_map(s) for s in _list(r.get("seeds"))]
        for s in seeds:
            lines.append(f"| {_cell(s.get('seed'))} | {_three(s.get('f1'))} | {_pct(s.get('follows'))}"
                         f" | {_pct(s.get('null_follows'))} | {_three(s.get('cosine'))} |")
        if not seeds:
            lines.append("| none recorded | | | | |")
        runs = ", ".join(_span(run) for run in _list(envelope.get("runs"))) or "none"
        gap = ", ".join(_num(step, ",") for step in _list(envelope.get("gap"))) or "none"
        lines += ["", f"The envelope: {runs}; its run to the final checkpoint {_span(envelope.get('final'))}. The probe"
                  f" reads the variable at 0.9 or more from step {_num(envelope.get('probe_reads_from'), ',')}, a start"
                  f" offset of {_num(checks.get('start_offset'), ',')} checkpoint(s). Step 0 outside the envelope:"
                  f" {_yes(checks.get('step0_outside'))}; one run to the end: {_yes(checks.get('one_final_run'))}. The gap,"
                  f" where the probe reads the variable but the name doesn't hold: {gap}.", "",
                  f"| Step | F1 | Follows | Null follows | Probe | {'Patch at 25' if two_hop else 'Patch'} | Accuracy | Holds |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        patch = {_map(p).get("step"): _map(p).get("rate") for p in _list(entry.get("patch"))}
        rows = sorted((_map(row) for row in _list(r.get("checkpoints"))),
                      key=lambda row: row.get("step") if _is_number(row.get("step")) else -1)
        for row in rows:
            step = row.get("step")
            lines.append(f"| {_num(step, ',')} | {_three(row.get('f1'))} | {_pct(row.get('follows'))}"
                         f" | {_pct(row.get('null_follows'))} | {_pct(row.get('probe'))}"
                         f" | {_pct(patch.get(step)) if step in patch else 'not measured'} | {_pct(row.get('accuracy'))}"
                         f" | {_yes(row.get('holds'))} |")
        if not rows:
            lines.append("| none recorded | | | | | | | |")
        if two_hop:
            lines += ["", "The patch moves position 25 only; G4.3's query patch moves positions 24 and 25."]
    lines += ["", "Over both names:", "",
              "- " + _bound("Every slice's F1, least", m["slice_f1_min"], bars, "slice_f1", at_least=True, shown=_three),
              "- " + _bound("Every slice's follows, least", m["slice_follows_min"], bars, "slice_follows", at_least=True),
              "- " + _bound("Seeds 1 and 2, F1, least", m["seed_f1_min"], bars, "seed_f1", at_least=True, shown=_three),
              "- " + _bound("Seeds 1 and 2, follows, least", m["seed_follows_min"], bars, "seed_follows", at_least=True),
              f"- Step 0 outside both envelopes: {_yes(m['step0_outside'])}. Each envelope one run to the final"
              f" checkpoint: {_yes(m['one_final_run'])}.",
              "- " + _bound("The envelope's start, checkpoints from the probe's first reading, most", m["start_offset_max"],
                            bars, "start_offset", at_least=False, shown=lambda v: _num(v, ","))]
    return lines


# ---------------------------------------------------------------------------
# G6.6
# ---------------------------------------------------------------------------


def _reference(m: Mapping[str, Any]) -> list[str]:
    blocks = _map(m["blocks"])
    settings = _map(m.get("settings"))
    lines = [f"G1.4's run at step {_num(m.get('step'), ',')}, with the TopK SAE after every block, on"
             f" {_num(m.get('held_out_windows'), ',')} held-out windows: recorded, never judged (D44). Per block,"
             f" {_num(settings.get('latents'), ',')} latents sampled and named; up to {_num(settings.get('values'), ',')}"
             f" named `token(p+1)` values steered, {_num(settings.get('per_value'), ',')} instances each; their sets opened"
             f" into the block below, {_num(settings.get('open'), ',')} instances.", ""]
    descriptions: list[str] = []
    for block in blocks.values():
        for d in _map(_map(_map(block).get("names")).get("by_description")):
            if d not in descriptions:
                descriptions.append(d)
    lines += ["| Block | Active of width | Sampled | Named | " + " | ".join(f"As `{_cell(d)}`" for d in descriptions)
              + " | Coverage, median | Values named | Values steered | Follows | Null follows | Unchanged | Ablated changes"
              " | Opened: within 128 parts | Seconds |",
              "| --- |" + " --- |" * (13 + len(descriptions))]
    for key in sorted(blocks, key=_number_order):
        b = _map(blocks[key])
        names, steering, opened = _map(b.get("names")), _map(b.get("steering")), b.get("open")
        pooled, by = _map(steering.get("pooled")), _map(names.get("by_description"))
        within = "not opened" if opened is None else _pct(_map(_map(_map(opened).get("orders")).get("absolute")).get("within"))
        seconds = _map(m.get("seconds")).get(key)
        lines.append(f"| {_cell(key)} | {_num(names.get('active'), ',')} of {_num(names.get('width'), ',')}"
                     f" | {_num(names.get('sampled'), ',')} | {_pct(names.get('named'))} | "
                     + " | ".join(_pct(by.get(d)) for d in descriptions)
                     + f"{' | ' if descriptions else ''}{_three(names.get('coverage_median'))}"
                     f" | {_num(steering.get('values_named'), ',')} | {len(_list(steering.get('values')))}"
                     f" | {_pct(pooled.get('follows'))} | {_pct(pooled.get('null_follows'))} | {_pct(pooled.get('unchanged'))}"
                     f" | {_pct(pooled.get('ablated_changed'))} | {within} | {_num(seconds, ',.1f')} |")
    if not blocks:
        lines.append("| none recorded |" + " |" * (13 + len(descriptions)))
    lines += ["", f"Every gesture on every block took {_num(_map(m.get('seconds')).get('total'), ',.1f')} seconds in all."]
    return lines


# ---------------------------------------------------------------------------
# G6.7
# ---------------------------------------------------------------------------


def _records(m: Mapping[str, Any]) -> list[str]:
    builds = _map(m.get("builds"))
    lines = [f"Every record built twice, the gestures panel written into each folder, and the badges file built twice,"
             f" byte-identical: {_yes(m['builds_identical'])}. External URLs in the viewer's files:"
             f" {_num(m['external_urls'], ',')}. Subjects without a badge: {_num(m['subjects_missing'], ',')}.", ""]
    if builds:
        lines += ["| Built | Identical | Files compared | Files differing |", "| --- | --- | --- | --- |"]
        for part in ["badges"] + [name for name in _runs(builds) if name != "badges"]:
            if part in builds:
                b = _map(builds[part])
                differing = _list(b.get("files_differing"))
                lines.append(f"| {RUN_NAMES.get(part, _cell(part))} | {_yes(b.get('identical'))}"
                             f" | {_num(b.get('files_compared'), ',')} | {', '.join(_cell(f) for f in differing) or 'none'} |")
        lines.append("")
    records, requests = _map(m["records"]), _map(m.get("requests"))
    lines += ["Each run's records by kind, as its first folder holds them, with the requests made before equal ones were"
              " merged.", "", "| Run | Name | Edge | Open-this-up | Generalize-this | Requests |", "| --- | --- | --- | --- | --- | --- |"]
    for run in _runs(records):
        held, asked = _map(records[run]), _map(requests.get(run))
        total = sum(v for v in asked.values() if _is_number(v)) if asked else None
        lines.append(f"| {RUN_NAMES.get(run, _cell(run))} | " + " | ".join(_num(held.get(k), ",") for k in KINDS)
                     + f" | {_num(total, ',')} |")
    if not records:
        lines.append("| none recorded | | | | | |")
    lines += ["", *_matches(m), "", *_intermediate(m.get("intermediate")), "", *_badges(m["badges"])]
    return lines


def _matches(m: Mapping[str, Any]) -> list[str]:
    matches = _map(m.get("matches"))
    lines = [f"The records the panel shows are the ones the gates judged: {_yes(m['reports_match'])}.", "",
             "| Judged by | Place | Record | In its run's records |", "| --- | --- | --- | --- |"]
    for part, gate in (("edges", "G6.3"), ("generalize", "G6.5")):
        for entry in (_map(e) for e in _list(matches.get(part))):
            lines.append(f"| {gate} | {_cell(entry.get('place', ''))} | `{_cell(str(entry.get('key', ''))[:12])}`"
                         f" | {_yes(entry.get('in_folder'))} |")
    for entry in (_map(e) for e in _list(matches.get("reference"))):
        lines.append(f"| G6.6 | reference block {_cell(entry.get('block'))}: steered values"
                     f" {', '.join(_cell(v) for v in _list(entry.get('g66'))) or 'none'} | G6.7's:"
                     f" {', '.join(_cell(v) for v in _list(entry.get('g67'))) or 'none'} | {_yes(entry.get('equal'))} |")
    if not any(_list(matches.get(part)) for part in ("edges", "generalize", "reference")):
        lines.append("| none recorded | | | |")
    return lines


def _intermediate(found: Any) -> list[str]:
    if not isinstance(found, Mapping):
        return ["The two-hop intermediate: not recorded."]
    g = _map(found.get("generalize"))
    envelope = _map(g.get("envelope"))
    runs = ", ".join(_span(run) for run in _list(envelope.get("runs"))) or "no checkpoint"
    final = _span(envelope.get("final")) if envelope.get("final") is not None else "none"
    reads = envelope.get("probe_reads_from")
    probe = f"from step {_num(reads, ',')}" if reads is not None else "at no checkpoint"
    lines = [f"The two-hop intermediate, {_cell(found.get('place', ''))}. Its generalize-this record"
             f" `{_cell(str(g.get('key', ''))[:12])}`: the envelope holds {runs}, its run to the final checkpoint"
             f" {final}; the probe reads the variable {probe}. Each value's name record, pointed at its best latent, with"
             " its meter from the panel: the name's rank from the bottom of its set's effect is 0 where the set ranks it"
             " last.", "", "| Value | Latent | Record | Label | Name | F1 | Rank from the bottom | Follows | Null follows |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    values = [_map(v) for v in _list(found.get("values"))]
    for v in values:
        meter = _map(v.get("meter"))
        named = (f"`{_cell(meter.get('description'))}` = {_cell(meter.get('value'))}"
                 if meter.get("description") is not None else "no name")
        lines.append(f"| {_cell(v.get('value'))} | {_cell(v.get('latent'))} | `{_cell(str(v.get('key', ''))[:12])}`"
                     f" | {_cell(meter.get('label', 'none'))} | {named} | {_three(meter.get('f1'))}"
                     f" | {_num(meter.get('rank_from_bottom'), ',')} | {_pct(meter.get('follows'))}"
                     f" | {_pct(meter.get('null_follows'))} |")
    if not values:
        lines.append("| none recorded | | | | | | | | |")
    return lines


def _badges(badges: Any) -> list[str]:
    badges = _map(badges)
    lines = [
        "Each badge, as G6.7 generated it from G4.1 to G4.4's, G5.1 to G5.4's and G6.1 to G6.5's reports: Phase 4's"
        " model-level badges, then each shipped dictionary kind's and `dpatch`'s, then each gesture's and `dsteer`'s."
        " Every value is the cited report's own; a rank is 0-based, 0 being the top token.",
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
# Order: runs as two-hop then induction, places by block, checkpoints by step, blocks by number
# ---------------------------------------------------------------------------


def _run_index(key: str) -> int:
    run = str(key).split(" ", 1)[0]
    return _RUNS.index(run) if run in _RUNS else len(_RUNS)


def _runs(mapping: Any) -> list[str]:
    """Run names as two-hop, induction and reference; anything else after, sorted."""
    return sorted((str(k) for k in mapping), key=lambda k: (_RUNS.index(k) if k in _RUNS else len(_RUNS), k))


def _place_order(key: str) -> tuple:
    found = _AFTER_BLOCK.search(str(key))
    return (_run_index(key), int(found.group(1)) if found else -1, str(key))


def _site_order(key: str) -> tuple:
    found = _BLOCK.search(str(key))
    return (_run_index(key), int(found.group(1)) if found else -1, str(key))


def _blocks_of(*places: Any) -> tuple:
    return tuple(p.get("block") if _is_number(p.get("block")) else -1 for p in places)


def _relation_order(key: str, entry: Any) -> tuple:
    """An edge by its run, then its upstream block, then its downstream block."""
    trained = _map(_map(entry).get("trained"))
    return (_run_index(key), *_blocks_of(_map(trained.get("up")), _map(trained.get("down"))), str(key))


def _target_order(key: str, entry: Any) -> tuple:
    """An open-this-up target by its run, then the block it opens into, then its own block."""
    entry = _map(entry)
    return (_run_index(key), *_blocks_of(_map(entry.get("planted")), _map(entry.get("target"))), str(key))


def _number_order(key: str) -> tuple:
    return (0, int(key), "") if str(key).isdigit() else (1, 0, str(key))


# ---------------------------------------------------------------------------


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


def _parts(value: Any) -> str:
    """A number of upstream parts: an integer, or to one decimal where a median falls between two."""
    if not _is_number(value):
        return "not measured" if value is None else _cell(value)
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _span(value: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"steps {_num(value[0], ',')} to {_num(value[1], ',')}"
    return "none" if value is None else _cell(value)


def _range(value: Any, shown: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"{shown(value[0])} to {shown(value[1])}"
    return "not measured"


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
