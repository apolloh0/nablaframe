"""The Phase 5 summary: reports/phase5/summary.md, generated from the six Phase 5 reports.

Same rules as gates/phase4/summarize.py: never written by hand, a missing or invalid report says
so rather than being skipped, and every number comes from a report. Its H5 evidence:

* G5.1: whether each kind trained byte-identically, the cache keys checked and whether they were
  distinct, and the identities exact of those checked, by site
* G5.2 and G5.3, by model: each dictionary's meters beside the bars (its FVU and its twin's, and the
  splice's recovery, or the change it makes where an ablation costs too little to judge); the
  faithful code beside the residual patch, by kind and test, with the single-latent patch beside
  them; the selective latents, trained against twin, with the margin; and, recorded, readability
  and seed robustness. G5.3 names the best swap block it read from G4.3
* G5.4: each MLP's edge meter, trained against twin, with its share of the scored edges and, where
  judged, its ratio beside the bar; the block pairs; the Jacobian SAE at each ``λ`` beside the TopK
  pair, with its Jacobian statistics and both sides' FVU
* G5.5: the reference run's meters, its two MLP pairs' meters, and each dictionary's training time
* G5.6: the badges, each with its status and facts, as Phase 4's summary lists them; whether the
  builds matched; each run's features payloads; and ``query_latents``, the latents active at ``⇒``
  in the two-hop TopK at the best swap block, with the intermediate of their top contexts

The verdict covers G5.1 to G5.4 and G5.6. G5.5 always passes, so its numbers are listed without a
verdict of their own.
"""

# summarize() raises ValueError, naming every problem, when a report is unreadable, invalid
# (validate_report, which includes the status rules), not a Phase 5 report, or when the inputs
# are not exactly one report per gate. main() prints the problems, writes nothing and returns 1.
#
# A pure function of the report files: no timestamps of its own, rows in gate order. Reports
# written from a dirty tree, or from different commits or lockfiles, are named as such. It never
# states a result that is not in a report. apolloh0 rules H5.
#
# Reports are canonical JSON, whose keys come back sorted, so the summary puts them back in the
# spec's order: kinds as topk, staircase, transcoder and jsae; sites and blocks by number; variables
# as the gates list them. Anything else follows, sorted. A rate, an FVU or a share is shown as a
# percentage; an F1, a ratio or a margin to three decimals; a loss or a change in nats to three
# decimals; an edge count or a time as a number. Badges follow views/badges.py's SUBJECTS, then
# views/dictionary_badges.py's, each fact in the badge's own order, its value shown by its unit.
#
# Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
# gates/phase5/summarize.py``.

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nablaframe.reports import PHASE5_GATE_IDS, validate_report
from nablaframe.views import badges as phase4_badges
from nablaframe.views import dictionary_badges

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "phase5"
SUMMARY_PATH = REPORTS_DIR / "summary.md"

GATE_NAMES = {
    "G5.1": "Determinism and the operation",
    "G5.2": "Induction",
    "G5.3": "Two-hop",
    "G5.4": "Edge sparsity",
    "G5.5": "Reference dictionaries",
    "G5.6": "Features view and badges",
}
# The spec's "On failure" column, docs/specs/phase-5.md, Gates.
ON_FAILURE = {
    "G5.1": "diagnose the trainer or the operation",
    "G5.2": "diagnose the dictionary that failed",
    "G5.3": "diagnose the dictionary that failed",
    "G5.4": "diagnose the meter",
    "G5.5": "not applicable",
    "G5.6": "diagnose the build",
}
RECORDED_GATES = ("G5.5",)
VERDICT_GATES = tuple(gate_id for gate_id in PHASE5_GATE_IDS if gate_id not in RECORDED_GATES)
STATUS_TEXT = {"pass": "pass", "conditional_pass": "conditional pass", "fail": "fail"}
# The spec's bars, docs/specs/phase-5.md, "Discrimination tests on the zoo" and "The edge-sparsity meter".
BARS = {"fvu": 0.1, "ablation_cost": 0.1, "recovered": 0.9, "faithful_ratio": 0.9, "break_excess": 0.05,
        "control_unchanged": 0.95, "selective_f1": 0.5, "edge_ratio": 0.2}
SUBJECTS = tuple(phase4_badges.SUBJECTS) + tuple(dictionary_badges.SUBJECTS)
MODEL_NAMES = {"induction": "Induction", "two_hop": "Two-hop", "reference": "Reference"}
RATES = ("swap", "transplant", "control_unchanged")
RATE_NAMES = {"swap": "swap", "transplant": "transplant", "control_unchanged": "control unchanged"}
PASSES = ("residual", "code", "latents")
TESTS = {"follow": "following", "break": "breaking"}
# The order names come back in: who trained it, the run, the kind, then blocks and numbers, then variables.
_WHO = ("trained", "twin", "seed")
_RUNS = ("induction", "two_hop", "reference")
_KINDS = ("topk", "staircase", "transcoder", "jsae")
_VARIABLES = ("cur", "prev", "next", "a", "b", "c")
_BLOCK = re.compile(r"blocks\.(\d+)\.(.+)")
_NUMBER = re.compile(r"(\d+)(?:\.\d+)?(?:->\d+)?")


def summarize(report_paths: list[Path]) -> str:
    reports = _load(report_paths)
    m = {gate_id: report["measurements"] for gate_id, report in reports.items()}
    status = {gate_id: report["status"] for gate_id, report in reports.items()}

    lines = [
        "# Phase 5 summary",
        "",
        "Generated by `gates/phase5/summarize.py` from the six Phase 5 gate reports. Do not edit by hand;"
        " regenerate with `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase5/summarize.py`.",
        "",
        "## Gates",
        "",
        "| Gate | Status | Key measurements |",
        "| --- | --- | --- |",
    ]
    for gate_id in PHASE5_GATE_IDS:
        shown = "always passes" if gate_id in RECORDED_GATES else STATUS_TEXT[status[gate_id]]
        lines.append(f"| {gate_id} {GATE_NAMES[gate_id]} | {shown} | {_cell(_key(gate_id, m[gate_id]))} |")

    lines += [
        "",
        "## Provenance",
        "",
        "| Gate | Commit | Dirty tree | Env | Chip | Threads | XLA_FLAGS | Lockfile SHA-256 | Report time (UTC) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for gate_id in PHASE5_GATE_IDS:
        r = reports[gate_id]
        flags = f"`{_cell(r['xla_flags'])}`" if r["xla_flags"] else "none"
        lines.append(
            f"| {gate_id} | `{r['git_commit'][:12]}` | {_yes(r['git_dirty'])} | {r['env_name']} | {_cell(r['chip'])}"
            f" | {r['thread_count']} | {flags} | `{r['env_lock_sha256'][:12]}` | {r['timestamp_utc']} |"
        )
    lines += ["", *_consistency(reports)]

    lines += [
        "",
        "## H5 evidence",
        "",
        "H5 asks whether every shipped dictionary carries its meters and a badge of zoo results, generated from the"
        " reports, and whether the features view shows the two-hop intermediate's latents at `⇒`, where patching"
        " finds it.",
        "",
        _verdict(status),
        "",
        "### Determinism and the operation (G5.1)",
        "",
        *_determinism(m["G5.1"]),
        "",
        "### Induction (G5.2)",
        "",
        *_zoo(m["G5.2"], "induction"),
        "",
        "### Two-hop (G5.3)",
        "",
        *_zoo(m["G5.3"], "two_hop"),
        "",
        "### Edge sparsity (G5.4)",
        "",
        *_edges(m["G5.4"]),
        "",
        "### Reference dictionaries (G5.5)",
        "",
        *_reference(m["G5.5"]),
        "",
        "### Features view and badges (G5.6)",
        "",
        *_features(m["G5.6"]),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    paths = sorted(args.reports_dir.glob("g5*.json"))
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
        if report["gate_id"] not in PHASE5_GATE_IDS:
            problems.append(f"{path}: {report['gate_id']} is not a Phase 5 gate")
            continue
        by_gate.setdefault(report["gate_id"], []).append((Path(path), report))
    for gate_id in PHASE5_GATE_IDS:
        found = by_gate.get(gate_id, [])
        if not found:
            problems.append(f"no report for {gate_id}")
        elif len(found) > 1:
            problems.append(f"{len(found)} reports for {gate_id}: {', '.join(str(p) for p, _ in found)}")
    if problems:
        raise ValueError("cannot summarize:\n" + "\n".join(f"  - {p}" for p in problems))
    return {gate_id: by_gate[gate_id][0][1] for gate_id in PHASE5_GATE_IDS}


def _verdict(status: dict[str, str]) -> str:
    not_passed = [gate_id for gate_id in VERDICT_GATES if status[gate_id] != "pass"]
    if not not_passed:
        return ("The gates support H5: G5.1, G5.2, G5.3, G5.4 and G5.6 all pass. G5.5 records, not judges; its numbers"
                " are below, and each badge's facts are listed at the end, with the latents active at `⇒`. Your own"
                " session with the features view is the rest of the evidence.")
    items = "\n".join(f"- {gate_id} {GATE_NAMES[gate_id]}: {STATUS_TEXT[status[gate_id]]}. The spec's next step:"
                       f" {ON_FAILURE[gate_id]}." for gate_id in not_passed)
    return f"The gates do **not** yet support H5. These gates did not pass:\n\n{items}"


def _key(gate_id: str, m: Mapping[str, Any]) -> str:
    if gate_id == "G5.1":
        return (f"trained twice byte-identically: {_yes(m['files_identical'])}; {_num(m['cache_keys_checked'], ',')} cache"
                f" keys, distinct: {_yes(m['cache_keys_distinct'])}; {_num(m['identities_exact'], ',')} of"
                f" {_num(m['identities_checked'], ',')} identities bit for bit")
    if gate_id in ("G5.2", "G5.3"):
        third = (f"breaking excess at most {_pct(m['break_excess_max'])}" if gate_id == "G5.2"
                 else f"control unchanged at least {_pct(m['control_unchanged_min'])}")
        return (f"FVU at most {_pct(m['fvu_max'])}; recovered at least {_pct(m['recovered_min'])}; faithful ratio at"
                f" least {_three(m['faithful_ratio_min'])}; {third}; selective F1 at least"
                f" {_three(m['selective_f1_min'])}, margin at least {_three(m['selective_margin_min'])}")
    if gate_id == "G5.4":
        judged = m["judged_mlps"] if isinstance(m["judged_mlps"], list) else []
        return (f"{len(judged)} judged MLPs ({', '.join(_cell(key) for key in judged) or 'none'}); greatest ratio to"
                f" the twin {_three(m['edge_ratio_max'])}")
    if gate_id == "G5.5":
        meters = _map(m["meters"])
        return f"recorded, not judged; {len(meters)} sites' meters, {len(_map(m['edge_pairs']))} MLP pairs"
    return (f"{_num(m['subjects_missing'], ',')} subjects and {_num(m['dictionaries_missing'], ',')} shipped dictionaries"
            f" without a badge; builds identical: {_yes(m['builds_identical'])};"
            f" {_num(m['external_urls'], ',')} external URLs")


def _consistency(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    commits = sorted({r["git_commit"] for r in reports.values()})
    locks = sorted({r["env_lock_sha256"] for r in reports.values()})
    lines.append(f"All six reports come from commit `{commits[0]}`." if len(commits) == 1
                 else f"The reports come from {len(commits)} different commits.")
    lines.append(f"All six reports come from lockfile `{locks[0]}`." if len(locks) == 1
                 else f"The reports come from {len(locks)} different lockfiles.")
    dirty = [gate_id for gate_id in PHASE5_GATE_IDS if reports[gate_id]["git_dirty"]]
    if dirty:
        lines.append(f"Written from a tree with uncommitted changes: {', '.join(dirty)}. Those reports can't be"
                     " reproduced from their commit alone.")
    return [" ".join(lines)]


# ---------------------------------------------------------------------------
# G5.1
# ---------------------------------------------------------------------------


def _determinism(m: Mapping[str, Any]) -> list[str]:
    by_kind = _map(m["identical_by_kind"])
    lines = [
        "Each kind trained twice from one key at the short settings, from scratch, and both files compared.",
        "",
        "| Kind | Byte-identical |",
        "| --- | --- |",
    ]
    lines += [f"| {_cell(kind)} | {_yes(by_kind[kind])} |" for kind in _ordered(by_kind)]
    if not by_kind:
        lines.append("| no kinds recorded | |")
    by_site = _map(m["identities_by_site"])
    lines += [
        "",
        f"Cache keys: {_num(m['cache_keys_checked'], ',')} checked, all distinct: {_yes(m['cache_keys_distinct'])}.",
        "",
        f"`dpatch`'s identities, from a source equal to the target and with no latents: {_num(m['identities_exact'], ',')}"
        f" of {_num(m['identities_checked'], ',')} bit for bit.",
        "",
        "| Site | Same source | No latents |",
        "| --- | --- | --- |",
    ]
    for site in _ordered(by_site):
        pair = by_site[site] if isinstance(by_site[site], list) else []
        same, none = (pair + [None, None])[:2]
        lines.append(f"| {_cell(site)} | {_exact(same)} | {_exact(none)} |")
    if not by_site:
        lines.append("| no sites recorded | | |")
    return lines


def _exact(value: Any) -> str:
    return "not measured" if value is None else ("exact" if value else "**not exact**")


# ---------------------------------------------------------------------------
# G5.2 and G5.3
# ---------------------------------------------------------------------------


def _zoo(m: Mapping[str, Any], model: str) -> list[str]:
    lines = []
    if model == "two_hop":
        lines += [f"G5.3 read G4.3's best swap block: block {_num(m['best_swap_block'], ',')}.", ""]
    lines += ["Meters, on the trained model and its random-weights twin, judged at every site.", "", *_meters(m, twin=True)]
    lines += ["", "**The faithful code.** A patch through each dictionary's code beside the residual patch from the same"
              " sources, with the single-latent patch beside them (recorded).", ""]
    lines += _faithful_induction(m) if model == "induction" else _faithful_two_hop(m)
    lines += ["", "**Selective latents.** Each value's best latent by F1, the mean of those, trained against twin.", "",
              *_selectivity(m)]
    lines += ["", "**Recorded: readability.** How well single latents and a ridge probe read each variable, from the"
              " site's input activation and from its code.", "", *_readability(m["recovery"], model)]
    lines += ["", "**Recorded: seed robustness.** The TopK SAEs at two more dictionary seeds.", "",
              *_robustness(m["robustness"])]
    lines += ["", "Training time, `from the cache` where an earlier gate trained it.", "", *_seconds(m["train_seconds"])]
    return lines


def _meters(m: Mapping[str, Any], *, twin: bool) -> list[str]:
    meters = _map(m["meters"])
    head = "| Site | FVU | " + ("Twin FVU | " if twin else "") + "Dead | " + ("Twin dead | " if twin else "")
    head += "Ablation cost (nats) | Splice |"
    lines = [head, "| --- |" + " --- |" * (6 if twin else 4)]
    for site in _ordered(meters):
        e = _map(meters[site])
        cells = [_cell(site), _pct(e.get("fvu"))] + ([_pct(e.get("twin_fvu"))] if twin else [])
        cells += [_pct(e.get("dead_share"))] + ([_pct(e.get("twin_dead_share"))] if twin else [])
        cells += [_three(e.get("cost")), _splice(e)]
        lines.append("| " + " | ".join(cells) + " |")
    if not meters:
        lines.append("| no sites recorded |" + " |" * (6 if twin else 4))
    lines += ["", f"FVU at most {_pct(m.get('fvu_max'))}, bar at most {_pct(BARS['fvu'])}: met"
              f" {_at_most(m.get('fvu_max'), BARS['fvu'])}. Splice recovers at least {_pct(m.get('recovered_min'))} where"
              f" an ablation costs at least {_three(BARS['ablation_cost'])} nats, bar at least {_pct(BARS['recovered'])}:"
              f" met {_at_least(m.get('recovered_min'), BARS['recovered'])}."]
    return lines


def _splice(e: Mapping[str, Any]) -> str:
    if _is_number(e.get("recovered")):
        return f"recovers {_pct(e['recovered'])}"
    clean, spliced = e.get("clean_loss"), e.get("spliced_loss")
    if _is_number(clean) and _is_number(spliced):
        return f"changes the loss by {spliced - clean:+.3f} nats (not judged)"
    return "not measured"


def _faithful_induction(m: Mapping[str, Any]) -> list[str]:
    faithful = _map(m["faithful"])
    lines = ["Induction, after block 0 (block 0's MLP output for the transcoder): following is how often the prediction"
             " is the patched-in token; breaking is how often it stays the unpatched pass's own.", "",
             "| Kind | Test | Copy target | Residual | Code | Single-latent |", "| --- | --- | --- | --- | --- | --- |"]
    targets = m.get("patch_targets") if isinstance(m.get("patch_targets"), list) else []
    for kind in _ordered(faithful):
        entry = _map(faithful[kind])
        for test in TESTS:
            by_pass = _map(entry.get(test))
            n = max((len(v) for v in by_pass.values() if isinstance(v, list)), default=0)
            for i in range(n):
                target = _cell(targets[i]) if i < len(targets) else str(i)
                rates = [(_list(by_pass.get(name)) + [None] * n)[i] for name in PASSES]
                lines.append(f"| {_cell(kind)} | {TESTS[test]} | {target} | " + " | ".join(_pct(r) for r in rates) + " |")
    if not faithful:
        lines.append("| no kinds recorded | | | | | |")
    moved = []
    for kind in _ordered(faithful):
        assigned = _map(_map(faithful[kind]).get("assigned"))
        counts = [f"{_num(assigned[v], ',')} for {_cell(v)}" for v in _ordered(assigned)]
        moved.append(f"{_cell(kind)}: {', '.join(counts) or 'none recorded'}")
    lines += ["", f"The single-latent patch moves the latents assigned to the current token when following and to the"
              f" previous token when breaking: {'; '.join(moved) or 'none recorded'}.", "",
              f"Code over residual where the residual patch moves the prediction at least half the time, least"
              f" {_three(m.get('faithful_ratio_min'))}, bar at least {_three(BARS['faithful_ratio'])}: met"
              f" {_at_least(m.get('faithful_ratio_min'), BARS['faithful_ratio'])}. Breaking, the code patch keeps the"
              f" prediction at most {_pct(m.get('break_excess_max'))} more often than the residual patch, bar at most"
              f" {_pct(BARS['break_excess'])}: met {_at_most(m.get('break_excess_max'), BARS['break_excess'])}."]
    return lines


def _faithful_two_hop(m: Mapping[str, Any]) -> list[str]:
    faithful = _map(m["faithful"])
    lines = ["Two-hop, at the query after each block, from G4.3's aligned sources: swap (the answer becomes the"
             " counterpart's successor), transplant (the source's answer), and from the controls, how often the answer"
             " stays the unpatched pass's own.", "",
             "| Kind | Block | Rate | Residual | Code | Single-latent |", "| --- | --- | --- | --- | --- | --- |"]
    moved = []
    for kind in _ordered(faithful):
        by_block = _map(faithful[kind])
        for block in _ordered(by_block):
            entry = _map(by_block[block])
            for rate in RATES:
                values = [_map(entry.get(name)).get(rate) for name in PASSES]
                lines.append(f"| {_cell(kind)} | {_cell(block)} | {RATE_NAMES[rate]} | "
                             + " | ".join(_pct(v) for v in values) + " |")
            assigned = _map(entry.get("assigned"))
            moved.append(f"{_cell(kind)} after block {_cell(block)}: "
                         + ", ".join(f"{_num(assigned[p], ',')} at {_cell(p)}" for p in _ordered(assigned)))
    if not faithful:
        lines.append("| no kinds recorded | | | | | |")
    lines += ["", f"The single-latent patch moves the latents assigned to `b` at each query position: {'; '.join(moved) or 'none recorded'}.",
              "", f"Code over residual where the residual patch moves the answer at least half the time, least"
              f" {_three(m.get('faithful_ratio_min'))}, bar at least {_three(BARS['faithful_ratio'])}: met"
              f" {_at_least(m.get('faithful_ratio_min'), BARS['faithful_ratio'])}. The control patch through the code"
              f" leaves the answer unchanged at least {_pct(m.get('control_unchanged_min'))}, bar at least"
              f" {_pct(BARS['control_unchanged'])}: met {_at_least(m.get('control_unchanged_min'), BARS['control_unchanged'])}."]
    return lines


def _selectivity(m: Mapping[str, Any]) -> list[str]:
    selectivity = _map(m["selectivity"])
    lines = ["| Site and variable | Trained | Twin | Margin over twice the twin |", "| --- | --- | --- | --- |"]
    for key in _ordered(selectivity):
        e = _map(selectivity[key])
        lines.append(f"| {_cell(key)} | {_three(e.get('trained'))} | {_three(e.get('twin'))} | {_three(e.get('margin'))} |")
    if not selectivity:
        lines.append("| none recorded | | | |")
    lines += ["", f"Mean best F1 at least {_three(m.get('selective_f1_min'))}, bar at least {_three(BARS['selective_f1'])}:"
              f" met {_at_least(m.get('selective_f1_min'), BARS['selective_f1'])}. Margin at least"
              f" {_three(m.get('selective_margin_min'))}, bar at least 0: met {_at_least(m.get('selective_margin_min'), 0)}."]
    return lines


def _readability(recovery: Any, model: str) -> list[str]:
    recovery = _map(recovery)
    lines = ["| Site | Position | Variable | Mean best F1 | Twin | Probe from activation | Twin | Probe from code | Twin |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for site in _ordered(recovery):
        by_position = _map(recovery[site]) if model == "two_hop" else {"mechanism": recovery[site]}
        for position in _ordered(by_position):
            by_variable = _map(by_position[position])
            for variable in _ordered(by_variable):
                sides = _map(by_variable[variable])
                t, w = _map(sides.get("trained")), _map(sides.get("twin"))
                cells = [_three(t.get("mean_best_f1")), _three(w.get("mean_best_f1")), _pct(t.get("probe_activation")),
                         _pct(w.get("probe_activation")), _pct(t.get("probe_code")), _pct(w.get("probe_code"))]
                where = "mechanism positions" if position == "mechanism" else _cell(position)
                lines.append(f"| {_cell(site)} | {where} | {_cell(variable)} | " + " | ".join(cells) + " |")
    if not recovery:
        lines.append("| none recorded |" + " |" * 8)
    return lines


def _robustness(robustness: Any) -> list[str]:
    robustness = _map(robustness)
    lines = ["| Seed | Site | FVU | Mean best F1 |", "| --- | --- | --- | --- |"]
    for seed in _ordered(robustness):
        by_site = _map(robustness[seed])
        for site in _ordered(by_site):
            e = _map(by_site[site])
            f1 = _map(e.get("mean_best_f1"))
            shown = ", ".join(f"{_cell(v)} {_three(f1[v])}" for v in _ordered(f1)) or "none recorded"
            lines.append(f"| {_cell(seed)} | {_cell(site)} | {_pct(e.get('fvu'))} | {shown} |")
    if not robustness:
        lines.append("| none recorded | | | |")
    return lines


def _seconds(seconds: Any) -> list[str]:
    seconds = _map(seconds)
    lines = ["| Dictionary | Training time (s) |", "| --- | --- |"]
    for name in _ordered(seconds):
        value = seconds[name]
        lines.append(f"| {_cell(name)} | {'from the cache' if value is None else _num(value, ',.0f')} |")
    if not seconds:
        lines.append("| none recorded | |")
    return lines


# ---------------------------------------------------------------------------
# G5.4 and G5.5
# ---------------------------------------------------------------------------


def _edges(m: Mapping[str, Any]) -> list[str]:
    pairs = _map(m["edge_pairs"])
    lines = ["The edge meter, the fewest edges after which the splice's KL stays within 10% of the way to its floor, for a"
             " TopK pair across each MLP, trained against twin. Judged where ablating the MLP costs at least"
             f" {_three(BARS['ablation_cost'])} nats at the readout.", "",
             "| MLP | Ablation cost (nats) | Judged | Edges | Share of scored | Twin edges | Twin share | Ratio | Met |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key in _ordered(pairs):
        e = _map(pairs[key])
        t, w = _map(e.get("trained")), _map(e.get("twin"))
        judged = e.get("judged")
        met = _at_most(e.get("ratio"), BARS["edge_ratio"]) if judged else "recorded"
        lines.append(f"| {_cell(key)} | {_three(e.get('cost'))} | {_yes(judged)} | {_edges_of(t)} | {_pct(t.get('share'), 2)}"
                     f" | {_edges_of(w)} | {_pct(w.get('share'), 2)} | {_three(e.get('ratio'))} | {met} |")
    if not pairs:
        lines.append("| none recorded |" + " |" * 8)
    lines += ["", f"Greatest ratio over the judged MLPs {_three(m['edge_ratio_max'])}, bar at most"
              f" {_three(BARS['edge_ratio'])}: met {_at_most(m['edge_ratio_max'], BARS['edge_ratio'])}.", "",
              "Recorded: across every block, where masking a latent also moves attention (D42).", "",
              "| Pair | Edges | Share of scored | Twin edges | Twin share |", "| --- | --- | --- | --- | --- |"]
    blocks = _map(m["block_pairs"])
    for key in _ordered(blocks):
        e = _map(blocks[key])
        t, w = _map(e.get("trained")), _map(e.get("twin"))
        lines.append(f"| {_cell(key)} | {_edges_of(t)} | {_pct(t.get('share'), 2)} | {_edges_of(w)} | {_pct(w.get('share'), 2)} |")
    if not blocks:
        lines.append("| none recorded | | | | |")
    lines += ["", "Recorded: the Jacobian SAE at each `λ` beside the TopK pair (`λ` 0), its FVU on each side and its"
              " Jacobian's statistics: the median number of entries holding 90% of its absolute mass, the share above"
              " 0.01 and the mean absolute entry (D43).", "",
              "| MLP | λ | Edges | Share of scored | FVU in | FVU out | Entries for 90% | Share above 0.01 | Mean absolute entry |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    jsae = _map(m["jsae"])
    for key in _ordered(jsae):
        by_lam = _map(jsae[key])
        for lam in _ordered(by_lam):
            e = _map(by_lam[lam])
            jac = _map(e.get("jacobian"))
            lines.append(f"| {_cell(key)} | {_cell(lam)} | {_edges_of(e)} | {_pct(e.get('share'), 2)} | {_pct(e.get('fvu_in'))}"
                         f" | {_pct(e.get('fvu_out'))} | {_num(jac.get('entries_for_90'), ',.1f')}"
                         f" | {_pct(jac.get('share_above_0.01'), 2)} | {_num(jac.get('mean_abs'), '.3g')} |")
    if not jsae:
        lines.append("| none recorded |" + " |" * 8)
    return lines


def _edges_of(reading: Mapping[str, Any]) -> str:
    meter, scored = reading.get("meter"), reading.get("scored_edges")
    if not _is_number(scored):
        return _num(meter, ",")
    if meter is None:
        return f"never settles ({scored:,} scored)"
    return f"{_num(meter, ',')} of {scored:,}"


def _reference(m: Mapping[str, Any]) -> list[str]:
    meters = _map(m["meters"])
    lines = ["Every judged kind on the reference run at G1.4's final checkpoint, with nothing to judge it against"
             " (D44): its meters, recorded.", "",
             "| Site | FVU | Dead | Loss | Spliced | Ablated | Ablation cost (nats) | Splice |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for site in _ordered(meters):
        e = _map(meters[site])
        lines.append(f"| {_cell(site)} | {_pct(e.get('fvu'))} | {_pct(e.get('dead_share'))} | {_three(e.get('clean_loss'))}"
                     f" | {_three(e.get('spliced_loss'))} | {_three(e.get('ablated_loss'))} | {_three(e.get('cost'))}"
                     f" | {_splice(e)} |")
    if not meters:
        lines.append("| none recorded |" + " |" * 7)
    pairs = _map(m["edge_pairs"])
    lines += ["", "The edge meter's TopK pairs across two MLPs, recorded.", "",
              "| MLP | Edges | Share of scored | Spliced-whole KL |", "| --- | --- | --- | --- |"]
    for key in _ordered(pairs):
        e = _map(pairs[key])
        lines.append(f"| {_cell(key)} | {_edges_of(e)} | {_pct(e.get('share'), 2)} | {_num(e.get('pair_kl'), '.3g')} |")
    if not pairs:
        lines.append("| none recorded | | | |")
    lines += ["", "Training time, `from the cache` where an earlier gate trained it.", "", *_seconds(m["train_seconds"])]
    return lines


# ---------------------------------------------------------------------------
# G5.6
# ---------------------------------------------------------------------------


def _features(m: Mapping[str, Any]) -> list[str]:
    builds = _map(m.get("builds"))
    lines = [f"Badges and every run's views, with their features payloads, built twice byte-identical:"
             f" {_yes(m['builds_identical'])}. External URLs in the viewer's files: {_num(m['external_urls'], ',')}."
             f" Subjects without a badge: {_num(m['subjects_missing'], ',')}; shipped dictionaries whose kind has no"
             f" badge: {_num(m['dictionaries_missing'], ',')}.", ""]
    if builds:
        lines += ["| Built | Identical | Files compared | Files differing |", "| --- | --- | --- | --- |"]
        for part in ["badges"] + [name for name in _ordered(builds) if name != "badges"]:
            if part in builds:
                b = _map(builds[part])
                differing = _list(b.get("files_differing"))
                lines.append(f"| {_cell(part)} | {_yes(b.get('identical'))} | {_num(b.get('files_compared'), ',')}"
                             f" | {', '.join(_cell(f) for f in differing) or 'none'} |")
        lines.append("")
    dictionaries, counts = _map(m["dictionaries"]), _map(m["features"])
    lines += ["Each run's shipped dictionaries, each with a features payload per site, its meters from its gate's report.",
              "", "| Run | Dictionary | Sites | Meters from | Content hash |", "| --- | --- | --- | --- | --- |"]
    for run in _ordered(dictionaries):
        for d in _list(dictionaries[run]):
            d = _map(d)
            sites = ", ".join(_cell(site[-1]) if isinstance(site, list) and site else _cell(site)
                              for site in _list(d.get("sites")))
            lines.append(f"| {MODEL_NAMES.get(run, _cell(run))} | {_cell(d.get('kind', ''))} | {sites}"
                         f" | {_cell(d.get('meters_gate', ''))} | `{_cell(str(d.get('content_hash', ''))[:12])}` |")
    if not dictionaries:
        lines.append("| none recorded | | | | |")
    lines += ["", "Features payloads: " + ("; ".join(f"{MODEL_NAMES.get(run, _cell(run))} {_num(counts[run], ',')}"
                                                   for run in _ordered(counts)) or "none recorded") + "."]
    lines += ["", *_query(m.get("query_latents")), "", *_badges(m["badges"])]
    return lines


def _query(query: Any) -> list[str]:
    if not isinstance(query, Mapping):
        return ["The latents active at `⇒`: not recorded."]
    lines = [f"The latents active at `⇒` (position {_num(query.get('position'), ',')}) in the two-hop TopK SAE after"
             f" block {_num(query.get('block'), ',')}, G5.3's best swap block, on each probe window, with the"
             " intermediate `b` of each of their top held-out contexts' windows, read from the features payload.", "",
             "| Window | Its `b` | Latent | Code | `b` of its top contexts |", "| --- | --- | --- | --- | --- |"]
    for window in _list(query.get("windows")):
        window = _map(window)
        latents = _list(window.get("latents"))
        first = f"| {_cell(window.get('window'))} | {_cell(window.get('b'))} |"
        if not latents:
            lines.append(f"{first} none active | | |")
        for i, latent in enumerate(latents):
            latent = _map(latent)
            contexts = ", ".join(_cell(b) for b in _list(latent.get("context_b"))) or "none"
            lines.append(f"{first if i == 0 else '| | |'} {_cell(latent.get('latent'))}"
                         f" | {_num(latent.get('code'), '.3f')} | {contexts} |")
    return lines


def _badges(badges: Any) -> list[str]:
    badges = _map(badges)
    lines = [
        "Each badge, as G5.6 generated it from G4.1 to G4.4's and G5.1 to G5.4's reports: Phase 4's model-level badges,"
        " then each shipped dictionary kind's and `dpatch`'s. Every value is the cited report's own; a rank is 0-based,"
        " 0 being the top token.",
        "",
        "| Subject | Status | Fact | Value | Gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    notes = []
    for subject in [s for s in SUBJECTS] + sorted(s for s in badges if s not in SUBJECTS):
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
    return _num(value, "g") if _is_number(value) else _cell(value)


# ---------------------------------------------------------------------------


def _token(token: str) -> tuple:
    if token in _WHO:
        return (0, _WHO.index(token), 0, "")
    if token in _RUNS:
        return (1, _RUNS.index(token), 0, "")
    if token in _KINDS:
        return (2, _KINDS.index(token), 0, "")
    block = _BLOCK.fullmatch(token)
    if block:
        return (3, int(block.group(1)), 0, block.group(2))
    number = _NUMBER.fullmatch(token)
    if number:
        return (3, int(number.group(1)), 0, token)
    if token in _VARIABLES:
        return (4, _VARIABLES.index(token), 0, "")
    return (5, 0, 0, token)


def _ordered(mapping: Any) -> list[str]:
    """Keys in the spec's order: kinds as topk, staircase, transcoder and jsae, blocks and numbers in order,
    variables as the gates list them; anything else after, sorted."""
    return sorted((str(key) for key in mapping), key=lambda key: (tuple(_token(t) for t in key.split()), key))


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
