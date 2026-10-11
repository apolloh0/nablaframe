"""Phase 5 delta 5: dictionary badges, generated from the Phase 5 reports (D38, D41, D44). Needs no JAX.

Contract: src/nablaframe/views/dictionary_badges.py.
"""

from __future__ import annotations

import copy
import json

import pytest

from nablaframe.reports import validate_report
from nablaframe.views import dictionary_badges as db  # looked up per test, so a missing name fails its test alone
from test_phase1_store_reports import _v2
from test_reports import passing_report
from test_views_badges import MEASUREMENTS as PHASE4

KINDS = ("topk", "staircase")
TAPS = {"topk": "hook_resid_post", "staircase": "hook_resid_post", "transcoder": "hook_mlp_out"}


def _f1(seed: float) -> dict:
    return {"trained": round(0.6 + seed, 4), "twin": round(0.1 + seed / 2, 4), "margin": round(0.4 + seed / 3, 4)}


def _selectivity(blocks: range, variables: tuple, base: float) -> dict:
    """Every kind, block and variable given, each with its own numbers, so a wrong path reads a wrong value."""
    out = {}
    for i, kind in enumerate(("topk", "staircase", "transcoder")):
        for l in blocks:
            for j, v in enumerate(variables):
                out[f"{kind} blocks.{l}.{TAPS[kind]} {v}"] = _f1(base + 0.1 * i + 0.01 * l + 0.001 * j)
    return out


def _faithful(blocks: range) -> dict:
    return {kind: {str(l): {"residual": {"swap": 0.5 + 0.1 * l, "transplant": 0.2, "control_unchanged": 0.99},
                            "code": {"swap": 0.49 + 0.1 * l + 0.001 * i, "transplant": 0.21, "control_unchanged": 0.985},
                            "latents": {"swap": 0.1 + 0.05 * l + 0.002 * i, "transplant": 0.0, "control_unchanged": 1.0},
                            "assigned": {"24": 2, "25": 3 + l}}
                   for l in blocks}
            for i, kind in enumerate(KINDS)}


def _pair(trained: int, twin: int, judged: bool) -> dict:
    return {"cost": 0.4 if judged else 0.01, "judged": judged, "ratio": trained / twin,
            "trained": {"meter": trained, "share": 0.01, "scored_edges": 60000, "pair_kl": 0.01, "curve": []},
            "twin": {"meter": twin, "share": 0.9, "scored_edges": 60000, "pair_kl": 0.002, "curve": []}}


MEASUREMENTS = {
    "G4.1": copy.deepcopy(PHASE4["G4.1"]),
    "G5.1": {"files_identical": True, "identical_by_kind": {"topk": True, "staircase": True, "transcoder": True, "jsae": True},
             "kinds_checked": ["topk", "staircase", "transcoder", "jsae"], "cache_keys_checked": 68,
             "cache_keys_distinct": True, "identities_checked": 36, "identities_exact": 36,
             "identities_by_site": {"induction topk blocks.0.hook_resid_post": [True, True]}},
    "G5.2": {"meters": {}, "fvu_max": 0.032, "recovered_min": 0.991,
             "faithful": {"topk": {"follow": {"residual": [1.0, 1.0, 1.0], "code": [1.0, 1.0, 1.0], "latents": [0.4, 0.3, 0.2]},
                                   "break": {"residual": [0.02, 0.03, 0.03], "code": [0.03, 0.03, 0.04],
                                             "latents": [0.5, 0.5, 0.5]},
                                   "assigned": {"cur": 40, "prev": 12}}},
             "faithful_ratio_min": 0.981, "break_excess_max": 0.013,
             "selectivity": _selectivity(range(2), ("cur", "prev", "next"), 0.2), "selective_f1_min": 0.621,
             "selective_margin_min": 0.33, "recovery": {}, "robustness": {}, "train_seconds": {}},
    "G5.3": {"meters": {}, "fvu_max": 0.028, "recovered_min": 0.993, "faithful": _faithful(range(4)),
             "faithful_ratio_min": 0.957, "control_unchanged_min": 0.986,
             "selectivity": _selectivity(range(4), ("a", "b", "c"), 0.1), "selective_f1_min": 0.726,
             "selective_margin_min": 0.24, "best_swap_block": 2, "recovery": {}, "robustness": {}, "train_seconds": {}},
    "G5.4": {"edge_pairs": {"induction mlp 0": _pair(606, 58391, True), "induction mlp 1": _pair(252, 22363, True),
                            "two_hop mlp 0": _pair(263, 3944, True), "two_hop mlp 2": _pair(47, 3815, False)},
             "judged_mlps": ["induction mlp 0", "induction mlp 1", "two_hop mlp 0"], "edge_ratio_max": 0.067,
             "block_pairs": {}, "jsae": {}},
}


def _reports(measurements=None, **statuses) -> dict:
    out = {}
    for gate_id, m in (measurements or MEASUREMENTS).items():
        report = _v2(passing_report("G0.3"))
        report.update(gate_id=gate_id, status=statuses.get(gate_id.replace(".", ""), "pass"), env_name="nf-jax",
                      measurements=copy.deepcopy(m))
        assert validate_report(report) == [], gate_id
        out[gate_id] = report
    return out


def _at(report: dict, path: list):
    value = report["measurements"]
    for segment in path:
        value = value[segment]
    return value


def _expected(best: int, last: int, judged: list) -> dict:
    """Each subject's facts as (gate, path, unit), in the contract's order."""
    def sel(gate, key):
        return [(gate, ["selectivity", key, "trained"], "score"), (gate, ["selectivity", key, "twin"], "score")]

    out = {}
    for kind in KINDS:
        facts = (sel("G5.2", f"{kind} blocks.0.hook_resid_post cur") + sel("G5.2", f"{kind} blocks.1.hook_resid_post next")
                 + sel("G5.3", f"{kind} blocks.{best}.hook_resid_post b")
                 + sel("G5.3", f"{kind} blocks.{last}.hook_resid_post c"))
        facts += [("G5.3", ["faithful", kind, str(best), p, "swap"], "fraction") for p in ("code", "residual", "latents")]
        facts += [("G5.2", ["faithful_ratio_min"], "ratio"), ("G5.2", ["break_excess_max"], "fraction"),
                  ("G5.3", ["faithful_ratio_min"], "ratio"), ("G5.3", ["control_unchanged_min"], "fraction")]
        if kind == "topk":
            facts += [("G5.4", ["edge_pairs", key, side, "meter"], "edges") for key in judged for side in ("trained", "twin")]
        out[kind] = facts
    out["transcoder"] = (sel("G5.2", "transcoder blocks.0.hook_mlp_out cur") + sel("G5.2", "transcoder blocks.1.hook_mlp_out next")
                         + sel("G5.3", f"transcoder blocks.{best}.hook_mlp_out b")
                         + [("G5.2", ["faithful_ratio_min"], "ratio"), ("G5.2", ["break_excess_max"], "fraction")])
    out["dpatch"] = [("G5.1", ["identities_exact"], "count"), ("G5.1", ["identities_checked"], "count"),
                     ("G5.2", ["faithful_ratio_min"], "ratio"), ("G5.2", ["break_excess_max"], "fraction"),
                     ("G5.3", ["faithful_ratio_min"], "ratio"), ("G5.3", ["control_unchanged_min"], "fraction")]
    return out


def _check(badges: dict, reports: dict, best: int, last: int, judged: list) -> None:
    assert list(badges) == list(db.SUBJECTS)
    expected = _expected(best, last, judged)
    for subject, badge in badges.items():
        assert set(badge) == {"status", "sources", "facts", "notes"}, subject
        assert [(x["gate"], x["path"], x["unit"]) for x in badge["facts"]] == expected[subject], subject
        for fact in badge["facts"]:
            assert set(fact) == {"label", "value", "unit", "gate", "path"}
            assert fact["value"] == _at(reports[fact["gate"]], fact["path"]), (subject, fact["path"])
            assert isinstance(fact["label"], str) and fact["label"].strip(), (subject, fact["path"])
        labels = [fact["label"] for fact in badge["facts"]]
        assert len(set(labels)) == len(labels), f"{subject}: each label names one fact"
        assert badge["sources"] == sorted({fact["gate"] for fact in badge["facts"]})
    assert json.loads(json.dumps(badges)) == badges, "plain JSON"


def test_subjects_units_and_notes():
    assert db.SUBJECTS == ("topk", "staircase", "transcoder", "dpatch")
    assert db.UNITS == ("fraction", "score", "ratio", "edges", "count")
    assert db.BOUNDS_NOTE == "A figure for any kind is its gate's bound over every kind that gate judges, not this kind's alone."
    assert db.TWIN_NOTE == ("Each twin figure is a dictionary trained the same way on the model's random-weights twin, "
                            "which has nothing to find.")
    assert db.EDGE_NOTE == ("The edge meter is judged across the MLPs whose mean ablation costs at least 0.1 nats. "
                            "Across a block it is recorded, not judged: masking a latent also moves attention (D42).")
    assert db.TRANSCODER_NOTE == ("Two-hop's MLP outputs carry no planted variable at the query, so a transcoder's code "
                                  "is patched on the induction model only.")
    assert db.DPATCH_NOTE == ("The identities show dpatch changes nothing where it should change nothing. The rates "
                              "judge a patch through a dictionary's code against the residual patch, so they judge the "
                              "operation and the dictionaries together.")


def test_badges_cite_the_reports():
    reports = _reports()
    badges = db.dictionary_badges(reports)
    _check(badges, reports, best=2, last=3, judged=MEASUREMENTS["G5.4"]["judged_mlps"])
    assert {s: b["notes"] for s, b in badges.items()} == {
        "topk": [db.BOUNDS_NOTE, db.TWIN_NOTE, db.EDGE_NOTE], "staircase": [db.BOUNDS_NOTE, db.TWIN_NOTE],
        "transcoder": [db.BOUNDS_NOTE, db.TWIN_NOTE, db.TRANSCODER_NOTE], "dpatch": [db.BOUNDS_NOTE, db.DPATCH_NOTE]}
    assert all(b["status"] == "judged: pass" for b in badges.values())
    assert badges["topk"]["sources"] == ["G5.2", "G5.3", "G5.4"] and badges["dpatch"]["sources"] == ["G5.1", "G5.2", "G5.3"]
    assert reports == _reports(), "the reports are read, never changed"


def test_paths_follow_the_reports_blocks_and_judged_mlps():
    m = copy.deepcopy(MEASUREMENTS)
    m["G5.3"]["best_swap_block"] = 1
    m["G4.1"]["models"]["two_hop"]["model_config"]["n_layers"] = 3
    m["G5.4"]["judged_mlps"] = ["two_hop mlp 0", "induction mlp 0"]
    reports = _reports(m)
    _check(db.dictionary_badges(reports), reports, best=1, last=2, judged=["two_hop mlp 0", "induction mlp 0"])


def test_status_follows_the_judging_gates():
    def status(**statuses):
        return {s: b["status"] for s, b in db.dictionary_badges(_reports(**statuses)).items()}

    passing, failing = "judged: pass", "judged: fail"
    assert status(G54="fail") == {"topk": failing, "staircase": passing, "transcoder": passing, "dpatch": passing}
    assert status(G51="fail") == {"topk": passing, "staircase": passing, "transcoder": passing, "dpatch": failing}
    assert status(G52="fail") == {s: failing for s in db.SUBJECTS}
    assert status(G53="fail") == {s: failing for s in db.SUBJECTS}


def test_bad_reports_are_refused():
    def without(gate_id):
        return {g: r for g, r in _reports().items() if g != gate_id}

    def changed(gate_id, edit):
        reports = _reports()
        edit(reports[gate_id]["measurements"])
        return reports

    def misfiled():
        reports = _reports()
        impostor = copy.deepcopy(reports["G5.3"])  # a valid G5.3 report holding every path G5.2's badge facts read
        impostor["measurements"]["selectivity"].update(reports["G5.2"]["measurements"]["selectivity"])
        impostor["measurements"]["break_excess_max"] = 0.013
        assert validate_report(impostor) == []
        reports["G5.2"] = impostor
        return reports

    def invalid():
        reports = _reports()
        del reports["G5.2"]["measurements"]["selectivity"]
        return reports

    cases = {
        "G4.1 missing": without("G4.1"),
        "G5.4 missing": without("G5.4"),
        "a G5.3 report filed as G5.2's": misfiled(),
        "an invalid report": invalid(),
        "a selectivity entry missing": changed("G5.2", lambda m: m["selectivity"].pop("topk blocks.0.hook_resid_post cur")),
        "the best swap block's faithful code missing": changed("G5.3", lambda m: m["faithful"]["staircase"].pop("2")),
        "a judged MLP without its pair": changed("G5.4", lambda m: m["edge_pairs"].pop("induction mlp 1")),
    }
    for label, reports in cases.items():
        with pytest.raises(ValueError):
            db.dictionary_badges(reports)
            pytest.fail(label)
    extra = _reports()
    extra["G4.2"] = {"not": "a report"}
    extra["G5.5"] = None
    assert db.dictionary_badges(extra) == db.dictionary_badges(_reports()), "other keys are ignored"
