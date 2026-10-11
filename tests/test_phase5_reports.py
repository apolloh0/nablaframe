"""Phase 5 report rules: gate IDs, status rules and required measurements. Contract: docs/deltas/phase-5-delta-4.md."""

from __future__ import annotations

from nablaframe import reports
from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

METERS = {"topk blocks.0.hook_resid_post": {"fvu": 0.03, "dead_share": 0.01, "twin_fvu": 0.15, "twin_dead_share": 0.0,
                                            "clean_loss": 0.001, "spliced_loss": 0.01, "ablated_loss": 4.2, "cost": 4.2,
                                            "recovered": 0.998}}
PASSING = {
    "G5.1": {"files_identical": True, "identical_by_kind": {"topk": True}, "kinds_checked": ["topk", "staircase"],
             "cache_keys_checked": 68, "cache_keys_distinct": True, "identities_checked": 36, "identities_exact": 36,
             "identities_by_site": {"induction topk blocks.0.hook_resid_post": [True, True]}},
    "G5.2": {"meters": METERS, "fvu_max": 0.05, "recovered_min": 0.95, "faithful": {}, "faithful_ratio_min": 0.97,
             "break_excess_max": 0.01, "selectivity": {}, "selective_f1_min": 0.8, "selective_margin_min": 0.3,
             "recovery": {}, "robustness": {}, "train_seconds": {"topk blocks.0.hook_resid_post": 287.5}},
    "G5.3": {"meters": METERS, "fvu_max": 0.05, "recovered_min": 0.95, "faithful": {}, "faithful_ratio_min": 0.95,
             "control_unchanged_min": 0.98, "selectivity": {}, "selective_f1_min": 0.76, "selective_margin_min": 0.26,
             "best_swap_block": 2, "recovery": {}, "robustness": {}, "train_seconds": {}},
    "G5.4": {"edge_pairs": {"induction mlp 0": {"ratio": 0.01}}, "judged_mlps": ["induction mlp 0"], "edge_ratio_max": 0.067,
             "block_pairs": {}, "jsae": {}},
    "G5.5": {"meters": METERS, "edge_pairs": {}, "train_seconds": {}},
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING[gate_id], **measurements})
    return report


def test_phase5_gate_ids():
    assert reports.PHASE5_GATE_IDS == ("G5.1", "G5.2", "G5.3", "G5.4", "G5.5", "G5.6")  # G5.6 from delta 5
    old = _g("G5.1")
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 5 report must use schema version 2"


def test_phase5_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    for label, report in {
        "G5.1 files differed": _g("G5.1", files_identical=False),
        "G5.1 two cache keys collided": _g("G5.1", cache_keys_distinct=False),
        "G5.1 an identity wasn't exact": _g("G5.1", identities_exact=35),
        "G5.1 no identity checked": _g("G5.1", identities_checked=0, identities_exact=0),
        "G5.2 a reconstruction too poor": _g("G5.2", fvu_max=0.11),
        "G5.2 a splice recovering too little": _g("G5.2", recovered_min=0.89),
        "G5.2 no splice judged": _g("G5.2", recovered_min=None),
        "G5.2 a code patch short of the residual's": _g("G5.2", faithful_ratio_min=0.89),
        "G5.2 no faithful test applied": _g("G5.2", faithful_ratio_min=None),
        "G5.2 breaking kept too often": _g("G5.2", break_excess_max=0.06),
        "G5.2 latents not selective": _g("G5.2", selective_f1_min=0.49),
        "G5.2 not twice the twin": _g("G5.2", selective_margin_min=-0.01),
        "G5.3 a reconstruction too poor": _g("G5.3", fvu_max=0.11),
        "G5.3 a splice recovering too little": _g("G5.3", recovered_min=0.89),
        "G5.3 a code patch short of the residual's": _g("G5.3", faithful_ratio_min=0.89),
        "G5.3 the control moved the answer": _g("G5.3", control_unchanged_min=0.94),
        "G5.3 latents not selective": _g("G5.3", selective_f1_min=0.49),
        "G5.3 not twice the twin": _g("G5.3", selective_margin_min=-0.01),
        "G5.4 the trained pair not a fifth of its twin's": _g("G5.4", edge_ratio_max=0.21),
        "G5.4 a trained meter that never settled": _g("G5.4", edge_ratio_max=None),
        "G5.5 is recorded, not judged: it can't fail": _g("G5.5", "fail"),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    for gate_id, values in (("G5.2", dict(fvu_max=0.1, recovered_min=0.9, faithful_ratio_min=0.9, break_excess_max=0.05,
                                          selective_f1_min=0.5, selective_margin_min=0.0)),
                            ("G5.3", dict(control_unchanged_min=0.95)), ("G5.4", dict(edge_ratio_max=0.2))):
        assert validate_report(_g(gate_id, **values)) == [], f"{gate_id} at its bars passes"
    assert validate_report(_g("G5.2", "fail", fvu_max=0.5)) == []
    for gate_id, key in (("G5.1", "identities_by_site"), ("G5.2", "faithful"), ("G5.2", "train_seconds"),
                         ("G5.3", "best_swap_block"), ("G5.3", "control_unchanged_min"), ("G5.4", "jsae"),
                         ("G5.4", "judged_mlps"), ("G5.5", "edge_pairs")):
        missing = _g(gate_id)
        del missing["measurements"][key]
        assert validate_report(missing), f"{gate_id} missing {key}"
