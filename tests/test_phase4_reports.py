"""Phase 4 report rules: gate IDs, status rules and required measurements. Contract: docs/deltas/phase-4-delta-3.md."""

from __future__ import annotations

from nablaframe import reports
from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {
    "G4.1": {"recompute_identical": True, "checkpoints_written": 62, "checkpoints_scheduled": 62, "events_recorded": 1576,
             "events_expected": 1576,
             "induction_mechanism_accuracy": 1.0, "induction_control_accuracy": 0.003, "induction_twin_accuracy": 0.017,
             "induction_chance": 1 / 64, "two_hop_accuracy": 0.99, "two_hop_broken_accuracy": 0.0,
             "two_hop_twin_accuracy": 0.008, "two_hop_chance": 1 / 48},
    "G4.2": {"mechanism_accuracy": 1.0, "chance": 1 / 64, "attention_by_head": [0.985, 0.977, 0.963, 0.986],
             "best_head_attention": 0.986, "induction_heads": [0, 1, 2, 3], "twin_best_attention": 0.048,
             "uniform_baseline": 0.046, "ablation_fraction_removed": 1.0, "fraction_removed_by_head": {"1.0": 0.0},
             "swap_step": 50, "swap_fraction_removed": 0.93, "patch_followed_by_tap": {"blocks.0.hook_resid_post": [1.0]},
             "patch_after_block0_min": 1.0, "patch_after_block1_max": 0.0},
    "G4.3": {"unpatched_accuracy": 0.99, "swap_by_site": {"query": [0.0, 0.7, 0.86, 0.0]},
             "transplant_by_site": {"query": [0.0, 0.03, 0.07, 0.99]},
             "control_unchanged_by_site": {"query": [0.98, 0.98, 0.99, 1.0]}, "control_unchanged_min": 0.98,
             "best_swap": {"site": "query", "block": 2, "rate": 0.86}},
    "G4.4": {"lens_ranks": {}, "jswap_by_layer": [], "attribution": {}},
    "G4.5": {"formation_loss_step": 600, "formation_attention_step": 600, "builds_identical": True, "external_urls": 0},
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING[gate_id], **measurements})
    return report


def test_phase4_gate_ids():
    assert reports.PHASE4_GATE_IDS == ("G4.1", "G4.2", "G4.3", "G4.4", "G4.5")
    old = _g("G4.1")
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 4 report must use schema version 2"


def test_phase4_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    for label, report in {
        "G4.1 recompute differed": _g("G4.1", recompute_identical=False),
        "G4.1 a checkpoint missing": _g("G4.1", checkpoints_written=61),
        "G4.1 a trace event missing": _g("G4.1", events_recorded=1575),
        "G4.1 induction below the bar": _g("G4.1", induction_mechanism_accuracy=0.89),
        "G4.1 induction control above twice chance": _g("G4.1", induction_control_accuracy=0.032),
        "G4.1 induction twin above twice chance": _g("G4.1", induction_twin_accuracy=0.032),
        "G4.1 two-hop below the bar": _g("G4.1", two_hop_accuracy=0.89),
        "G4.1 broken chains above twice chance": _g("G4.1", two_hop_broken_accuracy=0.042),
        "G4.1 two-hop twin above twice chance": _g("G4.1", two_hop_twin_accuracy=0.042),
        "G4.2 no head reaches the bar": _g("G4.2", best_head_attention=0.49),
        "G4.2 twin above twice uniform": _g("G4.2", twin_best_attention=0.093),
        "G4.2 ablation removes too little": _g("G4.2", ablation_fraction_removed=0.49),
        "G4.2 ablation not measured": _g("G4.2", ablation_fraction_removed=None),
        "G4.2 swap removes too little": _g("G4.2", swap_fraction_removed=0.49),
        "G4.2 patch after block 0 not followed": _g("G4.2", patch_after_block0_min=0.89),
        "G4.2 patch after block 1 still followed": _g("G4.2", patch_after_block1_max=0.11),
        "G4.3 the control moved the answer": _g("G4.3", control_unchanged_min=0.94),
        "G4.4 is recorded, not judged: it can't fail": _g("G4.4", "fail"),
        "G4.5 builds differed": _g("G4.5", builds_identical=False),
        "G4.5 the viewer reaches outside": _g("G4.5", external_urls=1),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    assert validate_report(_g("G4.3", best_swap={"site": "query", "block": 1, "rate": 0.2})) == [], \
        "the swap is recorded, not judged (D36)"
    assert validate_report(_g("G4.3", "fail", control_unchanged_min=0.5)) == []
    for gate_id, key in (("G4.2", "fraction_removed_by_head"), ("G4.3", "swap_by_site"), ("G4.4", "jswap_by_layer"),
                         ("G4.5", "formation_attention_step"), ("G4.1", "events_recorded")):
        missing = _g(gate_id)
        del missing["measurements"][key]
        assert validate_report(missing), f"{gate_id} missing {key}"
