"""Phase 3 report rules: gate IDs, status rules and required measurements. Contract: docs/deltas/phase-3-delta-2.md."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {
    "G3.1": {"identities_checked": 72, "identities_exact": 72, "identities_by_kind": {"null edit": 26}},
    "G3.2": {"sample_size": 12, "recompute_identical": 12, "cache_keys_distinct": True},
    "G3.3": {"layers": [0, 1, 2], "scale_factors": [0.25, 0.5, 1.0], "results_by_layer": {"0": {}}},
    "G3.4": {"meters_by_step": {"12208": {}}},
    "G3.5": {"builds_identical": True, "external_urls": 0, "payloads": 6},
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING[gate_id], **measurements})
    return report


def test_phase3_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    for label, report in {
        "G3.1 an identity not exact": _g("G3.1", identities_exact=71),
        "G3.2 a recompute differed": _g("G3.2", recompute_identical=11),
        "G3.2 two keys collided": _g("G3.2", cache_keys_distinct=False),
        "G3.3 is recorded, not judged: it can't fail": _g("G3.3", "fail"),
        "G3.4 is recorded, not judged: it can't fail": _g("G3.4", "fail"),
        "G3.5 builds differed": _g("G3.5", builds_identical=False),
        "G3.5 the viewer reaches outside": _g("G3.5", external_urls=1),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    missing = _g("G3.3")
    del missing["measurements"]["results_by_layer"]
    assert validate_report(missing), "a required measurement is missing (G3.3 always passes, so only the list catches it)"
    assert validate_report(_g("G3.1", "fail", identities_exact=3)) == []
