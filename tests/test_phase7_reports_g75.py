"""Phase 7 delta 4: G7.5's report rules. Contract: docs/deltas/phase-7-delta-4.md, "src/nablaframe/reports.py"."""

from __future__ import annotations

from nablaframe import reports
from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {"badges": {"crystallize": {"status": "judged: pass"}}, "subjects_missing": 0,
           "records": {"modular_addition": {"name": 171, "crystallize": 4}, "two_hop": {"name": 0, "crystallize": 2}},
           "builds_identical": True, "external_urls": 0, "reports_match": True}


def _g75(status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id="G7.5", status=status, env_name="nf-jax", measurements={**PASSING, **measurements})
    return report


def test_g75_status_rules():
    assert reports.PHASE7_GATE_IDS == ("G7.1", "G7.2", "G7.3", "G7.4", "G7.5")
    assert set(reports.REQUIRED_MEASUREMENTS["G7.5"]) == set(PASSING)
    assert validate_report(_g75()) == []
    for label, report in {
        "the builds differed": _g75(builds_identical=False),
        "no build compared": _g75(builds_identical=None),
        "the viewer reaches outside": _g75(external_urls=1),
        "a subject has no badge": _g75(subjects_missing=1),
        "the records shown aren't the ones measured": _g75(reports_match=False),
        "the records weren't matched": _g75(reports_match=None),
    }.items():
        assert validate_report(report), label
    assert validate_report(_g75("fail", builds_identical=False, reports_match=False)) == [], "G7.5 may fail"
    for key in PASSING:
        missing = _g75()
        del missing["measurements"][key]
        assert validate_report(missing), f"missing {key}"
    old = _g75()
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 7 report must use schema version 2"
