"""Phase 6 delta 5: G6.7's report rules. Contract: docs/deltas/phase-6-delta-5.md, "src/nablaframe/reports.py"."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {"badges": {"name": {"status": "judged: pass"}}, "subjects_missing": 0,
           "records": {"two_hop": {"name": 300, "edge": 3, "open": 96, "generalize": 1}},
           "builds_identical": True, "external_urls": 0, "reports_match": True}


def _g67(status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id="G6.7", status=status, env_name="nf-jax", measurements={**PASSING, **measurements})
    return report


def test_g67_status_rules():
    assert validate_report(_g67()) == []
    for label, report in {
        "the builds differed": _g67(builds_identical=False),
        "no build compared": _g67(builds_identical=None),
        "the viewer reaches outside": _g67(external_urls=1),
        "a gesture or an operation has no badge": _g67(subjects_missing=1),
        "the records shown aren't the ones judged": _g67(reports_match=False),
        "the records weren't matched": _g67(reports_match=None),
    }.items():
        assert validate_report(report), label
    assert validate_report(_g67("fail", builds_identical=False, reports_match=False)) == []
    for key in PASSING:
        missing = _g67()
        del missing["measurements"][key]
        assert validate_report(missing), f"missing {key}"
    old = _g67()
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 6 report must use schema version 2"
