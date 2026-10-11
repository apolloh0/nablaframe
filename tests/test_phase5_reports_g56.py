"""Phase 5 delta 5: G5.6's report rules. Contract: docs/deltas/phase-5-delta-5.md, "src/nablaframe/reports.py"."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {"badges": {"topk": {"status": "judged: pass"}}, "subjects_missing": 0,
           "dictionaries": {"reference": [{"kind": "topk", "content_hash": "ab" * 32}]}, "dictionaries_missing": 0,
           "features": {"reference": 12, "induction": 6, "two_hop": 12}, "builds_identical": True, "external_urls": 0}


def _g56(status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id="G5.6", status=status, env_name="nf-jax", measurements={**PASSING, **measurements})
    return report


def test_g56_status_rules():
    assert validate_report(_g56()) == []
    for label, report in {
        "the builds differed": _g56(builds_identical=False),
        "no build compared": _g56(builds_identical=None),
        "the viewer reaches outside": _g56(external_urls=1),
        "a shipped dictionary has no badge": _g56(dictionaries_missing=1),
        "a view or operation has no badge": _g56(subjects_missing=1),
    }.items():
        assert validate_report(report), label
    assert validate_report(_g56("fail", builds_identical=False, external_urls=2)) == []
    for key in PASSING:
        missing = _g56()
        del missing["measurements"][key]
        assert validate_report(missing), f"missing {key}"
    old = _g56()
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 5 report must use schema version 2"
