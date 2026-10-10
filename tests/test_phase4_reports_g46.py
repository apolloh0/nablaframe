"""Phase 4 delta 4: G4.6's report rules. Contract: docs/deltas/phase-4-delta-4.md, "src/nablaframe/reports.py"."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {"badges": {"logit_lens": {"status": "recorded"}}, "subjects_missing": 0, "builds_identical": True,
           "payloads": {"induction": 1, "two_hop": 2}}


def _g46(status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id="G4.6", status=status, env_name="nf-jax", measurements={**PASSING, **measurements})
    return report


def test_g46_status_rules():
    assert validate_report(_g46()) == []
    assert validate_report(_g46(builds_identical=False)), "the builds differed"
    assert validate_report(_g46(subjects_missing=1)), "a view or operation has no badge"
    assert validate_report(_g46("fail", builds_identical=False)) == []
    for key in PASSING:
        missing = _g46()
        del missing["measurements"][key]
        assert validate_report(missing), f"missing {key}"
    old = _g46()
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 4 report must use schema version 2"
