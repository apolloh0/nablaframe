"""Report rules for G2.10, the J-lens view. Contract: docs/deltas/phase-2-delta-4.md."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {"planted_recovery": 1.0, "identity_exact": True, "random_max_accuracy": 0.0003, "random_chance": 1 / 4096,
           "last_reading_kl_max": 2e-13, "comparison_by_step": {"12208": [{"layer": 0, "js_mean": 0.2, "top1_agreement": 0.4}]}}


def _g(status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id="G2.10", status=status, env_name="nf-jax", measurements={**PASSING, **measurements})
    return report


def test_jlens_view_status_rules():
    assert validate_report(_g()) == []
    for label, report in {
        "planted transport missed": _g(planted_recovery=0.98),
        "identity J differs from the logit lens": _g(identity_exact=False),
        "above twice chance on random weights": _g(random_max_accuracy=3 / 4096),
        "last reading drifts from the logits": _g(last_reading_kl_max=2e-5),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    missing = _g()
    del missing["measurements"]["comparison_by_step"]  # in no rule, so only the required list catches it
    assert validate_report(missing), "a required measurement is missing"
