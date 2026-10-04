"""Report rules for the J-lens gates, G2.7 to G2.9. Contract: docs/deltas/phase-2-delta-3.md."""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {
    "G2.7": ("nf-jax", {"step": 12208, "layer": 0, "windows": 8, "export_matches_params": True,
                        "jacobian_recompute_identical": True}),
    "G2.8": ("nf-torch", {"windows": 8, "tokens_identical": True, "export_logit_max_abs_diff": 3e-6,
                          "jacobian_relative_difference": 2e-7}),
    "G2.9": ("nf-jax", {"jacobians_written": 84, "jacobians_expected": 84, "recompute_identical": True,
                        "seconds": 14_000.0}),
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    env, passing = PASSING[gate_id]
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name=env, measurements={**passing, **measurements})
    return report


def test_jlens_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    contradictions = {
        "G2.7 export differs from the parameters": _g("G2.7", export_matches_params=False),
        "G2.7 J recomputes differently": _g("G2.7", jacobian_recompute_identical=False),
        "G2.8 the reference saw other tokens": _g("G2.8", tokens_identical=False),
        "G2.8 export logits drift": _g("G2.8", export_logit_max_abs_diff=2e-4),
        "G2.8 J differs from the reference": _g("G2.8", jacobian_relative_difference=2e-4),
        "G2.9 a Jacobian missing": _g("G2.9", jacobians_written=83),
        "G2.9 J recomputes differently": _g("G2.9", recompute_identical=False),
    }
    for label, report in contradictions.items():
        assert validate_report(report), f"accepted: {label}"
    missing = _g("G2.9")
    del missing["measurements"]["seconds"]  # in no rule, so only the required list catches it
    assert validate_report(missing), "a required measurement is missing"
    assert validate_report(_g("G2.8", "fail", jacobian_relative_difference=0.5)) == []
