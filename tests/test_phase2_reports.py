"""Phase 2 report rules: gate IDs, status rules and required measurements.

Spec: docs/specs/phase-2.md, "Gates"; contract: docs/deltas/phase-2-delta-2.md.
"""

from __future__ import annotations

from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING_G2 = {
    "G2.1": {"logits_events": 28, "logits_events_expected": 28, "other_events_written": 0, "sample_size": 20,
             "sample_matches": 20, "logits_bytes": 939_526_560, "logits_bytes_expected": 939_524_096},
    "G2.2": {"sample_size": 200, "sample_matches": 200, "builds_identical": True},
    "G2.3": {"planted_recovery": 0.999, "random_max_accuracy": 0.0003, "random_chance": 1 / 4096,
             "last_layer_kl_max": 2e-8},
    "G2.4": {"planted_aligned_credit": 1.0, "planted_perpendicular_quiet": 1.0, "decomposition_error_max": 4e-6},
    "G2.5": {"exact_positions": 2048, "estimator_errors": {"1": 0.2, "8": 0.07, "32": 0.04}, "fit_sizes": [2048, 8192, 32768],
             "fit_change_per_doubling": [0.05, 0.025], "projected_seconds": 5400.0, "crosscheck_feasible": False},
    "G2.6": {"builds_identical": True, "external_urls": 0, "export_bytes": 41_000_000},
}


def _g2(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING_G2[gate_id], **measurements})
    return report


def test_phase2_status_rules():
    for gate_id in PASSING_G2:
        assert validate_report(_g2(gate_id)) == [], gate_id
    contradictions = {
        "G2.1 a logits event missing": _g2("G2.1", logits_events=27),
        "G2.1 wrote a non-logits event": _g2("G2.1", other_events_written=1),
        "G2.1 a logits event failed to recompute": _g2("G2.1", sample_matches=19),
        "G2.1 logits bytes 2% over the formula": _g2("G2.1", logits_bytes=958_314_578),
        "G2.2 a readout differed from the trace": _g2("G2.2", sample_matches=199),
        "G2.2 builds differed": _g2("G2.2", builds_identical=False),
        "G2.3 planted token missed too often": _g2("G2.3", planted_recovery=0.98),
        "G2.3 lens above twice chance on random weights": _g2("G2.3", random_max_accuracy=3 / 4096),
        "G2.3 last-layer lens drifts from the logits": _g2("G2.3", last_layer_kl_max=2e-5),
        "G2.4 aligned part not credited": _g2("G2.4", planted_aligned_credit=0.98),
        "G2.4 perpendicular part credited": _g2("G2.4", planted_perpendicular_quiet=0.98),
        "G2.4 decomposition error too large": _g2("G2.4", decomposition_error_max=2e-3),
        "G2.5 not a pass": _g2("G2.5", "fail"),
        "G2.6 builds differed": _g2("G2.6", builds_identical=False),
        "G2.6 the viewer reaches outside": _g2("G2.6", external_urls=1),
    }
    old_schema = _g2("G2.2")
    old_schema["schema_version"] = 1
    for key in ("git_commit", "git_dirty"):  # a schema-1 report has no commit fields, so only the phase rule can refuse it
        del old_schema[key]
    contradictions["G2 report on schema 1"] = old_schema
    for label, report in contradictions.items():
        assert validate_report(report), f"accepted: {label}"
    missing = _g2("G2.5")
    del missing["measurements"]["projected_seconds"]
    assert validate_report(missing), "a required measurement is missing (G2.5 always passes, so only the list catches it)"
    assert validate_report(_g2("G2.3", "fail", planted_recovery=0.5)) == [], "a fail may contradict nothing"
