"""Phase 7 report rules: gate IDs, status rules and required measurements. Contract: docs/deltas/phase-7-delta-3.md."""

from __future__ import annotations

from nablaframe import reports
from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {
    "G7.1": {"recompute_identical": True, "checkpoints_written": 45, "checkpoints_scheduled": 45,
             "events_recorded": 450, "events_expected": 450, "held_accuracy": 1.0, "train_accuracy": 1.0,
             "control_accuracy": 0.0001, "twin_accuracy": 0.0088, "chance": 0.00885, "frequencies": 5,
             "energy": 0.795, "twin_frequencies": 39, "logit_r2": 0.966},
    "G7.2": {"fvu": 0.017, "twin_fvu": 0.046, "names": {}, "records": 231, "seconds": {"dictionary": 60.0}},
    "G7.3": {"identities_checked": 24, "identities_exact": 24, "identities_by_run": {"induction": {}},
             "request_hashes_checked": 64, "request_hashes_distinct": True, "request_hashes_stable": True,
             "records_identical": True, "records_by_form": {"table": [True, True]}},
    "G7.4": {"crystallizations": {}, "kl_max": 0.134, "ablation_min": 4.14, "wrong_min": 4.42,
             "reads_agreement_min": 0.938, "answer_agreement_max": 0.008, "twin_within_max": 0.003,
             "envelope_ok": True},
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING[gate_id], **measurements})
    return report


def test_phase7_gate_ids():
    assert reports.PHASE7_GATE_IDS == ("G7.1", "G7.2", "G7.3", "G7.4")
    for gate_id in reports.PHASE7_GATE_IDS:
        assert set(reports.REQUIRED_MEASUREMENTS[gate_id]) == set(PASSING[gate_id]), gate_id
        old = _g(gate_id)
        old["schema_version"] = 1
        del old["git_commit"], old["git_dirty"]
        assert validate_report(old), f"a Phase 7 report must use schema version 2: {gate_id}"


def test_phase7_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    for label, report in {
        "G7.1 the recompute moved": _g("G7.1", recompute_identical=False),
        "G7.1 a checkpoint missing": _g("G7.1", checkpoints_written=44),
        "G7.1 an event missing": _g("G7.1", events_recorded=449),
        "G7.1 held-out accuracy too low": _g("G7.1", held_accuracy=0.989),
        "G7.1 no held-out accuracy": _g("G7.1", held_accuracy=None),
        "G7.1 position 0 predicts": _g("G7.1", control_accuracy=0.0178),
        "G7.1 the twin predicts": _g("G7.1", twin_accuracy=0.0178),
        "G7.1 no twin read": _g("G7.1", twin_accuracy=None),
        "G7.1 no clock": _g("G7.1", frequencies=9),
        "G7.1 no frequencies counted": _g("G7.1", frequencies=None),
        "G7.1 the logits aren't the clock's": _g("G7.1", logit_r2=0.799),
        "G7.1 no fit": _g("G7.1", logit_r2=None),
        "G7.2 is recorded, not judged: it can't fail": _g("G7.2", "fail"),
        "G7.3 no identity checked": _g("G7.3", identities_checked=0, identities_exact=0),
        "G7.3 an identity wasn't exact": _g("G7.3", identities_exact=23),
        "G7.3 two requests hashed alike": _g("G7.3", request_hashes_distinct=False),
        "G7.3 a request's hash moved": _g("G7.3", request_hashes_stable=False),
        "G7.3 a record rebuilt differently": _g("G7.3", records_identical=False),
        "G7.4 a crystallization that doesn't hold": _g("G7.4", kl_max=0.251),
        "G7.4 no KL": _g("G7.4", kl_max=None),
        "G7.4 a component that doesn't move the prediction": _g("G7.4", ablation_min=0.99),
        "G7.4 a wrong function that holds": _g("G7.4", wrong_min=0.99),
        "G7.4 no wrong function": _g("G7.4", wrong_min=None),
        "G7.4 the counterfactual doesn't follow what it reads": _g("G7.4", reads_agreement_min=0.79),
        "G7.4 the counterfactual follows the answer": _g("G7.4", answer_agreement_max=0.11),
        "G7.4 no counterfactual": _g("G7.4", reads_agreement_min=None),
        "G7.4 the twin tells functions apart": _g("G7.4", twin_within_max=0.051),
        "G7.4 a broken envelope": _g("G7.4", envelope_ok=False),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    at_bars = {"G7.1": dict(held_accuracy=0.99, control_accuracy=0.0177, twin_accuracy=0.0177, chance=0.00885,
                            frequencies=8, logit_r2=0.8),
               "G7.3": dict(identities_checked=1, identities_exact=1),
               "G7.4": dict(kl_max=0.25, ablation_min=1.0, wrong_min=1.0, reads_agreement_min=0.8,
                            answer_agreement_max=0.1, twin_within_max=0.05)}
    for gate_id, values in at_bars.items():
        assert validate_report(_g(gate_id, **values)) == [], f"{gate_id} at its bars passes"
    for gate_id in ("G7.1", "G7.3", "G7.4"):
        assert validate_report(_g(gate_id, "fail", **{next(iter(PASSING[gate_id])): None})) == [], \
            f"{gate_id} may fail"
    for gate_id, key in (("G7.1", "energy"), ("G7.1", "twin_frequencies"), ("G7.1", "train_accuracy"),
                         ("G7.2", "names"), ("G7.2", "records"), ("G7.2", "twin_fvu"), ("G7.3", "identities_by_run"),
                         ("G7.3", "records_by_form"), ("G7.4", "crystallizations"), ("G7.4", "envelope_ok")):
        missing = _g(gate_id)
        del missing["measurements"][key]
        assert validate_report(missing), f"{gate_id} missing {key}"
