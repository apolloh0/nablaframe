"""Phase 6 report rules: gate IDs, status rules and required measurements. Contract: docs/deltas/phase-6-delta-4.md."""

from __future__ import annotations

from nablaframe import reports
from nablaframe.reports import validate_report
from test_phase1_store_reports import _v2
from test_reports import passing_report

PASSING = {
    "G6.1": {"identities_checked": 18, "identities_exact": 18,
             "identities_by_site": {"induction topk blocks.0.hook_resid_post": [True, True, True]},
             "request_hashes_checked": 64, "request_hashes_distinct": True, "request_hashes_stable": True,
             "records_identical": True, "records_by_kind": {"name": [True, True]}},
    "G6.2": {"names": {}, "named_as_planted_min": 0.81, "follows_min": 0.93, "null_follows_max": 0.008,
             "twin_named_max": 0.0, "twin_follows_max": 0.018},
    "G6.3": {"edges": {}, "follows_min": 0.939, "null_follows_max": 0.029, "output_follows_min": 0.934,
             "kept_max": 0.211, "null_kept_min": 0.962, "residual_alone_max": 0.025, "labels_computed": True,
             "twin_excess_max": 0.0},
    "G6.4": {"targets": {}, "reading_ratio_max": 0.5, "within_min": 0.93, "planted_top_min": 0.78,
             "random_within_max": 0.016, "twin_planted_top_max": 0.0},
    "G6.5": {"names": {}, "slice_f1_min": 0.78, "slice_follows_min": 0.902, "seed_f1_min": 0.747,
             "seed_follows_min": 0.92, "step0_outside": True, "one_final_run": True, "start_offset_max": 1},
    "G6.6": {"blocks": {}, "seconds": {"naming": 15.0}},
}


def _g(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING[gate_id], **measurements})
    return report


def test_phase6_gate_ids():
    assert reports.PHASE6_GATE_IDS == ("G6.1", "G6.2", "G6.3", "G6.4", "G6.5", "G6.6")
    old = _g("G6.2")
    old["schema_version"] = 1
    del old["git_commit"], old["git_dirty"]
    assert validate_report(old), "a Phase 6 report must use schema version 2"


def test_phase6_status_rules():
    for gate_id in PASSING:
        assert validate_report(_g(gate_id)) == [], gate_id
    for label, report in {
        "G6.1 an identity wasn't exact": _g("G6.1", identities_exact=17),
        "G6.1 no identity checked": _g("G6.1", identities_checked=0, identities_exact=0),
        "G6.1 two requests hashed alike": _g("G6.1", request_hashes_distinct=False),
        "G6.1 a request's hash moved": _g("G6.1", request_hashes_stable=False),
        "G6.1 a record rebuilt differently": _g("G6.1", records_identical=False),
        "G6.2 too few values named as planted": _g("G6.2", named_as_planted_min=0.69),
        "G6.2 no value named": _g("G6.2", named_as_planted_min=None),
        "G6.2 a set that doesn't steer": _g("G6.2", follows_min=0.79),
        "G6.2 no steering": _g("G6.2", follows_min=None),
        "G6.2 a null that steers": _g("G6.2", null_follows_max=0.051),
        "G6.2 the twin named": _g("G6.2", twin_named_max=0.11),
        "G6.2 the twin steered": _g("G6.2", twin_follows_max=0.051),
        "G6.3 an edge that doesn't follow": _g("G6.3", follows_min=0.79),
        "G6.3 a null edge that follows": _g("G6.3", null_follows_max=0.11),
        "G6.3 the output didn't follow": _g("G6.3", output_follows_min=0.79),
        "G6.3 no output read": _g("G6.3", output_follows_min=None),
        "G6.3 too much kept": _g("G6.3", kept_max=0.41),
        "G6.3 the null removed too much": _g("G6.3", null_kept_min=0.79),
        "G6.3 carried through the residual": _g("G6.3", residual_alone_max=0.11),
        "G6.3 no residual-alone pass": _g("G6.3", residual_alone_max=None),
        "G6.3 a judged edge labelled carried": _g("G6.3", labels_computed=False),
        "G6.3 the twin beat its null": _g("G6.3", twin_excess_max=0.11),
        "G6.4 a reading over its bar": _g("G6.4", reading_ratio_max=1.01),
        "G6.4 a median not within 128": _g("G6.4", reading_ratio_max=None),
        "G6.4 too few within 128": _g("G6.4", within_min=0.79),
        "G6.4 the planted set missed": _g("G6.4", planted_top_min=0.59),
        "G6.4 random order reproduced it": _g("G6.4", random_within_max=0.11),
        "G6.4 the twin found the planted set": _g("G6.4", twin_planted_top_max=0.11),
        "G6.5 a slice's F1 too low": _g("G6.5", slice_f1_min=0.49),
        "G6.5 a slice that doesn't steer": _g("G6.5", slice_follows_min=0.79),
        "G6.5 a seed's F1 too low": _g("G6.5", seed_f1_min=0.49),
        "G6.5 a seed that doesn't steer": _g("G6.5", seed_follows_min=0.79),
        "G6.5 the name held at step 0": _g("G6.5", step0_outside=False),
        "G6.5 a broken envelope": _g("G6.5", one_final_run=False),
        "G6.5 the envelope far from the probe": _g("G6.5", start_offset_max=3),
        "G6.5 no probe reading": _g("G6.5", start_offset_max=None),
        "G6.6 is recorded, not judged: it can't fail": _g("G6.6", "fail"),
    }.items():
        assert validate_report(report), f"accepted: {label}"
    at_bars = {"G6.2": dict(named_as_planted_min=0.7, follows_min=0.8, null_follows_max=0.05, twin_named_max=0.1,
                            twin_follows_max=0.05),
               "G6.3": dict(follows_min=0.8, null_follows_max=0.1, output_follows_min=0.8, kept_max=0.4,
                            null_kept_min=0.8, residual_alone_max=0.1, twin_excess_max=0.1),
               "G6.4": dict(reading_ratio_max=1.0, within_min=0.8, planted_top_min=0.6, random_within_max=0.1,
                            twin_planted_top_max=0.1),
               "G6.5": dict(slice_f1_min=0.5, slice_follows_min=0.8, seed_f1_min=0.5, seed_follows_min=0.8,
                            start_offset_max=2)}
    for gate_id, values in at_bars.items():
        assert validate_report(_g(gate_id, **values)) == [], f"{gate_id} at its bars passes"
    assert validate_report(_g("G6.3", "fail", follows_min=0.1)) == []
    for gate_id, key in (("G6.1", "identities_by_site"), ("G6.1", "records_by_kind"), ("G6.1", "request_hashes_checked"),
                         ("G6.2", "names"), ("G6.3", "edges"), ("G6.3", "labels_computed"), ("G6.4", "targets"),
                         ("G6.5", "names"), ("G6.5", "start_offset_max"), ("G6.6", "blocks"), ("G6.6", "seconds")):
        missing = _g(gate_id)
        del missing["measurements"][key]
        assert validate_report(missing), f"{gate_id} missing {key}"
