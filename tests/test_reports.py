"""Gate report tests: a report is valid only if its status agrees with its own numbers.

Spec: docs/specs/phase-0.md, "Gates". Rules: nablaframe/reports.py module docstring.
"""

from __future__ import annotations

import copy
import json

import pytest

from conftest import REPO_ROOT
from nablaframe.reports import GATE_IDS, REQUIRED_FIELDS, REQUIRED_MEASUREMENTS, validate_report, write_report

PASSING_MEASUREMENTS = {
    "G0.1": {"steps": 1000, "eval_loss_initial": 8.3, "eval_loss_final": 4.1, "loss_drop_fraction": 0.506, "nonfinite_seen": False},
    "G0.2": {"hash_run_a": "a" * 64, "hash_run_b": "a" * 64, "hash_resumed": "a" * 64, "identical": True,
             "max_abs_param_diff": 0.0, "single_threaded": False},
    "G0.3": {"tokens_per_s_S": 41000.0, "tokens_per_s_M": 9800.0},
    "G0.4": {"max_abs_logit_diff": 2.4e-5, "top1_agreement": 1.0, "positions_compared": 1024, "positions_near_tie": 3},
    "G0.5": {"runs": True, "declined_ops": [], "max_rel_loss_diff": 0.004, "tokens_per_s": 160000.0},
    "G0.6": {"bitwise_equal": False, "max_abs_logit_diff": 4.8e-7, "seeds": [0, 1, 2]},
}
ENVS = {"G0.1": "nf-jax", "G0.2": "nf-jax", "G0.3": "nf-jax", "G0.4": "nf-torch", "G0.5": "nf-metal", "G0.6": "nf-jax"}


def passing_report(gate_id: str) -> dict:
    return {
        "schema_version": 1,
        "gate_id": gate_id,
        "status": "pass",
        "measurements": copy.deepcopy(PASSING_MEASUREMENTS[gate_id]),
        "env_name": ENVS[gate_id],
        "env_lock_sha256": "0123456789abcdef" * 4,
        "xla_flags": "",
        "thread_count": 12,
        "chip": "Apple M3 Max",
        "timestamp_utc": "2026-09-27T12:00:00Z",
    }


def _with(report: dict, **changes) -> dict:
    out = copy.deepcopy(report)
    for key, value in changes.items():
        if key.startswith("m_"):
            out["measurements"][key[2:]] = value
        else:
            out[key] = value
    return out


def _invalid_cases() -> list[tuple[str, dict]]:
    base = passing_report("G0.1")
    cases = []
    for field in REQUIRED_FIELDS:
        missing = copy.deepcopy(base)
        del missing[field]
        cases.append((f"missing {field}", missing))
    for gate_id in GATE_IDS:
        for key in REQUIRED_MEASUREMENTS[gate_id]:
            report = passing_report(gate_id)
            del report["measurements"][key]
            cases.append((f"{gate_id} missing measurement {key}", report))
    cases += [
        ("extra field", _with(base, notes="hand-written")),
        ("schema version", _with(base, schema_version=2)),
        ("unknown gate", _with(base, gate_id="G0.7")),
        ("unknown status", _with(base, status="passed")),
        ("unknown env", _with(base, env_name="nf-cuda")),
        ("short lock hash", _with(base, env_lock_sha256="abc")),
        ("zero threads", _with(base, thread_count=0)),
        ("empty chip", _with(base, chip="")),
        ("timestamp without Z", _with(base, timestamp_utc="2026-09-27T12:00:00")),
        ("unparseable timestamp", _with(base, timestamp_utc="yesterday Z")),
        ("NaN measurement", _with(base, m_eval_loss_final=float("nan"))),
        ("G0.1 pass, loss barely moved", _with(base, m_loss_drop_fraction=0.2)),
        ("G0.1 pass, NaN seen", _with(base, m_nonfinite_seen=True)),
        ("G0.2 pass, hashes differ", _with(passing_report("G0.2"), m_identical=False, m_max_abs_param_diff=1e-7)),
        ("G0.2 conditional, diff too large", _with(passing_report("G0.2"), status="conditional_pass",
                                                   m_identical=False, m_max_abs_param_diff=1e-5)),
        ("G0.2 conditional, but identical", _with(passing_report("G0.2"), status="conditional_pass")),
        ("conditional outside G0.2", _with(passing_report("G0.3"), status="conditional_pass")),
        ("G0.3 pass, too slow", _with(passing_report("G0.3"), m_tokens_per_s_S=24999.0)),
        ("G0.4 pass, logits too far", _with(passing_report("G0.4"), m_max_abs_logit_diff=2e-3)),
        ("G0.4 pass, top-1 disagreement", _with(passing_report("G0.4"), m_top1_agreement=0.999)),
        ("G0.5 pass, loss drifted", _with(passing_report("G0.5"), m_max_rel_loss_diff=0.03)),
        ("G0.6 cannot fail", _with(passing_report("G0.6"), status="fail")),
    ]
    return cases


def test_gate_reports_validate(tmp_path):
    for gate_id in GATE_IDS:
        assert validate_report(passing_report(gate_id)) == [], gate_id

    # Honest failures and G0.2's conditional pass are valid reports.
    assert validate_report(_with(passing_report("G0.3"), status="fail", m_tokens_per_s_S=12000.0)) == []
    assert validate_report(_with(passing_report("G0.2"), status="conditional_pass",
                                 m_identical=False, m_max_abs_param_diff=4e-7)) == []

    for label, report in _invalid_cases():
        assert validate_report(report), f"accepted an invalid report: {label}"

    # write_report refuses invalid reports and existing files, and writes nothing then.
    target = tmp_path / "g03.json"
    with pytest.raises(ValueError):
        write_report(_with(passing_report("G0.3"), m_tokens_per_s_S=100.0), target)
    assert not target.exists()
    write_report(passing_report("G0.3"), target)
    assert json.loads(target.read_text()) == passing_report("G0.3")
    with pytest.raises(FileExistsError):
        write_report(passing_report("G0.3"), target)
    write_report(_with(passing_report("G0.3"), m_tokens_per_s_M=9900.0), target, overwrite=True)

    # Every committed gate report validates. None exist until apolloh0 runs the gates.
    for path in sorted((REPO_ROOT / "reports" / "phase0").glob("g0*.json")):
        problems = validate_report(json.loads(path.read_text()))
        assert problems == [], f"{path.name}: {problems}"
