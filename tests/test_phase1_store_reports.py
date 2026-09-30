"""Work package 1: Phase 0 carry-overs. Atomic shards, one writer per store, report schema 2.

Spec: docs/specs/phase-1.md, "Carried over from Phase 0".
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import nablaframe.trace.store as store_mod
from conftest import REPO_ROOT, make_event
from nablaframe.reports import validate_report
from nablaframe.trace.store import AppendOnlyViolation, StoreError, TraceStore
from test_reports import passing_report

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_shard_write_is_atomic_under_crash(tmp_path, monkeypatch):
    store = TraceStore(tmp_path)
    tensor = np.arange(6, dtype=np.float32)
    real_save = store_mod.save_file

    def dies_mid_write(tensors, path, *args, **kwargs):
        Path(path).write_bytes(b'\x08\x00\x00\x00\x00\x00\x00\x00{"tens')  # a truncated shard
        raise OSError("killed while writing the shard")

    monkeypatch.setattr(store_mod, "save_file", dies_mid_write)
    with pytest.raises(OSError):
        store.append(make_event(step=7), tensor)
    assert (tmp_path / "events.jsonl").read_bytes() == b""
    assert not list((tmp_path / "shards").glob("*.safetensors")), "a partial shard never carries its final name"

    monkeypatch.setattr(store_mod, "save_file", real_save)
    event = store.append(make_event(step=7), tensor)
    assert np.array_equal(store.load_tensor(event), tensor)


def test_second_writer_is_refused(tmp_path):
    first = TraceStore(tmp_path)
    stale = TraceStore(tmp_path)  # opened before any write, so its view of the log is empty
    first.append(make_event(step=1))
    assert [e.event_id for e in TraceStore(tmp_path).events()] == [e.event_id for e in first.events()], \
        "reading needs no lock"
    with pytest.raises(StoreError):
        stale.append(make_event(step=2))

    first.close()
    with pytest.raises(AppendOnlyViolation):
        stale.append(make_event(step=1))  # once it holds the lock, it rereads the log and sees the first write
    stale.append(make_event(step=2))
    stale.close()


def test_readonly_store_refuses_appends(tmp_path):
    with TraceStore(tmp_path) as writer:
        writer.append(make_event(step=1))
    reader = TraceStore(tmp_path, mode="r")
    assert len(reader.events()) == 1
    with pytest.raises(StoreError):
        reader.append(make_event(step=2))
    with pytest.raises(StoreError):
        reader.supersede(reader.events()[0].event_id, make_event(step=3))
    with pytest.raises(FileNotFoundError):
        TraceStore(tmp_path / "missing", mode="r")


def _v2(report: dict) -> dict:
    out = copy.deepcopy(report)
    out.update(schema_version=2, git_commit=COMMIT, git_dirty=False)
    return out


def test_report_v2_requires_git_commit():
    assert validate_report(passing_report("G0.3")) == [], "version 1 reports stay valid"
    good = _v2(passing_report("G0.3"))
    assert validate_report(good) == []
    for change in ({"git_commit": None}, {"git_commit": "abc"}, {"git_commit": COMMIT.upper()}, {"git_dirty": "no"}):
        assert validate_report({**good, **change}), change
    missing = dict(good)
    del missing["git_commit"]
    assert validate_report(missing)
    assert validate_report({**passing_report("G0.3"), "git_commit": COMMIT}), "version 1 has no commit fields"


PASSING_G1 = {
    "G1.1": {"vocab_selected": 4096, "unk_rate_4096": 0.004, "unk_rate_8192": 0.001, "unk_rate_4096_cased": 0.012,
             "builds_identical": True, "train_tokens": 600_000_000, "validation_tokens": 6_000_000, "stories_dropped": 120},
    "G1.2": {"n_params": 4273664, "n_params_expected": 4273664, "max_abs_logit_diff": 3e-6, "top1_agreement": 1.0,
             "positions_near_tie": 0},
    "G1.3": {"hash_run_a": "a" * 64, "hash_run_b": "a" * 64, "hash_resumed": "a" * 64, "identical": True,
             "max_abs_param_diff": 0.0, "single_threaded": False},
    "G1.4": {"steps": 12208, "tokens": 100_007_936, "nonfinite_seen": False, "checkpoints_written": 28,
             "checkpoints_scheduled": 28, "final_val_loss": 1.9, "bigram_val_loss": 3.4},
    "G1.5": {"bitwise_equal": True, "steps_checked": [0, 6000, 12208], "defect_detected": True},
    "G1.6": {"events_recorded": 980, "sample_size": 50, "sample_matches": 50, "retrace_new_events": 0,
             "store_bytes": 2_760_000_000, "store_bytes_expected": 2_760_000_000},
}


def _g1(gate_id: str, status: str = "pass", **measurements) -> dict:
    report = _v2(passing_report("G0.3"))
    report.update(gate_id=gate_id, status=status, env_name="nf-jax", measurements={**PASSING_G1[gate_id], **measurements})
    return report


def test_phase1_status_rules():
    for gate_id in PASSING_G1:
        assert validate_report(_g1(gate_id)) == [], gate_id
    old_schema = _g1("G1.1")
    old_schema["schema_version"] = 1
    for key in ("git_commit", "git_dirty"):
        del old_schema[key]
    contradictions = {
        "G1 report on schema 1": old_schema,
        "G1.1 picked 8,192 when 4,096 met the rule": _g1("G1.1", vocab_selected=8192),
        "G1.1 picked 4,096 above the 1% rule": _g1("G1.1", unk_rate_4096=0.02),
        "G1.1 builds differed": _g1("G1.1", builds_identical=False),
        "G1.2 parameter count off": _g1("G1.2", n_params=4273665),
        "G1.2 logits too far": _g1("G1.2", max_abs_logit_diff=2e-4),
        "G1.3 flag contradicts hashes": _g1("G1.3", hash_resumed="b" * 64),
        "G1.4 no better than bigram": _g1("G1.4", final_val_loss=3.4),
        "G1.4 checkpoint missing": _g1("G1.4", checkpoints_written=27),
        "G1.5 planted defect not caught": _g1("G1.5", defect_detected=False),
        "G1.6 recompute mismatch": _g1("G1.6", sample_matches=49),
        "G1.6 retrace wrote events": _g1("G1.6", retrace_new_events=1),
        "G1.6 store 20% over budget": _g1("G1.6", store_bytes=3_312_000_000),
    }
    for label, report in contradictions.items():
        assert validate_report(report), f"accepted: {label}"
    assert validate_report(_g1("G1.3", "conditional_pass", hash_resumed="b" * 64, identical=False,
                               max_abs_param_diff=4e-7)) == []


def test_phase0_reports_still_validate():
    reports = sorted((REPO_ROOT / "reports" / "phase0").glob("g0*.json"))
    for path in reports:
        assert validate_report(json.loads(path.read_text())) == [], path.name
