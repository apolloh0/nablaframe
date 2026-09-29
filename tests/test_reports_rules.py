"""Gate report rules added in Phase 0 delta 2.

Spec: docs/deltas/phase-0-delta-2.md, "Contract changes", reports.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nablaframe.reports import validate_report, write_report
from test_reports import passing_report


def _with(report: dict, status: str | None = None, **measurements) -> dict:
    out = json.loads(json.dumps(report))
    if status is not None:
        out["status"] = status
    out["measurements"].update(measurements)
    return out


def test_g01_pass_requires_the_full_run():
    assert validate_report(_with(passing_report("G0.1"), steps=1000)) == []
    assert validate_report(_with(passing_report("G0.1"), steps=800))
    # A run that stopped early is an honest fail.
    assert validate_report(_with(passing_report("G0.1"), "fail", steps=800)) == []


def test_g02_identical_flag_must_agree_with_hashes():
    base = passing_report("G0.2")
    assert validate_report(base) == []
    # Claims identical, hashes differ: invalid under every status.
    for status in ("pass", "fail"):
        assert validate_report(_with(base, status, hash_resumed="b" * 64)), status
    # Hashes differ and the flag says so: an honest conditional pass.
    assert validate_report(_with(base, "conditional_pass", hash_resumed="b" * 64,
                                 identical=False, max_abs_param_diff=4e-7)) == []
    # Hashes agree but the flag says they don't: invalid.
    assert validate_report(_with(base, "fail", identical=False, max_abs_param_diff=0.0))


def test_write_report_creates_missing_folders(tmp_path):
    target = tmp_path / "reports" / "phase0" / "g03_throughput.json"
    write_report(passing_report("G0.3"), target)
    assert json.loads(target.read_text()) == passing_report("G0.3")


def test_measurements_allow_null_but_not_numpy_scalars(tmp_path):
    failed_probe = _with(passing_report("G0.5"), "fail", runs=False, max_rel_loss_diff=None,
                         tokens_per_s=None, declined_ops=[])
    failed_probe["measurements"].update(macos_version="15.6", load_error="dlopen failed")
    assert validate_report(failed_probe) == []

    with_numpy = passing_report("G0.3")
    with_numpy["measurements"]["tokens_per_s_M"] = np.int64(9800)
    assert validate_report(with_numpy)
    with pytest.raises(ValueError):
        write_report(with_numpy, tmp_path / "g03.json")
    assert not (tmp_path / "g03.json").exists()
