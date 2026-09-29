"""D3: a tolerance record applies only to its own dtype, and cites a real gate report path.

Spec: docs/deltas/phase-0-delta-2.md, D3. These are the discrimination tests the
oracle-change rule requires: both fail on the oracle as merged in PR #2.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import pinned_tolerance
from nablaframe.oracle.mode_equality import compare_logits
from nablaframe.oracle.tolerance import ToleranceRecord


def test_canonical_policy_rejects_tolerance_for_another_dtype():
    logits = np.ones((2, 3), dtype=np.float16)
    with pytest.raises(ValueError):
        compare_logits(logits, logits.copy(), "instrumented_canonical", pinned_tolerance())  # a float32 record

    matching = ToleranceRecord(1e-3, "reports/phase0/g06_mode_probe.json", "cpu", "float16", 0.0)
    assert compare_logits(logits, logits.copy(), "instrumented_canonical", matching).passed


def _record(source: object) -> ToleranceRecord:
    return ToleranceRecord(1e-6, source, "cpu", "float32", 0.0)


@pytest.mark.parametrize("source", [
    "reports/phase0/g06_mode_probe.json",
    "reports/phase12/g01-cpu_train.json",
])
def test_source_report_accepts_gate_report_paths(source):
    assert _record(source).source_report == source


@pytest.mark.parametrize("source", [
    "reports/../x.json",
    "reports/phase0/../../x.json",
    "reports/phase0/sub/g06.json",
    "reports/g06_mode_probe.json",
    "reports/phaseX/g06.json",
    "reports/phase0/g06 mode.json",
    "reports/phase0/.json",
    "reports\\phase0\\g06.json",
])
def test_source_report_rejects_anything_else(source):
    with pytest.raises(ValueError):
        _record(source)
