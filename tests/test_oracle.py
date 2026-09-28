"""Oracle tests: a green oracle must mean something, so it is tested against planted defects.

Spec: docs/specs/phase-0.md, "Mode-equality oracle".
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import NumpyToy, pinned_tolerance
from nablaframe.oracle.mode_equality import check_mode_equality, compare_logits
from nablaframe.oracle.tolerance import ToleranceRecord

TAPS = ("blocks.0.hook_resid_post", "blocks.1.hook_resid_post", "ln_final.hook_normalized")
SEEDS = (0, 1, 2)


def _run(model, seed, policy, tolerance=None):
    return check_mode_equality(model, NumpyToy.init(seed), NumpyToy.tokens(seed), TAPS, policy, tolerance)


def test_oracle_noop_tap_passes():
    model = NumpyToy()  # identity tap
    for seed in SEEDS:
        bitwise = _run(model, seed, "bitwise")
        assert bitwise.passed, bitwise.reason
        assert bitwise.n_mismatched == 0 and bitwise.max_abs_diff == 0.0 and bitwise.reason == ""

        canonical = _run(model, seed, "instrumented_canonical", pinned_tolerance())
        assert canonical.passed, canonical.reason


def _adds_at_resid_post0(name, x):
    if name == "blocks.0.hook_resid_post":
        x += np.float32(1e-3)  # in place, as the planted defect in the spec
    return x


def test_oracle_mutating_tap_fails():
    model = NumpyToy(tap=_adds_at_resid_post0)
    for seed in SEEDS:
        bitwise = _run(model, seed, "bitwise")
        assert not bitwise.passed
        assert bitwise.n_mismatched > 0 and bitwise.reason

        canonical = _run(model, seed, "instrumented_canonical", pinned_tolerance())
        assert not canonical.passed
        assert canonical.max_abs_diff > 1e-5, "the perturbation must reach the logits"


def test_oracle_one_ulp_fails_under_bitwise():
    logits = np.random.default_rng(0).standard_normal((2, 5, 16)).astype(np.float32)
    nudged = logits.copy()
    nudged[1, 3, 7] = np.nextafter(nudged[1, 3, 7], np.float32(np.inf))

    result = compare_logits(logits, nudged, "bitwise")
    assert not result.passed
    assert result.n_mismatched == 1
    assert 0 < result.max_abs_diff < 1e-6

    # The same one-ulp difference is inside a pinned 1e-6 tolerance: the policies differ.
    assert compare_logits(logits, nudged, "instrumented_canonical", pinned_tolerance(1e-6)).passed

    # Bitwise means bit patterns: signed zeros differ.
    zeros = np.zeros(4, dtype=np.float32)
    assert not compare_logits(zeros, -zeros, "bitwise").passed

    # Non-finite values and mismatched shapes or dtypes fail under both policies.
    with_nan = logits.copy()
    with_nan[0, 0, 0] = np.nan
    assert not compare_logits(with_nan, with_nan.copy(), "bitwise").passed
    assert not compare_logits(with_nan, with_nan.copy(), "instrumented_canonical", pinned_tolerance()).passed
    assert not compare_logits(logits, logits.astype(np.float64), "bitwise").passed
    assert not compare_logits(logits, logits[:1], "instrumented_canonical", pinned_tolerance()).passed


def test_tolerance_record_requires_source_report():
    good = pinned_tolerance()
    assert ToleranceRecord.from_dict(good.to_dict()) == good

    for bad_source in ("", "notes.txt", "reports/phase0/g06.txt", "/abs/reports/phase0/g06.json", "g06.json", None):
        with pytest.raises(ValueError):
            ToleranceRecord(value=1e-6, source_report=bad_source, backend="cpu", dtype="float32", measured_max_abs_diff=0.0)

    for bad in (dict(value=-1.0), dict(value=float("nan")), dict(value=1e-7, measured_max_abs_diff=1e-6),
                dict(backend=""), dict(dtype="int32")):
        fields = dict(value=1e-6, source_report="reports/phase0/g06_mode_probe.json", backend="cpu",
                      dtype="float32", measured_max_abs_diff=0.0)
        fields.update(bad)
        with pytest.raises(ValueError):
            ToleranceRecord(**fields)

    # The canonical policy never runs on an unpinned tolerance; bitwise takes none.
    logits = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compare_logits(logits, logits, "instrumented_canonical")
    with pytest.raises(ValueError):
        compare_logits(logits, logits, "bitwise", good)
    with pytest.raises(ValueError):
        compare_logits(logits, logits, "approximately")
