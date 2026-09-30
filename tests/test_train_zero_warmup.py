"""Pins PR #11's ruled behavior: with no warmup steps, the schedule starts at the peak rate.

Spec: docs/deltas/phase-1-delta-3.md. Needs the jax lane.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

from nablaframe.train.loop import TrainConfig, learning_rate  # noqa: E402


def test_zero_warmup_starts_at_peak():
    config = TrainConfig(lr=1e-3, warmup_steps=0, total_steps=100, min_lr_ratio=0.1)
    assert learning_rate(config, 0) == pytest.approx(1e-3, rel=1e-5)
    assert learning_rate(config, 100) == pytest.approx(1e-4, rel=1e-5)
