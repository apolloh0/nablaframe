"""Phase 4 package 4: induction targets and scores, head ablation, and the formation steps."""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo.induction import head_zero, induction_scores, induction_targets, steepest_drop, steepest_rise, uniform_baseline


def test_induction_targets():
    w = np.array([[5, 6, 7, 5, 6, 9, 5]], np.int32)
    assert induction_targets(w, [3, 4, 5, 6]).tolist() == [[1, 2, -1, 4]], "after the most recent earlier occurrence"


def test_induction_scores():
    w = np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]], np.int32)
    positions = [3, 4]
    T = w.shape[1]
    causal = np.tril(np.ones((T, T))) / np.arange(1, T + 1)[:, None]
    pattern = np.broadcast_to(causal, (2, 3, T, T)).copy()
    targets = induction_targets(w, positions)
    for b in range(2):
        for i, q in enumerate(positions):
            pattern[b, 1, q] = 0.3 / q
            pattern[b, 1, q, targets[b, i]] = 0.7
            pattern[b, 1, q, q] = 0.0
    scores = induction_scores(pattern, w, positions)
    assert scores.shape == (3,) and scores[1] == pytest.approx(0.7)
    assert scores[0] == pytest.approx(uniform_baseline(w, positions)) == pytest.approx((1 / 4 + 1 / 5) / 2)


def test_formation_steps_and_head_ablation():
    steps = [0, 50, 100, 150, 200]
    assert steepest_drop(steps, [4.0, 3.9, 2.0, 1.8, 1.7]) == 100
    assert steepest_rise(steps, [0.1, 0.1, 0.15, 0.6, 0.62]) == 150
    assert steepest_drop(steps, [3.0, 2.0, 1.0, 1.0, 1.0]) == 50, "ties go to the earlier step"
    from nablaframe.interventions.spec import InterventionSpec, edit
    assert head_zero(1, 2) == InterventionSpec((edit("blocks.1.attn.hook_z", "all", "zero", head=2),))
