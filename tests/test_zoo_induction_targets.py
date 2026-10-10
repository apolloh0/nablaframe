"""Phase 4 delta 3: attention to every induction target, and the induction patch's sources (D36). Needs no JAX.

Contract: docs/deltas/phase-4-delta-3.md, "src/nablaframe/zoo/induction.py".
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo import induction  # functions are looked up per test, so a missing one fails its test alone

# Period 3: positions 3 to 5 have one earlier occurrence of their token, 6 and 7 have two.
W = np.array([[4, 5, 6, 4, 5, 6, 4, 5, 6]], np.int32)


def test_every_position_after_an_earlier_occurrence_is_a_target():
    mask = induction.induction_target_mask(W, [2, 3, 6, 7])
    assert mask.dtype == bool and mask.shape == (1, 4, 9)
    assert [np.flatnonzero(row).tolist() for row in mask[0]] == [[], [1], [1, 4], [2, 5]]


def test_scores_sum_the_attention_on_every_target():
    T, positions = 8, [3, 6, 7]  # the model reads 8 tokens; the window holds 9
    pattern = np.zeros((1, 3, T, T))
    for q in range(T):
        pattern[0, :, q, : q + 1] = 1 / (q + 1)  # uniform causal attention on every head
    pattern[0, 1, 6] = 0.0
    pattern[0, 1, 6, [1, 4]] = [0.25, 0.5]  # split across both targets, the most recent holding more
    pattern[0, 1, 6, 6] = 0.25
    scores = induction.induction_scores(pattern, W, positions)
    assert scores.shape == (3,) and scores.dtype == np.float64
    uniform = (1 / 4 + 2 / 7 + 2 / 8) / 3
    assert scores[0] == pytest.approx(uniform) == pytest.approx(induction.uniform_baseline(W, positions)), \
        "uniform attention scores the uniform baseline, which counts every target"
    assert scores[1] == pytest.approx((1 / 4 + 0.75 + 2 / 8) / 3), "both targets of position 6 count"
    assert induction.induction_scores(pattern, W[:, :T], positions) == pytest.approx(scores), "a window as long as the pattern"
    assert induction.induction_scores(pattern, W, [2, *positions]) == pytest.approx(scores), "a position without a target is left out"
    assert induction.uniform_baseline(W, [2, 3]) == pytest.approx(1 / 4), "pairs without a target are left out"
    for no_target in ([0, 1, 2],):
        with pytest.raises(ValueError):
            induction.induction_scores(pattern, W, no_target)
        with pytest.raises(ValueError):
            induction.uniform_baseline(W, no_target)


def test_new_token_sources():
    windows = np.array([np.arange(64) % 20, (np.arange(64) % 20) + 30], np.int32)
    sources = induction.new_token_sources(windows, 13, np.random.default_rng(13), 64)
    mirror = np.random.default_rng(13)
    expected = [int(mirror.choice(np.setdiff1d(np.arange(64), row))) for row in windows]
    assert sources.dtype == np.int32 and sources.shape == windows.shape
    assert sources[:, 13].tolist() == expected, "drawn in window order from the ids each window doesn't use"
    assert all(int(s[13]) not in w.tolist() for s, w in zip(sources, windows))
    assert np.array_equal(np.delete(sources, 13, axis=1), np.delete(windows, 13, axis=1)), "nothing else changes"
    assert windows[0, 13] == 13, "the windows themselves are left alone"
