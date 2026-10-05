"""Phase 3 package 4: the J-lens experiment's directions and scales.

Spec: docs/specs/phase-3.md, "The first experiment".
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.interventions.jlens_experiment import SCALE_FACTORS, jlens_direction, logit_lens_direction, steering_scales


def test_jlens_direction_is_the_pullback_of_the_unembedding():
    rng = np.random.default_rng(0)
    wte = rng.standard_normal((32, 16)).astype(np.float32)
    J = rng.standard_normal((16, 16)).astype(np.float32)
    v = jlens_direction(J, wte, 5)
    assert v.dtype == np.float32 and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    expected = J.T.astype(np.float64) @ wte[5]
    assert np.allclose(v, expected / np.linalg.norm(expected), atol=1e-6), "the direction J carries most onto token 5"
    gain = float(wte[5] @ J @ v)
    for _ in range(200):
        u = rng.standard_normal(16); u /= np.linalg.norm(u)
        assert wte[5] @ J @ u <= gain + 1e-5
    assert np.allclose(jlens_direction(np.eye(16, dtype=np.float32), wte, 5), logit_lens_direction(wte, 5), atol=1e-6)
    assert np.linalg.norm(logit_lens_direction(wte, 5)) == pytest.approx(1.0, abs=1e-6)


def test_steering_scales_follow_the_residual_norm():
    resid = np.random.default_rng(1).standard_normal((3, 7, 16)).astype(np.float32)
    mean_norm = float(np.linalg.norm(resid.astype(np.float64), axis=-1).mean())
    assert SCALE_FACTORS == (0.25, 0.5, 1.0)
    scales = steering_scales(resid)
    assert all(type(s) is float for s in scales)
    assert scales == pytest.approx(tuple(f * mean_norm for f in SCALE_FACTORS), rel=1e-12)
