"""Phase 2 work package 4: direct logit attribution, an exact decomposition (D14).

Spec: docs/specs/phase-2.md, "Views" and D14. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.views.attribution import attribution, attribution_from_trace, parts_from_trace  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402


def test_attribution_parts_sum_to_residual(tmp_path):
    run = traced_run(tmp_path)
    reader = TraceReader(run.root)
    parts, final = parts_from_trace(reader, 0, n_layers=TINY.n_layers)
    assert set(parts) == {"hook_embed", "hook_pos_embed", "blocks.0.hook_attn_out", "blocks.0.hook_mlp_out",
                          "blocks.1.hook_attn_out", "blocks.1.hook_mlp_out"}
    assert np.abs(sum(p.astype(np.float64) for p in parts.values()) - final).max() <= 1e-5

    contributions, error = attribution_from_trace(reader, 0, run.params, run.probe, n_layers=TINY.n_layers)
    assert set(contributions) == set(parts) | {"ln_bias"}
    assert type(error) is float and 0 <= error <= 1e-4
    for values in contributions.values():
        assert np.isnan(values[:, -1]).all(), "the last position has no next token"


def test_attribution_recovers_planted_component():
    """Two equally strong parts: one along the target token's direction, one exactly perpendicular to it.

    Attribution must credit the first and give the second essentially nothing. A test that only
    checked the planted part's share of the total would pass for any dominant part, whatever its direction.
    """
    rng = np.random.default_rng(0)
    d, vocab, n = 32, 64, 1_000
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(1, n))
    u = wte[targets].astype(np.float64)
    uc = u - u.mean(-1, keepdims=True)
    r = rng.standard_normal((1, n, d))
    r -= r.mean(-1, keepdims=True)
    r -= (r * uc).sum(-1, keepdims=True) / (uc * uc).sum(-1, keepdims=True) * uc  # perpendicular to the target
    aligned = 6.0 * u
    perpendicular = r * (np.linalg.norm(aligned, axis=-1, keepdims=True) / np.linalg.norm(r, axis=-1, keepdims=True))
    parts = {"aligned": aligned.astype(np.float32), "perpendicular": perpendicular.astype(np.float32),
             "quiet": (0.01 * rng.standard_normal((1, n, d))).astype(np.float32)}
    out = attribution(parts, sum(parts.values()), np.ones(d, np.float32), np.zeros(d, np.float32), wte, targets)
    assert (out["aligned"] > 0).mean() >= 0.99, "the part along the target's direction is credited"
    assert (np.abs(out["perpendicular"]) <= 1e-3 * np.abs(out["aligned"])).mean() >= 0.99, \
        "an equally strong part perpendicular to the target gets essentially nothing"


def test_attribution_refuses_a_different_probe(tmp_path):
    """A wrong probe means wrong targets, and the meter can't see it: both sides use the same targets."""
    run = traced_run(tmp_path)
    with pytest.raises(ValueError):
        attribution_from_trace(TraceReader(run.root), 0, run.params, run.probe[::-1], n_layers=TINY.n_layers)


def test_attribution_error_is_reported(tmp_path):
    run = traced_run(tmp_path)
    _, error = attribution_from_trace(TraceReader(run.root), 0, run.params, run.probe, n_layers=TINY.n_layers)
    assert np.isfinite(error) and error <= 1e-4
