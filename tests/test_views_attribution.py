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
    rng = np.random.default_rng(0)
    d, vocab, n = 32, 64, 1_000
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(1, n))
    planted = (6.0 * wte[targets]).astype(np.float32)
    quiet = {f"q{i}": (0.01 * rng.standard_normal((1, n, d))).astype(np.float32) for i in range(3)}
    parts = {"planted": planted, **quiet}
    final = sum(parts.values())
    out = attribution(parts, final, np.ones(d, np.float32), np.zeros(d, np.float32), wte, targets)
    total = sum(out.values())
    assert (out["planted"] / total >= 0.99).mean() >= 0.99


def test_attribution_error_is_reported(tmp_path):
    run = traced_run(tmp_path)
    _, error = attribution_from_trace(TraceReader(run.root), 0, run.params, run.probe, n_layers=TINY.n_layers)
    assert np.isfinite(error) and error <= 1e-4
