"""Phase 2 work package 2: readout views, exact decodes of the trace.

Spec: docs/specs/phase-2.md, "Views". Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.trace.store import TraceStore  # noqa: E402
from nablaframe.views.readouts import TraceReader, attention_pattern, per_token_loss, residual_norms  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402


def _traced(run, tap):
    store = TraceStore(run.root, mode="r")
    [event] = [e for e in store.current() if e.tap == tap and e.step == 0]
    return store.load_tensor(event)


def test_readout_equals_traced_tensor(tmp_path):
    run = traced_run(tmp_path)
    reader = TraceReader(run.root)
    assert reader.steps() == (0,)
    pattern = _traced(run, "blocks.1.attn.hook_pattern")
    assert np.array_equal(attention_pattern(reader, 0, layer=1, head=0, sequence=1), pattern[1, 0])

    norms = residual_norms(reader, 0, n_layers=TINY.n_layers)
    assert norms.shape == (TINY.n_layers + 1, 2, TINY.seq) and norms.dtype == np.float32
    for i, tap in enumerate(["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre", "blocks.1.hook_resid_post"]):
        expected = np.linalg.norm(_traced(run, tap).astype(np.float64), axis=-1).astype(np.float32)
        assert np.array_equal(norms[i], expected), tap


def test_per_token_loss_leaves_last_position_blank(tmp_path):
    run = traced_run(tmp_path)
    reader = TraceReader(run.root)
    loss = per_token_loss(reader, 0, run.probe)
    assert loss.shape == run.probe.shape and loss.dtype == np.float32
    assert np.isnan(loss[:, -1]).all() and np.isfinite(loss[:, :-1]).all()

    logits = _traced(run, "logits").astype(np.float64)
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    expected = -np.take_along_axis(logp[:, :-1], run.probe[:, 1:, None], axis=-1)[..., 0]
    assert np.allclose(loss[:, :-1], expected, atol=1e-6)

    with pytest.raises(ValueError):
        per_token_loss(reader, 0, run.probe[::-1])  # not the probe this trace was recorded on


def test_readout_build_is_deterministic(tmp_path):
    run = traced_run(tmp_path)
    a, b = TraceReader(run.root), TraceReader(run.root)
    assert per_token_loss(a, 0, run.probe).tobytes() == per_token_loss(b, 0, run.probe).tobytes()
    assert residual_norms(a, 0, n_layers=2).tobytes() == residual_norms(b, 0, n_layers=2).tobytes()
