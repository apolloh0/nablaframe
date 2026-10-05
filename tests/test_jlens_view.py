"""Phase 2 delta 3b: the J-lens view, its meter, its comparison with the logit lens, and D14's tests.

Spec: docs/specs/phase-2.md, "The J-lens"; rulings D17 and D20. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.trace.store import TraceStore  # noqa: E402
from nablaframe.views.jlens import block_map, summed_jacobian  # noqa: E402
from nablaframe.views.jlens_view import JLENS_BADGE, jlens_lens, jlens_payload, jlens_readings, lens_divergence  # noqa: E402
from nablaframe.views.lens import lens_meter, lens_params, lens_readings, logit_lens, next_token_accuracy  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402


def _recovery(readings, targets):
    return float((readings.argmax(-1) == targets).mean())


def test_identity_jacobian_is_the_logit_lens():
    rng = np.random.default_rng(0)
    resid = rng.standard_normal((2, 8, 16)).astype(np.float32)
    w, b = (1 + 0.1 * rng.standard_normal(16)).astype(np.float32), (0.1 * rng.standard_normal(16)).astype(np.float32)
    wte = rng.standard_normal((32, 16)).astype(np.float32)
    assert np.array_equal(jlens_lens(resid, np.eye(16, dtype=np.float32), w, b, wte), logit_lens(resid, w, b, wte))


def test_jlens_recovers_planted_transport():
    """Plant source directions that J carries to each token's unembedding: only the J-lens can read them."""
    rng = np.random.default_rng(1)
    d, vocab, n = 32, 24, 2_000
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    source = rng.standard_normal((vocab, d))
    J = (wte.T.astype(np.float64) @ np.linalg.pinv(source.T)).astype(np.float32)  # J @ source[t] == wte[t]
    targets = rng.integers(0, vocab, size=(1, n))
    resid = (4.0 * source[targets] + 0.1 * rng.standard_normal((1, n, d))).astype(np.float32)
    ones, zeros = np.ones(d, np.float32), np.zeros(d, np.float32)
    assert _recovery(jlens_lens(resid, J, ones, zeros, wte), targets) >= 0.99
    assert _recovery(logit_lens(resid, ones, zeros, wte), targets) < 0.5, "without J the planted signal is unreadable"


def test_lens_divergence():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((3, 2, 5, 16)).astype(np.float32)
    b = rng.standard_normal((3, 2, 5, 16)).astype(np.float32)
    same = lens_divergence(a, a)
    assert [m["layer"] for m in same] == [0, 1, 2]
    assert all(m["js_mean"] <= 1e-12 and m["top1_agreement"] == 1.0 for m in same)
    ab, ba = lens_divergence(a, b), lens_divergence(b, a)
    for x, y in zip(ab, ba):
        assert set(x) == {"layer", "js_mean", "top1_agreement"}
        assert type(x["js_mean"]) is float and type(x["top1_agreement"]) is float
        assert 0 <= x["js_mean"] <= np.log(2) and x["js_mean"] == pytest.approx(y["js_mean"], rel=1e-12), "symmetric"
    p = np.exp(a[0].astype(np.float64)); p /= p.sum(-1, keepdims=True)
    q = np.exp(b[0].astype(np.float64)); q /= q.sum(-1, keepdims=True)
    m = (p + q) / 2
    js = 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)
    assert ab[0]["js_mean"] == pytest.approx(float(js.mean()), rel=1e-9), "Jensen-Shannon divergence in nats"


def test_jlens_payload(tmp_path):
    run = traced_run(tmp_path)
    reader = TraceReader(run.root)
    J = {0: (np.eye(16) + 0.3 * np.random.default_rng(3).standard_normal((16, 16))).astype(np.float32)}
    readings = jlens_readings(reader, 0, run.params, J, TINY.n_layers)
    logit = lens_readings(reader, 0, run.params, TINY.n_layers)
    assert readings.shape == (TINY.n_layers, 2, TINY.seq, TINY.vocab)
    w, b, wte = lens_params(run.params)
    assert np.array_equal(readings[0], jlens_lens(reader.tensor(0, "blocks.0.hook_resid_post"), J[0], w, b, wte))
    assert np.array_equal(readings[-1], logit[-1]), "the last block's J is the identity"

    payload = jlens_payload(reader, 0, run.params, J, run.probe, TINY.n_layers)
    assert payload["step"] == 0 and payload["badge"] == JLENS_BADGE
    assert payload["layers"] == ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]
    assert np.asarray(payload["top_tokens"]).shape == (TINY.n_layers, 2, TINY.seq, 5)
    assert payload["meter"] == lens_meter(readings, reader.tensor(0, "logits"))
    assert payload["meter"][-1]["kl_mean"] <= 1e-6
    assert payload["comparison"] == lens_divergence(readings, logit[1:]), "against the logit lens at the same residuals"
    ranks = payload["target_rank"]
    assert len(ranks) == TINY.n_layers and all(row[-1] is None for layer in ranks for row in layer)
    with pytest.raises(ValueError):
        jlens_payload(reader, 0, run.params, J, run.probe[::-1], TINY.n_layers)  # not this trace's probe
    with pytest.raises(KeyError):
        jlens_payload(reader, 0, run.params, {}, run.probe, TINY.n_layers)  # layer 0's J is missing


def test_jlens_is_empty_on_random_weights():
    cfg = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=24)
    model, params = GPT2Model(cfg), init_params(cfg, seed=3)
    stream = np.arange(8_192) % cfg.vocab  # each token is the previous one plus one
    starts = np.random.default_rng(1).integers(0, len(stream) - cfg.seq, size=64)
    tokens = np.stack([stream[s: s + cfg.seq] for s in starts]).astype(np.int32)
    _, taps = model.instrumented(params, tokens, ["blocks.0.hook_resid_post"])
    resid = np.asarray(getattr(taps["blocks.0.hook_resid_post"], "array", taps["blocks.0.hook_resid_post"]))
    J = summed_jacobian(block_map(model, params, layer=0, n_layers=2), resid[:8])
    w, b, wte = lens_params(params)
    accuracy = next_token_accuracy(jlens_lens(resid, J, w, b, wte), tokens)
    assert accuracy <= 2 / cfg.vocab, f"{accuracy:.3f} on random weights"
