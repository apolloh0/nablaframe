"""Phase 2 work package 3: the logit lens and its meter, with D14's discrimination tests.

Spec: docs/specs/phase-2.md, "Views" and D14. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from nablaframe.trace.store import TraceStore  # noqa: E402
from nablaframe.views.lens import lens_meter, lens_params, lens_readings, logit_lens, next_token_accuracy  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402


def test_lens_recovers_planted_token():
    rng = np.random.default_rng(0)
    d, vocab, n = 32, 64, 2_000
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(1, n))
    resid = (4.0 * wte[targets] + 0.3 * rng.standard_normal((1, n, d))).astype(np.float32)
    lens = logit_lens(resid, np.ones(d, np.float32), np.zeros(d, np.float32), wte)
    assert lens.shape == (1, n, vocab) and lens.dtype == np.float32
    assert (lens.argmax(-1) == targets).mean() >= 0.99


def test_lens_is_empty_on_random_weights():
    model, params = GPT2Model(TINY), init_params(TINY, seed=3)
    stream = np.arange(4_096) % TINY.vocab  # perfectly predictable: each token is the previous one plus one
    starts = np.random.default_rng(1).integers(0, len(stream) - TINY.seq, size=64)
    tokens = np.stack([stream[s: s + TINY.seq] for s in starts]).astype(np.int32)
    _, taps = model.instrumented(params, tokens, ["blocks.0.hook_resid_pre", "blocks.0.hook_resid_post",
                                                  "blocks.1.hook_resid_post"])
    ln_w, ln_b, wte = lens_params(params)
    chance = 1 / TINY.vocab
    for name, resid in taps.items():
        accuracy = next_token_accuracy(logit_lens(np.asarray(resid), ln_w, ln_b, wte), tokens)
        assert accuracy <= 2 * chance, f"{name}: {accuracy:.3f} on random weights"


def test_last_layer_lens_matches_logits(tmp_path):
    run = traced_run(tmp_path)
    readings = lens_readings(TraceReader(run.root), 0, run.params, n_layers=TINY.n_layers)
    assert readings.shape == (TINY.n_layers + 1, 2, TINY.seq, TINY.vocab)
    store = TraceStore(run.root, mode="r")
    [event] = [e for e in store.current() if e.tap == "logits"]
    logits = store.load_tensor(event)
    assert np.abs(readings[-1] - logits).max() <= 1e-4
    assert (readings[-1].argmax(-1) == logits.argmax(-1)).all()


def test_lens_meter_is_reported_per_layer(tmp_path):
    run = traced_run(tmp_path)
    readings = lens_readings(TraceReader(run.root), 0, run.params, n_layers=TINY.n_layers)
    store = TraceStore(run.root, mode="r")
    [event] = [e for e in store.current() if e.tap == "logits"]
    meter = lens_meter(readings, store.load_tensor(event))
    assert [m["layer"] for m in meter] == list(range(TINY.n_layers + 1))
    for m in meter:
        assert set(m) == {"layer", "kl_mean", "top1_agreement"}
        assert np.isfinite(m["kl_mean"]) and m["kl_mean"] >= 0 and 0 <= m["top1_agreement"] <= 1
        assert type(m["kl_mean"]) is float and type(m["top1_agreement"]) is float
    assert meter[-1]["kl_mean"] <= 1e-6 and meter[-1]["top1_agreement"] == 1.0

    final = store.load_tensor(event).astype(np.float64)
    lens0 = readings[0].astype(np.float64)
    log_p = final - final.max(-1, keepdims=True); log_p -= np.log(np.exp(log_p).sum(-1, keepdims=True))
    log_q = lens0 - lens0.max(-1, keepdims=True); log_q -= np.log(np.exp(log_q).sum(-1, keepdims=True))
    expected = float((np.exp(log_p) * (log_p - log_q)).sum(-1).mean())
    assert meter[0]["kl_mean"] == pytest.approx(expected, rel=1e-9), "KL(model || lens), in that direction"
