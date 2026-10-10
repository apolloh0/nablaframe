"""Phase 5 delta 1: training a dictionary, and the Jacobian SAE's Jacobian through an MLP (D40, D43).

Contract: src/nablaframe/dictionaries/train.py. Each test trains on small planted data, whose features
are known, so a dictionary that trains is seen to find them. Needs the jax lane.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from conftest import REPO_ROOT  # noqa: E402
from nablaframe.dictionaries import core, train  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

D, K_TRUE, ACTIVE = 16, 24, 3
RESID = [("blocks.0.hook_resid_post",) * 2]


def planted(n, seed, features=None, count=K_TRUE):
    """Rows that are sums of ACTIVE unit features out of ``count``, magnitudes 1 to 2, and the features."""
    rng = np.random.default_rng(seed)
    if features is None:
        features = rng.standard_normal((K_TRUE, D))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
    coef = np.zeros((n, K_TRUE))
    for row in coef:
        row[rng.choice(count, ACTIVE, replace=False)] = rng.uniform(1, 2, ACTIVE)
    return (coef @ features).astype(np.float32), features, coef


def fvu(d, site, x, y):
    r = core.reconstruct(d, site, x)
    return float(((y - r) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def best_cosines(d, features, unit=0):
    W = d.arrays["W_dec"][unit].astype(np.float64)
    return (features @ W.T / np.linalg.norm(W, axis=1)).max(1)


SETTINGS = dict(steps=1500, batch=256, lr=1e-2)


def test_training_is_deterministic():
    x, _, _ = planted(2048, 0)
    s = core.Settings("topk", 48, ACTIVE, steps=60, batch=128, lr=3e-3)
    a, b = (train.train(s, RESID, [(x, x)]) for _ in range(2))
    assert core.content_hash(a) == core.content_hash(b), "the same settings and data give the same bytes"
    other = train.train(core.Settings("topk", 48, ACTIVE, steps=60, batch=128, lr=3e-3, seed=1), RESID, [(x, x)])
    assert not np.array_equal(other.arrays["W_dec"], a.arrays["W_dec"]), "the seed reaches the weights"


def test_initialisation_follows_the_seed():
    x, _, _ = planted(512, 14)
    w = _toy_mlp(15)
    y = train.mlp(w, x)
    cases = [("topk", RESID, [(x, x)], None), ("transcoder", [("blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out")],
                                                [(x, y)], None),
             ("jsae", [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2], [(x, x), (y, y)], w)]
    for kind, sites, data, weights in cases:
        # One step at a learning rate too small to move anything: the weights are as initialised.
        d = train.train(core.Settings(kind, 48, ACTIVE, steps=1, batch=8, lr=1e-9, seed=3), sites, data, mlp_weights=weights)
        keys = jax.random.split(jax.random.PRNGKey(3), len(d.arrays["W_dec"]))
        for u, k in enumerate(keys):
            W = np.asarray(jax.random.normal(k, (48, D)), np.float64)
            W /= np.linalg.norm(W, axis=1, keepdims=True)
            assert np.allclose(d.arrays["W_dec"][u], W, atol=1e-6), f"{kind} unit {u}'s decoder"
            assert np.allclose(d.arrays["W_enc"][u], W.T * (0.1 if kind == "transcoder" else 1), atol=1e-6), kind
        assert np.abs(d.arrays["b_enc"]).max() < 1e-6
        for s, (xs, ys) in enumerate(data):
            assert np.allclose(d.arrays["b_pre"][s], (xs / d.arrays["scale_in"][s]).mean(0), atol=1e-6)
            assert np.allclose(d.arrays["b_dec"][s], (ys / d.arrays["scale_out"][s]).mean(0), atol=1e-6)


def test_every_setting_reaches_the_trainer():
    x, _, _ = planted(2048, 13)
    # Wide, with small batches, so latents go dead and AuxK acts; warmup under steps // 10, so it counts.
    base = dict(kind="topk", width=192, k=ACTIVE, steps=40, batch=16, lr=3e-3, warmup=2, dead_after=3)
    weights = lambda **change: train.train(core.Settings(**(base | change)), RESID, [(x, x)]).arrays["W_dec"]  # noqa: E731
    reference = weights()
    for change in (dict(seed=1), dict(lr=1e-3), dict(steps=41), dict(batch=15), dict(warmup=3),
                   dict(decay_fraction=0.6), dict(aux_weight=0.5), dict(dead_after=6)):
        assert not np.array_equal(weights(**change), reference), f"{change} changes what trains"


def test_a_topk_sae_finds_planted_features():
    x, features, _ = planted(8192, 1)
    seen = []
    d = train.train(core.Settings("topk", 48, ACTIVE, **SETTINGS), RESID, [(x, x)], key="d" * 64,
                    progress=lambda step, steps, losses: seen.append((step, steps, losses)))
    assert d.key == "d" * 64 and d.sites == tuple(map(tuple, RESID))
    assert [s for s, _, _ in seen] == list(range(150, 1501, 150)) and all(len(l) == 1 for _, _, l in seen)
    assert seen[-1][2][0] < seen[0][2][0], "the reconstruction term falls"
    rms = np.sqrt((x.astype(np.float64) ** 2).sum(1).mean() / D)
    assert np.isclose(d.arrays["scale_in"][0], rms, rtol=1e-6) and np.isclose(d.arrays["scale_out"][0], rms, rtol=1e-6)
    assert np.allclose(np.linalg.norm(d.arrays["W_dec"], axis=2), 1, atol=1e-5), "decoder rows have unit norm"
    held, _, _ = planted(2048, 2, features)
    assert fvu(d, 0, held, held) < 0.1
    assert best_cosines(d, features).min() > 0.9, "every planted feature has a latent along it"


def test_a_staircase_sae_trains_every_block():
    x0, features, _ = planted(8192, 3, count=12)  # block 0 holds the first 12 features
    x1, _, _ = planted(8192, 4, features)  # block 1 all 24
    sites = [(f"blocks.{layer}.hook_resid_post",) * 2 for layer in range(2)]
    d = train.train(core.Settings("staircase", 48, ACTIVE, **SETTINGS), sites, [(x0, x0), (x1, x1)])
    assert d.arrays["W_enc"].shape == (1, D, 48) and d.arrays["b_enc"].shape == (2, 48)
    assert not core.encode(d, 0, x0)[:, 24:].any(), "block 0 uses its first 24 latents only"
    assert fvu(d, 0, x0, x0) < 0.1 and fvu(d, 1, x1, x1) < 0.1


def test_a_transcoder_learns_a_sparse_map():
    x, features, coef = planted(8192, 5)
    rng = np.random.default_rng(6)
    targets = rng.standard_normal((K_TRUE, D))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    y = (coef @ targets).astype(np.float32)  # each input feature maps to its own output direction
    sites = [("blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out")]
    d = train.train(core.Settings("transcoder", 48, ACTIVE, **SETTINGS), sites, [(x, y)])
    assert np.isclose(d.arrays["scale_out"][0], np.sqrt((y.astype(np.float64) ** 2).sum(1).mean() / D), rtol=1e-6)
    assert fvu(d, 0, x, y) < 0.1
    assert (best_cosines(d, targets) > 0.9).sum() >= K_TRUE - 2, "nearly every output direction has a latent along it"


def _toy_mlp(seed):
    rng = np.random.default_rng(seed)
    return {"W_fc": (rng.standard_normal((D, 32)) / 4).astype(np.float32), "b_fc": rng.normal(0, 0.1, 32).astype(np.float32),
            "W_proj": (rng.standard_normal((32, D)) / 6).astype(np.float32), "b_proj": rng.normal(0, 0.1, D).astype(np.float32)}


def test_mlp_matches_the_model():
    params = _perturbed(init_params(TINY, seed=0))
    tokens = np.random.default_rng(7).integers(0, TINY.vocab, (2, TINY.seq)).astype(np.int32)
    _, rec = GPT2Model(TINY).instrumented(params, tokens, ["blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out"])
    x = np.asarray(getattr(rec["blocks.1.ln2.hook_normalized"], "array", rec["blocks.1.ln2.hook_normalized"]))
    y = np.asarray(getattr(rec["blocks.1.hook_mlp_out"], "array", rec["blocks.1.hook_mlp_out"]))
    w = train.mlp_weights(params, 1)
    assert {n: v.shape for n, v in w.items()} == {"W_fc": (TINY.d_model, TINY.d_mlp), "b_fc": (TINY.d_mlp,),
                                                   "W_proj": (TINY.d_mlp, TINY.d_model), "b_proj": (TINY.d_model,)}
    got = train.mlp(w, x.reshape(-1, TINY.d_model)).reshape(y.shape)
    assert np.allclose(got, y, rtol=1e-5, atol=1e-5), "the same MLP, GELU's tanh approximation included"
    with pytest.raises(ValueError):
        train.mlp_weights(params, TINY.n_layers)


def test_the_pair_jacobian_is_the_derivative():
    rng = np.random.default_rng(8)
    m, k = 20, 4
    W_dec = rng.standard_normal((2, m, D))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    # The output site's biases are lowered, so some of the k output latents kept don't fire.
    arrays = {"W_enc": rng.standard_normal((2, D, m)), "W_dec": W_dec, "b_enc": rng.normal(0, 0.3, (2, m)) - [[0], [6]],
              "b_pre": rng.normal(0, 0.1, (2, D)), "b_dec": rng.normal(0, 0.1, (2, D)),
              "scale_in": np.array([1.3, 0.8]), "scale_out": np.array([1.3, 0.8])}
    d = core.Dictionary(core.Settings("jsae", m, k, steps=10, batch=4, lam=1.0),
                        [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2],
                        {n: v.astype(np.float32) for n, v in arrays.items()})
    w = _toy_mlp(9)
    codes = core.encode(d, 0, rng.standard_normal((6, D)).astype(np.float32))
    jac, idx_in, idx_out = train.pair_jacobian(d, codes, w)
    assert jac.shape == (6, k, k) and jac.dtype == np.float32

    def pre_out(a, wj, c):  # the contract's composition, written out independently, in float64
        x_hat = (c @ a["W_dec"][0] + a["b_dec"][0]) * a["scale_out"][0]
        y = jax.nn.gelu(x_hat @ wj["W_fc"] + wj["b_fc"], approximate=True) @ wj["W_proj"] + wj["b_proj"]
        return (y / a["scale_in"][1] - a["b_pre"][1]) @ a["W_enc"][1] + a["b_enc"][1]

    with jax.enable_x64(True):
        a = {n: jnp.asarray(v, jnp.float64) for n, v in d.arrays.items()}
        wj = {n: jnp.asarray(v, jnp.float64) for n, v in w.items()}
        silent = 0
        for b in range(6):
            c = jnp.asarray(codes[b], jnp.float64)
            full = np.asarray(jax.jacfwd(lambda c: pre_out(a, wj, c))(c))  # (m out, m in)
            pre = np.asarray(pre_out(a, wj, c))
            assert full.dtype == np.float64
            assert idx_in[b].tolist() == np.argsort(-codes[b], kind="stable")[:k].tolist()
            assert idx_out[b].tolist() == np.argsort(-pre, kind="stable")[:k].tolist()
            want = full[np.ix_(idx_out[b], idx_in[b])].T * (pre[idx_out[b]] > 0)[None, :]
            assert np.allclose(jac[b], want, rtol=1e-3, atol=1e-4), f"row {b}"
            silent += int((pre[idx_out[b]] <= 0).sum())
        assert 0 < silent < 6 * k, "the test covers kept output latents that fire and ones that don't"
    with pytest.raises(ValueError):
        train.pair_jacobian(core.Dictionary(core.Settings("topk", m, k, steps=10, batch=4), RESID,
                                            {n: (v[:1] if n in ("W_enc", "W_dec", "b_enc", "b_pre", "b_dec", "scale_in", "scale_out") else v).astype(np.float32)
                                             for n, v in arrays.items()}), codes, w)


def test_the_penalty_shrinks_the_jacobian():
    w = _toy_mlp(10)
    x, _, _ = planted(8192, 11)
    y = train.mlp(w, x)
    sites = [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2]
    means = {}
    for lam in (0.0, 1.0):
        d = train.train(core.Settings("jsae", 48, ACTIVE, steps=600, batch=256, lr=3e-3, lam=lam, jac_batch=128),
                        sites, [(x, x), (y, y)], mlp_weights=w)
        jac, _, _ = train.pair_jacobian(d, core.encode(d, 0, x[:1024]), w)
        means[lam] = float(np.abs(jac).mean())
        assert fvu(d, 0, x, x) < 0.4 and fvu(d, 1, y, y) < 0.4, f"lam {lam} still reconstructs"
    assert means[1.0] < 0.5 * means[0.0], means


def test_bad_inputs_are_refused():
    x, _, _ = planted(256, 12)
    s = core.Settings("topk", 48, ACTIVE, steps=5, batch=32)
    cases = [
        lambda: train.train(s, RESID, [(x, x), (x, x)]),
        lambda: train.train(s, RESID, [(x, x + 1)]),
        lambda: train.train(s, RESID, [(x.astype(np.float64), x.astype(np.float64))]),
        lambda: train.train(s, RESID, [(np.where(np.arange(D) == 0, np.nan, x).astype(np.float32),) * 2]),
        lambda: train.train(s, RESID, [(x, x)], mlp_weights=_toy_mlp(0)),
        lambda: train.train(core.Settings("jsae", 48, ACTIVE, steps=5, batch=32),
                            [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2], [(x, x), (x, x)]),
        lambda: train.train(core.Settings("staircase", 48, ACTIVE, steps=5, batch=32),
                            [(f"blocks.{i}.hook_resid_post",) * 2 for i in range(2)], [(x, x), (x[:100], x[:100])]),
    ]
    for case in cases:
        with pytest.raises(ValueError):
            case()


def test_code_identity_is_the_modules_blob_hash():
    path = REPO_ROOT / "src" / "nablaframe" / "dictionaries" / "train.py"
    blob = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    assert train.code_identity() == blob
