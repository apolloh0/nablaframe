"""Phase 5 delta 2: measuring a dictionary on a model (D41).

Contract: src/nablaframe/dictionaries/measure.py. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.dictionaries import core, measure  # noqa: E402
from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

D, M, K = TINY.d_model, 24, 4
RESID0 = [("blocks.0.hook_resid_post",) * 2]


def dictionary(kind, sites, seed=0, m=M, k=K, b_enc=None):
    rng = np.random.default_rng(seed)
    S = len(sites)
    W_dec = rng.standard_normal((1, m, D))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    arrays = {"W_enc": rng.standard_normal((1, D, m)), "W_dec": W_dec, "b_enc": rng.normal(0, 0.1, (S, m)),
              "b_pre": rng.normal(0, 0.1, (S, D)), "b_dec": rng.normal(0, 0.1, (S, D)),
              "scale_in": rng.uniform(0.5, 2, S), "scale_out": rng.uniform(0.5, 2, S)}
    if b_enc is not None:
        arrays["b_enc"] = b_enc
    return core.Dictionary(core.Settings(kind, m, k, steps=10, batch=4), sites,
                           {n: np.asarray(v, dtype=np.float32) for n, v in arrays.items()})


def identity(sites=RESID0):
    """A TopK SAE whose reconstruction is its input: codes [x⁺, (−x)⁺], decoded by [I; −I]."""
    eye = np.eye(D)
    arrays = {"W_enc": np.concatenate([eye, -eye], 1)[None], "W_dec": np.concatenate([eye, -eye], 0)[None],
              "b_enc": np.zeros((1, 2 * D)), "b_pre": np.zeros((1, D)), "b_dec": np.zeros((1, D)),
              "scale_in": np.ones(1), "scale_out": np.ones(1)}
    return core.Dictionary(core.Settings("topk", 2 * D, 2 * D, steps=10, batch=4), sites,
                           {n: v.astype(np.float32) for n, v in arrays.items()})


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    tokens = rng.integers(0, TINY.vocab, (5, TINY.seq)).astype(np.int32)
    sources = rng.integers(0, TINY.vocab, (5, TINY.seq)).astype(np.int32)
    return model, params, tokens, sources


def value(model, params, tokens, tap):
    _, rec = model.instrumented(params, tokens, [tap])
    return np.asarray(getattr(rec[tap], "array", rec[tap]))


def test_activations_come_in_chunks(world):
    model, params, tokens, _ = world
    taps = ["blocks.1.hook_resid_post", "blocks.0.attn.hook_z", "blocks.1.hook_resid_post"]
    got = measure.activations(model, params, tokens, taps, chunk=2)
    assert list(got) == ["blocks.1.hook_resid_post", "blocks.0.attn.hook_z"]
    for tap in got:
        want = value(model, params, tokens, tap)
        assert got[tap].dtype == np.float32 and got[tap].shape == want.shape, "every sequence, the last chunk short"
        assert np.allclose(got[tap], want, rtol=1e-6, atol=1e-6)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            measure.activations(model, params, tokens, taps, chunk=bad)


def test_fvu():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, D)).astype(np.float32)
    assert measure.fvu(identity(), 0, x, x) < 1e-10, "a perfect reconstruction leaves nothing unexplained"
    silent = dictionary("topk", RESID0, b_enc=np.full((1, M), -1e4))
    mean_only = core.Dictionary(silent.settings, silent.sites, {**dict(silent.arrays), "scale_out": np.ones(1, np.float32),
                                                               "b_dec": x.mean(0, keepdims=True)})
    assert np.isclose(measure.fvu(mean_only, 0, x, x), 1.0, atol=1e-6), "a code that never fires explains nothing"
    d = dictionary("transcoder", [("blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out")], seed=2)
    y = rng.standard_normal((300, D)).astype(np.float32) * 3 + 1
    y_hat = core.reconstruct(d, 0, x).astype(np.float64)
    want = ((y - y_hat) ** 2).sum() / ((y - y.astype(np.float64).mean(0)) ** 2).sum()
    assert np.isclose(measure.fvu(d, 0, x, y), want, rtol=1e-9) and isinstance(measure.fvu(d, 0, x, y), float)
    for bad in ((x, y[:-1]), (x, np.ones_like(y)), (x[0], y[0])):
        with pytest.raises(ValueError):
            measure.fvu(d, 0, *bad)


def test_dead_share():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4096, D)).astype(np.float32)
    b = np.zeros((1, M))
    b[0, :6] = -1e4  # six latents that can never fire
    assert measure.dead_share(dictionary("topk", RESID0, seed=4, b_enc=b), 0, x) == pytest.approx(6 / M)
    b = np.zeros((2, M))
    b[0, :3] = -1e4  # site 0 uses the first 12 latents: three of them dead
    b[1, 20:] = -1e4  # site 1 uses all 24: four dead
    stair = dictionary("staircase", [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(2)], seed=4, b_enc=b)
    assert measure.dead_share(stair, 0, x) == pytest.approx(3 / 12), "a Staircase site counts only its own share"
    assert measure.dead_share(stair, 1, x) == pytest.approx(4 / 24)


def test_readout_loss():
    logits = np.zeros((2, 3, 4), np.float32)
    logits[0, 1] = [2.0, 2.0, 0.0, -1.0]  # a tie: the lower id wins
    logits[1, 1] = [0.0, 1.0, 3.0, 0.5]
    logits[0, 2] = [1.0, 0.0, 0.0, 0.0]
    logits[1, 2] = [0.0, 0.0, 0.0, 4.0]
    windows = np.array([[0, 3, 0, 2], [1, 1, 2, 3]])
    loss, accuracy = measure.readout_loss(logits, windows, [1, 2])

    def nll(row, target):
        p = np.exp(row.astype(np.float64)) / np.exp(row.astype(np.float64)).sum()
        return -np.log(p[target])

    targets = {(0, 1): 0, (1, 1): 2, (0, 2): 2, (1, 2): 3}  # windows[:, p + 1]
    assert np.isclose(loss, np.mean([nll(logits[b, p], t) for (b, p), t in targets.items()]), rtol=1e-12)
    assert accuracy == 0.75, "right at (0, 1) by the tie, (1, 1) and (1, 2); wrong at (0, 2)"
    assert measure.readout_loss(logits, windows, [2]) == (pytest.approx(np.mean([nll(logits[0, 2], 2), nll(logits[1, 2], 3)])), 0.5)
    for bad in ([], [3], [-1]):
        with pytest.raises(ValueError):
            measure.readout_loss(logits, windows, bad)
    with pytest.raises(ValueError):
        measure.readout_loss(logits, windows[:, :3], [2])  # no target after position 2


def test_spliced_logits(world):
    model, params, tokens, _ = world
    clean = np.asarray(model.instrumented(params, tokens, [])[0])

    def by_hand(d, site):
        reads, writes = d.sites[site]
        x = value(model, params, tokens, reads)
        recon = core.reconstruct(d, site, x.reshape(-1, D)).reshape(x.shape[:2] + (D,))
        return np.asarray(model.instrumented(params, tokens, [], {writes: lambda _: recon})[0])

    cases = [(dictionary("topk", RESID0, seed=5), 0),
             (dictionary("transcoder", [("blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out")], seed=6), 0),
             (dictionary("staircase", [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(2)], seed=7), 1)]
    for d, site in cases:
        got = measure.spliced_logits(model, params, tokens, d, site)
        assert got.dtype == np.float32 and got.shape == clean.shape
        assert np.allclose(got, by_hand(d, site), rtol=1e-5, atol=1e-5), f"{d.settings.kind} site {site}"
        assert not np.allclose(got, clean, atol=1e-3), "a random dictionary's splice changes the logits"
    assert np.allclose(measure.spliced_logits(model, params, tokens, identity(), 0), clean, rtol=1e-5, atol=1e-5), \
        "splicing a perfect reconstruction changes nothing"
    with pytest.raises(ValueError):
        measure.spliced_logits(model, params, tokens, cases[0][0], 1)


def test_ablated_logits(world):
    model, params, tokens, _ = world
    tap = "blocks.0.hook_mlp_out"
    mean = value(model, params, tokens, tap).astype(np.float64).mean(0, keepdims=True).astype(np.float32)
    want = np.asarray(model.instrumented(params, tokens, [], {tap: lambda v: np.broadcast_to(mean, v.shape)})[0])
    got = measure.ablated_logits(model, params, tokens, tap, n_layers=TINY.n_layers)
    assert got.dtype == np.float32 and np.allclose(got, want, rtol=1e-5, atol=1e-5), "each position's mean over the windows"
    assert not np.allclose(got, np.asarray(model.instrumented(params, tokens, [])[0]), atol=1e-3)


def test_f1_table():
    codes = np.array([[1, 0, 0], [1, 0, 2], [0, 3, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    values = np.array([0, 0, 1, 1, 2, 0, 3])
    f1, ids = measure.f1_table(codes, values, 4, min_count=2)
    assert ids.tolist() == [0, 1], "values 2 and 3 occur once"
    # latent 0: active on rows 0, 1, 5; latent 1 on 2, 3; latent 2 on 1, 4.
    want = np.array([[2 * 3 / (3 + 3), 0.0], [0.0, 2 * 2 / (2 + 2)], [2 * 1 / (2 + 3), 0.0]])
    assert f1.dtype == np.float64 and np.allclose(f1, want, rtol=1e-12)
    assert measure.mean_best_f1(f1) == pytest.approx(1.0)
    f1, ids = measure.f1_table(-codes, values, 4, min_count=2)
    assert not f1.any(), "a latent is active only where its code is above zero"
    assert measure.f1_table(codes, values, 4, min_count=1)[1].tolist() == [0, 1, 2, 3]
    for bad in ((codes, values[:-1], 4), (codes, values, 3), (codes, values.astype(float), 4), (codes[0], values, 4)):
        with pytest.raises(ValueError):
            measure.f1_table(*bad, min_count=2)
    with pytest.raises(ValueError):
        measure.f1_table(codes, values, 4, min_count=4)  # no value occurs four times


def test_mean_best_f1():
    assert measure.mean_best_f1(np.array([[0.2, 0.9], [0.6, 0.1]])) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        measure.mean_best_f1(np.zeros((3, 0)))


def test_assigned():
    tables = {"a": np.array([[0.9, 0.1], [0.2, 0.3], [0.6, 0.0], [0.1, 0.0]]),
              "b": np.array([[0.5], [0.7], [0.6], [0.4]])}
    assert measure.assigned(tables, "a") == [0, 2], "latent 2 ties at 0.6 and goes to the first variable"
    assert measure.assigned(tables, "b") == [1]
    assert measure.assigned(tables, "a", floor=0.8) == [0] and measure.assigned(tables, "b", floor=0.8) == []
    assert measure.assigned(tables, "b", floor=0.4) == [1, 3], "a floor is reached by equalling it"
    assert measure.assigned({"b": tables["b"], "a": tables["a"]}, "b") == [1, 2]
    with pytest.raises(ValueError):
        measure.assigned(tables, "c")


def test_probe_accuracy():
    rng = np.random.default_rng(8)
    values = rng.integers(0, 5, 200)
    clear = np.eye(5)[values] * 3 + rng.normal(0, 0.1, (200, 5))
    assert measure.probe_accuracy(clear, values, 5) == 1.0
    x = rng.standard_normal((60, 20))
    values = rng.integers(0, 4, 60)

    def by_hand(x, values, ridge):
        fit = np.arange(len(x)) % 2 == 0
        xc = x - x[fit].mean(0)
        g = xc[fit].T @ xc[fit]
        w = np.linalg.solve(g + ridge * np.trace(g) / len(g) * np.eye(len(g)), xc[fit].T @ np.eye(4)[values[fit]])
        return float(((xc[~fit] @ w).argmax(1) == values[~fit]).mean())

    for ridge in (1e-2, 10.0):
        assert measure.probe_accuracy(x, values, 4, ridge=ridge) == by_hand(x, values, ridge)
    assert by_hand(x, values, 1e-2) != by_hand(x, values, 10.0), "the test can tell the ridge apart"
    for bad in ((x[:1], values[:1]), (x, values[:-1]), (x[0], values)):
        with pytest.raises(ValueError):
            measure.probe_accuracy(*bad, 4)


def test_patch_predictions(world):
    model, params, _, _ = world
    rng = np.random.default_rng(10)
    tokens, sources = (rng.integers(0, TINY.vocab, (64, TINY.seq)).astype(np.int32) for _ in range(2))
    tap, positions, read, n = "blocks.0.hook_resid_post", [2, 5], TINY.seq - 1, TINY.n_layers
    loud = dictionary("topk", RESID0, seed=9)  # a large output scale, so a code's change moves predictions
    d = core.Dictionary(loud.settings, loud.sites, {**dict(loud.arrays), "scale_out": np.full(1, 20, np.float32)})
    store = {batch_hash(tokens): tokens, batch_hash(sources): sources, core.content_hash(d): d}

    def argmax(*edits):
        logits, _ = run_intervened(model, params, tokens, InterventionSpec(edits), n_layers=n, resolve=store.__getitem__)
        return np.asarray(logits)[:, read].argmax(-1)

    src, h = batch_hash(sources), core.content_hash(d)
    latents = {5: [7, 0], 2: [3]}
    got = measure.patch_predictions(model, params, tokens, sources, tap=tap, positions=positions, read=read, n_layers=n,
                                    dictionary=d, latents=latents)
    assert sorted(got) == ["code", "latents", "residual"] and all(v.dtype == np.int64 and v.shape == (64,) for v in got.values())
    assert np.array_equal(got["residual"], argmax(edit(tap, positions, "patch", source=src)))
    assert np.array_equal(got["code"], argmax(edit(tap, positions, "dpatch", dictionary=h, source=src)))
    assert np.array_equal(got["latents"], argmax(edit(tap, [2], "dpatch", dictionary=h, source=src, latents=[3]),
                                                 edit(tap, [5], "dpatch", dictionary=h, source=src, latents=[0, 7])))
    same = measure.patch_predictions(model, params, tokens, sources, tap=tap, positions=positions, read=read, n_layers=n,
                                     dictionary=d, latents={2: [3], 5: [3]})
    assert not np.array_equal(same["latents"], got["latents"]), "each position's own latents count"
    every = measure.patch_predictions(model, params, tokens, sources, tap=tap, positions=positions, read=read, n_layers=n,
                                      dictionary=d, latents={2: range(M), 5: range(M)})
    assert np.array_equal(every["latents"], got["code"]), "every latent at every position is the whole code"
    whole = measure.patch_predictions(model, params, tokens, sources, tap=tap, positions=range(TINY.seq), read=read, n_layers=n)
    assert list(whole) == ["residual"]
    source_clean = np.asarray(model.instrumented(params, sources, [])[0])[:, read].argmax(-1)
    assert np.array_equal(whole["residual"], source_clean), "the residual patch at every position is the source's pass"
    for bad in (dict(latents={2: [1]}), dict(dictionary=d, latents={2: [1]}), dict(dictionary=d, latents={2: [1], 5: [], 6: []})):
        with pytest.raises(ValueError):
            measure.patch_predictions(model, params, tokens, sources, tap=tap, positions=positions, read=read, n_layers=n, **bad)
