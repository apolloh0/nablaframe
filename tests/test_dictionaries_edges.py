"""Phase 5 delta 3: the edge-sparsity meter, and the Jacobian across an MLP (D42, D43).

Contract: src/nablaframe/dictionaries/edges.py. Each derivative and ablation is checked against a
direct computation written out here. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from nablaframe.dictionaries import core, edges, train  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

D, M, K = TINY.d_model, 24, 4
LN2, MLP_OUT = "blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out"
RESID = [f"blocks.{l}.hook_resid_post" for l in range(TINY.n_layers)]
SHIFT = -5.0

# Round 2's curves on the induction model (feasibility, 2026-10-10): [edges, KL] and the pair spliced in whole.
TRAINED_0 = ([[1, 5.39134], [2, 5.40727], [5, 5.28701], [11, 5.25493], [25, 4.19323], [55, 3.40318], [122, 2.33083],
              [272, 0.92346], [606, 0.27164], [1349, 0.09391], [3005, 0.03356], [6694, 0.02939], [14910, 0.013],
              [33209, 0.00803]], 0.01064)
TWIN_0 = ([[1, 0.00692], [2, 0.0069], [5, 0.00684], [13, 0.00686], [29, 0.00684], [68, 0.00651], [158, 0.00612],
           [369, 0.0062], [857, 0.00551], [1994, 0.00465], [4639, 0.00383], [10791, 0.00335], [25102, 0.00253],
           [58391, 0.00182]], 0.00165)
TRAINED_1 = ([[1, 1.75785], [2, 1.76232], [5, 1.76464], [11, 1.77665], [24, 1.79769], [52, 1.41034], [115, 0.64437],
              [252, 0.15121], [556, 0.02904], [1226, 0.00796], [2702, 0.00581], [5955, 0.00569], [13125, 0.00482],
              [28924, 0.0045]], 0.00408)
TWIN_1 = ([[1, 0.00233], [2, 0.00232], [5, 0.00227], [12, 0.00219], [28, 0.00211], [65, 0.002], [150, 0.00183],
           [345, 0.00169], [794, 0.00152], [1829, 0.00132], [4213, 0.0011], [9707, 0.00095], [22363, 0.00085],
           [51522, 0.00075]], 0.00072)


def arrays(S, U, m, seed, scale=1.0, bias_shift=0.0):
    rng = np.random.default_rng(seed)
    W_dec = rng.standard_normal((U, m, D))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    return {"W_enc": rng.standard_normal((U, D, m)) * 0.5, "W_dec": W_dec, "b_enc": rng.normal(0, 0.2, (S, m)) + bias_shift,
            "b_pre": rng.normal(0, 0.1, (S, D)), "b_dec": rng.normal(0, 0.1, (S, D)),
            "scale_in": rng.uniform(0.5, 1.5, S), "scale_out": rng.uniform(0.5, 1.5, S) * scale}


def make(kind, sites, seed, m=M, **kw):
    U = 2 if kind == "jsae" else 1
    return core.Dictionary(core.Settings(kind, m, K, steps=10, batch=4), sites,
                           {n: np.asarray(v, np.float32) for n, v in arrays(len(sites), U, m, seed, **kw).items()})


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    tokens = rng.integers(0, TINY.vocab, (2, TINY.seq)).astype(np.int32)
    w = train.mlp_weights(params, 1)
    up = (make("topk", [(LN2, LN2)], 1), 0)
    down = (make("topk", [(MLP_OUT, MLP_OUT)], 2, scale=3.0), 0)
    resid = [(make("topk", [(RESID[l], RESID[l])], 3 + l, scale=2.0), 0) for l in range(2)]
    stair = make("staircase", [(t, t) for t in RESID], 5)
    return model, params, tokens, w, up, down, resid, stair


def value(model, params, tokens, tap, interventions=None):
    logits, rec = model.instrumented(params, tokens, [tap], interventions or {})
    return np.asarray(logits), np.asarray(getattr(rec[tap], "array", rec[tap]))


def replaced(model, tap, read=None):
    """A compiled pass with ``tap`` replaced by an argument: the logits, or the tap ``read``."""

    def forward(params, tokens, v):
        logits, rec = model.instrumented(params, tokens, [read] if read else [], {tap: lambda _: v})
        return logits if read is None else getattr(rec[read], "array", rec[read])

    return jax.jit(forward)


def log_softmax(x):
    x = np.asarray(x, np.float64)
    return x - x.max(-1, keepdims=True) - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True))


def kl_by_hand(clean, other):
    p, q = log_softmax(clean), log_softmax(other)
    return float((np.exp(p) * (p - q)).sum(-1).mean())


def test_edge_counts():
    assert edges.EDGE_POINTS == 14 and edges.IG_STEPS == 5 and edges.SETTLE == 0.1
    for curve, _ in (TRAINED_0, TWIN_0, TRAINED_1, TWIN_1):
        assert edges.edge_counts(curve[-1][0]) == [n for n, _ in curve], "round 2's edge counts"
    assert edges.edge_counts(1) == [1] and edges.edge_counts(3) == [1, 2, 3]
    assert edges.edge_counts(1000, points=4) == [1, 10, 100, 1000]
    for bad in ((0, 14), (10, 0), (10.0, 14), (True, 14)):
        with pytest.raises(ValueError):
            edges.edge_counts(*bad)


def test_meter():
    assert [edges.meter(*c) for c in (TRAINED_0, TWIN_0, TRAINED_1, TWIN_1)] == [606, 58391, 252, 22363], \
        "round 2's readings, which separated each trained MLP from its twin"
    assert edges.meter([[1, 1.0], [10, 0.05], [100, 0.2], [1000, 0.01]], 0.01) == 1000, "a point that rises again resets it"
    assert edges.meter([[1, 1.0], [10, 0.105], [100, 0.1]], 0.1) == 10, "within the tenth, its edge included"
    assert edges.meter([[1, 1.0], [10, 0.5], [100, 0.4]], 0.0) is None, "a curve that never settles near the pair"
    assert edges.meter([[1, 0.001], [10, 0.0015]], 0.002) == 1, "a curve that starts at the floor needs one edge"
    assert edges.meter([[1, 1.0], [10, 0.12], [100, 0.01]], 0.1) == 100, "the last point can be the floor"
    assert edges.meter([[1, 2.0], [10, 0.2], [100, 0.0]], 0.5) == 10, "a point exactly at the bound counts"
    with pytest.raises(ValueError):
        edges.meter([], 0.1)


def test_kl():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 3, 5)).astype(np.float32), rng.standard_normal((2, 3, 5)).astype(np.float32)
    assert edges.kl(a, a) == pytest.approx(0.0, abs=1e-12)
    assert edges.kl(a, b) == pytest.approx(kl_by_hand(a, b), rel=1e-12) and edges.kl(a, b) != pytest.approx(edges.kl(b, a))
    with pytest.raises(ValueError):
        edges.kl(a, b[:, :2])


def test_mlp_jacobian_is_the_derivative(world):
    _, _, _, w, up, _, _, _ = world
    down = (make("topk", [(MLP_OUT, MLP_OUT)], 2, bias_shift=SHIFT), 0)  # lowered, so some kept latents don't fire
    rng = np.random.default_rng(2)
    codes = core.encode(*up, rng.standard_normal((6, D)).astype(np.float32) * 2)
    jac, idx_in, idx_out = edges.mlp_jacobian(up, down, w, codes)
    assert jac.shape == (6, K, K) and jac.dtype == np.float32
    with jax.enable_x64(True):
        pu = {n: jnp.asarray(v, jnp.float64) for n, v in up[0].arrays.items()}
        pd = {n: jnp.asarray(v, jnp.float64) for n, v in down[0].arrays.items()}
        wj = {n: jnp.asarray(v, jnp.float64) for n, v in w.items()}

        def pre_out(c):
            x = (c @ pu["W_dec"][0] + pu["b_dec"][0]) * pu["scale_out"][0]
            y = jax.nn.gelu(x @ wj["W_fc"] + wj["b_fc"], approximate=True) @ wj["W_proj"] + wj["b_proj"]
            return (y / pd["scale_in"][0] - pd["b_pre"][0]) @ pd["W_enc"][0] + pd["b_enc"][0]

        silent = 0
        for t in range(6):
            c = jnp.asarray(codes[t], jnp.float64)
            full, pre = np.asarray(jax.jacfwd(pre_out)(c)), np.asarray(pre_out(c))
            assert idx_in[t].tolist() == np.argsort(-codes[t], kind="stable")[:K].tolist()
            assert idx_out[t].tolist() == np.argsort(-pre, kind="stable")[:K].tolist()
            want = full[np.ix_(idx_out[t], idx_in[t])].T * (pre[idx_out[t]] > 0)[None, :]
            assert np.allclose(jac[t], want, rtol=1e-3, atol=1e-4), f"row {t}"
            silent += int((pre[idx_out[t]] <= 0).sum())
    assert 0 < silent < 6 * K, "the test covers kept downstream latents that fire and ones that don't"
    jsae = make("jsae", [(LN2, LN2), (MLP_OUT, MLP_OUT)], 6)
    codes = core.encode(jsae, 0, rng.standard_normal((5, D)).astype(np.float32))
    got, want = edges.mlp_jacobian((jsae, 0), (jsae, 1), w, codes), train.pair_jacobian(jsae, codes, w)
    assert all(np.allclose(g, h, rtol=1e-5, atol=1e-6) for g, h in zip(got, want)), "a Jacobian SAE's own pair Jacobian"


def test_jacobian_stats():
    jac = np.array([[[0.5, -0.3], [0.2, 0.0]],  # |.| 0.5, 0.3, 0.2, 0: 90% of 1.0 needs three entries
                    [[0.0, 0.0], [0.0, 0.0]],  # none
                    [[-1.0, 0.005], [0.0, 0.0]]])  # one
    stats = edges.jacobian_stats(jac)
    assert stats == {"entries_for_90": 1.0, "share_above_0.01": pytest.approx((3 / 4 + 0 + 1 / 4) / 3),
                     "mean_abs": pytest.approx((1.0 + 0 + 1.005) / 12)}
    assert edges.jacobian_stats(jac[:1])["entries_for_90"] == 3.0
    assert edges.jacobian_stats(np.array([[[0.45, 0.45, 0.1]]]))["entries_for_90"] == 2.0, "reaching 90% exactly counts"
    with pytest.raises(ValueError):
        edges.jacobian_stats(jac[0])


def test_mlp_edge_scores(world):
    model, params, tokens, w, up, down, _, _ = world
    scores = edges.mlp_edge_scores(model, params, tokens, up, down, w)
    assert scores.shape == (M, M) and scores.dtype == np.float64
    _, x = value(model, params, tokens, LN2)
    u = core.encode(*up, x.reshape(-1, D))
    with jax.enable_x64(True):
        pu = {n: jnp.asarray(v, jnp.float64) for n, v in up[0].arrays.items()}
        pd = {n: jnp.asarray(v, jnp.float64) for n, v in down[0].arrays.items()}
        wj = {n: jnp.asarray(v, jnp.float64) for n, v in w.items()}

        def codes_out(c):  # the downstream code, as encode computes it, of the MLP on the upstream decode
            x = (c @ pu["W_dec"][0] + pu["b_dec"][0]) * pu["scale_out"][0]
            y = jax.nn.gelu(x @ wj["W_fc"] + wj["b_fc"], approximate=True) @ wj["W_proj"] + wj["b_proj"]
            pre = (y / pd["scale_in"][0] - pd["b_pre"][0]) @ pd["W_enc"][0] + pd["b_enc"][0]
            values, index = jax.lax.top_k(pre, K)
            return jnp.zeros(M).at[index].set(jnp.maximum(values, 0.0))

        total = np.zeros((M, M))
        for t in range(len(u)):
            c = jnp.asarray(u[t], jnp.float64)
            mean = sum(np.asarray(jax.jacfwd(codes_out)(c * a)) for a in (np.arange(5) + 0.5) / 5) / 5  # (down, up)
            total += (u[t][None, :] * mean).T ** 2
    want = np.sqrt(total / len(u))
    assert np.allclose(scores, want, rtol=1e-3, atol=1e-5 * want.max()), "integrated gradients from the origin, per position"
    assert (scores > 0).sum() > M, "the test scores many edges"
    assert not np.allclose(edges.mlp_edge_scores(model, params, tokens, up, down, w, ig_steps=1), scores, rtol=1e-3)


def test_block_edge_scores(world):
    model, params, tokens, _, _, _, resid, _ = world
    (du, _), (dd, _) = resid
    scores = edges.block_edge_scores(model, params, tokens, *resid)
    assert scores.shape == (M, M) and scores.dtype == np.float64
    _, x = value(model, params, tokens, RESID[0])
    u = core.encode(du, 0, x.reshape(-1, D)).reshape(x.shape[:2] + (M,))
    a_up, a_dn = du.arrays, dd.arrays

    def summed(c, window):  # written out: the upstream decode at every position, the downstream code summed
        v = ((c @ a_up["W_dec"][0] + a_up["b_dec"][0]) * a_up["scale_out"][0])[None]
        _, rec = model.instrumented(params, window, [RESID[1]], {RESID[0]: lambda _: v})
        y = getattr(rec[RESID[1]], "array", rec[RESID[1]])[0]
        pre = (y / a_dn["scale_in"][0] - a_dn["b_pre"][0]) @ a_dn["W_enc"][0] + a_dn["b_enc"][0]
        values, index = jax.lax.top_k(pre, K)
        return jnp.zeros(pre.shape).at[jnp.arange(pre.shape[0])[:, None], index].set(jnp.maximum(values, 0.0)).sum(0)

    total = np.zeros((M, M))
    for b in range(len(tokens)):
        jac = sum(np.asarray(jax.jacfwd(summed)(jnp.asarray(u[b] * a), tokens[b:b + 1]), np.float64)
                  for a in (np.arange(5) + 0.5) / 5) / 5  # (down, pos, up), forward mode
        total += (jac * u[b][None]).sum(1).T ** 2
    want = np.sqrt(total / len(tokens))
    assert np.allclose(scores, want, rtol=1e-3, atol=1e-4 * want.max()), "summed over positions, RMS over windows"
    assert (scores > 0).sum() > M


def by_hand_mlp_curve(model, params, tokens, up, down, w, scores):
    clean, x = value(model, params, tokens, LN2)
    u = core.encode(*up, x.reshape(-1, D))
    order = np.argsort(-scores, axis=None, kind="stable")
    n_scored = int((scores > 0).sum())

    spliced = replaced(model, MLP_OUT)

    def splice(codes):
        return kl_by_hand(clean, spliced(params, tokens, core.decode(*down, codes).reshape(x.shape)))

    curve = []
    for n in edges.edge_counts(n_scored):
        keep = np.zeros(scores.size, bool)
        keep[order[:n]] = True
        keep = keep.reshape(scores.shape)
        codes = np.zeros((len(u), M), np.float32)
        for j in range(M):
            if keep[:, j].any():
                y = train.mlp(w, core.decode(*up, u * keep[:, j]))
                codes[:, j] = core.encode(*down, y)[:, j]
        curve.append([n, splice(codes)])
    whole = core.encode(*down, train.mlp(w, core.decode(*up, u)))
    return curve, splice(whole), n_scored


def test_mlp_curve(world):
    model, params, tokens, w, up, down, _, _ = world
    rng = np.random.default_rng(3)
    scores = rng.random((M, M)) * (rng.random((M, M)) < 0.4)  # some edges unscored
    scores[0, :3] = scores[0, 3]  # ties go to the lower ids
    got = edges.mlp_curve(model, params, tokens, up, down, w, scores)
    curve, pair_kl, n_scored = by_hand_mlp_curve(model, params, tokens, up, down, w, scores)
    assert sorted(got) == ["curve", "pair_kl", "scored_edges"] and got["scored_edges"] == n_scored
    assert [n for n, _ in got["curve"]] == [n for n, _ in curve]
    assert np.allclose([v for _, v in got["curve"]], [v for _, v in curve], rtol=1e-4, atol=1e-7)
    assert got["pair_kl"] == pytest.approx(pair_kl, rel=1e-4, abs=1e-7)
    assert len({round(v, 6) for _, v in curve}) > 3, "the test's curve moves"
    flat = np.ones((M, M))  # every edge tied: the order is the tie-break alone
    got, want = edges.mlp_curve(model, params, tokens, up, down, w, flat), by_hand_mlp_curve(model, params, tokens, up, down, w, flat)
    assert np.allclose([v for _, v in got["curve"]], [v for _, v in want[0]], rtol=1e-4, atol=1e-7), "ties to the lower ids"
    short = edges.mlp_curve(model, params, tokens, up, down, w, scores, points=3)
    assert [n for n, _ in short["curve"]] == edges.edge_counts(n_scored, 3)
    for bad in (scores[:-1], np.zeros((M, M))):
        with pytest.raises(ValueError):
            edges.mlp_curve(model, params, tokens, up, down, w, bad)


def test_block_curve(world):
    model, params, tokens, _, _, _, resid, stair = world
    rng = np.random.default_rng(4)
    for up, down in (resid, ((stair, 0), (stair, 1))):
        m_up, m_dn = up[0].settings.width, down[0].settings.width
        scores = rng.random((m_up, m_dn)) * (rng.random((m_up, m_dn)) < 0.3)
        got = edges.block_curve(model, params, tokens, up, down, scores, points=6)
        clean, x = value(model, params, tokens, RESID[0])
        u = core.encode(*up, x.reshape(-1, D)).reshape(x.shape[:2] + (m_up,))
        through, spliced = replaced(model, RESID[0], RESID[1]), replaced(model, RESID[1])

        def downstream(code):
            v = core.decode(*up, code.reshape(-1, m_up)).reshape(x.shape)
            y = np.asarray(through(params, tokens, v))
            return core.encode(*down, y.reshape(-1, D)).reshape(x.shape[:2] + (m_dn,))

        def splice(codes):
            v = core.decode(*down, codes.reshape(-1, m_dn)).reshape(x.shape)
            return kl_by_hand(clean, spliced(params, tokens, v))

        order = np.argsort(-scores, axis=None, kind="stable")
        want = []
        for n in edges.edge_counts(int((scores > 0).sum()), 6):
            keep = np.zeros(scores.size, bool)
            keep[order[:n]] = True
            keep = keep.reshape(scores.shape)
            codes = np.zeros(x.shape[:2] + (m_dn,), np.float32)
            for j in np.flatnonzero(keep.any(0)):
                codes[..., j] = downstream(u * keep[:, j])[..., j]  # masked at every position
            want.append([n, splice(codes)])
        assert [n for n, _ in got["curve"]] == [n for n, _ in want], up[0].settings.kind
        assert np.allclose([v for _, v in got["curve"]], [v for _, v in want], rtol=1e-4, atol=1e-7), up[0].settings.kind
        assert got["pair_kl"] == pytest.approx(splice(downstream(u)), rel=1e-4, abs=1e-7)
        assert got["scored_edges"] == int((scores > 0).sum())


def test_pairs_are_checked(world):
    model, params, tokens, w, up, down, resid, stair = world
    other_mlp = (make("topk", [("blocks.0.hook_mlp_out",) * 2], 7), 0)
    narrow = core.Dictionary(core.Settings("topk", M, K, steps=10, batch=4), [(RESID[1],) * 2],
                             {n: np.asarray(v[..., :8] if n in ("W_dec", "b_dec") else v, np.float32)
                              for n, v in arrays(1, 1, M, 8).items()} | {"W_enc": np.zeros((1, 8, M), np.float32),
                                                                         "b_pre": np.zeros((1, 8), np.float32)})
    scores = np.ones((M, M))
    cases = [
        lambda: edges.mlp_edge_scores(model, params, tokens, up, other_mlp, w),  # another block's MLP
        lambda: edges.mlp_edge_scores(model, params, tokens, down, up, w),  # backwards
        lambda: edges.mlp_edge_scores(model, params, tokens, *resid, w),  # not an MLP pair
        lambda: edges.mlp_jacobian(up, down, {**w, "W_fc": w["W_fc"][:8]}, np.zeros((1, M), np.float32)),
        lambda: edges.mlp_jacobian(up, down, w, np.zeros((1, M + 1), np.float32)),
        lambda: edges.block_edge_scores(model, params, tokens, resid[1], resid[0]),  # backwards
        lambda: edges.block_edge_scores(model, params, tokens, (stair, 0), (stair, 0)),  # not the next block
        lambda: edges.block_edge_scores(model, params, tokens, up, down),  # not a block pair
        lambda: edges.block_edge_scores(model, params, tokens, resid[0], (narrow, 0)),  # widths that don't meet
        lambda: edges.block_curve(model, params, tokens, *resid, scores[:, :-1]),
        lambda: edges.block_curve(model, params, tokens, (stair, 0), (stair, 2), scores),  # no site 2
    ]
    for case in cases:
        with pytest.raises(ValueError):
            case()
