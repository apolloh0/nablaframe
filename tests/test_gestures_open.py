"""Phase 6 delta 3: open-this-up (D51).

Contract: src/nablaframe/gestures/open.py. The toy models are in phase6_fixtures.py: a target after block 1 reads its
token's direction from block 0's four latents at its own position, and nothing moves between positions. Needs the jax
lane.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import phase6_fixtures as F  # noqa: E402
from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.gestures import open as op  # noqa: E402
from nablaframe.gestures import records, sets  # noqa: E402


@pytest.fixture(scope="module")
def identity():
    return F.world("identity")


@pytest.fixture(scope="module")
def mapping():
    return F.world("mapping")


def opened(w, key, cls="every", **kwargs):
    """``open_up`` from block 0 into block 1, once per world and key."""
    if key not in w.cache:
        w.cache[key] = op.open_up(F.reading(w, 0), F.reading(w, 1), cls, **kwargs)
    return w.cache[key]


def recordable(results):
    assert json.loads(json.dumps(results)) == results
    args = {"dictionary": "1" * 64, "site": 0, "class": "every", "upstream": "4" * 64, "upstream_site": 0, "latent": 0}
    request = records.GestureRequest(kind="open", run="0" * 64, step=0, args=args, instances=12, seed=0)
    records.GestureRecord(request, {"states": {"0": "2" * 64}, "dictionaries": ["1" * 64, "4" * 64],
                                    "windows": ["3" * 64]}, {"nablaframe.gestures.open": "a" * 40}, results)


def forward(w, up, down, target, codes):
    """The target's pre-activation with the upstream output tap holding decode(codes) + err, built from the
    contract with numpy and the model's own pass."""
    i, p = target.window, target.position
    tokens = up.tokens[[i]]
    D, m = up.dictionary, up.dictionary.settings.width

    def padded(c):
        out = np.zeros((F.SEQ, m), np.float32)
        out[:, :up.width] = c
        return out

    _, rec = w.model.instrumented(w.params, tokens, [up.output_tap])
    x = np.asarray(rec[up.output_tap])[0]
    held = core.decode(D, up.site, padded(np.asarray(codes, np.float32))) + (x - core.decode(D, up.site, padded(up.codes[i])))
    _, rec = w.model.instrumented(w.params, tokens, [down.input_tap], {up.output_tap: lambda v: held[None]})
    return float(op.pre_activation(down.dictionary, down.site, np.asarray(rec[down.input_tap])[0, p][None],
                                   target.latents)[0])


def offset(d):
    """``d`` with a pre-encoder bias, a decoder bias and an output scale, none of which changes its codes or what its
    latents write: the biases lie along directions no token uses, and the scale is undone in the decoder."""
    E = F.directions()
    a = d.arrays
    arrays = dict(a, b_pre=a["b_pre"] + 0.4 * E[13], b_dec=a["b_dec"] + 0.3 * E[14], W_dec=a["W_dec"] / 1.5,
                  scale_out=a["scale_out"] * 1.5)
    return core.Dictionary(d.settings, d.sites, {n: np.array(v, np.float32) for n, v in arrays.items()})


def random_dictionary(kind, sites, seed=0, m=24, d=F.D_MODEL):
    rng = np.random.default_rng(seed)
    S, U = len(sites), 2 if kind == "jsae" else 1
    arrays = {"W_enc": rng.standard_normal((U, d, m)), "W_dec": rng.standard_normal((U, m, d)),
              "b_enc": rng.normal(0, 0.3, (S, m)), "b_pre": rng.normal(0, 0.3, (S, d)), "b_dec": rng.normal(0, 0.1, (S, d)),
              "scale_in": rng.uniform(0.5, 2, S), "scale_out": rng.uniform(0.5, 2, S)}
    return core.Dictionary(core.Settings(kind, m, 4, steps=1, batch=1), sites,
                           {n: v.astype(np.float32) for n, v in arrays.items()})


def test_the_pre_activation(identity, mapping):
    rng = np.random.default_rng(1)
    y = rng.standard_normal((5, F.D_MODEL))
    for kind, sites, site in (("topk", [("blocks.0.hook_resid_post",) * 2], 0),
                              ("jsae", [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2], 1)):
        d = random_dictionary(kind, sites)
        a, u = d.arrays, d.unit(site)
        expected = sum(((y / a["scale_in"][site] - a["b_pre"][site]) @ a["W_enc"][u][:, l].astype(np.float64)
                        + a["b_enc"][site, l]) for l in (3, 7))
        got = op.pre_activation(d, site, y, [3, 7])
        assert got.dtype == np.float64 and got.shape == (5,)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
    stair = random_dictionary("staircase", [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(2)])
    op.pre_activation(stair, 0, y, [11])
    for bad in ([12], [], [-1], [True]):
        with pytest.raises(ValueError):
            op.pre_activation(stair, 0, y, bad)

    # evaluate with the unintervened codes is the downstream latents' own pre-activation; with others, the contract's
    shifted = F.world("mapping")
    shifted.dictionaries = {b: offset(d) for b, d in shifted.dictionaries.items()}
    np.testing.assert_allclose(F.reading(shifted, 0).codes, F.reading(mapping, 0).codes, atol=1e-5)
    for w in (identity, mapping, shifted):
        up, down = F.reading(w, 0), F.reading(w, 1)
        direction = (lambda t: t) if w is identity else F.G.__getitem__
        targets = [op.Target(i, p, (F.latent(direction(int(down.tokens[i, p]))), F.SHARED_LATENT))
                   for i, p in ((0, 1), (5, 4), (9, 0))]
        got = op.evaluate(up, down, targets, up.codes[[t.window for t in targets]], chunk=2)
        assert got.dtype == np.float64
        for t, g in zip(targets, got):
            _, rec = w.model.instrumented(w.params, up.tokens[[t.window]], [down.input_tap])
            y_t = np.asarray(rec[down.input_tap])[0, t.position][None]
            assert g == pytest.approx(op.pre_activation(down.dictionary, 0, y_t, t.latents)[0], abs=1e-4)
        codes = up.codes[[t.window for t in targets]] * np.float32(0.4)
        got = op.evaluate(up, down, targets, codes)
        for t, g, c in zip(targets, got, codes):
            assert g == pytest.approx(forward(w, up, down, t, c), abs=1e-4)
    # a downstream dictionary with random weights, biases and scales: its pre-activation is computed in full
    up = F.reading(mapping, 0)
    rand = random_dictionary("topk", [("blocks.1.hook_resid_post",) * 2], seed=5)
    down = sets.read_site(mapping.model, mapping.params, n_layers=2, declaration=mapping.declaration, dictionary=rand,
                          site=0, windows=mapping.windows)
    targets = [op.Target(i, p, (3, 7)) for i, p in ((0, 1), (5, 4))]
    for codes in (up.codes[[0, 5]], up.codes[[0, 5]] * np.float32(0.3)):
        got = op.evaluate(up, down, targets, codes)
        for t, g, c in zip(targets, got, codes):
            assert g == pytest.approx(forward(mapping, up, down, t, c), rel=1e-5, abs=1e-4)

    up, down = F.reading(identity, 0), F.reading(identity, 1)
    t = [op.Target(0, 1, (F.latent(2),))]
    for bad in (lambda: op.evaluate(down, up, t, up.codes[[0]]), lambda: op.attribute(down, up, t),
                lambda: op.attribute(up, F.reading(mapping, 1), t),
                lambda: op.attribute(up, down, [op.Target(0, 1, (F.WIDTH,))]),
                lambda: op.attribute(up, down, [op.Target(0, 1, ())]), lambda: op.attribute(up, down, [])):
        with pytest.raises(ValueError):
            bad()


def test_attribution_matches_finite_differences(identity, mapping):
    w = mapping
    up, down = F.reading(w, 0), F.reading(w, 1)
    targets = [op.Target(i, p, (F.latent(F.G[int(up.tokens[i, p])]), F.latent(F.G[int(up.tokens[i, p])], 2)))
               for i, p in ((2, 3), (7, 5), (11, 0))]
    A = op.attribute(up, down, targets, chunk=2)
    assert A.codes.dtype == np.float32 and A.attributions.dtype == np.float64
    assert A.codes.shape == A.attributions.shape == (3, F.SEQ, F.WIDTH)
    eps = 0.02
    for k, t in enumerate(targets):
        np.testing.assert_array_equal(A.codes[k], up.codes[t.window])
        assert not A.attributions[k, t.position + 1:].any(), "no attribution after the target's position"
        assert A.full[k] == pytest.approx(forward(w, up, down, t, A.codes[k]), abs=1e-4)
        assert A.none[k] == pytest.approx(forward(w, up, down, t, np.zeros_like(A.codes[k])), abs=1e-4)
        token = int(up.tokens[t.window, t.position])
        parts = [(t.position, F.latent(token, j)) for j in range(4)]
        for p, l in parts[:2]:
            slopes = []
            for s in range(op.IG_STEPS):
                at = A.codes[k] * np.float32((s + 0.5) / op.IG_STEPS)
                hi, lo = at.copy(), at.copy()
                hi[p, l] += eps
                lo[p, l] -= eps
                slopes.append((forward(w, up, down, t, hi) - forward(w, up, down, t, lo)) / (2 * eps))
            assert A.attributions[k, p, l] == pytest.approx(A.codes[k, p, l] * np.mean(slopes), rel=0.03, abs=2e-3)
        assert A.attributions[k, t.position, parts[0][1]] != 0

    # where the pre-activation is linear in the codes, the attributions add up to the whole change
    up, down = F.reading(identity, 0), F.reading(identity, 1)
    targets = [op.Target(i, p, (F.latent(int(up.tokens[i, p])),)) for i, p in ((1, 2), (4, 5))]
    A = op.attribute(up, down, targets)
    np.testing.assert_allclose(A.attributions.sum(axis=(1, 2)), A.full - A.none, rtol=1e-4)


def test_absolute_order_and_ties():
    a = np.array([[0.5, -2.0, 0.0], [2.0, -0.5, 0.0]])
    codes = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert op.order(a, codes, "absolute").tolist() == [1, 3, 0, 4, 2, 5], "ties to the lower position, then latent"
    assert op.order(a, codes, "signed").tolist() == [3, 0, 2, 5, 4, 1], "largest first, with the same ties"
    got = op.order(a, codes, "random", np.random.default_rng(3))
    rng = np.random.default_rng(3)
    assert got.tolist() == rng.permutation([0, 1, 4, 5]).tolist() + [2, 3], "the active parts shuffled, then the rest"
    assert got.dtype == np.int64
    for bad in (lambda: op.order(a, codes[:, :2], "absolute"), lambda: op.order(a[0], codes[0], "absolute"),
                lambda: op.order(a, codes, "loudest"), lambda: op.order(a, codes, "random")):
        with pytest.raises(ValueError):
            bad()


def test_the_error_curve_and_reading(identity):
    res = opened(identity, "five", latent=F.latent(5), instances=12, seed=0)
    up, down = F.reading(identity, 0), F.reading(identity, 1)
    assert (res["class"], res["steps"], res["parts"]) == ("every", op.IG_STEPS, list(op.PARTS))
    assert res["target"] == {"latents": [F.latent(5)], "latent": F.latent(5), "description": None, "value": None}
    assert res["up"] == {"site": 0, "tap": "blocks.0.hook_resid_post", "block": 0}
    assert res["down"] == {"site": 0, "tap": "blocks.1.hook_resid_post", "block": 1}
    assert res["candidates"] == int((up.tokens == 5).sum()) == 54
    assert (res["instances"], res["no_span"], res["measured"]) == (12, 0, 12)
    # each of the four latents writes a quarter of the token's direction, so one leaves 3/4 of the change unmade
    a = res["orders"]["absolute"]
    np.testing.assert_allclose(a["error_median"], [0.75, 0.5, 0, 0, 0, 0, 0, 0], atol=1e-4)
    assert a["readings"] == [4] * 12 and a["reading_median"] == 4.0 and a["within"] == 1.0
    # one instance, from the contract
    i, p = res["windows"][0]
    t = op.Target(i, p, (F.latent(5),))
    A = op.attribute(up, down, [t])
    order = op.order(A.attributions[0, :p + 1], A.codes[0, :p + 1], "absolute")
    errors = []
    for size in op.PARTS:
        kept = np.zeros_like(A.codes[0])
        flat, src = kept.reshape(-1), A.codes[0].reshape(-1)
        flat[order[:size]] = src[order[:size]]
        errors.append(abs(forward(identity, up, down, t, kept) - A.full[0]) / abs(A.full[0] - A.none[0]))
    assert next(s for s, e in zip(op.PARTS, errors) if e <= op.ERROR) == a["readings"][0]
    recordable(res)

    named = opened(identity, "five set", description="token(p)", value=5, instances=12, seed=0)
    assert named["target"] == {"latents": [F.latent(5, j) for j in range(4)], "latent": None,
                               "description": "token(p)", "value": 5}
    assert named["orders"]["absolute"]["readings"] == [4] * 12
    # a name's instances are where its set fires and its description holds its value: token(p+1) = 5's set is
    # token 4's latents, which fire on every 4 but precede a 5 only sometimes
    S = list(down.table("every").set_for("token(p+1)", 5))
    fires = down.codes[:, :, S].sum(axis=-1) > 0
    nxt = op.open_up(up, down, "every", description="token(p+1)", value=5, instances=2, seed=0)
    assert nxt["candidates"] == int((fires & (down.described["token(p+1)"] == 5)).sum()) < int(fires.sum())
    assert opened(identity, "five", latent=F.latent(5), instances=12, seed=0) == res
    assert op.open_up(up, down, "every", latent=F.latent(5), instances=12, seed=0) == res, "one seed, one result"
    for bad in (dict(), dict(latent=1, description="token(p)", value=5), dict(latent=F.WIDTH), dict(latent=True),
                dict(description="token(p)", value=0), dict(description="token(p)"), dict(latent=1, instances=0),
                dict(latent=1, seed=-1), dict(latent=1, cls="nope")):
        kwargs = dict(instances=12, seed=0, cls="every") | bad
        with pytest.raises(ValueError):
            op.open_up(up, down, kwargs.pop("cls"), **kwargs)


def test_signed_and_random_orders(identity):
    # decoders that write minus the direction: every attribution is negative, so signed order puts the set last
    negative = F.world("identity", decoder="negative")
    res = op.open_up(F.reading(negative, 0), F.reading(negative, 1), "late", latent=F.latent(5), instances=8, seed=1)
    assert res["orders"]["absolute"]["readings"] == [4] * res["measured"]
    assert res["orders"]["signed"]["readings"] == [None] * res["measured"]
    assert res["orders"]["signed"]["reading_median"] is None and res["orders"]["signed"]["within"] == 0.0
    assert all(p["attribution"] < 0 for parts in res["top_parts"] for p in parts[:4])

    up, down = F.reading(identity, 0), F.reading(identity, 1)
    res = op.open_up(up, down, "late", latent=F.latent(6), instances=10, seed=2)
    candidates = [(i, p) for i in range(99) for p in (3, 4, 5) if up.tokens[i, p] == 6]
    rng = np.random.default_rng(2)
    if len(candidates) > 10:
        candidates = [candidates[k] for k in sorted(rng.choice(len(candidates), size=10, replace=False))]
    targets = [op.Target(i, p, (F.latent(6),)) for i, p in candidates]
    orders = [op.order(np.zeros((p + 1, F.WIDTH)), up.codes[i, :p + 1], "random", rng) for i, p in candidates]
    A = op.attribute(up, down, targets)
    kept = []
    for t, o, c in zip(targets, orders, A.codes):
        for size in op.PARTS:
            k = np.zeros_like(c)
            k.reshape(-1)[o[:size]] = c.reshape(-1)[o[:size]]
            kept.append(k)
    got = op.evaluate(up, down, [t for t in targets for _ in op.PARTS], np.stack(kept)).reshape(len(targets), -1)
    errors = np.abs(got - A.full[:, None]) / np.abs(A.full - A.none)[:, None]
    expected = [next((s for s, e in zip(op.PARTS, row) if e <= op.ERROR), None) for row in errors]
    assert res["windows"] == [list(c) for c in candidates]
    assert res["orders"]["random"]["readings"] == expected, "each instance's shuffle drawn in turn from the seed"
    assert res["orders"]["random"]["reading_median"] > res["orders"]["absolute"]["reading_median"] == 4.0
    assert res["orders"]["signed"]["readings"] == res["orders"]["absolute"]["readings"]


def test_no_span_is_left_out(identity):
    res = opened(identity, "shared", latent=F.SHARED_LATENT, instances=12, seed=0)
    assert res["instances"] == res["no_span"] == 12 and res["measured"] == 0, "a constant pre-activation is left out"
    assert res["windows"] == [] and res["top_parts"] == []
    assert res["orders"] == {kind: {"error_median": None, "readings": [], "reading_median": None, "within": None}
                             for kind in op.ORDERS}
    recordable(res)
    group = opened(identity, "group", latent=F.GROUP_LATENT, instances=12, seed=0)
    assert group["no_span"] == 0 and group["measured"] == 12
    dead = opened(identity, "dead", latent=F.DEAD_LATENT, instances=12, seed=0)
    assert (dead["candidates"], dead["instances"], dead["measured"]) == (0, 0, 0)


def test_top_parts_carry_their_names(identity):
    classes = SimpleNamespace(run="toy", classes={"every": range(6), "late": (3, 4, 5), "mid": (2, 3, 4)})
    assert [op.class_of(classes, p) for p in range(6)] == ["every", "every", "mid", "late", "late", "late"], \
        "the smallest class holding the position, ties to the one declared first"
    with pytest.raises(ValueError):
        op.class_of(classes, 6)

    res = opened(identity, "seven", latent=F.latent(7), instances=12, seed=3)
    up, down = F.reading(identity, 0), F.reading(identity, 1)
    targets = [op.Target(i, p, (F.latent(7),)) for i, p in res["windows"]]
    A = op.attribute(up, down, targets)
    assert len(res["top_parts"]) == res["measured"] == 12
    for (i, p), parts, attr in zip(res["windows"], res["top_parts"], A.attributions):
        assert len(parts) == op.TOP_PARTS
        top = parts[:4]
        assert {(x["position"], x["latent"]) for x in top} == {(p, F.latent(7, j)) for j in range(4)}
        assert all(x["attribution"] == pytest.approx(attr[x["position"], x["latent"]], abs=1e-12) for x in parts)
        cls = op.class_of(identity.declaration, p)
        assert all(x["name"] == {"description": "token(p)", "value": 7, "f1": 1.0, "label": "named"} for x in top), cls
        rest = sorted((q, l) for q in range(p + 1) for l in range(F.WIDTH) if (q, l) not in
                      {(x["position"], x["latent"]) for x in top})[:4]
        assert [(x["position"], x["latent"]) for x in parts[4:]] == rest, "zero attributions in part order"

    # the shared latent, made to write a little of token 7's direction, ranks fifth at the target's position; it is
    # named at the smallest class there, and only weakly, by the token it fires most often beside
    leaky = F.world("identity")
    d = leaky.dictionaries[0]
    W_dec = d.arrays["W_dec"].copy()
    W_dec[0, F.SHARED_LATENT] = 0.2 * F.directions()[7]
    leaky.dictionaries[0] = core.Dictionary(d.settings, d.sites, dict(d.arrays, W_dec=W_dec))
    up = F.reading(leaky, 0)
    res = op.open_up(up, F.reading(leaky, 1), "every", latent=F.latent(7), instances=12, seed=3)
    names = {cls: up.table(cls).candidates(F.SHARED_LATENT)[0] for cls in ("every", "late")}
    assert names["every"][:2] != names["late"][:2] and max(f for *_, f in names.values()) < 0.5
    assert {p for _, p in res["windows"]} & {3, 4, 5} and {p for _, p in res["windows"]} - {3, 4, 5}
    for (i, p), parts in zip(res["windows"], res["top_parts"]):
        fifth = parts[4]
        d_, v_, f_ = names[op.class_of(leaky.declaration, p)]
        assert (fifth["position"], fifth["latent"]) == (p, F.SHARED_LATENT) and fifth["attribution"] > 0
        assert fifth["name"] == {"description": d_, "value": v_, "f1": f_, "label": "weak"}
