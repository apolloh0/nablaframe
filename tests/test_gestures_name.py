"""Phase 6 delta 2: name-this-direction (D49), and the site readings, F1 tables, null and steering it shares with
assert-this-edge.

Contracts: src/nablaframe/gestures/sets.py and name.py. The toy models are in phase6_fixtures.py. Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import phase6_fixtures as F  # noqa: E402
from nablaframe.dictionaries import core, measure  # noqa: E402
from nablaframe.dictionaries.edges import kl  # noqa: E402
from nablaframe.gestures import declare, records  # noqa: E402
from nablaframe.gestures import name as nm  # noqa: E402
from nablaframe.gestures import sets  # noqa: E402

TAP0 = "blocks.0.hook_resid_post"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def identity():
    return F.world("identity")


@pytest.fixture(scope="module")
def mapping():
    return F.world("mapping")


def plain(x) -> bool:
    """Whether ``x`` is made only of JSON's own types."""
    if isinstance(x, dict):
        return all(type(k) is str and plain(v) for k, v in x.items())
    if isinstance(x, list):
        return all(plain(v) for v in x)
    return x is None or type(x) in (str, int, float, bool)


def recordable(results, kind="name", args=None):
    """The results make a gesture record."""
    assert plain(results) and json.loads(json.dumps(results)) == results
    args = args or {"dictionary": "1" * 64, "site": 0, "class": "every", "latent": 0}
    request = records.GestureRequest(kind=kind, run="0" * 64, step=0, args=args, instances=40, seed=1)
    records.GestureRecord(request, {"states": {"0": "2" * 64}, "dictionaries": ["1" * 64], "windows": ["3" * 64]},
                          {"nablaframe.gestures.name": "a" * 40}, results)


def synthetic():
    """Codes and descriptions with known F1s at a class of three of five positions (1, 2 and 4)."""
    rng = np.random.default_rng(3)
    B, seq, m = 30, 5, 12
    positions = (1, 2, 4)
    a = rng.integers(0, 3, (B, seq))
    a[:5, 1] = -1  # rows with nothing count among a latent's active rows
    e = np.full((B, seq), -1)
    e[:6, 2] = 3  # one value at least 5 times at the class: doesn't count
    e[6:10, 2] = 4
    e[:, 0] = 4  # many more, at a position outside the class
    f = np.zeros((B, seq), np.int64)
    f[:5, 4] = 1  # exactly 5 times: counts
    described = {"a": a, "b": a.copy(), "c": np.full((B, seq), 7), "e": e, "f": f}
    codes = np.zeros((B, seq, m), np.float32)
    live = lambda shape: rng.uniform(0.5, 2.0, shape).astype(np.float32)  # noqa: E731
    codes[..., 0] = np.where(a == 1, live((B, seq)), 0)
    codes[:, 0, 0] = 1.0  # outside the class: doesn't count
    codes[..., 1] = np.where((a == 1) | (a == -1), live((B, seq)), 0)
    codes[..., 2] = live((B, seq))
    rows = [(i, p) for i in range(B) for p in positions]
    for latent, n_active in ((3, 81), (4, 80)):  # 81 of 90 is 0.9, shared; 80 isn't
        for i, p in rows[:n_active]:
            codes[i, p, latent] = 1.5
    codes[:, 3, 5] = 1.0  # active only outside the class
    codes[..., 6] = codes[..., 0]  # ties with latent 0
    codes[..., 7:] = np.where(rng.random((B, seq, 5)) < 0.3, live((B, seq, 5)), 0)
    return codes, described, positions


def brute_f1(codes, values, positions, keep):
    rows = codes[:, list(positions)].reshape(-1, codes.shape[-1]) > 0
    vals = values[:, list(positions)].reshape(-1)
    out = np.zeros((codes.shape[-1], len(keep)))
    for i in range(codes.shape[-1]):
        for j, v in enumerate(keep):
            out[i, j] = 2 * np.sum(rows[:, i] & (vals == v)) / (rows[:, i].sum() + np.sum(vals == v))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_candidates_rank_by_f1():
    codes, described, positions = synthetic()
    T = sets.ClassTable(codes, described, positions)
    assert T.rows == 90
    assert T.descriptions == ("a", "b", "f"), "a constant, and a value counted 4 times, count for nothing"
    np.testing.assert_array_equal(T.values("f"), [0, 1])
    rows = codes[:, list(positions)].reshape(-1, 12) > 0
    np.testing.assert_array_equal(T.active, rows.sum(axis=0))
    assert T.active.dtype == np.int64
    assert T.shared == (2, 3)
    brute = {}
    for d in T.descriptions:
        keep = declare.values_at(described[d], list(positions))
        np.testing.assert_array_equal(T.values(d), keep)
        brute[d] = brute_f1(codes, described[d], positions, keep)
        assert T.f1(d).dtype == np.float64 and T.f1(d).shape == (12, len(keep))
        np.testing.assert_allclose(T.f1(d), brute[d], rtol=0, atol=1e-12)
    assert T.f1("a")[1, list(T.values("a")).index(1)] < 1, "rows where a is -1 count against a latent"
    for latent in range(12):
        if latent == 5:
            continue
        everything = sorted((-brute[d][latent, j], i, int(v), d) for i, d in enumerate(T.descriptions)
                            for j, v in enumerate(T.values(d)))
        got = T.candidates(latent)
        assert [(d, v) for d, v, _ in got] == [(d, v) for _, _, v, d in everything[:sets.CANDIDATES]], latent
        np.testing.assert_allclose([f for *_, f in got], [-f for f, *_ in everything[:sets.CANDIDATES]], atol=1e-12)
        assert all(type(d) is str and type(v) is int and type(f) is float for d, v, f in got)
    assert [d for d, _, _ in T.candidates(0)[:2]] == ["a", "b"], "a tie goes to the earlier description"
    assert T.candidates(5) == (), "a latent active only outside the class has no candidates"
    assert T.has("a", 1) and T.has("f", 1)
    for d, v in (("c", 7), ("e", 3), ("a", 5), ("zz", 1), ("a", True)):
        assert not T.has(d, v)
    for bad in (lambda: T.ranked("c", 7), lambda: T.mean_codes("a", 9), lambda: T.set_for("e", 3),
                lambda: T.name_f1("zz", 1, [0]), lambda: T.candidates(12), lambda: T.candidates(-1)):
        with pytest.raises(ValueError):
            bad()
    for bad in (lambda: sets.ClassTable(codes, described, (1, 1)), lambda: sets.ClassTable(codes, described, (5,)),
                lambda: sets.ClassTable(codes[0], described, positions),
                lambda: sets.ClassTable(codes, {"a": described["a"][:, :4]}, positions)):
        with pytest.raises(ValueError):
            bad()


def test_named_and_weak(identity):
    r = F.reading(identity, 0)
    named = nm.name(r, "every", F.latent(5), instances=24, seed=1)
    assert (named["class"], named["latent"], named["windows"], named["rows"]) == ("every", F.latent(5), 99, 594)
    assert named["candidates"][0] == {"description": "token(p)", "value": 5, "f1": 1.0}
    assert len(named["candidates"]) == sets.CANDIDATES
    assert named["name"] == {"description": "token(p)", "value": 5, "f1": 1.0, "label": nm.NAMED}

    # tokens 9, 10 and 11 sit at the class equally often, so the group latent's F1 for each is exactly 0.5
    group = nm.name(r, "every", F.GROUP_LATENT, instances=24, seed=1)
    assert group["name"] == {"description": "token(p)", "value": 9, "f1": 0.5, "label": nm.NAMED}, \
        "the floor itself names, and a tie goes to the lower value"

    weak = nm.name(r, "every", F.SHARED_LATENT, instances=24, seed=1)
    assert weak["name"]["label"] == nm.WEAK and weak["name"]["f1"] < sets.FLOOR
    assert weak["candidates"][0]["f1"] == weak["name"]["f1"]
    for key in ("set", "coverage", "effect", "steering"):
        assert weak[key] is not None, f"a weak name still gets its {key}"

    dead = nm.name(r, "every", F.DEAD_LATENT, instances=24, seed=1)
    assert dead == {"class": "every", "latent": F.DEAD_LATENT, "windows": 99, "rows": 594, "candidates": [],
                    "name": None}

    for results in (named, group, weak, dead):
        recordable(results)
    for bad in (dict(latent=F.WIDTH), dict(latent=-1), dict(latent=True), dict(cls="nope"), dict(instances=0),
                dict(instances=True), dict(seed=-1), dict(seed=1.0)):
        kwargs = dict(cls="every", latent=F.latent(5), instances=24, seed=1) | bad
        with pytest.raises(ValueError):
            nm.name(r, kwargs.pop("cls"), kwargs.pop("latent"), **kwargs)


def test_the_set_is_the_top_four(identity):
    codes, described, positions = synthetic()
    T = sets.ClassTable(codes, described, positions)
    col = list(T.values("a")).index(1)
    expected = np.argsort(-brute_f1(codes, described["a"], positions, T.values("a"))[:, col], kind="stable")
    ranked = T.ranked("a", 1)
    assert ranked.dtype == np.int64
    np.testing.assert_array_equal(ranked, expected)
    assert list(ranked[:2]) == [0, 6], "ties go to the lower latent"
    assert T.set_for("a", 1) == tuple(int(i) for i in expected[:sets.SET_SIZE])
    assert all(type(i) is int for i in T.set_for("a", 1))
    assert T.name_f1("a", 1, [1, 7, 8]) == pytest.approx(max(T.f1("a")[i, col] for i in (1, 7, 8)), abs=1e-12)
    rows = codes[:, list(positions)].reshape(-1, 12)
    mean = T.mean_codes("a", 1)
    assert mean.dtype == np.float32
    np.testing.assert_array_equal(
        mean, rows[described["a"][:, list(positions)].reshape(-1) == 1].astype(np.float64).mean(axis=0).astype(np.float32))

    r = F.reading(identity, 0)
    s = nm.name(r, "every", F.latent(7, 2), instances=24, seed=1)["set"]
    assert s == {"latents": [F.latent(7, j) for j in range(4)], "f1": [1.0] * 4, "name_f1": 1.0, "pointed_rank": 2,
                 "in_set": True}
    s = nm.name(r, "every", F.GROUP_LATENT, instances=24, seed=1)["set"]
    assert s["latents"] == [F.latent(9, j) for j in range(4)] and s["name_f1"] == 1.0
    assert (s["pointed_rank"], s["in_set"]) == (4, False), "the pointed latent need not be in its name's set"


def test_coverage(identity, mapping):
    r, w = F.reading(identity, 0), identity
    tokens = w.windows[:, :F.SEQ]
    assert r.tokens.dtype == np.int32 and r.windows.dtype == np.int64
    np.testing.assert_array_equal(r.tokens, tokens)
    assert (r.input_tap, r.output_tap, r.block, r.width, r.site) == (TAP0, TAP0, 0, F.WIDTH, 0)
    x = measure.activations(w.model, w.params, tokens.astype(np.int32), [TAP0])[TAP0]
    expected = core.encode(w.dictionaries[0], 0, x.reshape(-1, F.D_MODEL)).reshape(99, F.SEQ, -1)
    assert r.codes.dtype == np.float32 and r.codes.shape == (99, F.SEQ, F.WIDTH)
    np.testing.assert_allclose(r.codes, expected, atol=1e-6)
    assert list(r.described) == list(declare.DESCRIPTIONS)
    assert list(F.reading(mapping, 1).described) == list(declare.DESCRIPTIONS) + ["g"]
    np.testing.assert_allclose(F.read(w, 0, chunk=7).codes, r.codes, atol=1e-6)
    stair = sets.read_site(w.model, w.params, n_layers=2, declaration=w.declaration, dictionary=F.staircase(), site=0,
                           windows=w.windows)
    assert stair.width == F.WIDTH and stair.codes.shape == (99, F.SEQ, F.WIDTH), "only the site's active latents"
    np.testing.assert_allclose(stair.codes, r.codes, atol=1e-6)
    assert r.table("every") is r.table("every")
    for bad in (lambda: r.table("nope"), lambda: F.read(w, 0, chunk=0)):
        with pytest.raises(ValueError):
            bad()
    for windows in (w.windows[:, :F.SEQ], w.windows[0], w.windows.astype(np.float32), w.windows[:0]):
        with pytest.raises(ValueError):
            sets.read_site(w.model, w.params, n_layers=2, declaration=w.declaration, dictionary=w.dictionaries[0],
                           site=0, windows=windows)
    with pytest.raises(ValueError):
        sets.read_site(w.model, w.params, n_layers=2, declaration=w.declaration, dictionary=w.dictionaries[0], site=1,
                       windows=w.windows)

    def brute(latents, value):
        mass = r.codes[:, :, latents].astype(np.float64).sum(axis=-1)
        return {d: float(mass[arr == value].sum() / mass.sum()) for d, arr in r.described.items()}

    one = nm.coverage(r, [F.latent(5)], 5)
    assert list(one) == list(r.described)
    assert one == pytest.approx(brute([F.latent(5)], 5), abs=1e-12)
    assert one["token(p)"] == pytest.approx(1.0)
    four = [F.latent(5, j) for j in range(4)]
    assert nm.coverage(r, four, 5) == pytest.approx(brute(four, 5), abs=1e-12)
    assert nm.coverage(r, [F.DEAD_LATENT], 5) == {d: None for d in r.described}

    late = nm.name(r, "late", F.latent(5), instances=24, seed=1)
    assert late["coverage"]["latent"] == pytest.approx(brute([F.latent(5)], 5), abs=1e-12)
    assert late["coverage"]["set"] == pytest.approx(brute(four, 5), abs=1e-12)
    mass = r.codes[:, 3:, F.latent(5)].astype(np.float64)
    in_class = float(mass[r.described["token(p+1)"][:, 3:] == 5].sum() / mass.sum())
    assert late["coverage"]["latent"]["token(p+1)"] != pytest.approx(in_class), \
        "coverage is over every held-out position, not only the class's"


def test_the_set_effect_at_both_ends(identity):
    r, w = F.reading(identity, 0), identity
    latents = [F.latent(5), F.SHARED_LATENT, F.latent(2, 3)]
    one_hot = np.zeros((3, F.WIDTH), np.float32)
    one_hot[np.arange(3), latents] = 1
    e = core.decode_delta(w.dictionaries[0], 0, one_hot).astype(np.float64)
    e -= e.mean(axis=1, keepdims=True)
    expected = (e * np.asarray(w.params.ln_f.weight, np.float64)) @ np.asarray(w.params.wte, np.float64).T
    got = nm.latent_effects(w.dictionaries[0], 0, w.params, latents)
    assert got.dtype == np.float64 and got.shape == (3, F.VOCAB)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    effect = np.array([0.5, 2.0, -1.0, 2.0, 0.0, -1.0, 3.0, 0.5])
    assert nm.ends(effect, 3) == {"top": [[6, 3.0], [1, 2.0], [3, 2.0], [0, 0.5], [7, 0.5]],
                                  "bottom": [[2, -1.0], [5, -1.0], [4, 0.0], [0, 0.5], [7, 0.5]],
                                  "rank_from_top": 2, "rank_from_bottom": 6}
    assert (nm.ends(effect, 5)["rank_from_bottom"], nm.ends(effect, 1)["rank_from_top"]) == (1, 1)
    for bad in (8, -1, 2.0):
        with pytest.raises(ValueError):
            nm.ends(effect, bad)

    res = nm.name(r, "every", F.latent(5, 1), instances=24, seed=1)
    S = res["set"]["latents"]
    weights = r.table("every").mean_codes("token(p)", 5)[S].astype(np.float64)
    effects = nm.latent_effects(w.dictionaries[0], 0, w.params, S + [F.latent(5, 1)])
    assert res["effect"]["set"] == nm.ends(weights @ effects[:4], 5), "the set's effect, weighted by its mean codes"
    assert res["effect"]["latent"] == {"latent": F.latent(5, 1), **nm.ends(effects[4], 5)}
    assert res["effect"]["vocabulary"] == F.VOCAB
    assert res["effect"]["set"]["rank_from_top"] == 0 and res["effect"]["set"]["top"][0][0] == 5

    negative = F.world("identity", decoder="negative")
    res = nm.name(F.read(negative, 0), "every", F.latent(5, 1), instances=24, seed=1)
    assert res["effect"]["set"]["rank_from_bottom"] == 0 and res["effect"]["set"]["bottom"][0][0] == 5, \
        "a set that writes minus its token's direction ranks the token last"


def test_steering_follows_a_planted_variable(identity, mapping):
    q1 = F.reading(mapping, 1)
    res = nm.name(q1, "every", F.latent(5), instances=40, seed=3)
    assert res["name"] == {"description": "g", "value": 5, "f1": 1.0, "label": nm.NAMED}
    st = res["steering"]
    assert st["causal"] is True and st["instances"] == 40
    assert (st["follows"], st["unchanged"], st["ablated_changed"], st["null_follows"]) == (1.0, 0.0, 1.0, 0.0)
    assert st["null_unchanged"] >= 0.95
    assert nm.name(q1, "every", F.latent(5), instances=40, seed=3) == res, "one seed, one result"
    st = nm.name(F.reading(identity, 0), "every", F.latent(8), instances=40, seed=3)["steering"]
    assert (st["causal"], st["follows"], st["unchanged"], st["null_follows"]) == (True, 1.0, 0.0, 0.0)
    st = nm.name(F.reading(mapping, 0), "every", F.latent(3), instances=40, seed=3)["steering"]
    assert (st["causal"], st["follows"]) == (True, 1.0), "token(p) after block 0 predicts g of its new value"
    elsewhere = F.world("elsewhere")
    st = nm.name(F.read(elsewhere, 0), "every", F.latent(3), instances=40, seed=3)["steering"]
    assert (st["follows"], st["unchanged"], st["ablated_changed"]) == (1.0, 1.0, 0.0), \
        "the output is read where the run says, here the next position, which the edit can't reach"

    # the instances, from the contract
    T, g = q1.table("every"), q1.described["g"]
    values = [int(x) for x in T.values("g")]
    eligible = []
    for i in range(99):
        options = [u for u in values if u not in set(g[i].tolist())]
        eligible += [(i, p, options) for p in range(F.SEQ) if g[i, p] == 5 and options]
    causal, every = nm.draw_instances(q1, "every", "g", 5, limit=10_000, rng=np.random.default_rng(0))
    assert causal is True and len(every) == len(eligible) > 40
    assert [(x.window, x.position) for x in every] == [(i, p) for i, p, _ in eligible]
    for x, (_, _, options) in zip(every, eligible):
        assert x.w in options and (x.read, x.target) == (x.position, x.w)
        assert type(x.w) is int and type(x.read) is int and type(x.target) is int
    rng = np.random.default_rng(7)
    kept = [eligible[k] for k in sorted(rng.choice(len(eligible), size=10, replace=False))]
    expected = [(i, p, options[rng.integers(len(options))]) for i, p, options in kept]
    causal, ten = nm.draw_instances(q1, "every", "g", 5, limit=10, rng=np.random.default_rng(7))
    assert [(x.window, x.position, x.w) for x in ten] == expected
    other = nm.draw_instances(q1, "every", "g", 5, limit=10, rng=np.random.default_rng(8))[1]
    assert other != ten
    r0 = F.reading(identity, 0)
    for i in (0, 7):
        assert sets.taken(r0.described, "token(p+1)", i) == frozenset(r0.tokens[i].tolist()), \
            "for a token description, every token the window holds is taken"
        assert sets.taken(q1.described, "g", i) == frozenset(q1.described["g"][i].tolist())
    causal, nxt = nm.draw_instances(r0, "every", "token(p+1)", 5, limit=1000, rng=np.random.default_rng(0))
    assert causal is False and len(nxt) > 40
    assert all(x.w not in set(r0.tokens[x.window].tolist()) for x in nxt)
    for bad in (dict(v=2), dict(d="token(p-9)"), dict(limit=0), dict(limit=2.5)):
        kwargs = dict(d="g", v=5, limit=10) | bad
        with pytest.raises(ValueError):
            nm.draw_instances(q1, "every", kwargs["d"], kwargs["v"], limit=kwargs["limit"], rng=np.random.default_rng(0))


def test_the_null_avoids_the_sets_and_shared_latents():
    rng = np.random.default_rng(4)
    B, seq, m = 40, 4, 50
    a = rng.integers(0, 5, (B, seq))
    # latents 2 to 41: eight for each value, mostly active where a is that value, so each value's top 8 are its own
    # and the shared latents 0 and 1 fall outside every top 8; 42 to 44 active at random; 45 to 49 never
    p_active = np.where((np.arange(m)[None, None, :] - 2) // 8 == a[..., None], 0.8, 0.1)
    p_active[..., 42:45], p_active[..., 45:] = 0.4, 0.0
    codes = np.where(rng.random((B, seq, m)) < p_active, rng.uniform(0.2, 2, (B, seq, m)), 0).astype(np.float32)
    codes[..., 0] = 1.0
    codes[..., 1] = np.where(rng.random((B, seq)) < 0.95, 1.0, 0)
    T = sets.ClassTable(codes, {"a": a}, range(seq))
    assert set(T.shared) == {0, 1}
    for v in range(5):
        assert set(T.ranked("a", v)[:sets.EXCLUDE].tolist()) == set(range(2 + 8 * v, 10 + 8 * v))
    rows = [(int(i), int(p)) for i, p in zip(rng.integers(0, B, 300), rng.integers(0, seq, 300))]
    z = np.stack([codes[i, p] for i, p in rows])
    # a few rows whose only active latents are shared or among v's top 8, so nothing is left to remove
    for k in range(5):
        z[k] = 0
        z[k, [0, 1]] = 1.0
        z[k, T.ranked("a", int(a[rows[k]]))[:3]] = 1.0
    vs = [int(a[i, p]) for i, p in rows]
    ws = [int((v + 1 + rng.integers(4)) % 5) for v in vs]
    removed, added = sets.null_sets(T, "a", z, vs, ws, np.random.default_rng(5))
    assert len(removed) == len(added) == 300
    pool = set(np.flatnonzero(T.active > 0).tolist())
    assert pool == set(range(45))
    assert any(len(r) < sets.SET_SIZE for r in removed[5:]) and any(len(r) == sets.SET_SIZE for r in removed)
    shared = set(T.shared)
    draw = np.random.default_rng(5)
    for k in range(300):
        live = set(np.flatnonzero(z[k] > 0).tolist())
        r_ok = sorted(live - set(T.ranked("a", vs[k])[:sets.EXCLUDE].tolist()) - shared)
        a_ok = sorted(pool - set(T.ranked("a", ws[k])[:sets.EXCLUDE].tolist()) - shared - live)
        assert set(removed[k]) <= set(r_ok) and len(set(removed[k])) == len(removed[k]) == min(sets.SET_SIZE, len(r_ok))
        assert set(added[k]) <= set(a_ok) and len(set(added[k])) == len(added[k]) == min(sets.SET_SIZE, len(a_ok))
        assert all(type(i) is int for i in removed[k] + added[k])
        # and exactly as drawn by the contract, so a seed always gives the same null
        expect_r = [int(i) for i in draw.choice(r_ok, size=min(4, len(r_ok)), replace=False)] if r_ok else []
        expect_a = [int(i) for i in draw.choice(a_ok, size=min(4, len(a_ok)), replace=False)] if a_ok else []
        assert (removed[k], added[k]) == (expect_r, expect_a), k
    assert removed[:5] == [[]] * 5, "nothing eligible, nothing removed"
    with pytest.raises(ValueError):
        sets.null_sets(T, "a", z, vs[:-1], ws, np.random.default_rng(5))


def test_ablation(identity):
    st = nm.name(F.reading(identity, 0), "every", F.latent(4), instances=24, seed=2)["steering"]
    assert st["ablated_changed"] == 1.0, "removing a token's set leaves only token 0's direction"
    hollow = F.world("identity", decoder="hollow")
    st = nm.name(F.read(hollow, 0), "every", F.latent(4), instances=24, seed=2)["steering"]
    assert (st["ablated_changed"], st["follows"], st["unchanged"]) == (0.0, 0.0, 1.0), \
        "a set whose decoder writes nothing changes nothing when removed: the edit goes through the dictionary"

    # run_steered's arithmetic, from the contract
    r, w = F.reading(identity, 0), identity
    index, positions = [0, 1, 2, 3, 4], [1, 1, 2, 3, 3]
    remove = [[F.latent(int(r.tokens[0, 1]))], [], list(range(F.latent(int(r.tokens[2, 2])),
                                                               F.latent(int(r.tokens[2, 2])) + 4)), [], [F.SHARED_LATENT]]
    add = [[], [F.latent(7), F.latent(7, 1)], [F.latent(3)], [], [F.latent(2)]]
    amounts = [[], [0.5, 1.5], [0.95], [], [2.0]]
    logits, recorded = sets.run_steered(r, index, positions, remove, add, amounts, taps=[TAP0])
    assert logits.dtype == np.float32 and logits.shape == (5, F.SEQ, F.VOCAB)
    change = np.zeros((5, F.SEQ, F.WIDTH), np.float32)
    for k in range(5):
        for l in remove[k]:
            change[k, positions[k], l] -= r.codes[index[k], positions[k], l]
        for l, amount in zip(add[k], amounts[k]):
            change[k, positions[k], l] += amount
    delta = core.decode_delta(w.dictionaries[0], 0, change.reshape(-1, F.WIDTH)).reshape(5, F.SEQ, -1)
    toks = r.tokens[index]
    expect, rec = w.model.instrumented(w.params, toks, [TAP0], {TAP0: lambda v: v + delta})
    np.testing.assert_allclose(logits, np.asarray(expect), atol=1e-4)
    np.testing.assert_allclose(recorded[TAP0], np.asarray(rec[TAP0]), atol=1e-5)
    none = [[]] * 5
    clean, _ = sets.run_steered(r, index, positions, none, none, none)
    np.testing.assert_array_equal(clean, np.asarray(w.model.instrumented(w.params, toks, [])[0]))
    np.testing.assert_array_equal(sets.argmax(clean[:, 1]), toks[:, 1])
    assert sets.argmax(np.array([[1.0, 3.0, 3.0]])).tolist() == [1], "ties go to the lower token"
    for bad in (lambda: sets.run_steered(r, index[:4], positions, remove, add, amounts),
                lambda: sets.run_steered(r, index, positions, remove, add, [[]] * 5),
                lambda: sets.run_steered(r, [], [], [], [], []),
                lambda: sets.run_steered(r, index, positions, none, none, none, freeze=(1,))):
        with pytest.raises(ValueError):
            bad()


def test_no_prediction_records_the_kl(identity):
    silent = F.world("silent")
    r = F.read(silent, 0)
    st = nm.name(r, "every", F.latent(5), instances=20, seed=4)["steering"]
    assert st["causal"] is False and st["note"] == nm.NO_PREDICTION and st["instances"] == 20
    assert set(st) == {"causal", "note", "instances", "kl", "null_kl", "examples"}
    rng = np.random.default_rng(4)
    causal, inst = nm.draw_instances(r, "every", "token(p)", 5, limit=20, rng=rng)
    assert causal is False and all(x.read is None and x.target is None for x in inst)
    T = r.table("every")
    nr, na = sets.null_sets(T, "token(p)", r.codes[[x.window for x in inst], [x.position for x in inst]], [5] * 20,
                            [x.w for x in inst], rng)
    index, pos = [x.window for x in inst], [x.position for x in inst]
    Sv = list(T.set_for("token(p)", 5))
    Sw = [list(T.set_for("token(p)", x.w)) for x in inst]
    amounts = [T.mean_codes("token(p)", x.w)[s] for x, s in zip(inst, Sw)]
    none = [[]] * 20
    clean, _ = sets.run_steered(r, index, pos, none, none, none)
    swapped, _ = sets.run_steered(r, index, pos, [Sv] * 20, Sw, amounts)
    null, _ = sets.run_steered(r, index, pos, nr, na, [a[:len(x)] for a, x in zip(amounts, na)])
    k = np.arange(20)
    assert st["kl"] == pytest.approx(kl(clean[k, pos], swapped[k, pos]), rel=1e-4)
    assert st["null_kl"] == pytest.approx(kl(clean[k, pos], null[k, pos]), rel=1e-4, abs=1e-6)
    assert st["kl"] > 1.0 > st["null_kl"]
    assert len(st["examples"]) == nm.EXAMPLES
    for ex, x in zip(st["examples"], inst):
        assert (ex["window"], ex["position"], ex["w"]) == (x.window, x.position, x.w)
        assert len(ex["unsteered"]) == len(ex["steered"]) == nm.TOP_TOKENS
        assert ex["unsteered"][0] == 5 and ex["steered"][0] == x.w
        assert ex["unsteered"] == [int(t) for t in np.argsort(-clean[inst.index(x), x.position], kind="stable")[:5]]
    recordable(nm.name(r, "every", F.latent(5), instances=20, seed=4))

    # where the run predicts at some instances, only those are steered
    late = F.world("identity", predict_when=lambda d, window, p, w: p >= 3)
    rl = F.read(late, 0)
    causal, inst = nm.draw_instances(rl, "every", "token(p)", 5, limit=1000, rng=np.random.default_rng(0))
    assert causal is True and inst and all(x.position >= 3 and x.read == x.position for x in inst)
    assert len(inst) == int((rl.described["token(p)"][:, 3:] == 5).sum())
    st = nm.name(rl, "every", F.latent(5), instances=20, seed=4)["steering"]
    assert st["causal"] is True and st["follows"] == 1.0
