"""Phase 6 delta 2: assert-this-edge (D50).

Contract: src/nablaframe/gestures/edge.py, built on sets.py. The toy models are in phase6_fixtures.py: in the identity
model a token's direction is carried from block 0 to the last block, and in the mapping model the last block's MLP
computes G of it. Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import phase6_fixtures as F  # noqa: E402
from nablaframe.gestures import edge as ed  # noqa: E402
from nablaframe.gestures import records, sets  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model  # noqa: E402


class Spy:
    """The model, noting the taps each forward pass edits."""

    def __init__(self, model):
        self.model, self.hooked = model, []

    def instrumented(self, params, tokens, taps, interventions=None):
        if interventions:
            self.hooked.append(frozenset(interventions))
        return self.model.instrumented(params, tokens, taps, interventions)


@pytest.fixture(scope="module")
def identity():
    return F.world("identity")


@pytest.fixture(scope="module")
def mapping():
    return F.world("mapping")


def measured(w, relation):
    """The relation's edge on the world's first and last blocks, measured once per world."""
    if relation not in w.cache:
        w.cache[relation] = ed.edge(F.reading(w, 0), F.reading(w, w.n_layers - 1), relation, instances=30, seed=2)
    return w.cache[relation]


def recordable(results):
    assert json.loads(json.dumps(results)) == results
    args = {"relation": "carry", "up": "1" * 64, "up_site": 0, "down": "4" * 64, "down_site": 0}
    request = records.GestureRequest(kind="edge", run="0" * 64, step=0, args=args, instances=30, seed=2)
    records.GestureRecord(request, {"states": {"0": "2" * 64}, "dictionaries": ["1" * 64, "4" * 64],
                                    "windows": ["3" * 64]}, {"nablaframe.gestures.edge": "a" * 40}, results)


def eligible(up, down, relation):
    """The candidates and their options, from the contract."""
    d = up.declaration
    R = d.relation(relation)
    du, dd = R.up.description, R.down.description
    Tu, Td = up.table(R.up.cls), down.table(R.down.cls)
    out = []
    for i in range(len(up.windows)):
        window = up.windows[i]
        taken = set(up.described[du][i].tolist())
        for pu, pd in d.pairs(relation, window):
            v = int(up.described[du][i, pu])
            gv = d.value_map(relation, window, v)
            if not Tu.has(du, v) or gv is None or not Td.has(dd, gv) or down.described[dd][i, pd] != gv:
                continue
            if down.codes[i, pd, list(Td.set_for(dd, gv))].sum() <= 0:
                continue
            options = [(u, d.value_map(relation, window, u)) for u in Tu.values(du).tolist() if u not in taken]
            options = [(u, gu) for u, gu in options if gu is not None and Td.has(dd, gu) and gu != gv]
            predicting = [(u, gu) for u, gu in options if d.predict(du, window, pu, u) is not None]
            options = predicting or options
            if options:
                out.append((i, pu, pd, v, gv, options))
    return out


def test_edge_follows_and_kept(identity, mapping):
    res = measured(identity, "carry")
    assert (res["relation"], res["judged"], res["instances"], res["pairs"]) == ("carry", False, 30, 99 * 2)
    assert res["up"] == {"description": "token(p)", "block": 0, "class": "every", "site": 0,
                         "tap": "blocks.0.hook_resid_post"}
    assert res["down"] == {"description": "token(p)", "block": 1, "class": "every", "site": 0,
                           "tap": "blocks.1.hook_resid_post"}
    assert (res["follows"], res["null_follows"]) == (1.0, 0.0)
    assert res["kept"] == pytest.approx(0.0, abs=1e-6) and res["null_kept"] == pytest.approx(1.0, abs=1e-6)
    assert res["output"] == {"instances": 30, "follows": 1.0, "null_follows": 0.0}
    recordable(res)
    assert ed.edge(F.reading(identity, 0), F.reading(identity, 1), "carry", instances=30, seed=2) == res, "one seed, one result"

    res = measured(mapping, "map")
    assert (res["judged"], res["follows"], res["null_follows"], res["output"]["follows"]) == (True, 1.0, 0.0, 1.0)
    assert res["kept"] == pytest.approx(0.0, abs=1e-6) and res["null_kept"] > 0.9
    recordable(res)

    never = ed.edge(F.reading(mapping, 0), F.reading(mapping, 1), "never", instances=30, seed=2)
    assert never["instances"] == 0 and never["pairs"] == 99 * 2
    for key in ("follows", "null_follows", "kept", "null_kept", "output", "residual_alone", "label", "single"):
        assert never[key] is None
    recordable(never)

    r0, r1 = F.reading(identity, 0), F.reading(identity, 1)
    other = F.world("identity")
    fewer = sets.read_site(identity.model, identity.params, n_layers=2, declaration=identity.declaration,
                           dictionary=identity.dictionaries[1], site=0, windows=identity.windows[:50])
    for bad in (lambda: ed.edge(r1, r0, "carry", instances=30, seed=2),
                lambda: ed.edge(r0, r1, "nope", instances=30, seed=2),
                lambda: ed.edge(r0, F.read(other, 1), "carry", instances=30, seed=2),
                lambda: ed.edge(r0, fewer, "carry", instances=30, seed=2),
                lambda: ed.edge(r0, r1, "carry", instances=0, seed=2),
                lambda: ed.edge(r0, r1, "carry", instances=30, seed=-1),
                lambda: ed.draw_edge_instances(r0, r1, "carry", limit=0, rng=np.random.default_rng(0))):
        with pytest.raises(ValueError):
            bad()


def test_edge_value_map(mapping):
    q0, q1 = F.reading(mapping, 0), F.reading(mapping, 1)
    expected = eligible(q0, q1, "map")
    got = ed.draw_edge_instances(q0, q1, "map", limit=10_000, rng=np.random.default_rng(0))
    assert len(got) == len(expected) > 30
    assert [(x.window, x.up, x.down, x.v, x.gv) for x in got] == [c[:5] for c in expected]
    for x, c in zip(got, expected):
        assert x.gv == F.G[x.v] and x.gw == F.G[x.w] and x.gw != x.gv, "g maps both values, and they differ"
        assert (x.w, x.gw) in c[5]
        assert x.w not in set(q0.tokens[x.window].tolist()), "w is a token the window doesn't hold"
        assert (x.read, x.target) == (x.down, x.gw)
    rng = np.random.default_rng(9)
    kept = [expected[k] for k in sorted(rng.choice(len(expected), size=12, replace=False))]
    draws = [(c[0], c[1], c[5][rng.integers(len(c[5]))][0]) for c in kept]
    got = ed.draw_edge_instances(q0, q1, "map", limit=12, rng=np.random.default_rng(9))
    assert [(x.window, x.up, x.w) for x in got] == draws

    T = q1.table("every")
    assert T.set_for("token(p)", 1) == T.set_for("token(p)", 2), "the partner's downstream set is v's own"
    wrong = ed.edge(q0, q1, "wrong map", instances=30, seed=2)
    assert wrong["instances"] == 0 and wrong["follows"] is None, "the downstream value must be g(v)"

    quiet = F.world("identity", silent={1: (3,)})
    u0, u1 = F.read(quiet, 0), F.read(quiet, 1)
    got = ed.draw_edge_instances(u0, u1, "carry", limit=10_000, rng=np.random.default_rng(0))
    assert len(got) == len(eligible(u0, u1, "carry")) and not any(x.v == 3 for x in got), \
        "an instance needs the downstream set for g(v) active"

    even = F.world("mapping", predict_when=lambda d, window, p, w: d != "token(p)" or w % 2 == 0)
    e0, e1 = F.read(even, 0), F.read(even, 1)
    got = ed.draw_edge_instances(e0, e1, "map", limit=10_000, rng=np.random.default_rng(0))
    assert got and all(x.w % 2 == 0 for x in got), "where the run predicts for some options, only those are drawn"


def test_the_residual_alone_freezes_the_blocks_between(mapping):
    spy = Spy(GPT2Model(F.config(3)))
    w = F.world("identity", n_layers=3, model=spy)
    r0, r2 = F.read(w, 0), F.read(w, 2)
    spy.hooked.clear()
    res = ed.edge(r0, r2, "carry", instances=30, seed=2)
    ra = res["residual_alone"]
    assert ra["blocks"] == [1, 2] and ra["instances"] == 30
    frozen = frozenset({"blocks.0.hook_resid_post", "blocks.1.hook_attn_out", "blocks.1.hook_mlp_out",
                        "blocks.2.hook_attn_out", "blocks.2.hook_mlp_out"})
    assert spy.hooked.count(frozen) == len(F.PAIRED), "one frozen pass per position, freezing blocks 1 and 2"
    assert all(h in (frozen, frozenset({"blocks.0.hook_resid_post"})) for h in spy.hooked)
    assert ra["follows"] == ra["follows_unfrozen"] == 1.0 and ra["output_follows"] == 1.0

    ra = measured(mapping, "map")["residual_alone"]
    assert ra["blocks"] == [1] and ra["instances"] == measured(mapping, "map")["instances"]
    assert (ra["follows"], ra["follows_unfrozen"]) == (0.0, 1.0), \
        "with block 1's MLP held, g of the new value is never computed"
    with pytest.raises(ValueError):
        sets.run_steered(r0, [0, 1], [4, 5], [[], []], [[], []], [[], []], freeze=(1,))


def test_carried_and_computed(identity, mapping):
    assert measured(identity, "carry")["label"] == ed.CARRIED
    assert measured(mapping, "map")["label"] == ed.COMPUTED
    res = ed.edge(F.reading(identity, 0), F.reading(identity, 1), "next", instances=30, seed=2)
    assert res["instances"] == 30 and res["residual_alone"] is None
    assert res["follows"] == 0.0, "nothing moves between positions in the toy model"
    assert res["label"] == ed.COMPUTED, "an edge across positions can only be computed"
    assert res["output"] is None, "the run declares no prediction for token(p-1)"


def test_single_latents_beside_sets(identity):
    res = measured(identity, "carry")
    assert (res["follows"], res["single"]["follows"]) == (1.0, 0.0)
    assert res["kept"] == pytest.approx(0.0, abs=1e-6)
    # each token's best latent writes a quarter of its direction; with it gone, the best downstream latent's code
    # falls from gain - 0.05 to 0.75 gain - 0.05
    inst = ed.draw_edge_instances(F.reading(identity, 0), F.reading(identity, 1), "carry", limit=30, rng=np.random.default_rng(2))
    kept = [(0.75 * F.gain(x.v, 0) + F.BIAS) / F.code(x.v, 0) for x in inst]
    assert res["single"]["kept"] == pytest.approx(float(np.median(kept)), abs=1e-4)
