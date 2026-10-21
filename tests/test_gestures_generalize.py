"""Phase 6 delta 3: generalize-this (D52).

Contract: src/nablaframe/gestures/generalize.py. The toy run is phase6_fixtures.py's identity model with four
checkpoints, each built here:

- step 0: every token embeds as another token's direction. The final dictionary's sets name the wrong tokens there,
  though a probe still reads them.
- step 5: the final layer norm's gain is negated, so the sets still read the tokens but steering them moves the output
  nowhere useful.
- steps 3 and 7: the final model.

Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import phase6_fixtures as F  # noqa: E402
from nablaframe.dictionaries import core, measure  # noqa: E402
from nablaframe.gestures import edge as ed  # noqa: E402
from nablaframe.gestures import generalize as ge  # noqa: E402
from nablaframe.gestures import records, sets  # noqa: E402

STEPS = [0, 3, 5, 7]


def scrambled(params):
    """Token t embeds as direction t + 1, and 11 as 1. The model still outputs the token it reads, since its
    embeddings are tied, but the dictionaries' latents for direction t now fire on token t - 1."""
    wte = np.asarray(params.wte).copy()
    wte[1:] = np.roll(wte[1:], -1, axis=0)
    return type(params)(wte=jnp.asarray(wte), wpe=params.wpe, blocks=params.blocks, ln_f=params.ln_f)


def unread(params):
    """The final layer norm's gain negated: every token's direction now scores its own token lowest."""
    ln = params.ln_f
    return type(params)(wte=params.wte, wpe=params.wpe, blocks=params.blocks, ln_f=type(ln)(-ln.weight, ln.bias))


def renumbered(d, seed):
    """``d`` with its latents in another order: the same latents under other ids, as another training seed might
    give them."""
    perm = np.random.default_rng(seed).permutation(d.settings.width)
    a = d.arrays
    arrays = dict(a, W_enc=a["W_enc"][:, :, perm], W_dec=a["W_dec"][:, perm], b_enc=a["b_enc"][:, perm])
    return core.Dictionary(d.settings, d.sites, {n: np.array(v, np.float32) for n, v in arrays.items()})


@pytest.fixture(scope="module")
def toy():
    w = F.world("identity")
    w.at = {0: scrambled(w.params), 3: w.params, 5: unread(w.params), 7: w.params}
    return w


def params_at(w):
    return w.at.__getitem__


def seed_reading(w, block=0, decoder="token", seed=1, windows=None):
    return sets.read_site(w.model, w.params, n_layers=2, declaration=w.declaration,
                          dictionary=renumbered(F.dictionary(block, decoder=decoder), seed), site=0,
                          windows=w.windows if windows is None else windows)


def generalized(w):
    """token(p) after block 0 at every position, generalized once per world."""
    if "general" not in w.cache:
        w.cache["general"] = ge.generalize_name(
            F.reading(w, 0), "every", "token(p)", instances=20, seed=4, steps=STEPS, params_at=params_at(w),
            seed_readings={1: seed_reading(w), 2: seed_reading(w, decoder="negative", seed=2)},
            extras=lambda step, params: {"double": 2 * step})
    return w.cache["general"]


def recordable(results, args):
    assert json.loads(json.dumps(results)) == results
    request = records.GestureRequest(kind="generalize", run="0" * 64, step=7, args=args, instances=20, seed=4)
    records.GestureRecord(request, {"states": {"7": "2" * 64}, "dictionaries": ["1" * 64], "windows": ["3" * 64]},
                          {"nablaframe.gestures.generalize": "a" * 40}, results)


def test_slices(toy):
    r = F.reading(toy, 0)
    T = r.table("every")
    res = generalized(toy)
    assert (res["description"], res["class"], res["site"], res["tap"]) == ("token(p)", "every", 0,
                                                                            "blocks.0.hook_resid_post")
    assert res["values"] == 11
    parts = np.array_split(np.arange(99), ge.SLICES)
    rows = res["slices"]["rows"]
    assert [x["windows"] for x in rows] == [len(p) for p in parts] and len(rows) == 8
    assert all(x["f1"] == 1.0 and x["follows"] == 1.0 and x["null_follows"] == 0.0 for x in rows)
    assert res["slices"]["f1"] == [1.0, 1.0] and res["slices"]["follows"] == [1.0, 1.0]
    recordable(res, {"dictionary": "1" * 64, "site": 0, "class": "every", "description": "token(p)"})

    # token(p+1) is read only loosely by token latents, so its F1 differs slice by slice; the run predicts nothing
    # for it, so no slice follows
    nxt = ge.generalize_name(r, "every", "token(p+1)", instances=20, seed=4, steps=[7], params_at=params_at(toy))
    expected = []
    for p in parts:
        Ts = sets.ClassTable(r.codes[p], {k: v[p] for k, v in r.described.items()}, range(F.SEQ))
        expected.append(ge.description_f1(Ts, T, "token(p+1)"))
    got = [x["f1"] for x in nxt["slices"]["rows"]]
    assert got == pytest.approx(expected, abs=1e-12) and len(set(np.round(got, 6))) > 1
    assert nxt["slices"]["f1"] == pytest.approx([min(expected), max(expected)], abs=1e-12)
    assert all(x["follows"] is None for x in nxt["slices"]["rows"]) and nxt["slices"]["follows"] is None
    assert nxt["checkpoints"][0]["causal"] is False and nxt["checkpoints"][0]["holds"] is False

    # description_f1: the mean over shared values of the best F1 within each value's set
    col = {v: j for j, v in enumerate(T.values("token(p+1)").tolist())}
    best = [max(T.f1("token(p+1)")[l, col[v]] for l in T.set_for("token(p+1)", v)) for v in col]
    assert ge.description_f1(T, T, "token(p+1)") == pytest.approx(float(np.mean(best)), abs=1e-12)
    assert ge.description_f1(T, T, "nothing") is None

    # steering one slice draws only from its windows
    eligible = sum(1 for i in parts[0] for p in range(F.SEQ))
    st = ge.steer_description(r, "every", "token(p)", instances=10_000, rng=np.random.default_rng(0),
                              windows=parts[0].tolist())
    assert st == {"causal": True, "instances": eligible, "follows": 1.0, "null_follows": 0.0}
    for bad in (dict(cls="nope"), dict(d="token(p-9)"), dict(instances=0)):
        kwargs = dict(cls="every", d="token(p)", instances=5) | bad
        with pytest.raises(ValueError):
            ge.steer_description(r, kwargs["cls"], kwargs["d"], instances=kwargs["instances"],
                                 rng=np.random.default_rng(0))


def test_seeds_match_by_name(toy):
    res = generalized(toy)
    one, two = res["seeds"]
    # seed 1 holds the same latents under other ids: matched by name, every number is seed 0's
    assert (one["seed"], one["f1"], one["follows"], one["null_follows"]) == (1, 1.0, 1.0, 0.0)
    assert one["cosine"] == pytest.approx(1.0, abs=1e-6)
    # seed 2's latents read the same tokens but write minus their directions: the name holds by F1, steering with its
    # own decoders doesn't follow, and its decoders point away from seed 0's
    assert (two["seed"], two["f1"], two["follows"]) == (2, 1.0, 0.0)
    assert two["cosine"] == pytest.approx(-1.0, abs=1e-6)
    r = F.reading(toy, 0)
    for bad in ({1: F.reading(toy, 1)}, {1: seed_reading(toy, windows=toy.windows[:50])}):
        with pytest.raises(ValueError):
            ge.generalize_name(r, "every", "token(p)", instances=5, seed=0, steps=[7], params_at=params_at(toy),
                               seed_readings=bad)


def test_checkpoint_rows(toy):
    res = generalized(toy)
    rows = res["checkpoints"]
    assert [x["step"] for x in rows] == STEPS
    early, unread_row = rows[0], rows[2]
    # through the final dictionary's sets: at step 0 they fire on other tokens, so the name doesn't hold
    assert (early["f1"], early["causal"], early["follows"], early["holds"]) == (0.0, True, 0.0, False)
    # at step 5 the sets read their tokens, but steering them doesn't move the output there: F1 alone doesn't hold
    assert (unread_row["f1"], unread_row["follows"], unread_row["holds"]) == (1.0, 0.0, False)
    for x in (rows[1], rows[3]):
        assert (x["f1"], x["causal"], x["follows"], x["null_follows"], x["holds"]) == (1.0, True, 1.0, 0.0, True)
    assert [x["extras"] for x in rows] == [{"double": 2 * s} for s in STEPS]
    r = F.reading(toy, 0)
    for x, step in zip(rows, STEPS):
        params = params_at(toy)(step)
        acts = measure.activations(toy.model, params, r.tokens, [r.input_tap])[r.input_tap]
        feats, vals = acts.reshape(-1, F.D_MODEL), r.described["token(p)"].reshape(-1)
        assert x["probe"] == pytest.approx(measure.probe_accuracy(feats, vals, int(vals.max()) + 1), abs=1e-12)
        logits = np.asarray(toy.model.instrumented(params, r.tokens, [])[0])
        assert x["accuracy"] == pytest.approx(measure.readout_loss(logits, toy.windows, range(F.SEQ))[1], abs=1e-12)
    assert early["probe"] >= 0.9, "the probe reads the token at step 0, where the name doesn't hold"
    assert res["envelope"] == {"steps": [3, 7], "runs": [[3, 3], [7, 7]], "final": [7, 7], "gap": [0, 5],
                               "probe_reads_from": 0}
    for bad in ([], [3, 0], [3, 3], [-1], [1.5]):
        with pytest.raises(ValueError):
            ge.generalize_name(r, "every", "token(p)", instances=5, seed=0, steps=bad, params_at=params_at(toy))

    # an edge, at each slice, seed and checkpoint
    r0, r1 = F.reading(toy, 0), F.reading(toy, 1)
    edge = ge.generalize_edge(r0, r1, "next", instances=10, seed=1, steps=[0, 7], params_at=params_at(toy),
                              seed_readings={1: (seed_reading(toy), seed_reading(toy, block=1))})
    assert edge["relation"] == "next"
    keys = ("instances", "follows", "null_follows", "kept", "null_kept", "label")
    assert [x["windows"] for x in edge["slices"]] == [len(p) for p in np.array_split(np.arange(99), 8)]
    part = np.array_split(np.arange(99), 8)[2]
    alone = ed.edge(*(sets.read_site(toy.model, toy.params, n_layers=2, declaration=toy.declaration,
                                     dictionary=toy.dictionaries[b], site=0, windows=toy.windows[part]) for b in (0, 1)),
                    "next", instances=10, seed=1)
    assert edge["slices"][2] == {"windows": len(part), **{k: alone[k] for k in keys}}
    assert [x["seed"] for x in edge["seeds"]] == [1] and [x["step"] for x in edge["checkpoints"]] == [0, 7]
    assert all(x["label"] == ed.COMPUTED for x in edge["slices"] + edge["seeds"] + edge["checkpoints"])
    assert edge["checkpoints"][1] == {"step": 7, **{k: ed.edge(r0, r1, "next", instances=10, seed=1)[k] for k in keys}}
    recordable(edge, {"relation": "next", "up": "1" * 64, "up_site": 0, "down": "4" * 64, "down_site": 0})


def test_the_envelope_and_the_gap():
    def rows(holds, probes, steps=None):
        steps = steps or [100 * j for j in range(len(holds))]
        return [{"step": s, "holds": h, "probe": p} for s, h, p in zip(steps, holds, probes)]

    env = ge.envelope(rows([False, False, True, True, False, True, True], [0.1, 0.95, 0.95, 0.2, 0.95, None, 1.0]))
    assert env == {"steps": [200, 300, 500, 600], "runs": [[200, 300], [500, 600]], "final": [500, 600],
                   "gap": [100, 400], "probe_reads_from": 100}
    env = ge.envelope(rows([True, True, False], [0.9, 0.3, 0.2]))
    assert env == {"steps": [0, 100], "runs": [[0, 100]], "final": None, "gap": [], "probe_reads_from": 0}, \
        "an envelope that stops before the last checkpoint has no final run"
    env = ge.envelope(rows([False, True, False, True], [None, 0.89, 0.9, 0.95], steps=[0, 5, 9, 12]))
    assert env == {"steps": [5, 12], "runs": [[5, 5], [12, 12]], "final": [12, 12], "gap": [9], "probe_reads_from": 9}
    assert ge.envelope(rows([False], [None])) == {"steps": [], "runs": [], "final": None, "gap": [],
                                                  "probe_reads_from": None}

    # a name holds where its F1 and, where it can be steered, its follows both reach 0.5
    assert ge.holds(0.5, True, 0.5) and ge.holds(1.0, True, 0.9)
    assert not ge.holds(0.49, True, 1.0) and not ge.holds(1.0, True, 0.49) and not ge.holds(1.0, True, None)
    assert ge.holds(0.6, False, None), "where the run predicts nothing, F1 alone decides"
    assert not ge.holds(0.4, False, None)
    assert not ge.holds(0.6, None, None) and not ge.holds(None, True, 1.0), "no steering instance, or no F1: no"
