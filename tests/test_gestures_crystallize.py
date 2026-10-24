"""Phase 7 delta 2: crystallize-this (D56 to D58).

Contract: src/nablaframe/gestures/crystallize.py's docstring. Needs the jax lane. The models are small random ones
(``phase7_fixtures``); each measurement is checked against the same quantity computed here another way.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import phase7_fixtures as F  # noqa: E402
from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.dictionaries.edges import kl  # noqa: E402
from nablaframe.gestures import crystallize as C  # noqa: E402
from nablaframe.gestures.name import NAMED, WEAK  # noqa: E402
from nablaframe.gestures.records import GestureRecord, GestureRequest  # noqa: E402
from nablaframe.interventions import spec  # noqa: E402
from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit  # noqa: E402
from nablaframe.zoo.functions import Case, Function, Key, Read, RunFunctions  # noqa: E402

ATTN, MLP = "blocks.0.hook_attn_out", "blocks.0.hook_mlp_out"


@pytest.fixture(scope="module")
def ma():
    return F.world("modular_addition")


@pytest.fixture(scope="module")
def ind():
    return F.world("induction")


def close(a, b, *, rtol=1e-5, atol=1e-7):
    return a is not None and b is not None and np.isclose(a, b, rtol=rtol, atol=atol)


def run(w, *, part, function, form, params=None, fit=None, held=None, functions=None, declaration=None, **kwargs):
    f = (functions or w.functions).functions[function]
    return C.crystallize(w.model, w.params if params is None else params, n_layers=w.n_layers,
                         functions=functions or w.functions, declaration=declaration or w.declaration, block=f.block,
                         part=part,
                         cls=f.cls, function=function, form=form, fit_windows=w.fit if fit is None else fit,
                         held_windows=w.held if held is None else held, seed=kwargs.pop("seed", 0), **kwargs)


def clean(w, params, windows, taps):
    """The unintervened logits and the taps, float64."""
    logits, rec = w.model.instrumented(params, np.asarray(windows)[:, :w.config.seq].astype(np.int32), list(taps))
    return np.asarray(logits, np.float64), {t: np.asarray(getattr(rec[t], "array", rec[t]), np.float64) for t in taps}


def intervened(w, params, windows, edits, store):
    tokens = np.asarray(windows)[:, :w.config.seq].astype(np.int32)
    logits, _ = run_intervened(w.model, params, tokens, InterventionSpec(tuple(edits)), n_layers=w.n_layers,
                               resolve=dict(store).__getitem__)
    return np.asarray(logits, np.float64)


def putting(store, values):
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    store[spec.array_hash(v)] = v
    return spec.array_hash(v)


def ma_table(w, params, key, fit_windows):
    """Modular addition's write at 2 as a table by ``key``, fitted on ``fit_windows``: (table, overall mean)."""
    _, rec = clean(w, params, fit_windows, (ATTN, MLP))
    Y = (rec[ATTN] + rec[MLP])[:, 2]
    k = w.functions.keys[key].values(fit_windows)[:, 2]
    T = np.tile(Y.mean(0), (w.task.p, 1))
    for v in np.unique(k):
        T[v] = Y[k == v].mean(0)
    return T, Y


def ma_write(w, params, values, windows=None):
    """Logits at 2 with the attention output at 2 set to ``values`` and the MLP output there zeroed."""
    windows = w.held if windows is None else windows
    store = {}
    h = putting(store, np.asarray(values)[:, None, :])
    return intervened(w, params, windows, (edit(ATTN, [2], "set", values=h), edit(MLP, [2], "zero")), store)[:, 2]


def test_table_form(ma):
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 5, (40, 1))
    keys[keys == 3] = 4  # 3 is never shown
    Y = rng.standard_normal((40, 6))
    f = C.fit("table", keys, Y, (5,))
    assert f.frequencies is None and [s.tolist() for s in f.seen] == [[True, True, True, False, True]]
    for v in (0, 1, 2, 4):
        np.testing.assert_allclose(f.value([[v]])[0], Y[keys[:, 0] == v].mean(0), rtol=0, atol=1e-12)
    got = f.value(keys)
    assert got.dtype == np.float64 and got.shape == (40, 6)
    for bad in (dict(keys=np.c_[keys, keys], n_values=(5, 5)), dict(k=(1,)), dict(keys=np.where(keys == 4, 5, keys)),
                dict(keys=-keys), dict(Y=Y[:39]), dict(keys=keys[:0], Y=Y[:0]), dict(keys=keys.astype(float))):
        args = {"keys": keys, "Y": Y, "n_values": (5,), **bad}
        with pytest.raises(ValueError):
            C.fit("table", args["keys"], args["Y"], args["n_values"], k=args.get("k"))
    with pytest.raises(ValueError):
        C.fit("lookup", keys, Y, (5,))

    # Through crystallize: modular addition's write at 2, a table by the sum.
    r = run(ma, part="write", function="sum", form="table")
    assert (r["function"], r["block"], r["part"], r["class"], r["form"], r["keys"]) == ("sum", 0, "write", "answer",
                                                                                      "table", ["sum"])
    assert r["reads"] == [["token(p-2)", "across"], ["token(p-1)", "across"]] and r["relation"] is None
    assert r["taps"] == [ATTN, MLP] and r["rows"] == {"fit": len(ma.fit), "held": len(ma.held)}
    T, _ = ma_table(ma, ma.params, "sum", ma.fit)
    _, rec = clean(ma, ma.params, ma.held, (ATTN, MLP))
    Yh = (rec[ATTN] + rec[MLP])[:, 2]
    Fh = T[ma.functions.keys["sum"].values(ma.held)[:, 2]]
    r2 = 1 - ((Yh - Fh) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum()
    assert close(r["r2"], r2, rtol=1e-9, atol=1e-12)
    sums = ma.functions.keys["sum"].values(ma.fit)[:, 2]
    assert r["values"] == {"seen": [len(np.unique(sums))], "unseen": 0}
    assert r["frequencies"] is None and r["twin"] is None and r["checkpoints"] is None and r["envelope"] is None
    assert r["counterfactual"] is None and r["cites"] is None
    assert r == run(ma, part="write", function="sum", form="table"), "the same request gives the same results"


def test_unseen_keys(ma):
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 4, (30, 2))
    keys[:, 0] = np.where(keys[:, 0] == 2, 1, keys[:, 0])  # the first key never shows 2
    Y = rng.standard_normal((30, 3))
    table = C.fit("table", keys[:, :1], Y, (4,))
    np.testing.assert_allclose(table.value([[2]])[0], Y.mean(0), rtol=0, atol=1e-12)
    additive = C.fit("additive", keys, Y, (4, 4))
    X = np.concatenate([np.eye(4)[keys[:, 0]], np.eye(4)[keys[:, 1]]], axis=1)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    np.testing.assert_allclose(additive.value([[2, 3]])[0], W[4 + 3], rtol=0, atol=1e-9,
                               err_msg="an unseen value adds nothing: its least-squares coefficient is 0")
    assert additive.seen[0].tolist() == [True, True, False, True]

    # Fitted without one sum, the held-out rows with that sum are counted, and read the overall mean.
    sums = ma.functions.keys["sum"].values(ma.fit)[:, 2]
    fit = ma.fit[sums != 3]
    r = run(ma, part="write", function="sum", form="table", fit=fit)
    held_sums = ma.functions.keys["sum"].values(ma.held)[:, 2]
    assert r["values"] == {"seen": [len(np.unique(sums[sums != 3]))], "unseen": int((held_sums == 3).sum())} != \
        {"seen": [r["values"]["seen"][0]], "unseen": 0}
    T, Y = ma_table(ma, ma.params, "sum", fit)
    assert np.allclose(T[3], Y.mean(0))
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    assert close(r["meter"]["kl"], kl(base, ma_write(ma, ma.params, T[held_sums])))


def test_additive_form(ma):
    rng = np.random.default_rng(2)
    keys = rng.integers(0, 5, (60, 3))
    A, B, D = rng.standard_normal((5, 4)), rng.standard_normal((5, 4)), rng.standard_normal((5, 4))
    Y = A[keys[:, 0]] + B[keys[:, 1]] + D[keys[:, 2]]
    f = C.fit("additive", keys, Y, (5, 5, 5))
    np.testing.assert_allclose(f.value(keys), Y, rtol=0, atol=1e-9)
    other = rng.integers(0, 5, (10, 3))
    np.testing.assert_allclose(f.value(other), A[other[:, 0]] + B[other[:, 1]] + D[other[:, 2]], rtol=0, atol=1e-9)
    noisy = Y + 0.3 * rng.standard_normal(Y.shape)
    g = C.fit("additive", keys, noisy, (5, 5, 5))
    X = np.concatenate([np.eye(5)[keys[:, j]] for j in range(3)], axis=1)
    W, *_ = np.linalg.lstsq(X, noisy, rcond=None)
    np.testing.assert_allclose(g.value(other), np.concatenate([np.eye(5)[other[:, j]] for j in range(3)], axis=1) @ W,
                               rtol=0, atol=1e-9, err_msg="the minimum-norm least-squares fit on the one-hot design")
    with pytest.raises(ValueError):
        C.fit("additive", keys[:, :1], Y, (5,))  # one key is a table
    with pytest.raises(ValueError):
        C.fit("additive", keys, Y, (5, 5, 5, 5))  # the keys and their ranges don't line up

    # Through crystallize: modular addition's MLP, additive in the sum and both operands.
    r = run(ma, part="mlp", function="sum and operands", form="additive")
    assert r["keys"] == ["sum", "a", "b"] and r["taps"] == [MLP]
    keys_of = lambda w: np.stack([ma.functions.keys[n].values(w)[:, 2] for n in ("sum", "a", "b")], axis=1)  # noqa: E731
    Yf = clean(ma, ma.params, ma.fit, (MLP,))[1][MLP][:, 2]
    p = ma.task.p
    design = lambda k: np.concatenate([np.eye(p)[k[:, j]] for j in range(3)], axis=1)  # noqa: E731
    W, *_ = np.linalg.lstsq(design(keys_of(ma.fit)), Yf, rcond=None)
    base, rec = clean(ma, ma.params, ma.held, (MLP,))
    Fh = design(keys_of(ma.held)) @ W
    store = {}
    logits = intervened(ma, ma.params, ma.held, (edit(MLP, [2], "set", values=putting(store, Fh[:, None])),), store)
    assert close(r["meter"]["kl"], kl(base[:, 2], logits[:, 2]))
    Yh = rec[MLP][:, 2]
    assert close(r["r2"], 1 - ((Yh - Fh) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum(), rtol=1e-9, atol=1e-12)


def test_fourier_form_and_its_frequencies(ma):
    p = 11
    v = np.arange(p)
    u = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 4)))[0]
    E = (3 * np.cos(2 * np.pi * 2 * v / p)[:, None] * u[0] + 2 * np.sin(2 * np.pi * 5 * v / p)[:, None] * u[1]
         + np.cos(2 * np.pi * 1 * v / p)[:, None] * u[2] + 0.7 * u[3])
    got = C.frequencies(E, p)
    assert got["k"] == [2, 5] and np.allclose(got["shares"], [9 / 14, 4 / 14]) and np.isclose(got["held"], 13 / 14)
    p23 = 23
    w23 = np.arange(p23)
    amplitude = 1 + 0.01 * (11 - np.arange(1, 12))  # each frequency a little weaker than the one before
    flat = np.stack([a * np.cos(2 * np.pi * k * w23 / p23) for k, a in zip(range(1, 12), amplitude)], axis=1)
    many = C.frequencies(flat, p23)
    held = (amplitude[:8] ** 2).sum() / (amplitude ** 2).sum()
    assert held < 0.75 and many["k"] == list(range(1, 9)) and np.isclose(many["held"], held), "cut at MAX_FREQUENCIES"
    for bad in ((E[:10], 10), (E, 12), (E, 1), (E[:9], p)):
        with pytest.raises(ValueError):
            C.frequencies(*bad)

    rng = np.random.default_rng(4)
    keys = rng.integers(0, p, (50, 1))
    mu, U, V = rng.standard_normal((3, 5))
    s = keys[:, 0]
    Y = mu + np.cos(2 * np.pi * 2 * s / p)[:, None] * U + np.sin(2 * np.pi * 5 * s / p)[:, None] * V
    f = C.fit("fourier", keys, Y, (p,), k=(2, 5))
    assert f.frequencies == (2, 5)
    np.testing.assert_allclose(f.value(v[:, None]), mu + np.cos(2 * np.pi * 2 * v / p)[:, None] * U
                               + np.sin(2 * np.pi * 5 * v / p)[:, None] * V, rtol=0, atol=1e-9)
    for bad in (dict(k=None), dict(k=()), dict(k=(2, 2))):
        with pytest.raises(ValueError):
            C.fit("fourier", keys, Y, (p,), **bad)

    # Through crystallize: K from the request's params; the twin keeps it.
    r = run(ma, part="write", function="sum", form="fourier", twin=ma.twin)
    want = C.frequencies(np.asarray(ma.params.wte)[:p], p)
    assert r["frequencies"]["k"] == want["k"] == r["twin"]["frequencies"]["k"]
    assert np.allclose(r["frequencies"]["embedding"], want["shares"])
    assert np.isclose(sum(r["frequencies"]["coefficients"]), 1.0)
    twin_shares = C.frequencies(np.asarray(ma.twin.wte)[:p], p)
    every = {k: s for k, s in zip(twin_shares["k"], twin_shares["shares"])}
    for k, share in zip(r["twin"]["frequencies"]["k"], r["twin"]["frequencies"]["embedding"]):
        assert k not in every or np.isclose(share, every[k]), "the twin's own embedding shares at K"
    sums = lambda w: ma.functions.keys["sum"].values(w)[:, 2]  # noqa: E731
    _, Yf = ma_table(ma, ma.params, "sum", ma.fit)
    design = lambda x: np.stack([np.ones(len(x))] + [g(2 * np.pi * k * x / p) for k in want["k"]  # noqa: E731
                                                     for g in (np.cos, np.sin)], axis=1)
    W, *_ = np.linalg.lstsq(design(sums(ma.fit)), Yf, rcond=None)
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    assert close(r["meter"]["kl"], kl(base, ma_write(ma, ma.params, design(sums(ma.held)) @ W)))
    mass = np.array([(W[1 + 2 * i] ** 2).sum() + (W[2 + 2 * i] ** 2).sum() for i in range(len(want["k"]))])
    np.testing.assert_allclose(r["frequencies"]["coefficients"], mass / mass.sum(), rtol=1e-6, atol=1e-9,
                               err_msg="each frequency's share of the cosine and sine coefficients")
    # At a checkpoint whose params are the twin's, the form is refitted there with the request's K.
    s = run(ma, part="write", function="sum", form="fourier", twin=ma.twin, steps=[5], params_at=lambda step: ma.twin)
    assert close(s["checkpoints"][0]["kl"], r["twin"]["meter"]["kl"], rtol=1e-9, atol=1e-12)


def test_write_is_attention_set_and_mlp_zeroed(ma):
    assert C.part_taps(2, "attention") == ("blocks.2.hook_attn_out",)
    assert C.part_taps(2, "mlp") == ("blocks.2.hook_mlp_out",)
    assert C.part_taps(2, "write") == ("blocks.2.hook_attn_out", "blocks.2.hook_mlp_out")
    for bad in ((2, "head"), (-1, "mlp"), (True, "mlp"), (1.0, "mlp")):
        with pytest.raises(ValueError):
            C.part_taps(*bad)
    h = "1" * 64
    assert C.part_edits(1, "write", [3, 4], h) == (edit("blocks.1.hook_attn_out", [3, 4], "set", values=h),
                                                     edit("blocks.1.hook_mlp_out", [3, 4], "zero"))
    assert C.part_edits(1, "attention", [3], h) == (edit("blocks.1.hook_attn_out", [3], "set", values=h),)
    assert C.part_edits(1, "mlp", "all", h) == (edit("blocks.1.hook_mlp_out", "all", "set", values=h),)

    # The residual after the block is the residual before it plus the function's value.
    T, _ = ma_table(ma, ma.params, "sum", ma.fit)
    Fh = T[ma.functions.keys["sum"].values(ma.held)[:, 2]]
    store = {}
    tokens = ma.held[:, :3].astype(np.int32)
    _, rec = run_intervened(ma.model, ma.params, tokens, InterventionSpec((
        edit(ATTN, [2], "set", values=putting(store, Fh[:, None])), edit(MLP, [2], "zero"))), n_layers=1,
        resolve=store.__getitem__, taps=["blocks.0.hook_resid_pre", "blocks.0.hook_resid_post"])
    pre, post = (np.asarray(getattr(rec[t], "array", rec[t]), np.float64)[:, 2]
                 for t in ("blocks.0.hook_resid_pre", "blocks.0.hook_resid_post"))
    np.testing.assert_allclose(post, pre + Fh, rtol=0, atol=1e-5)
    r = run(ma, part="write", function="sum", form="table")
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    assert close(r["meter"]["kl"], kl(base, ma_write(ma, ma.params, Fh)))

    # The attention part alone leaves the MLP running on what it reads.
    a = run(ma, part="attention", function="operands", form="additive")
    p = ma.task.p
    keys_of = lambda w: np.stack([ma.functions.keys[n].values(w)[:, 2] for n in ("a", "b")], axis=1)  # noqa: E731
    design = lambda k: np.concatenate([np.eye(p)[k[:, 0]], np.eye(p)[k[:, 1]]], axis=1)  # noqa: E731
    W, *_ = np.linalg.lstsq(design(keys_of(ma.fit)), clean(ma, ma.params, ma.fit, (ATTN,))[1][ATTN][:, 2], rcond=None)
    store = {}
    logits = intervened(ma, ma.params, ma.held,
                        (edit(ATTN, [2], "set", values=putting(store, (design(keys_of(ma.held)) @ W)[:, None])),), store)
    assert a["taps"] == [ATTN] and close(a["meter"]["kl"], kl(base, logits[:, 2]))


def test_kl_and_the_ablation(ma, ind):
    r = run(ma, part="write", function="sum", form="table")
    T, Yf = ma_table(ma, ma.params, "sum", ma.fit)
    held_sums = ma.functions.keys["sum"].values(ma.held)[:, 2]
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    crystal = ma_write(ma, ma.params, T[held_sums])
    ablated = ma_write(ma, ma.params, np.tile(Yf.mean(0), (len(ma.held), 1)))
    m = r["meter"]
    assert close(m["kl"], kl(base, crystal)) and close(m["ablation"], kl(base, ablated))
    assert m["ablation"] > 0.05, "the fixture's write moves the prediction"
    assert close(m["removed"], 1 - m["kl"] / m["ablation"], rtol=1e-9, atol=1e-12)
    assert m["agreement"] == np.mean(np.argmax(crystal, -1) == np.argmax(base, -1))
    assert m["label"] == C.label(m["kl"], m["ablation"])

    # A class of several positions: induction's block 1 attention at every mechanism position.
    q = list(ind.task.mechanism_positions)
    tap = "blocks.1.hook_attn_out"
    r = run(ind, part="attention", function="copied", form="table")
    assert r["rows"] == {"fit": len(ind.fit) * len(q), "held": len(ind.held) * len(q)}
    copied = lambda w: ind.functions.keys["copied"].values(w)[:, q]  # noqa: E731
    Yf = clean(ind, ind.params, ind.fit, (tap,))[1][tap][:, q].reshape(-1, ind.config.d_model)
    kf = copied(ind.fit).reshape(-1)
    T = np.tile(Yf.mean(0), (ind.task.vocab_size, 1))
    for v in np.unique(kf):
        T[v] = Yf[kf == v].mean(0)
    base = clean(ind, ind.params, ind.held, ())[0][:, q]
    for values, name in ((T[copied(ind.held)], "kl"), (np.broadcast_to(Yf.mean(0), (len(ind.held), len(q), 16)), "ablation")):
        store = {}
        logits = intervened(ind, ind.params, ind.held, (edit(tap, q, "set", values=putting(store, values)),), store)
        assert close(r["meter"][name], kl(base.reshape(-1, base.shape[-1]), logits[:, q].reshape(-1, base.shape[-1]))), name

    # Where a key is undefined at some of the class's positions, the part keeps its own output there.
    even = Key("even", ind.task.vocab_size, lambda w: np.where(np.arange(ind.task.seq) % 2 == 0,
                                                               ind.functions.keys["copied"].values(w), -1))
    fn = Function("even", 1, "mechanism", ("even",), (Read("token(p)", "across"),), None, (), None, ("table",), True)
    halves = RunFunctions("induction", ind.task.seq, {"even": even}, {"even": fn}, {})
    r = run(ind, part="attention", function="even", form="table", functions=halves)
    kept = [x for x in q if x % 2 == 0]
    assert r["rows"] == {"fit": len(ind.fit) * len(kept), "held": len(ind.held) * len(kept)}
    Yf = clean(ind, ind.params, ind.fit, (tap,))[1][tap][:, kept].reshape(-1, 16)
    kf = copied(ind.fit)[:, [q.index(x) for x in kept]].reshape(-1)
    T = np.tile(Yf.mean(0), (ind.task.vocab_size, 1))
    for v in np.unique(kf):
        T[v] = Yf[kf == v].mean(0)
    base, rec = clean(ind, ind.params, ind.held, (tap,))
    values = rec[tap][:, q].copy()
    for j, x in enumerate(q):
        if x % 2 == 0:
            values[:, j] = T[copied(ind.held)[:, j]]
    store = {}
    logits = intervened(ind, ind.params, ind.held, (edit(tap, q, "set", values=putting(store, values)),), store)
    assert close(r["meter"]["kl"], kl(base[:, kept].reshape(-1, base.shape[-1]), logits[:, kept].reshape(-1, base.shape[-1])))

    # A part that writes nothing: no KL, no ablation, nothing removed, no R².
    import equinox as eqx
    import jax.numpy as jnp
    proj = ma.params.blocks[0].attn.c_proj
    silent = eqx.tree_at(lambda q: (q.blocks[0].attn.c_proj.weight, q.blocks[0].attn.c_proj.bias), ma.params,
                         (jnp.zeros_like(proj.weight), jnp.zeros_like(proj.bias)))
    z = run(ma, part="attention", function="operands", form="additive", params=silent)
    assert z["meter"]["kl"] == 0 and z["meter"]["ablation"] == 0 and z["meter"]["removed"] is None
    assert z["meter"]["label"] == C.INERT and z["r2"] is None

    # Requests the run doesn't declare are refused before anything runs.
    for bad in (dict(part="head"), dict(function="nothing"), dict(block=1), dict(cls="every"), dict(form="additive"),
                dict(seed=-1), dict(seed=True), dict(fit=ma.fit.astype(float)), dict(held=ma.held[:, :3]),
                dict(held=-ma.held), dict(fit=ma.fit[:0]), dict(steps=[1]), dict(steps=[2, 1], params_at=lambda s: ma.params),
                dict(steps=[], params_at=lambda s: ma.params), dict(declaration=ind.declaration)):
        args = dict(model=ma.model, params=ma.params, n_layers=1, functions=ma.functions, declaration=ma.declaration,
                    block=0, part="write", cls="answer", function="sum", form="table", fit_windows=ma.fit,
                    held_windows=ma.held, seed=0)
        for k, v in bad.items():
            args[{"fit": "fit_windows", "held": "held_windows"}.get(k, k)] = v
        with pytest.raises(ValueError):
            C.crystallize(**args)
    table_only = ma.functions.functions["sum"]
    table_only = Function("sum", 0, "answer", ("sum",), table_only.reads, None, table_only.wrong, None, ("table",), True)
    import dataclasses
    with pytest.raises(ValueError):  # a declaration of the same shape, for another run
        run(ma, part="write", function="sum", form="table", declaration=dataclasses.replace(ma.declaration, run="other"))
    with pytest.raises(ValueError):  # a form the function could take but doesn't admit
        run(ma, part="write", function="sum", form="fourier",
            functions=RunFunctions("modular_addition", 3, dict(ma.functions.keys), {"sum": table_only}, {}))


def test_loss_delta_is_recorded(ma):
    r = run(ma, part="write", function="sum", form="table")
    T, _ = ma_table(ma, ma.params, "sum", ma.fit)
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    crystal = ma_write(ma, ma.params, T[ma.functions.keys["sum"].values(ma.held)[:, 2]])
    target = ma.held[:, 3]

    def loss(logits):
        z = logits - logits.max(-1, keepdims=True)
        return float(-(z[np.arange(len(z)), target] - np.log(np.exp(z).sum(-1))).mean())

    assert close(r["loss"]["model"], loss(base)) and close(r["loss"]["crystallized"], loss(crystal))
    assert close(r["meter"]["loss_delta"], r["loss"]["crystallized"] - r["loss"]["model"], rtol=1e-9, atol=1e-12)
    assert r["accuracy"] == {"model": np.mean(np.argmax(base, -1) == target),
                             "crystallized": np.mean(np.argmax(crystal, -1) == target)}
    # A table keyed by the answer predicts the answer, whatever the model does: the loss is recorded, never judged.
    answer = Key("answer", ma.task.p, lambda w: np.where(np.arange(3) == 2, np.asarray(w)[:, [3]], -1).astype(np.int64))
    fn = Function("answer", 0, "answer", ("answer",), (Read("token(p-2)", "across"),), None, (), None, ("table",), True)
    keyed = RunFunctions("modular_addition", 3, {"answer": answer}, {"answer": fn}, {})
    a = run(ma, part="write", function="answer", form="table", functions=keyed)
    assert a["meter"]["label"] == C.label(a["meter"]["kl"], a["meter"]["ablation"])
    assert a["meter"]["loss_delta"] == a["loss"]["crystallized"] - a["loss"]["model"]


def test_wrong_functions(ma):
    r = run(ma, part="write", function="sum", form="table", twin=ma.twin)
    assert [x["key"] for x in r["wrong"]] == ["difference", "a", "b"] == list(ma.functions.functions["sum"].wrong)
    base = clean(ma, ma.params, ma.held, ())[0][:, 2]
    for x in r["wrong"]:
        T, _ = ma_table(ma, ma.params, x["key"], ma.fit)
        logits = ma_write(ma, ma.params, T[ma.functions.keys[x["key"]].values(ma.held)[:, 2]])
        assert close(x["kl"], kl(base, logits)) and x["agreement"] == np.mean(np.argmax(logits, -1) == np.argmax(base, -1))
    twin_base = clean(ma, ma.twin, ma.held, ())[0][:, 2]
    T, _ = ma_table(ma, ma.twin, "a", ma.fit)
    assert close(r["twin"]["wrong"][1]["kl"], kl(twin_base, ma_write(ma, ma.twin, T[ma.held[:, 0]])))

    # A key that only relabels the function's own is refused: refitted, it is the function (D58).
    keys = dict(ma.functions.keys)
    p = ma.task.p
    keys["shifted"] = Key("shifted", p, lambda w: np.where(np.arange(3) == 2, (np.asarray(w)[:, [0]] + np.asarray(w)[:, [1]]
                                                                                + 1) % p, -1).astype(np.int64))
    keys["partial"] = Key("partial", p, lambda w: np.where((np.arange(3) == 2) & (np.asarray(w)[:, [0]] > 0),
                                                           np.asarray(w)[:, [0]], -1).astype(np.int64))
    base_fn = ma.functions.functions["sum"]
    for wrong in (("shifted",), ("partial",), ("a", "shifted")):
        fn = Function("sum", 0, "answer", ("sum",), base_fn.reads, None, wrong, None, ("table", "fourier"), True)
        bad = RunFunctions("modular_addition", 3, keys, {"sum": fn}, {})
        for form in ("table", "fourier"):
            with pytest.raises(ValueError):
                run(ma, part="write", function="sum", form=form, functions=bad)
    # Relabelling the function's keys together is refused too; a key that merges values isn't a relabelling.
    keys["half"] = Key("half", p, lambda w: np.where(np.arange(3) == 2, np.asarray(w)[:, [0]] // 2, -1).astype(np.int64))
    fn = Function("operands", 0, "answer", ("a", "b"), base_fn.reads, None, ("half",), None, ("additive",), False)
    ok = RunFunctions("modular_addition", 3, keys, {"operands": fn}, {})
    assert [x["key"] for x in run(ma, part="attention", function="operands", form="additive", functions=ok)["wrong"]] \
        == ["half"]


def test_counterfactuals(ma, ind):
    # Modular addition's MLP, with the attention output patched from another pair.
    r = run(ma, part="mlp", function="sum and operands", form="additive", seed=5, twin=ma.twin)
    case = ma.functions.counterfactuals["attention patched"](ma.held, np.random.default_rng(5))
    p = ma.task.p
    design = lambda k: np.concatenate([np.eye(p)[k[:, j]] for j in range(3)], axis=1)  # noqa: E731

    def expect(params):
        keys_of = lambda w: np.stack([ma.functions.keys[n].values(w)[:, 2] for n in ("sum", "a", "b")], axis=1)  # noqa: E731
        W, *_ = np.linalg.lstsq(design(keys_of(ma.fit)), clean(ma, params, ma.fit, (MLP,))[1][MLP][:, 2], rcond=None)
        src = case.patch[2][:, :3].astype(np.int32)
        store = {batch_hash(src): src}
        patch = edit(ATTN, [2], "patch", source=batch_hash(src))
        base = intervened(ma, params, case.windows, (patch,), store)[:, 2]
        out = {}
        for which in ("reads", "answer"):
            k = np.stack([getattr(case, which)[n] for n in ("sum", "a", "b")], axis=1)
            st = dict(store)
            logits = intervened(ma, params, case.windows,
                                (patch, edit(MLP, [2], "set", values=putting(st, (design(k) @ W)[:, None]))), st)[:, 2]
            out[which] = {"kl": kl(base, logits), "agreement": np.mean(np.argmax(logits, -1) == np.argmax(base, -1))}
        return out

    for got, params in ((r["counterfactual"], ma.params), (r["twin"]["counterfactual"], ma.twin)):
        want = expect(params)
        assert got["name"] == "attention patched" and got["rows"] == len(ma.held)
        for which in ("reads", "answer"):
            assert close(got[which]["kl"], want[which]["kl"]) and got[which]["agreement"] == want[which]["agreement"]
    assert run(ma, part="mlp", function="sum and operands", form="additive", seed=6)["counterfactual"] != \
        r["counterfactual"], "the seed draws the case"

    # Induction's broken windows: one row per window, at j - 1, the rest of the class keyed by the changed window.
    q = list(ind.task.mechanism_positions)
    tap = "blocks.1.hook_attn_out"
    r = run(ind, part="attention", function="copied", form="table", seed=2)
    case = ind.functions.counterfactuals["broken"](ind.held, np.random.default_rng(2))
    copied = lambda w: ind.functions.keys["copied"].values(w)  # noqa: E731
    Yf = clean(ind, ind.params, ind.fit, (tap,))[1][tap][:, q].reshape(-1, 16)
    kf = copied(ind.fit)[:, q].reshape(-1)
    T = np.tile(Yf.mean(0), (ind.task.vocab_size, 1))
    for v in np.unique(kf):
        T[v] = Yf[kf == v].mean(0)
    base = clean(ind, ind.params, case.windows, ())[0]
    rows = case.rows
    assert r["counterfactual"]["rows"] == len(rows) == len(ind.held)
    for which in ("reads", "answer"):
        keys = copied(case.windows)[:, q].copy()
        keys[np.arange(len(rows)), [q.index(int(x)) for x in rows[:, 1]]] = getattr(case, which)["copied"]
        store = {}
        logits = intervened(ind, ind.params, case.windows, (edit(tap, q, "set", values=putting(store, T[keys])),), store)
        at = lambda x: x[rows[:, 0], rows[:, 1]]  # noqa: E731
        got = r["counterfactual"][which]
        assert close(got["kl"], kl(at(base), at(logits))), which
        assert got["agreement"] == np.mean(np.argmax(at(logits), -1) == np.argmax(at(base), -1)), which

    # A case that changes the windows the model reads: every pass reads the case's windows.
    def moved(windows, rng):
        w = np.array(windows, dtype=np.int32)
        w[:, 0] = (w[:, 0] + 1) % p
        sums_of = lambda x: (x[:, 0].astype(np.int64) + x[:, 1]) % p  # noqa: E731
        return Case(w, None, np.c_[np.arange(len(w)), np.full(len(w), 2)].astype(np.int64), {"sum": sums_of(w)},
                    {"sum": sums_of(np.asarray(windows))})

    fn = ma.functions.functions["sum"]
    fn = Function("sum", 0, "answer", ("sum",), fn.reads, None, fn.wrong, "moved", fn.forms, True)
    shifted_case = RunFunctions("modular_addition", 3, dict(ma.functions.keys), {"sum": fn}, {"moved": moved})
    r = run(ma, part="write", function="sum", form="table", functions=shifted_case)
    w = moved(ma.held, None).windows
    T, _ = ma_table(ma, ma.params, "sum", ma.fit)
    base = clean(ma, ma.params, w, ())[0][:, 2]
    sums_of = lambda x: (x[:, 0].astype(np.int64) + x[:, 1]) % p  # noqa: E731
    for which, sums in (("reads", sums_of(w)), ("answer", sums_of(ma.held))):
        logits = ma_write(ma, ma.params, T[sums], windows=w)
        assert close(r["counterfactual"][which]["kl"], kl(base, logits)), which

    # A case row outside the class's positions is refused.
    def outside(windows, rng):
        good = ma.functions.counterfactuals["attention patched"](windows, rng)
        return Case(good.windows, good.patch, np.c_[good.rows[:, 0], np.ones(len(good.rows), np.int64)], good.reads,
                    good.answer)

    fn = ma.functions.functions["sum and operands"]
    bad = RunFunctions("modular_addition", 3, dict(ma.functions.keys), {fn.name: fn}, {"attention patched": outside})
    with pytest.raises(ValueError):
        run(ma, part="mlp", function="sum and operands", form="additive", functions=bad)


def test_labels():
    assert (C.HOLDS, C.MISSES, C.INERT) == ("holds", "doesn't hold", "inert")
    assert (C.KL_HOLDS, C.MOVES, C.ACCURATE) == (0.25, 1.0, 0.9)
    assert (C.ENERGY, C.MAX_FREQUENCIES, C.NAMES_HOLD, C.CHUNK) == (0.75, 8, 0.7, 256)
    assert C.PARTS == ("attention", "mlp", "write")
    assert C.label(0.25, 1.0) == C.HOLDS and C.label(0.0, 7.0) == C.HOLDS
    assert C.label(0.2500001, 1.0) == C.MISSES and C.label(3.0, 4.0) == C.MISSES
    assert C.label(0.0, 0.999) == C.INERT and C.label(5.0, 0.5) == C.INERT, "an ablation below 1 is inert, whatever the KL"


def test_envelope_and_gap(ma):
    rows = [{"step": s, "label": lab, "accuracy": acc} for s, lab, acc in
            ((0, C.MISSES, 0.1), (100, C.HOLDS, 0.5), (200, C.MISSES, 0.9), (300, C.HOLDS, 0.92), (400, C.HOLDS, 0.97),
             (500, C.HOLDS, 0.99))]
    assert C.envelope(rows) == {"steps": [100, 300, 400, 500], "runs": [[100, 100], [300, 500]], "final": [300, 500],
                                "gap": [200], "accurate_from": 200, "before": [100]}
    rows = [{"step": s, "label": lab, "accuracy": acc} for s, lab, acc in
            ((0, C.INERT, None), (10, C.HOLDS, 0.3), (20, C.HOLDS, 0.89), (30, C.MISSES, 0.5))]
    assert C.envelope(rows) == {"steps": [10, 20], "runs": [[10, 20]], "final": None, "gap": [],
                                "accurate_from": None, "before": [10, 20]}
    assert C.envelope([]) == {"steps": [], "runs": [], "final": None, "gap": [], "accurate_from": None, "before": []}

    # Across training the form is fitted again at each checkpoint: each row is that checkpoint's own crystallization.
    at = {0: ma.twin, 7: ma.params}
    r = run(ma, part="write", function="sum", form="table", steps=[0, 7], params_at=at.__getitem__)
    for row, params in zip(r["checkpoints"], (ma.twin, ma.params)):
        own = run(ma, part="write", function="sum", form="table", params=params)
        assert set(row) == {"step", "kl", "ablation", "label", "accuracy"}
        assert close(row["kl"], own["meter"]["kl"], rtol=1e-9, atol=1e-12)
        assert close(row["ablation"], own["meter"]["ablation"], rtol=1e-9, atol=1e-12)
        assert row["label"] == own["meter"]["label"] and row["accuracy"] == own["accuracy"]["model"]
    assert [x["step"] for x in r["checkpoints"]] == [0, 7] and r["envelope"] == C.envelope(r["checkpoints"])


def record(kind, run_id, args, results):
    """A record of the gesture ``kind``, made up: only its request and results matter to a citation."""
    return GestureRecord(GestureRequest(kind, run_id, 100, args, 8, 0),
                         {"states": {"100": "f" * 64}, "dictionaries": ["d" * 64], "windows": ["b" * 64]},
                         {"nablaframe.gestures.name": "0" * 40}, results)


def test_citations(ma, ind):
    assert C.describes(ma.declaration, ma.functions, "sum", ma.held) == ("token(p+1)",)
    assert C.describes(ma.declaration, ma.functions, "sum and operands", ma.held) == ("token(p+1)", "token(p-2)",
                                                                                      "token(p-1)")
    assert C.describes(ind.declaration, ind.functions, "copied", ind.held) == ("token(p+1)",)
    even = Key("even", ind.task.vocab_size, lambda w: np.where(np.arange(ind.task.seq) % 2 == 0,
                                                               ind.functions.keys["copied"].values(w), -1))
    halves = RunFunctions("induction", ind.task.seq, {"even": even}, {"even": Function(
        "even", 1, "mechanism", ("even",), (Read("token(p)", "across"),), None, (), None, ("table",), True)}, {})
    assert C.describes(ind.declaration, halves, "even", ind.held) == ("token(p+1)",), "compared only where defined"
    hop = F.world("two_hop")
    assert C.describes(hop.declaration, hop.functions, "hop", hop.held) == ("token(p+1)",)
    keys = dict(ma.functions.keys)
    keys["nowhere"] = Key("nowhere", 11, lambda w: np.full((len(w), 3), -1, np.int64))
    fn = Function("odd", 0, "answer", ("difference", "a", "nowhere"), (Read("token(p-2)", "across"),), None, (), None,
                  ("additive",), False)
    odd = RunFunctions("modular_addition", 3, keys, {"odd": fn}, {})
    assert C.describes(ma.declaration, odd, "odd", ma.held) == (None, "token(p-2)", None)

    run_id, other_run, D, E = "a" * 64, "c" * 64, "d" * 64, "e" * 64
    named = lambda d, v, label=NAMED: {"name": {"description": d, "value": v, "f1": 0.8, "label": label}}  # noqa: E731
    name_args = lambda latent, dictionary=D, site=0, cls="answer": {"dictionary": dictionary, "site": site,  # noqa: E731
                                                                     "class": cls, "latent": latent}
    sums = sorted(set(ma.functions.keys["sum"].values(ma.held)[:, 2].tolist()))
    names = [record("name", run_id, name_args(i), named("token(p+1)", v)) for i, v in enumerate(sums[:7])]
    names.append(record("name", run_id, name_args(40), named("token(p+1)", sums[7], WEAK)))  # weak: not as planted
    elsewhere = [record("name", run_id, name_args(50, site=1), named("token(p+1)", 0)),
                 record("name", run_id, name_args(51, dictionary=E), named("token(p+1)", 0)),
                 record("name", run_id, name_args(52, cls="every"), named("token(p+1)", 0)),
                 record("name", run_id, name_args(53), named("token(p)", 0)),
                 record("name", run_id, name_args(54), {"name": None})]
    edge = lambda relation, down=D, site=0: record("edge", run_id, {"relation": relation, "up": E, "up_site": 0,  # noqa: E731
                                                                     "down": down, "down_site": site}, {"label": "computed"})
    edges = [edge("hop"), edge("copy"), edge("hop", down=E), edge("hop", site=1)]
    everything = names + elsewhere + edges
    got = C.cite(everything, cls="answer", descriptions=("token(p+1)",), dictionary=D, site=0, relation="hop")
    assert got == {"names": sorted(r.key for r in names), "edges": [edges[0].key]}
    assert C.cite(everything, cls="answer", descriptions=("token(p+1)", None), dictionary=D, site=0,
                  relation=None)["edges"] == []
    with pytest.raises(ValueError):
        C.cite(everything + [record("name", other_run, name_args(60), named("token(p+1)", 0))], cls="answer",
               descriptions=("token(p+1)",), dictionary=D, site=0, relation=None)
    with pytest.raises(ValueError):
        C.cite(everything + ["not a record"], cls="answer", descriptions=("token(p+1)",), dictionary=D, site=0,
               relation=None)

    # In the record: the share of the key's values named as planted, and the note where it falls short.
    r = run(ma, part="write", function="sum", form="table", cited=names)
    (entry,) = r["cites"]["names"]
    share = 7 / len(sums)
    assert entry == {"key": "sum", "description": "token(p+1)", "records": sorted(x.key for x in names),
                     "values": len(sums), "named": share, "note": C.NAMES_FALL_SHORT if share < 0.7 else None}
    assert entry["note"] == C.NAMES_FALL_SHORT and r["cites"]["edges"] == []
    more = names + [record("name", run_id, name_args(70 + i), named("token(p+1)", v)) for i, v in enumerate(sums[7:])]
    assert run(ma, part="write", function="sum", form="table", cited=more)["cites"]["names"][0]["named"] == 1.0
    assert run(ma, part="write", function="sum", form="table", cited=more)["cites"]["names"][0]["note"] is None
    assert run(ma, part="write", function="sum", form="table", cited=[])["cites"] == {"names": [{
        "key": "sum", "description": "token(p+1)", "records": [], "values": len(sums), "named": 0.0,
        "note": C.NAMES_FALL_SHORT}], "edges": []}
    for bad in ([edges[0]], [elsewhere[3]], names + [record("name", other_run, name_args(61), named("token(p+1)", 1))]):
        with pytest.raises(ValueError):
            run(ma, part="write", function="sum", form="table", cited=bad)

    # Two-hop's hop reads b at its own position, and cites the hop's edge record.
    f = hop.functions.functions["hop"]
    hop_edge = record("edge", run_id, {"relation": "hop", "up": E, "up_site": 0, "down": D, "down_site": 0},
                      {"label": "computed"})
    r = C.crystallize(hop.model, hop.params, n_layers=4, functions=hop.functions, declaration=hop.declaration,
                      block=3, part="write", cls=f.cls, function="hop", form="table", fit_windows=hop.fit,
                      held_windows=hop.held, seed=0, cited=[hop_edge])
    assert r["cites"]["edges"] == [{"key": hop_edge.key, "relation": "hop", "label": "computed"}]
    assert r["cites"]["names"][0]["records"] == [] and r["relation"] == "hop"
