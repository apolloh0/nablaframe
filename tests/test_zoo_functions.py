"""Phase 7 delta 1: what each run offers crystallize-this, its keys, functions and counterfactuals (D56, D57).

Contract: src/nablaframe/zoo/functions.py. Numpy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo import declarations, functions
from nablaframe.zoo.functions import Case, Function, Key, Read, RunFunctions
from nablaframe.zoo.tasks import InductionTask, ModularAdditionTask, TwoHopTask
from nablaframe.zoo.two_hop import aligned_sources

INDUCTION, TWO_HOP, MODADD = InductionTask(), TwoHopTask(), ModularAdditionTask()


def induction_windows(n, seed=0):
    return INDUCTION.batch_for_step(seed, 0, n)


def two_hop_windows(n, seed=0):
    return TWO_HOP.batch_for_step(seed, 0, n)


def modadd_windows(n, seed=0):
    return MODADD.held_out()[np.random.default_rng(seed).choice(8938, size=n, replace=False)]


def fn(**over):
    base = dict(name="f", block=0, cls="answer", keys=("sum",), reads=(Read("token(p-2)", "across"),), relation=None,
                wrong=("a",), counterfactual=None, forms=("table",), judged=True)
    base.update(over)
    return Function(**base)


def test_functions_keys_and_reads():
    ind = functions.induction(INDUCTION)
    assert isinstance(ind, RunFunctions) and ind.run == "induction" and ind.seq == 63
    assert set(ind.keys) == {"copied", "current", "previous"}
    assert all(k.name == name and k.n_values == 64 for name, k in ind.keys.items())
    w = induction_windows(16)
    mech = list(range(32, 63))
    got = {name: k.values(w) for name, k in ind.keys.items()}
    for name, v in got.items():
        assert v.dtype == np.int64 and v.shape == (16, 63)
        assert (v[:, :32] == -1).all(), f"{name} is defined at the mechanism positions only"
    for i, window in enumerate(w):
        P = declarations.period(window)
        for q in mech:
            assert got["copied"][i, q] == window[q - P + 1], "the token after the previous occurrence"
            assert got["current"][i, q] == window[q] and got["previous"][i, q] == window[q - 1]
    lone = w[:1].copy()
    lone[0, 1:] = np.where(lone[0, 1:] == lone[0, 0], (lone[0, 0] + 1) % 64, lone[0, 1:])  # w[0] never recurs
    v = ind.keys["copied"].values(lone)
    assert (v == -1).all(), "no period read yet: the copied token is undefined"
    assert np.array_equal(ind.keys["current"].values(lone)[0, 32:], lone[0, 32:63])
    late = lone.copy()
    late[0, 45] = late[0, 0]  # w[0] recurs first at 45: before then no period has been read
    v = ind.keys["copied"].values(late)[0]
    assert (v[32:45] == -1).all() and all(v[q] == late[0, q - 45 + 1] for q in range(45, 63)), \
        "the period as far as q has read it, never one found later"
    f = ind.functions["copied"]
    assert f == Function("copied", 1, "mechanism", ("copied",), (Read("token(p)", "across"),), None,
                         ("current", "previous"), "broken", ("table",), True)
    assert set(ind.functions) == {"copied"} and set(ind.counterfactuals) == {"broken"}

    th = functions.two_hop(TWO_HOP)
    assert th.run == "two_hop" and th.seq == 26 and set(th.keys) == {"hop", "a", "b"}
    assert all(k.n_values == 51 for k in th.keys.values())
    w = two_hop_windows(32)
    got = {name: k.values(w) for name, k in th.keys.items()}
    for name, v in got.items():
        assert (v[:, :25] == -1).all(), f"{name} is defined at the query only"
    for i, window in enumerate(w):
        p = TWO_HOP.parse(window)
        assert (got["hop"][i, 25], got["a"][i, 25], got["b"][i, 25]) == (p.c, p.a, p.b)
        assert p.c == p.facts[p.b], "the hop's object is the object of b's fact"
    assert th.functions["hop"] == Function("hop", 3, "query", ("hop",), (Read("b", "same"), Read("token(p)", "across")),
                                           "hop", ("a", "b"), "b patched", ("table",), True)

    ma = functions.modular_addition(MODADD)
    assert ma.run == "modular_addition" and ma.seq == 3
    assert set(ma.keys) == {"sum", "difference", "shifted", "a", "b"} and all(k.n_values == 113 for k in ma.keys.values())
    w = modadd_windows(64)
    got = {name: k.values(w) for name, k in ma.keys.items()}
    a, b = w[:, 0].astype(np.int64), w[:, 1].astype(np.int64)
    want = {"sum": (a + b) % 113, "difference": (a - b) % 113, "shifted": (a + b + 1) % 113, "a": a, "b": b}
    for name, v in got.items():
        assert v.shape == (64, 3) and (v[:, :2] == -1).all() and np.array_equal(v[:, 2], want[name]), name
    across = (Read("token(p-2)", "across"), Read("token(p-1)", "across"))
    assert ma.functions["sum"] == Function("sum", 0, "answer", ("sum",), across, None,
                                           ("difference", "shifted", "a", "b"), None, ("table", "fourier"), True)
    assert ma.functions["sum and operands"] == Function(
        "sum and operands", 0, "answer", ("sum", "a", "b"), (Read("token(p-2)", "same"), Read("token(p-1)", "same")),
        None, (), "attention patched", ("additive",), False), "reads at its own position with no edge: recorded (D53)"
    assert ma.functions["operands"] == Function("operands", 0, "answer", ("a", "b"), across, None, (), None,
                                                ("additive",), False)
    assert set(ma.counterfactuals) == {"attention patched"}

    # A key at p reads only tokens up to p: changing later tokens, the answer among them, leaves it alone.
    for run, task, w in ((ind, INDUCTION, induction_windows(8, 3)), (th, TWO_HOP, two_hop_windows(8, 3)),
                         (ma, MODADD, modadd_windows(8, 3))):
        for name, k in run.keys.items():
            v = k.values(w)
            for p in range(run.seq):
                changed = w.copy()
                changed[:, p + 1:] = (changed[:, p + 1:] + 1) % len(task.vocab)
                assert np.array_equal(k.values(changed)[:, p], v[:, p]), f"{run.run} {name} at {p} reads ahead"
        for bad in (w[:, :-1], w[0], w.astype(float), -w - 1):
            with pytest.raises(ValueError):
                next(iter(run.keys.values())).values(bad)

    # for_run, and each function against its run's declaration
    for name, task, built in (("induction", INDUCTION, ind), ("two_hop", TWO_HOP, th), ("modular_addition", MODADD, ma)):
        again = functions.for_run(name, task)
        assert again.run == built.run and dict(again.functions) == dict(built.functions) and set(again.keys) == set(built.keys)
        decl = {"induction": declarations.induction, "two_hop": declarations.two_hop,
                "modular_addition": declarations.modular_addition}[name](task)
        for f in built.functions.values():
            assert f.cls in decl.classes
            assert f.relation is None or decl.relation(f.relation) is not None
            assert all(r.description in ("token(p)", "token(p-1)", "token(p-2)", "token(p-3)", "token(p+1)")
                       + decl.variables for r in f.reads)
    for bad in (("reference", INDUCTION), ("induction", TWO_HOP), ("two_hop", MODADD), ("modular_addition", INDUCTION)):
        with pytest.raises(ValueError):
            functions.for_run(*bad)
    for builder, task in ((functions.induction, TWO_HOP), (functions.two_hop, MODADD), (functions.modular_addition, None)):
        with pytest.raises(ValueError):
            builder(task)
    with pytest.raises(TypeError):
        ind.keys["copied"] = ind.keys["current"]
    with pytest.raises(TypeError):
        ma.functions["x"] = ma.functions["sum"]

    # what a Function refuses
    fn()  # the case they all vary is fine
    assert fn(keys=["sum"], wrong=["a"]).keys == ("sum",), "tuples are stored as tuples"
    for bad in (dict(name=""), dict(block=-1), dict(block=True), dict(keys=()), dict(keys=("sum", "sum"), forms=("additive",)),
                dict(keys=("sum", "b"), forms=("table",)), dict(keys=("sum", "b"), forms=("fourier",)),
                dict(forms=("additive",)), dict(forms=()), dict(forms=("lookup",)), dict(forms=("table", "table")),
                dict(reads=()), dict(reads=("token(p)",)),
                dict(relation="hop"), dict(reads=(Read("b", "same"),)), dict(wrong=("sum",)), dict(judged=1)):
        with pytest.raises(ValueError):
            fn(**bad)
    for bad in (lambda: Read("token(p)", "elsewhere"), lambda: Read("", "same"),
                lambda: fn(reads=(Read("token(p)", "across"), "token(p-1)"))):
        with pytest.raises(ValueError):
            bad()
    fn(reads=(Read("b", "same"),), judged=False)  # a same-position read with nothing to cite is fine when recorded
    fn(reads=(Read("b", "same"),), relation="hop")
    key = ma.keys["sum"]
    with pytest.raises(ValueError):
        RunFunctions("x", 3, {"sum": key}, {"f": fn()}, {})  # wrong function "a" isn't a key
    with pytest.raises(ValueError):
        RunFunctions("x", 3, {"sum": key, "a": ma.keys["a"]}, {"f": fn(counterfactual="nope")}, {})
    RunFunctions("x", 3, {"sum": key, "a": ma.keys["a"]}, {"f": fn()}, {})


def test_functions_wrong_and_counterfactual():
    # Wrong functions disagree with the right one almost everywhere.
    for run, w in ((functions.induction(INDUCTION), induction_windows(32)), (functions.two_hop(TWO_HOP), two_hop_windows(64)),
                   (functions.modular_addition(MODADD), modadd_windows(256))):
        for f in run.functions.values():
            right = run.keys[f.keys[0]].values(w)
            defined = right >= 0
            for name in f.wrong:
                other = run.keys[name].values(w)
                assert np.array_equal(other >= 0, defined), f"{name} is defined where {f.keys[0]} is"
                assert np.mean(other[defined] != right[defined]) > 0.9, f"{run.run}: {name} is a different function"

    # induction: broken windows
    ind = functions.induction(INDUCTION)
    w = induction_windows(64)
    original = w.copy()
    case = ind.counterfactuals["broken"](w, np.random.default_rng(3))
    assert isinstance(case, Case) and np.array_equal(w, original), "the windows given are left alone"
    assert case.patch is None and case.windows.dtype == np.int32 and case.windows.shape == w.shape
    rng = np.random.default_rng(3)
    for i in range(64):
        j = int(rng.integers(33, 63))
        new = int(rng.choice(np.delete(np.arange(64), w[i, j])))
        changed = np.flatnonzero(case.windows[i] != w[i])
        assert changed.tolist() == [j] and case.windows[i, j] == new, "one token, drawn as the contract says"
        assert case.rows[i].tolist() == [i, j - 1]
    assert case.rows.dtype == np.int64 and set(case.reads) == set(case.answer) == {"copied"}
    copied = ind.keys["copied"].values(case.windows)
    assert np.array_equal(case.reads["copied"], copied[case.rows[:, 0], case.rows[:, 1]])
    assert np.array_equal(case.reads["copied"], ind.keys["copied"].values(w)[case.rows[:, 0], case.rows[:, 1]]), \
        "the model can't see the change at j - 1: the copied token is the window's own"
    assert np.array_equal(case.answer["copied"], case.windows[case.rows[:, 0], case.rows[:, 1] + 1])
    assert (case.reads["copied"] != case.answer["copied"]).all(), "what it reads and the answer part ways"
    again = ind.counterfactuals["broken"](w, np.random.default_rng(3))
    assert np.array_equal(again.windows, case.windows) and np.array_equal(again.rows, case.rows)
    with pytest.raises(ValueError):
        ind.counterfactuals["broken"](w[:, :10], np.random.default_rng(3))

    # two-hop: b patched at the query from G4.3's aligned source
    th = functions.two_hop(TWO_HOP)
    w = two_hop_windows(64)
    case = th.counterfactuals["b patched"](w, np.random.default_rng(5))
    sources, _, counterparts = aligned_sources(TWO_HOP, w, np.random.default_rng(5))
    assert np.array_equal(case.windows, w) and case.windows.dtype == np.int32
    tap, position, src = case.patch
    assert (tap, position) == ("blocks.2.hook_resid_post", 25) and np.array_equal(src, sources) and src.dtype == np.int32
    assert case.rows.tolist() == [[i, 25] for i in range(64)]
    for i, window in enumerate(w):
        p = TWO_HOP.parse(window)
        assert case.reads["hop"][i] == p.facts[int(counterparts[i])], "the object of the counterpart's fact"
        assert case.answer["hop"][i] == p.c
    assert (case.reads["hop"] != case.answer["hop"]).all()

    # modular addition: the attention output patched from another pair with another sum
    ma = functions.modular_addition(MODADD)
    w = modadd_windows(128)
    case = ma.counterfactuals["attention patched"](w, np.random.default_rng(11))
    rng = np.random.default_rng(11)
    s = (w[:, 0].astype(np.int64) + w[:, 1]) % 113
    src = rng.integers(0, 128, size=128)
    while (same := s[src] == s).any():
        src[same] = rng.integers(0, 128, size=int(same.sum()))
    tap, position, sources = case.patch
    assert (tap, position) == ("blocks.0.hook_attn_out", 2) and np.array_equal(sources, w[src])
    assert np.array_equal(case.windows, w) and case.rows.tolist() == [[i, 2] for i in range(128)]
    assert set(case.reads) == set(case.answer) == {"sum", "a", "b"}
    assert np.array_equal(case.reads["sum"], s[src]) and np.array_equal(case.answer["sum"], s)
    assert np.array_equal(case.reads["a"], w[src, 0]) and np.array_equal(case.answer["b"], w[:, 1])
    assert (case.reads["sum"] != case.answer["sum"]).all()
    # with few sums, many windows draw a source with their own sum, and only those draw again
    held = MODADD.held_out()
    few = held[np.isin((held[:, 0].astype(np.int64) + held[:, 1]) % 113, range(6))][:60]
    s = (few[:, 0].astype(np.int64) + few[:, 1]) % 113
    rng = np.random.default_rng(2)
    src = rng.integers(0, 60, size=60)
    rounds = 0
    while (same := s[src] == s).any():
        src[same] = rng.integers(0, 60, size=int(same.sum()))
        rounds += 1
    assert rounds >= 2
    case = ma.counterfactuals["attention patched"](few, np.random.default_rng(2))
    assert np.array_equal(case.patch[2], few[src]) and (case.reads["sum"] != case.answer["sum"]).all()
    one_sum = MODADD.windows(np.array([1 * 113 + 4, 2 * 113 + 3, 0 * 113 + 5]))
    with pytest.raises(ValueError):
        ma.counterfactuals["attention patched"](one_sum, np.random.default_rng(0))
