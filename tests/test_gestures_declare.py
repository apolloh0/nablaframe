"""Phase 6 delta 1: what each run declares to the gestures (D49, D50, "What each run declares").

Contracts: src/nablaframe/gestures/declare.py and src/nablaframe/zoo/declarations.py. Numpy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.gestures import declare
from nablaframe.gestures.declare import (DESCRIPTIONS, Declaration, Place, Planted, Relation, counts_at,
                                         token_descriptions, values_at)
from nablaframe.zoo import declarations
from nablaframe.zoo.tasks import InductionTask, TwoHopTask

TWO_HOP, INDUCTION = TwoHopTask(), InductionTask()


def two_hop_windows(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([TWO_HOP.window(rng) for _ in range(n)])


def induction_window(period, seed=0):
    return INDUCTION.draw_window(np.random.default_rng(seed), period=period)


def test_descriptions_and_classes():
    assert DESCRIPTIONS == ("token(p)", "token(p-1)", "token(p-2)", "token(p-3)", "token(p+1)")
    windows = np.arange(2 * 9).reshape(2, 9)
    d = token_descriptions(windows, 8)
    assert set(d) == set(DESCRIPTIONS) and all(v.shape == (2, 8) and v.dtype == np.int64 for v in d.values())
    assert d["token(p)"][1].tolist() == list(range(9, 17))
    assert d["token(p-1)"][0].tolist() == [-1] + list(range(0, 7))
    assert d["token(p-3)"][0].tolist() == [-1, -1, -1, 0, 1, 2, 3, 4]
    assert d["token(p+1)"][0].tolist() == list(range(1, 9))
    for bad in (windows[:, :8], windows[0], windows.astype(float), -windows - 1):
        with pytest.raises(ValueError):
            token_descriptions(bad, 8)
    with pytest.raises(ValueError):
        token_descriptions(windows, 0)

    th = declarations.two_hop(TWO_HOP)
    assert th.run == "two_hop" and th.seq == 26 and th.variables == ("b",)
    assert dict(th.classes) == {"query": (25,), "subjects": (0, 4, 8, 12, 16, 20), "objects": (2, 6, 10, 14, 18, 22),
                                "every": tuple(range(26))}
    w = two_hop_windows(4)
    got = th.describe(w)
    assert set(got) == set(DESCRIPTIONS) | {"b"}
    for i, window in enumerate(w):
        b = TWO_HOP.parse(window).b
        assert got["b"][i, 24] == got["b"][i, 25] == b and (got["b"][i, :24] == -1).all()
        assert got["token(p+1)"][i, 25] == TWO_HOP.parse(window).c and got["token(p-1)"][i, 25] == window[24]
    assert th.planted == (Planted(Place("b", 2, "query"), True), Planted(Place("token(p+1)", 3, "query"), True),
                          Planted(Place("b", 1, "query"), False))

    ind = declarations.induction(INDUCTION)
    assert ind.run == "induction" and ind.seq == 63 and ind.variables == ()
    assert dict(ind.classes) == {"mechanism": tuple(range(32, 63)), "every": tuple(range(63))}
    assert set(ind.describe(induction_window(9)[None])) == set(DESCRIPTIONS)
    assert ind.planted == (Planted(Place("token(p)", 0, "every"), True),
                           Planted(Place("token(p+1)", 1, "mechanism"), True))

    ref = declare.reference()
    assert ref.run == "reference" and ref.seq == 256 and dict(ref.classes) == {"every": tuple(range(256))}
    assert ref.planted == () and ref.relations == () and ref.variables == ()
    with pytest.raises(TypeError):
        th.classes["query"] = (24,)  # read-only

    # A declaration checks what it's given.
    noop = (lambda windows: {}, lambda *a: None, lambda *a: (), lambda *a: None)
    good = dict(run="r", seq=8, classes={"all": range(8)}, variables=(), planted=(), relations=())
    Declaration(**good, describe_variables=noop[0], predict=noop[1], pairs=noop[2], value_map=noop[3])
    for change in (dict(run=""), dict(seq=0), dict(classes={}), dict(classes={"all": [8]}), dict(classes={"all": [1, 1]}),
                   dict(classes={"all": []}), dict(variables=("token(p)",)), dict(variables=("v", "v")),
                   dict(planted=(Planted(Place("nope", 0, "all"), True),)),
                   dict(planted=(Planted(Place("token(p)", 0, "nowhere"), True),)),
                   dict(planted=(Planted(Place("token(p)", -1, "all"), True),)),
                   dict(relations=(Relation("r", Place("token(p)", 0, "all"), Place("token(p)", 1, "all"), True),) * 2)):
        with pytest.raises(ValueError):
            Declaration(**(good | change), describe_variables=noop[0], predict=noop[1], pairs=noop[2],
                        value_map=noop[3])
    wrong = Declaration(**(good | dict(variables=("v",))), describe_variables=lambda windows: {"u": np.zeros((1, 8), np.int64)},
                        predict=noop[1], pairs=noop[2], value_map=noop[3])
    with pytest.raises(ValueError):
        wrong.describe(np.zeros((1, 9), np.int64))  # the variables come back named other than declared


def test_a_constant_description_names_nothing():
    th = declarations.two_hop(TWO_HOP)
    d = th.describe(two_hop_windows(400))
    query = th.classes["query"]
    assert not counts_at(d["token(p)"], query), "the query position always holds ⇒"
    assert not counts_at(d["token(p-2)"], query), "two back from ⇒ is always ;"
    for name in ("token(p-1)", "token(p-3)", "token(p+1)", "b"):
        assert counts_at(d[name], query), f"{name} varies at the query"
    assert not counts_at(d["b"], th.classes["subjects"]), "b is declared only at 24 and 25"

    values = np.array([[3, 3, 3, 3, 3, -1], [7, 7, 7, 7, 7, 9]])
    assert values_at(values, [0, 1, 2, 3, 4, 5]).tolist() == [3, 7]
    assert values_at(values, [0, 1, 2, 3, 4, 5], min_count=6).tolist() == []
    assert values_at(values, [4, 5], min_count=1).tolist() == [3, 7, 9], "-1 is nothing, never a value"
    assert values_at(values, [0, 1, 2, 3]).dtype == np.int64
    assert not counts_at(values, [0, 1, 2, 3, 4, 5], min_count=6)
    assert counts_at(values, [0, 1, 2, 3, 4, 5])
    assert not counts_at(values[:1], [0, 1, 2, 3, 4]), "one value is a constant"


def test_predictions_and_relations():
    th = declarations.two_hop(TWO_HOP)
    w = two_hop_windows(1, seed=3)[0]
    p = TWO_HOP.parse(w)
    distractors = [s for s in p.facts if s not in (p.a, p.b)]
    for s in distractors:
        assert th.predict("b", w, 25, s) == (25, p.facts[s]), "b moved to a distractor subject answers its object"
    absent = next(e for e in range(TWO_HOP.entities) if e not in p.facts)
    for none in (p.a, p.b, absent):
        assert th.predict("b", w, 25, none) is None
    assert th.predict("token(p+1)", w, 25, 7) == (25, 7)
    assert th.predict("b", w, 24, distractors[0]) is None, "b's prediction is declared at the query only"
    assert th.predict("token(p+1)", w, 10, 7) is None and th.predict("token(p)", w, 25, 7) is None
    for bad in (("c", w, 25, 1), ("b", w[:-1], 25, 1), ("b", w, 26, 1), ("b", w, -1, 1), ("b", w, 25, -1),
                ("b", w.astype(float), 25, 1), ("b", w, 25, 1.0)):
        with pytest.raises(ValueError):
            th.predict(*bad)

    assert [r.name for r in th.relations] == ["hop", "carry", "two back"]
    hop, carry, two_back = th.relations
    assert hop == Relation("hop", Place("b", 2, "query"), Place("token(p+1)", 3, "query"), True)
    assert carry == Relation("carry", Place("b", 1, "query"), Place("b", 2, "query"), False)
    assert two_back == Relation("two back", Place("token(p)", 0, "objects"), Place("token(p-2)", 2, "subjects"), False)
    assert th.relation("hop") is hop
    assert th.pairs("hop", w) == ((25, 25),) == th.pairs("carry", w)
    assert th.pairs("two back", w) == ((2, 4), (6, 8), (10, 12), (14, 16), (18, 20))
    assert th.value_map("hop", w, p.b) == p.c, "the hop maps b to the answer"
    for s in distractors:
        assert th.value_map("hop", w, s) == p.facts[s]
    assert th.value_map("hop", w, absent) is None
    assert th.value_map("carry", w, 5) == 5 and th.value_map("two back", w, 5) == 5
    for call in (lambda: th.relation("copy"), lambda: th.pairs("copy", w), lambda: th.value_map("copy", w, 1),
                 lambda: th.pairs("hop", w[:-1])):
        with pytest.raises(ValueError):
            call()

    ind = declarations.induction(INDUCTION)
    window = induction_window(20, seed=1)
    for q in (32, 47, 62):
        assert ind.predict("token(p+1)", window, q, 5) == (q, 5)
    assert ind.predict("token(p+1)", window, 31, 5) is None, "next-token predictions are at mechanism positions"
    for s in declarations.sole_targets(INDUCTION, window):
        assert ind.predict("token(p)", window, int(s), 5) == (int(s) + 19, 5)
    assert ind.predict("token(p)", window, 5, 5) is None, "5 + 19 is no mechanism position"
    assert ind.predict("token(p)", window, 40, 5) is None, "a mechanism position is no sole target"
    assert ind.predict("token(p-1)", window, 40, 5) is None
    with pytest.raises(ValueError):
        ind.predict("b", window, 40, 5)
    assert [r.name for r in ind.relations] == ["copy"]
    assert ind.relations[0] == Relation("copy", Place("token(p)", 0, "every"), Place("token(p+1)", 1, "mechanism"), True)
    assert ind.pairs("copy", window) == ((13, 32), (14, 33), (15, 34), (16, 35), (17, 36), (18, 37), (19, 38), (20, 39))
    assert ind.value_map("copy", window, 9) == 9
    for call in (lambda: ind.pairs("hop", window), lambda: ind.value_map("hop", window, 1)):
        with pytest.raises(ValueError):
            call()

    ref = declare.reference(seq=16)
    window = np.arange(17)
    assert ref.predict("token(p+1)", window, 3, 9) == (3, 9) and ref.predict("token(p+1)", window, 15, 2) == (15, 2)
    assert all(ref.predict(d, window, 3, 9) is None for d in DESCRIPTIONS if d != "token(p+1)")
    for call in (lambda: ref.predict("token(p+1)", window, 16, 1), lambda: ref.predict("b", window, 3, 1),
                 lambda: ref.pairs("hop", window), lambda: ref.value_map("hop", window, 1), lambda: ref.relation("hop")):
        with pytest.raises(ValueError):
            call()


def test_sole_targets():
    assert declarations.period(np.array([4, 5, 6, 4, 5, 6, 4])) == 3
    with pytest.raises(ValueError):
        declarations.period(np.array([1, 2, 3]))
    expected = {8: [], 16: [], 17: [16, 17], 20: list(range(13, 21)), 32: list(range(1, 32))}
    for P, want in expected.items():
        window = induction_window(P, seed=P)
        got = declarations.sole_targets(INDUCTION, window)
        assert got.dtype == np.int64 and got.tolist() == want, f"period {P}"
        for s in got:
            q = s + P - 1
            assert 32 <= q <= 62 and window[q + 1] == window[s], "the copy predicts the token at s"
            earlier = [i for i in range(q) if window[i] == window[q]]
            assert earlier == [s - 1], "the token at s + P - 1 occurred once before, at s - 1"
    with pytest.raises(ValueError):
        declarations.sole_targets(INDUCTION, induction_window(20)[:-1])
