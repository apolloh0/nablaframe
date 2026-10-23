"""Phase 7 delta 1: the modular-addition task and its declaration (D54, D57).

Contracts: src/nablaframe/zoo/tasks.py (ModularAdditionTask) and src/nablaframe/zoo/declarations.py
(modular_addition). Numpy only.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nablaframe.data.tokenizer import Tokenizer
from nablaframe.gestures.declare import DESCRIPTIONS, Place, Planted, values_at
from nablaframe.zoo import declarations
from nablaframe.zoo.tasks import ModularAdditionTask

P = 113
TASK = ModularAdditionTask()


def test_modadd_vocab_split_and_windows():
    t = TASK
    assert (t.p, t.train_fraction, t.split_seed) == (113, 0.3, 0)
    assert len(t.vocab) == P + 1 and t.vocab[0] == "0" and t.vocab[112] == "112" and t.vocab[P] == "="
    assert t.vocab[:P] == tuple(str(i) for i in range(P))
    assert t.window == 4 and t.seq == 3 and t.chance == 1 / P
    assert list(t.mechanism_positions) == [2] and list(t.control_positions) == [0]
    assert json.loads(t.tokenizer_json()) == json.loads(Tokenizer(t.vocab).to_json())

    pairs = t.pairs
    assert pairs.shape == (P * P, 2) and pairs.dtype == np.int64
    for i in (0, 1, 112, 113, 114, 5000, P * P - 1):
        assert pairs[i].tolist() == [i // P, i % P]

    train, held = t.split
    assert train.dtype == held.dtype == np.int64
    assert len(train) == 3831 and len(held) == 8938, "round(0.3 * 12,769) = 3,831 train, the rest held out"
    assert np.all(np.diff(train) > 0) and np.all(np.diff(held) > 0), "each part ascending"
    assert np.array_equal(np.union1d(train, held), np.arange(P * P)) and np.intersect1d(train, held).size == 0
    order = np.random.default_rng(0).permutation(P * P)
    assert np.array_equal(train, np.sort(order[:3831])) and np.array_equal(held, np.sort(order[3831:]))
    other = ModularAdditionTask(split_seed=1).split[0]
    assert not np.array_equal(other, train), "the split seed picks the split"
    small = ModularAdditionTask(p=7, train_fraction=0.3)
    assert len(small.split[0]) == 15 and len(small.split[1]) == 34, "round(0.3 * 49) = 15"
    assert len(ModularAdditionTask(p=7, train_fraction=1 / 7).split[0]) == 7

    index = np.array([5000, 0, 12768, 114])
    w = t.windows(index)
    assert w.dtype == np.int32 and w.shape == (4, 4)
    for row, i in zip(w, index):
        a, b = divmod(int(i), P)
        assert row.tolist() == [a, b, P, (a + b) % P], "in the order given: a, b, =, (a + b) mod p"
    assert t.windows(np.array([], dtype=np.int64)).shape == (0, 4)
    assert np.array_equal(t.training(), t.windows(train)) and np.array_equal(t.held_out(), t.windows(held))
    assert np.array_equal(small.windows(np.arange(49))[:, 3], (np.arange(49) // 7 + np.arange(49) % 7) % 7)

    for bad in (dict(p=2), dict(p=True), dict(p=7.0), dict(p=-5), dict(train_fraction=0.0), dict(train_fraction=1.0),
                dict(p=3, train_fraction=0.05), dict(p=3, train_fraction=0.97), dict(train_fraction="0.3"),
                dict(split_seed=-1), dict(split_seed=1.5), dict(split_seed=False)):
        with pytest.raises(ValueError):
            ModularAdditionTask(**bad).split
    for bad in (np.array([P * P]), np.array([-1]), np.array([0.0]), np.array([[0, 1]])):
        with pytest.raises(ValueError):
            t.windows(bad)


def test_modadd_batches_are_seeded_training_pairs():
    t = TASK
    train, _ = t.split
    b = t.batch_for_step(0, 5, 512)
    assert b.dtype == np.int32 and b.shape == (512, 4)
    rng = np.random.default_rng([0, 5])
    assert np.array_equal(b, t.windows(rng.choice(train, size=512, replace=True))), "drawn with replacement"
    index = b[:, 0].astype(np.int64) * P + b[:, 1]
    assert np.isin(index, train).all(), "only training pairs"
    assert np.array_equal(b[:, 3], (b[:, 0] + b[:, 1]) % P) and (b[:, 2] == P).all()
    assert np.array_equal(t.batch_for_step(0, 5, 512), b), "a step's batch is the same every time"
    assert not np.array_equal(t.batch_for_step(0, 6, 512), b) and not np.array_equal(t.batch_for_step(1, 5, 512), b)
    big = t.batch_for_step(0, 0, 8192)
    assert big.shape == (8192, 4) and len(np.unique(big[:, 0].astype(np.int64) * P + big[:, 1])) < 3831, \
        "8,192 draws with replacement from 3,831 pairs repeat some and miss others"


def test_modadd_declaration():
    d = declarations.modular_addition(TASK)
    assert d.run == "modular_addition" and d.seq == 3 and d.variables == ()
    assert dict(d.classes) == {"answer": (2,), "every": (0, 1, 2)}
    assert d.planted == (Planted(Place("token(p+1)", 0, "answer"), False),), "recorded, never judged (C11)"
    assert d.relations == ()

    w = TASK.held_out()[:200]
    got = d.describe(w)
    assert set(got) == set(DESCRIPTIONS)
    assert np.array_equal(got["token(p+1)"][:, 2], w[:, 3]) and np.array_equal(got["token(p-2)"][:, 2], w[:, 0])
    assert np.array_equal(got["token(p-1)"][:, 2], w[:, 1]) and (got["token(p)"][:, 2] == P).all()
    assert len(values_at(d.describe(TASK.held_out())["token(p+1)"], [2])) == P, "every sum counts at the answer"

    window = w[0]
    assert d.predict("token(p+1)", window, 2, 5) == (2, 5)
    assert d.predict("token(p+1)", window, 2, 112) == (2, 112)
    for desc, pos in (("token(p+1)", 1), ("token(p+1)", 0), ("token(p)", 2), ("token(p-2)", 2), ("token(p-1)", 2)):
        assert d.predict(desc, window, pos, 5) is None
    for args in (("b", window, 2, 5), ("token(p+1)", window[:3], 2, 5), ("token(p+1)", window, 3, 5),
                 ("token(p+1)", window, 2, -1), ("token(p+1)", window.astype(float), 2, 5)):
        with pytest.raises(ValueError):
            d.predict(*args)
    for name in ("sum", "hop"):
        with pytest.raises(ValueError, match=name):
            d.pairs(name, window)
        with pytest.raises(ValueError, match=name):
            d.value_map(name, window, 3)
