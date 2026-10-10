"""Phase 4 package 1: zoo tasks, their mechanism positions, and the trainer's batch source (D29, D32).

Spec: docs/specs/phase-4.md, "The zoo".
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.data.tokenizer import Tokenizer
from nablaframe.zoo.tasks import InductionTask, TwoHopTask


def test_induction_windows():
    t = InductionTask()
    b = t.batch_for_step(seed=0, step=3, batch=5)
    assert b.dtype == np.int32 and b.shape == (5, 64) and (t.window, t.seq) == (64, 63)
    assert np.array_equal(b, t.batch_for_step(seed=0, step=3, batch=5)), "deterministic"
    assert not np.array_equal(b, t.batch_for_step(seed=0, step=4, batch=5))
    assert not np.array_equal(b, t.batch_for_step(seed=1, step=3, batch=5))
    for row in b:  # D36: a block of P distinct tokens, P from 8 to 32, repeated to fill the window
        period = next(P for P in range(8, 33) if (row[P:] == row[:-P]).all())
        assert len(set(row[:period].tolist())) == period, "no token repeats within a period"
    assert 0 <= b.min() and b.max() < 64
    assert t.mechanism_positions.tolist() == list(range(32, 63)) and t.control_positions.tolist() == list(range(7))
    assert t.chance == 1 / 64 and len(t.vocab) == 64
    assert Tokenizer.from_json(t.tokenizer_json()).vocab == t.vocab


def test_two_hop_windows():
    t = TwoHopTask()
    b = t.batch_for_step(seed=0, step=0, batch=50)
    assert b.dtype == np.int32 and b.shape == (50, 27) and t.seq == 26 and t.mechanism_positions.tolist() == [25]
    arrow, sep, query = (t.vocab.index(s) for s in ("→", ";", "⇒"))
    for w in b:
        facts = w[:24].reshape(6, 4)
        assert (facts[:, 1] == arrow).all() and (facts[:, 3] == sep).all() and w[25] == query
        assert len(set(facts[:, 0].tolist())) == 6 and facts[:, 0].max() < 48 and facts[:, 2].max() < 48
        p = t.parse(w)
        assert p.a == w[24] and p.facts[p.a] == p.b and p.facts[p.b] == p.c == w[26], "the answer is two hops away"
    assert len({int(np.where(w[0:24:4] == w[24])[0][0]) for w in b}) > 1, "the facts are shuffled"
    rng = np.random.default_rng(5)
    p = t.parse(t.window(rng, intermediate=7))
    assert p.b == 7 and p.a != 7 and p.facts[7] == p.c
    for _ in range(20):
        w = t.window(rng, broken=True)
        facts = dict(zip(w[0:24:4].tolist(), w[2:24:4].tolist()))
        assert facts[int(w[24])] not in facts, "the chain's second fact is missing"
        assert int(w[26]) not in w[:24].tolist(), "so the answer appears nowhere in the facts"
    assert t.chance == 1 / 48 and len(t.vocab) == 51
    assert Tokenizer.from_json(t.tokenizer_json()).vocab == t.vocab


def test_trainer_batch_source_leaves_the_default_unchanged():
    pytest.importorskip("jax")
    from nablaframe.data.corpus import batch_for_step
    from nablaframe.train.checkpoint import state_hash
    from nablaframe.train.loop import run_until
    from test_train import CONFIG, STREAM, TINY, _fresh
    plain = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=4)
    corpus = lambda k: batch_for_step(STREAM, CONFIG.data_seed, k, CONFIG.batch, CONFIG.seq)  # noqa: E731
    assert state_hash(run_until(_fresh(), None, TINY, CONFIG, until_step=4, batch_source=corpus)) == state_hash(plain)
    other = lambda k: np.random.default_rng(k).integers(0, TINY.vocab, (CONFIG.batch, CONFIG.seq + 1)).astype(np.int32)  # noqa: E731
    assert state_hash(run_until(_fresh(), None, TINY, CONFIG, until_step=4, batch_source=other)) != state_hash(plain)
    for bad in (lambda k: np.zeros((CONFIG.batch, CONFIG.seq), np.int32), lambda k: np.zeros((CONFIG.batch, CONFIG.seq + 1), np.int64)):
        with pytest.raises(ValueError):
            run_until(_fresh(), None, TINY, CONFIG, until_step=2, batch_source=bad)
