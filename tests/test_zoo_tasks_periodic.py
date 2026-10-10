"""Phase 4 delta 3: the periodic induction task (D36). Needs no JAX.

Contract: docs/deltas/phase-4-delta-3.md, "src/nablaframe/zoo/tasks.py".
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo.tasks import InductionTask


def _drawn_as_the_feasibility_run_drew(seed: int, step: int, batch: int) -> np.ndarray:
    rng = np.random.default_rng([seed, step])
    rows = []
    for _ in range(batch):
        period = int(rng.integers(8, 33))
        block = rng.choice(64, size=period, replace=False)
        rows.append([block[i % period] for i in range(64)])
    return np.array(rows, np.int32)


def test_periodic_windows_draw_in_a_fixed_order():
    t = InductionTask()
    assert (t.min_period, t.max_period) == (8, 32)
    b = t.batch_for_step(seed=0, step=3, batch=6)
    assert b.dtype == np.int32 and b.shape == (6, 64)
    assert np.array_equal(b, _drawn_as_the_feasibility_run_drew(0, 3, 6)), \
        "the period first, then the block, window by window: G4.1 retraces the feasibility run's batches"
    periods = {next(P for P in range(8, 33) if (row[P:] == row[:-P]).all()) for row in t.batch_for_step(0, 0, 64)}
    assert len(periods) > 5, "the period varies from window to window"


def test_a_fixed_period_draws_no_period():
    t = InductionTask()
    rng, mirror = np.random.default_rng(4), np.random.default_rng(4)
    w = t.draw_window(rng, period=20)
    block = mirror.choice(64, size=20, replace=False)
    assert w.dtype == np.int32 and w.tolist() == [int(block[i % 20]) for i in range(64)]
    assert rng.integers(1 << 30) == mirror.integers(1 << 30), "only the block was drawn"
    for bad in (7, 33):
        with pytest.raises(ValueError):
            t.draw_window(np.random.default_rng(0), period=bad)


def test_the_task_requires_its_mechanism():
    """D30's first condition: every mechanism position is predictable only by matching content."""
    t = InductionTask()
    assert t.mechanism_positions.tolist() == list(range(32, 63)) and t.control_positions.tolist() == list(range(7))
    for w in t.batch_for_step(seed=2, step=0, batch=64):
        for p in t.mechanism_positions:
            earlier = np.flatnonzero(w[:p] == w[p])
            assert earlier.size and (w[earlier + 1] == w[p + 1]).all(), \
                "every earlier occurrence of the current token is followed by the next token"
        for p in t.control_positions:
            assert w[p + 1] not in w[: p + 1], "a control position's next token hasn't appeared yet"


def test_periods_are_checked():
    for kwargs in ({"min_period": 1}, {"min_period": 9, "max_period": 8}, {"max_period": 65},
                   {"max_period": 63}, {"window": 33}):
        with pytest.raises(ValueError):
            InductionTask(**kwargs)
    small = InductionTask(vocab_size=16, window=20, min_period=3, max_period=6)
    assert small.mechanism_positions.tolist() == list(range(6, 19)) and small.control_positions.tolist() == [0, 1]
