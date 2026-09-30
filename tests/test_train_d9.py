"""D9: warmup starts at lr / warmup_steps, so the first update moves the parameters.

Spec: docs/deltas/phase-1-delta-2.md, D9. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.model.gpt2 import GPT2Config  # noqa: E402
import nablaframe.train.loop as loop  # noqa: E402
from nablaframe.train.loop import TrainConfig, init_state, run_until  # noqa: E402

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)
CONFIG = TrainConfig(batch=4, seq=8, lr=1e-2, warmup_steps=4, total_steps=20, data_seed=1, param_seed=0)
STREAM = np.random.default_rng(0).integers(0, TINY.vocab, size=4096).astype(np.uint16)


def test_first_update_moves_params():
    before = init_state(TINY, CONFIG)
    after = run_until(before, STREAM, TINY, CONFIG, until_step=1)
    leaves = zip(jax.tree_util.tree_leaves(before.params), jax.tree_util.tree_leaves(after.params))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in leaves), \
        "update 1 must change the model, so step-1 and step-0 checkpoints differ"


def test_learning_rate_follows_d9():
    config = TrainConfig(lr=1e-3, warmup_steps=500, total_steps=12_208, min_lr_ratio=0.1)
    learning_rate = loop.learning_rate  # looked up here, so a missing function fails only this test
    assert learning_rate(config, 0) == pytest.approx(1e-3 / 500, rel=1e-5)
    assert learning_rate(config, 500) == pytest.approx(1e-3, rel=1e-5)
    assert learning_rate(config, 12_208) == pytest.approx(1e-4, rel=1e-5)
    assert learning_rate(config, 0) < learning_rate(config, 250) < learning_rate(config, 500)
    assert isinstance(learning_rate(config, 0), float)
