"""Phase 4 package 2: behavioural ground truth, from logits alone (D30)."""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo.ground_truth import TWIN_SEED, next_token_accuracy


def test_next_token_accuracy():
    windows = np.array([[1, 2, 3, 4], [3, 2, 1, 0]], np.int32)
    logits = np.zeros((2, 3, 5), np.float32)
    logits[0, 0, 2] = logits[0, 1, 0] = logits[0, 2, 4] = 1  # right, wrong, right
    logits[1, 0, 2] = logits[1, 1, 1] = logits[1, 2, 3] = 1  # right, right, wrong
    assert next_token_accuracy(logits, windows, [0, 1, 2]) == pytest.approx(4 / 6)
    assert next_token_accuracy(logits, windows, [0]) == 1.0 and next_token_accuracy(logits, windows, [1]) == 0.5
    tie = np.zeros((1, 1, 5), np.float32)
    assert next_token_accuracy(tie, np.array([[9, 0]], np.int32), [0]) == 1.0, "a tie goes to the lowest id"
    with pytest.raises(ValueError):
        next_token_accuracy(logits, windows[:, :3], [0])


def test_twin_params():
    pytest.importorskip("jax")
    import jax
    from nablaframe.model.gpt2 import GPT2Config, init_params
    from nablaframe.zoo.ground_truth import twin_params
    config = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)
    assert TWIN_SEED == 1
    same = jax.tree_util.tree_leaves(twin_params(config)), jax.tree_util.tree_leaves(init_params(config, seed=1))
    assert all(np.array_equal(a, b) for a, b in zip(*same)), "the untrained twin, seeded apart from training's"
