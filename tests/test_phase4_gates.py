"""Phase 4 gate helpers: the zoo's settings, seeds, schedules and bars, fixed before any gate runs (D36, D37).

The settings are the feasibility runs', so G4.1 retraces them. Only the last test needs JAX.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from nablaframe.zoo.tasks import InductionTask, TwoHopTask


def _common():
    gates = REPO_ROOT / "gates" / "phase4"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase4_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_settings_and_bars_are_fixed():
    c = _common()
    assert c.MODELS == {
        "induction": {"vocab": 64, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_mlp": 256, "seq": 63},
        "two_hop": {"vocab": 51, "d_model": 128, "n_layers": 4, "n_heads": 4, "d_mlp": 512, "seq": 26},
    }
    assert c.TRAINING == {
        "induction": {"batch": 32, "seq": 63, "lr": 1e-3, "warmup_steps": 100, "total_steps": 4000, "data_seed": 0, "param_seed": 0},
        "two_hop": {"batch": 64, "seq": 26, "lr": 1e-3, "warmup_steps": 500, "total_steps": 20000, "data_seed": 0, "param_seed": 0},
    }
    assert (c.FIT_SEED, c.EVAL_SEED, c.BROKEN_SEED, c.PATCH_SEED, c.SOURCE_SEED) == (1, 2, 3, 4, 5)
    assert (c.EVAL_WINDOWS, c.PROBE_WINDOWS, c.FIT_WINDOWS) == (256, 8, 64)
    assert (c.PATCH_PERIOD, c.PATCH_TARGETS, c.PATCH_LAG) == (20, (13, 16, 19), 19)
    assert c.BARS == {"behaviour": 0.9, "control_ratio": 2.0, "induction_attention": 0.5, "twin_attention_ratio": 2.0,
                      "ablation_fraction": 0.5, "swap_fraction": 0.5, "patch_followed": 0.9, "patch_after_read": 0.1,
                      "control_unchanged": 0.95}
    assert c.MIN_ABOVE_CHANCE == 0.1
    assert c.TWO_HOP_SITES == {"query": (24, 25), "fact positions": tuple(range(24)), "every position": tuple(range(26))}


def test_schedules():
    c = _common()
    induction = c.checkpoint_steps("induction")
    assert induction[:4] == [0, 50, 100, 150] and 1500 in induction and 1550 not in induction
    assert induction[-3:] == [3500, 3750, 4000] and induction == sorted(set(induction)) and len(induction) == 41
    assert c.checkpoint_steps("two_hop") == list(range(0, 20_001, 1000))
    with pytest.raises(ValueError):
        c.checkpoint_steps("modular_addition")


def test_fraction_removed_and_the_last_checkpoint_at_chance():
    c = _common()
    assert c.fraction_removed(1.0, 0.25, 0.0) == pytest.approx(0.75)
    assert c.fraction_removed(0.9, 0.9, 0.1) == 0.0 and type(c.fraction_removed(0.9, 0.5, 0.1)) is float
    assert c.fraction_removed(0.9, 0.1, 0.1) == pytest.approx(1.0)
    assert c.fraction_removed(0.9, 0.05, 0.1) == pytest.approx(0.85 / 0.8), "below chance removes more than all of it"
    for too_close in ((0.15, 0.0, 0.1), (0.199, 0.1, 0.1), (0.05, 0.0, 0.1)):
        with pytest.raises(ValueError):
            c.fraction_removed(*too_close)
    steps = [0, 50, 100, 150, 200]
    assert c.last_at_chance(steps, [0.01, 0.03, 0.02, 0.5, 0.03], 1 / 64) == 200, "the last, even after a rise"
    assert c.last_at_chance(steps, [0.01, 0.031, 0.5, 0.9, 1.0], 1 / 64) == 50, "at most twice chance"
    assert c.last_at_chance(steps, [0.01, 0.0313, 0.5, 0.9, 1.0], 1 / 64) == 0
    assert c.last_at_chance([0, 50], [0.0, 2 / 64], 1 / 64) == 50, "exactly twice chance counts"
    with pytest.raises(ValueError):
        c.last_at_chance(steps, [0.5] * 5, 1 / 64)
    with pytest.raises(ValueError):
        c.last_at_chance(steps, [0.01] * 4, 1 / 64)


def test_the_gates_windows():
    c = _common()
    ind, two = InductionTask(), TwoHopTask()
    for task in (ind, two):
        assert np.array_equal(c.eval_windows(task), task.batch_for_step(2, 0, 256))
        assert np.array_equal(c.probe(task), c.eval_windows(task)[:8]), "the probe is the first eight evaluation windows"
    assert np.array_equal(c.fit_windows(two), two.batch_for_step(1, 0, 64)[:, :-1]), "what the model reads"
    rng = np.random.default_rng([3, 0])
    assert np.array_equal(c.broken_windows(two), np.stack([two.window(rng, broken=True) for _ in range(256)]))
    rng = np.random.default_rng([4, 0])
    patch = c.induction_patch_windows(ind)
    assert np.array_equal(patch, np.stack([ind.draw_window(rng, period=20) for _ in range(256)]))
    for row in patch[:4]:
        for j in c.PATCH_TARGETS:
            q = j + c.PATCH_LAG
            assert np.flatnonzero(row[:q] == row[q]).tolist() == [j - 1], "the query's token occurred once, just before the target"
            assert np.flatnonzero(row[: q + 1] == row[j]).tolist() == [j], "the target's token appears once up to the query"


def test_configs_build_the_settings():
    pytest.importorskip("jax")
    c = _common()
    for name in ("induction", "two_hop"):
        model, train = c.configs(name)
        assert {k: getattr(model, k) for k in c.MODELS[name]} == c.MODELS[name]
        assert {k: getattr(train, k) for k in c.TRAINING[name]} == c.TRAINING[name]
    with pytest.raises(ValueError):
        c.configs("modular_addition")
