"""Work package 4: trainer, checkpoints and heartbeat.

Spec: docs/specs/phase-1.md, "Trainer and checkpoints". Needs the jax lane.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import nablaframe.train.checkpoint as checkpoint_mod  # noqa: E402
from nablaframe.data.corpus import batch_for_step  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config  # noqa: E402
from nablaframe.train.checkpoint import load_checkpoint, save_checkpoint, state_hash  # noqa: E402
from nablaframe.train.heartbeat import Heartbeat  # noqa: E402
from nablaframe.train.loop import TrainConfig, init_state, make_train_step, run_until  # noqa: E402

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)
CONFIG = TrainConfig(batch=4, seq=8, lr=1e-2, warmup_steps=2, total_steps=10, data_seed=1, param_seed=0)
STREAM = np.random.default_rng(0).integers(0, TINY.vocab, size=4096).astype(np.uint16)
RUN_ID = "r" * 64


def _fresh():
    return init_state(TINY, CONFIG)


def test_train_step_is_deterministic():
    a = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=5)
    b = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=5)
    assert a.step == 5 and state_hash(a) == state_hash(b)
    other = dataclasses.replace(CONFIG, data_seed=2)
    assert state_hash(run_until(init_state(TINY, other), STREAM, TINY, other, until_step=5)) != state_hash(a)


def test_checkpoint_roundtrip_is_bitwise(tmp_path):
    state = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=3)
    back = load_checkpoint(save_checkpoint(state, tmp_path, run_id=RUN_ID), like=_fresh())
    assert back.step == 3 and state_hash(back) == state_hash(state)
    for x, y in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(back)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_resume_matches_uninterrupted(tmp_path):
    full = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=8)
    half = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=4)
    resumed = run_until(load_checkpoint(save_checkpoint(half, tmp_path, run_id=RUN_ID), like=_fresh()),
                        STREAM, TINY, CONFIG, until_step=8)
    assert state_hash(resumed) == state_hash(full)


def test_checkpoint_writes_are_atomic(tmp_path, monkeypatch):
    state = run_until(_fresh(), STREAM, TINY, CONFIG, until_step=2)

    def dies_mid_write(tensors, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("killed while writing the checkpoint")

    monkeypatch.setattr(checkpoint_mod, "save_file", dies_mid_write)
    with pytest.raises(OSError):
        save_checkpoint(state, tmp_path, run_id=RUN_ID)
    assert not list(tmp_path.glob("step-*.safetensors")) and not list(tmp_path.glob("step-*.json"))


def test_step_n_holds_n_updates(tmp_path):
    step_fn = make_train_step(TINY, CONFIG)
    manual = _fresh()
    for k in range(3):
        manual, _loss = step_fn(manual, batch_for_step(STREAM, CONFIG.data_seed, k, CONFIG.batch, CONFIG.seq))
    run_until(_fresh(), STREAM, TINY, CONFIG, until_step=3, checkpoint_dir=tmp_path, save_every=3)
    loaded = load_checkpoint(tmp_path / "step-3.safetensors", like=_fresh())
    assert manual.step == loaded.step == 3 and state_hash(loaded) == state_hash(manual)


def test_heartbeat_prints_on_interval():
    lines, now = [], [0.0]
    heartbeat = Heartbeat(total_steps=100, interval_s=30.0, clock=lambda: now[0], write=lines.append)
    for step in range(1, 101):
        now[0] = float(step)
        heartbeat.tick(step=step, loss=2.0, tokens=256)
    assert len(lines) == 3, lines
    assert lines[0].startswith("step 30/100 | loss 2.000 | 256 tokens/s | about ") and lines[0].endswith(" left")
