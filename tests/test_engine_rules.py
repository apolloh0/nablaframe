"""Phase 1 delta 3: the engine refuses conflicting retraces and foreign checkpoints,
and the store's writer lock is public.

Spec: docs/deltas/phase-1-delta-3.md. Needs the jax lane. New names are looked up
inside each test, so a missing one fails only that test, not the whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import nablaframe.trace.engine as engine  # noqa: E402
from conftest import make_event, make_manifest  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.trace.engine import TraceRun  # noqa: E402
from nablaframe.trace.store import StoreError, TraceStore  # noqa: E402
from nablaframe.train.checkpoint import save_params  # noqa: E402

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)
PROBE = np.random.default_rng(7).integers(0, TINY.vocab, size=(2, TINY.seq)).astype(np.int32)
MANIFEST = make_manifest()


def _save(checkpoints, step, seed, run_id):
    for old in checkpoints.glob(f"step-{step}.*"):
        old.unlink()
    save_params(init_params(TINY, seed=seed), checkpoints, step=step, run_id=run_id)


def _open(tmp_path):
    return TraceRun.open(tmp_path / "trace", MANIFEST, GPT2Model(TINY), checkpoint_dir=tmp_path / "checkpoints",
                         probe=PROBE, like=init_params(TINY, seed=0))


def test_conflicting_retrace_raises_and_writes_nothing(tmp_path):
    not_reproducible = engine.NotReproducible
    checkpoints = tmp_path / "checkpoints"
    _save(checkpoints, 0, seed=0, run_id=MANIFEST.run_id)
    with _open(tmp_path) as run:
        assert len(run.record(step=0, taps=["hook_embed"])) == 1
        log = (tmp_path / "trace" / "events.jsonl").read_bytes()

        _save(checkpoints, 0, seed=1, run_id=MANIFEST.run_id)  # step 0 now computes something else
        with pytest.raises(not_reproducible):
            run.record(step=0, taps=["ln_final.hook_normalized", "hook_embed"])
        assert (tmp_path / "trace" / "events.jsonl").read_bytes() == log, \
            "no event for any tap, not even the one that wasn't recorded yet"

        _save(checkpoints, 0, seed=0, run_id=MANIFEST.run_id)  # back to what was recorded
        assert run.record(step=0, taps=["hook_embed"]) == ()


def test_checkpoint_from_another_run_is_refused(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    _save(checkpoints, 0, seed=0, run_id="f" * 64)
    with _open(tmp_path) as run:
        with pytest.raises(ValueError):
            run.record(step=0, taps=["hook_embed"])
        assert (tmp_path / "trace" / "events.jsonl").read_bytes() == b""

        _save(checkpoints, 1, seed=0, run_id=MANIFEST.run_id)
        event = run.record(step=1, taps=["hook_embed"])[0]
        _save(checkpoints, 1, seed=0, run_id="f" * 64)  # same weights, but provenance from another run
        with pytest.raises(ValueError):
            run.rematerialize(event)

        _save(checkpoints, 2, seed=0, run_id=None)
        with pytest.raises(ValueError):
            run.record(step=2, taps=["hook_embed"])


def test_store_lock_is_public_and_taken_at_open(tmp_path):
    store = TraceStore(tmp_path / "store")
    store.acquire()
    store.acquire()  # holding it already is fine
    other = TraceStore(tmp_path / "store")
    with pytest.raises(StoreError):
        other.acquire()
    store.close()
    other.acquire()
    other.close()
    with pytest.raises(StoreError):
        TraceStore(tmp_path / "store", mode="r").acquire()

    _save(tmp_path / "checkpoints", 0, seed=0, run_id=MANIFEST.run_id)
    with _open(tmp_path):
        with pytest.raises(StoreError):
            TraceStore(tmp_path / "trace").acquire()  # the run became the one writer at open
