"""Work package 5: the trace engine.

Spec: docs/specs/phase-1.md, "Trace engine". Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from conftest import make_manifest  # noqa: E402
from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.trace.engine import OracleFailure, TraceRun  # noqa: E402
from nablaframe.trace.registry import tap_names  # noqa: E402
from nablaframe.trace.schema import tensor_sha256  # noqa: E402
from nablaframe.train.checkpoint import save_params  # noqa: E402

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)
PROBE = np.random.default_rng(7).integers(0, TINY.vocab, size=(2, TINY.seq)).astype(np.int32)
TAPS = tap_names(TINY.n_layers)


class Broken(GPT2Model):
    """Instrumentation that changes the computation: the oracle must stop it."""

    def instrumented(self, params, tokens, taps, interventions=None):
        return super().instrumented(params, tokens, taps,
                                    interventions={"blocks.0.hook_resid_post": lambda x: x + 1e-3})


def _open(tmp_path, model=None, manifest=None):
    params = init_params(TINY, seed=0)
    checkpoints = tmp_path / "checkpoints"
    manifest = manifest or make_manifest()
    if not (checkpoints / "step-0.safetensors").exists():
        save_params(params, checkpoints, step=0, run_id=manifest.run_id)
    return TraceRun.open(tmp_path / "trace", manifest, model or GPT2Model(TINY),
                         checkpoint_dir=checkpoints, probe=PROBE, like=params)


def test_manifest_is_written_once(tmp_path):
    with _open(tmp_path):
        pass
    written = (tmp_path / "trace" / "manifest.json").read_bytes()
    with _open(tmp_path):
        pass
    assert (tmp_path / "trace" / "manifest.json").read_bytes() == written
    with pytest.raises(ValueError):
        _open(tmp_path, manifest=make_manifest(backend="other"))


def test_oracle_failure_writes_no_events(tmp_path):
    with _open(tmp_path, model=Broken(TINY)) as run:
        with pytest.raises(OracleFailure):
            run.record(step=0, taps=TAPS)
    assert (tmp_path / "trace" / "events.jsonl").read_bytes() == b""


def test_retrace_writes_nothing(tmp_path):
    with _open(tmp_path) as run:
        assert len(run.record(step=0, taps=TAPS)) == len(TAPS)
        log = (tmp_path / "trace" / "events.jsonl").read_bytes()
        assert run.record(step=0, taps=TAPS) == ()
    assert (tmp_path / "trace" / "events.jsonl").read_bytes() == log


def test_rematerialized_tensor_matches_event(tmp_path):
    with _open(tmp_path) as run:
        for event in run.record(step=0, taps=TAPS):
            assert tensor_sha256(run.rematerialize(event)) == event.sha256
            assert run.verify(event)


def test_probe_batch_hash_is_stable(tmp_path):
    manifest = make_manifest()
    with _open(tmp_path, manifest=manifest) as run:
        events = run.record(step=0, taps=TAPS)
    assert {e.batch_hash for e in events} == {batch_hash(PROBE)}
    assert all(e.run_id == manifest.run_id and e.step == 0 and e.kind == "forward" and e.mode == "instrumented"
               for e in events)
