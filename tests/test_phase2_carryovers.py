"""Phase 2 work packages 0 and 1: G1.1 pins its dataset revision; the trace gains logits (D12).

Spec: docs/specs/phase-2.md, "Carried over from Phase 1" and D12. Needs the jax lane.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import nablaframe.trace.registry as registry  # noqa: E402
from conftest import REPO_ROOT  # noqa: E402
from nablaframe.trace.engine import TraceRun  # noqa: E402
from nablaframe.trace.registry import tap_names, tap_spec  # noqa: E402
from nablaframe.trace.schema import Axis, TapSpec  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def _load_gate(name: str):
    gates = REPO_ROOT / "gates" / "phase1"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location(f"phase1_{name}", gates / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g11_downloads_by_revision():
    g11 = _load_gate("g11_tokenizer")
    url = g11.download_url("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", REVISION)
    assert url == f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/TinyStoriesV2-GPT4-valid.txt"
    assert "/resolve/main/" not in url


def test_logits_tap_is_registered():
    logits = registry.LOGITS
    assert logits == "logits"
    assert logits not in tap_names(TINY.n_layers), "tap_names stays TransformerLens-only"
    spec = tap_spec(logits, batch=2, seq=8, d_model=16, n_heads=2, d_head=8, vocab=32)
    assert [(a.name, a.size) for a in spec.axes] == [("Batch", 2), ("Pos", 8), ("Vocab", 32)]
    assert spec.site == "unembed"
    TapSpec("logits", (Axis("Batch", 2),), "float32", "unembed")  # the new site is valid
    with pytest.raises(ValueError):
        tap_spec(logits, batch=2, seq=8, d_model=16, n_heads=2, d_head=8)  # logits need a vocabulary


def test_logits_event_matches_model_output(tmp_path):
    run = traced_run(tmp_path)
    [event] = [e for e in run.events if e.tap == registry.LOGITS]
    from nablaframe.trace.store import TraceStore
    stored = TraceStore(run.root, mode="r").load_tensor(event)
    assert np.array_equal(stored, np.asarray(run.model.fast(run.params, run.probe))), "the model's own output, bit for bit"
    with TraceRun.open(run.root, run.manifest, run.model, checkpoint_dir=run.checkpoints, probe=run.probe,
                       like=run.params) as reopened:
        assert reopened.verify(event)


def test_retrace_adds_only_logits(tmp_path):
    run = traced_run(tmp_path, with_logits=False)
    assert len(run.events) == len(tap_names(TINY.n_layers))
    log = (run.root / "events.jsonl").read_bytes()
    with TraceRun.open(run.root, run.manifest, run.model, checkpoint_dir=run.checkpoints, probe=run.probe,
                       like=run.params) as reopened:
        added = reopened.record(step=0, taps=tap_names(TINY.n_layers) + (registry.LOGITS,))
    assert [e.tap for e in added] == [registry.LOGITS]
    assert (run.root / "events.jsonl").read_bytes().startswith(log), "existing events untouched"
