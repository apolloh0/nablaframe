"""Shared Phase 2 test helpers: a tiny traced run. Not a test module."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np

import nablaframe.trace.registry as registry
from conftest import make_manifest
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params
from nablaframe.trace.engine import TraceRun
from nablaframe.trace.registry import tap_names
from nablaframe.train.checkpoint import save_params

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)


def probe_tokens(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, TINY.vocab, size=(2, TINY.seq)).astype(np.int32)


def _perturbed(params):
    """Nudge every parameter, so layer-norm weights aren't exactly 1 and biases aren't exactly 0.

    At initialization they are, and a view that ignored them would pass unnoticed.
    """
    rng = np.random.default_rng(11)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return jax.tree_util.tree_unflatten(
        treedef, [leaf + 0.05 * rng.standard_normal(np.shape(leaf)).astype(np.float32) for leaf in leaves])


def traced_run(tmp_path, *, with_logits: bool = True, param_seed: int = 0) -> SimpleNamespace:
    """Record every canonical tap, and the logits unless asked not to, at step 0."""
    manifest = make_manifest()
    params = _perturbed(init_params(TINY, seed=param_seed))
    checkpoints = tmp_path / "checkpoints"
    save_params(params, checkpoints, step=0, run_id=manifest.run_id)
    probe = probe_tokens()
    taps = tap_names(TINY.n_layers) + ((registry.LOGITS,) if with_logits else ())
    with TraceRun.open(tmp_path / "trace", manifest, GPT2Model(TINY), checkpoint_dir=checkpoints,
                       probe=probe, like=params) as run:
        events = run.record(step=0, taps=taps)
    return SimpleNamespace(root=tmp_path / "trace", params=params, probe=probe, manifest=manifest,
                           checkpoints=checkpoints, events=events, model=GPT2Model(TINY))
