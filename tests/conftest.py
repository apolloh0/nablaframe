"""Shared fixtures for the Phase 0 tests.

These tests use numpy only, so they run in any environment, including cloud
threads. Nothing here imports JAX, PyTorch or TransformerLens.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from nablaframe.oracle.tolerance import ToleranceRecord
from nablaframe.trace.schema import RunManifest, Seeds, TraceEvent

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"


def make_manifest(**overrides) -> RunManifest:
    fields = dict(
        model_config={"n_layers": 2, "d_model": 128, "n_heads": 4, "vocab": 4096},
        optimizer={"name": "adamw", "lr": 3e-4, "schedule": "cosine", "warmup": 100},
        seeds=Seeds(param_init=0, data_order=1, dropout=2),
        corpus_id="tinystories-v2",
        corpus_sha256="a" * 64,
        tokenizer_id="words-4096",
        tokenizer_sha256="b" * 64,
        git_commit="c" * 40,
        git_dirty=False,
        env_lock_sha256="d" * 64,
        backend="cpu",
        xla_flags="",
        checkpoint_schedule=(0, 100, 1000),
        created_at="2026-09-27T12:00:00Z",
    )
    fields.update(overrides)
    return RunManifest(**fields)


def make_event(**overrides) -> TraceEvent:
    fields = dict(
        run_id="e" * 64,
        step=100,
        kind="forward",
        tap="blocks.0.hook_resid_post",
        batch_hash="f" * 64,
        mode="instrumented",
    )
    fields.update(overrides)
    return TraceEvent(**fields)


def pinned_tolerance(value: float = 1e-6) -> ToleranceRecord:
    return ToleranceRecord(
        value=value,
        source_report="reports/phase0/g06_mode_probe.json",
        backend="cpu",
        dtype="float32",
        measured_max_abs_diff=0.0,
    )


Tap = Callable[[str, np.ndarray], np.ndarray]


def identity_tap(name: str, x: np.ndarray) -> np.ndarray:
    return x


class NumpyToy:
    """A 2-layer residual toy with hook points named like the registry. Test-only.

    ``fast`` and ``instrumented`` run the same float32 operations in the same
    order, so a correct instrumented path is bitwise equal to the fast one. The
    ``tap`` is applied at every hook point in instrumented mode only, which is how
    the tests plant defects.
    """

    HOOKS: tuple[str, ...] = (
        "hook_embed",
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_post",
        "blocks.1.hook_resid_pre",
        "blocks.1.hook_resid_post",
        "ln_final.hook_normalized",
    )

    def __init__(self, tap: Tap = identity_tap) -> None:
        self.tap = tap

    @staticmethod
    def init(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "E": rng.standard_normal((16, 8)).astype(np.float32),
            "W0": (0.5 * rng.standard_normal((8, 8))).astype(np.float32),
            "W1": (0.5 * rng.standard_normal((8, 8))).astype(np.float32),
            "U": rng.standard_normal((8, 16)).astype(np.float32),
        }

    @staticmethod
    def tokens(seed: int) -> np.ndarray:
        return np.random.default_rng(seed + 1000).integers(0, 16, size=(2, 5))

    def _forward(self, params, tokens, hook: Callable[[str, np.ndarray], np.ndarray]) -> np.ndarray:
        x = params["E"][tokens]
        x = hook("hook_embed", x)
        for layer in (0, 1):
            x = hook(f"blocks.{layer}.hook_resid_pre", x)
            x = x + np.tanh(x @ params[f"W{layer}"])
            x = hook(f"blocks.{layer}.hook_resid_post", x)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        x = (x - mu) / np.sqrt(var + np.float32(1e-5))
        x = hook("ln_final.hook_normalized", x)
        return x @ params["U"]

    def fast(self, params, tokens) -> np.ndarray:
        return self._forward(params, tokens, lambda name, x: x)

    def instrumented(self, params, tokens, taps: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        unknown = set(taps) - set(self.HOOKS)
        if unknown:
            raise KeyError(sorted(unknown))
        recorded: dict[str, np.ndarray] = {}

        def hook(name: str, x: np.ndarray) -> np.ndarray:
            x = self.tap(name, x)
            if name in taps:
                recorded[name] = x.copy()
            return x

        return self._forward(params, tokens, hook), recorded


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
