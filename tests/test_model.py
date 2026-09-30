"""Work package 3: the reference model.

Spec: docs/specs/phase-1.md, "The reference model". Needs the jax lane:
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.oracle.mode_equality import check_mode_equality, compare_logits  # noqa: E402
from nablaframe.trace.registry import tap_names, tap_spec  # noqa: E402

TINY = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=8)


def _tokens(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, TINY.vocab, size=(3, TINY.seq)).astype(np.int32)


def _count(params) -> int:
    return sum(int(np.prod(np.shape(leaf))) for leaf in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize("size,vocab,expected", [
    ("S", 4096, 4_273_664), ("S", 8192, 5_322_240), ("M", 4096, 14_838_784), ("M", 8192, 16_935_936)])
def test_param_count_matches_levanter(size, vocab, expected):
    config = GPT2Config.from_size(size, vocab)
    assert config.n_params() == expected
    if size == "S":  # building M too would only slow the suite down
        assert _count(init_params(config, seed=0)) == expected
    assert _count(init_params(TINY, seed=0)) == TINY.n_params()


def test_tap_names_match_registry():
    model = GPT2Model(TINY)
    params = init_params(TINY, seed=0)
    _, taps = model.instrumented(params, _tokens(), tap_names(TINY.n_layers))
    assert set(taps) == set(tap_names(TINY.n_layers))
    _, some = model.instrumented(params, _tokens(), ["hook_embed", "ln_final.hook_normalized"])
    assert set(some) == {"hook_embed", "ln_final.hook_normalized"}
    with pytest.raises(KeyError):
        model.instrumented(params, _tokens(), ["blocks.9.hook_resid_pre"])


def test_tap_shapes_match_tapspecs():
    model = GPT2Model(TINY)
    logits, taps = model.instrumented(init_params(TINY, seed=0), _tokens(), tap_names(TINY.n_layers))
    assert np.asarray(logits).shape == (3, TINY.seq, TINY.vocab) and np.asarray(logits).dtype == np.float32
    for name, tensor in taps.items():
        spec = tap_spec(name, batch=3, seq=TINY.seq, d_model=TINY.d_model, n_heads=TINY.n_heads,
                        d_head=TINY.d_model // TINY.n_heads)
        assert np.asarray(tensor).shape == tuple(axis.size for axis in spec.axes), name
        assert np.asarray(tensor).dtype == np.float32, name


def test_fast_and_instrumented_logits_bitwise():
    model = GPT2Model(TINY)
    for seed in (0, 1, 2):
        result = check_mode_equality(model, init_params(TINY, seed=seed), _tokens(seed),
                                     tap_names(TINY.n_layers), "bitwise")
        assert result.passed, result.reason


def test_mutating_tap_breaks_oracle_on_real_model():
    model = GPT2Model(TINY)
    params = init_params(TINY, seed=0)
    tap = "blocks.0.hook_resid_post"
    fast = model.fast(params, _tokens())
    nudged, recorded = model.instrumented(params, _tokens(), [tap], interventions={tap: lambda x: x + 1e-3})
    assert not compare_logits(fast, nudged, "bitwise").passed
    _, clean = model.instrumented(params, _tokens(), [tap])
    assert np.array_equal(np.asarray(recorded[tap]), np.asarray(clean[tap]) + np.float32(1e-3)), \
        "a tap records the value after its intervention, which is what flows downstream"
