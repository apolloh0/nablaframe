"""Phase 2 delta 3: the summed Jacobian of D17, exactly as anthropics/jacobian-lens computes it.

Reference: jlens/fitting.py at 581d398. One mask, positions 16 to T - 2, places the one-hot
cotangents and selects the source positions averaged; each sequence's J is averaged over its
valid positions, then sequences are averaged with equal weight. Needs the jax lane.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from conftest import REPO_ROOT  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.views.jlens import (  # noqa: E402
    SKIP_FIRST, block_map, code_identity, jacobian_key, load_jacobian, save_jacobian, summed_jacobian, valid_positions,
)
from phase2_fixtures import _perturbed  # noqa: E402

CFG = GPT2Config(vocab=32, d_model=16, n_layers=2, n_heads=2, d_mlp=64, seq=24)  # long enough to leave valid positions
D, T = 6, 24
A = np.random.default_rng(0).standard_normal((D, D)).astype(np.float32)
B = 2.0 * np.random.default_rng(1).standard_normal((D, D)).astype(np.float32)


def linear(x):
    """y[t] = A x[t] + B x[t-1]."""
    shifted = jnp.concatenate([jnp.zeros((1, D), x.dtype), x[:-1]], axis=0)
    return x @ A.T + shifted @ B.T


def _model():
    return GPT2Model(CFG), _perturbed(init_params(CFG, seed=0))


def _resid(model, params, tokens, tap):
    _, taps = model.instrumented(params, tokens, [tap])
    value = taps[tap]
    return np.asarray(getattr(value, "array", value))


def _tokens(n=3):
    return np.random.default_rng(4).integers(0, CFG.vocab, size=(n, CFG.seq)).astype(np.int32)


def test_valid_positions_match_the_reference():
    assert SKIP_FIRST == 16
    assert valid_positions(256).tolist() == list(range(16, 255)), "positions 16 to T - 2, as valid_position_mask"
    assert valid_positions(18).tolist() == [16]
    with pytest.raises(ValueError):
        valid_positions(17)  # no valid position left


def test_summed_jacobian_closed_form():
    xs = np.random.default_rng(2).standard_normal((8, T, D)).astype(np.float32)
    J = summed_jacobian(linear, xs)
    n = T - 1 - SKIP_FIRST  # 7 valid positions; the last valid source has no later valid target
    assert J.shape == (D, D) and J.dtype == np.float32
    assert np.allclose(J, A + B * (n - 1) / n, atol=1e-5), "a cotangent at every valid target sums each later effect"
    assert np.array_equal(J, summed_jacobian(linear, xs)), "deterministic"


def test_block_map_reproduces_the_forward():
    model, params = _model()
    tokens = _tokens()
    after0 = _resid(model, params, tokens, "blocks.0.hook_resid_post")
    final = _resid(model, params, tokens, "blocks.1.hook_resid_post")
    f = block_map(model, params, layer=0, n_layers=CFG.n_layers)
    assert np.allclose(np.stack([np.asarray(f(h)) for h in after0]), final, atol=1e-5)
    identity = block_map(model, params, layer=1, n_layers=CFG.n_layers)
    assert np.array_equal(np.asarray(identity(final[0])), final[0]), "the last block's output maps to itself"


def test_summed_jacobian_matches_the_full_jacobian():
    model, params = _model()
    xs = _resid(model, params, _tokens(), "blocks.0.hook_resid_post")
    f = block_map(model, params, layer=0, n_layers=CFG.n_layers)
    valid = valid_positions(CFG.seq)
    expected = []
    for x in xs:
        full = np.asarray(jax.jacrev(f)(jnp.asarray(x)), np.float64)  # (out pos, out dim, in pos, in dim)
        expected.append(full[valid][:, :, valid].sum(axis=0).mean(axis=1))  # sum over targets, mean over sources
    assert np.allclose(summed_jacobian(f, xs), np.mean(expected, axis=0), atol=1e-5)


def test_jacobian_cache(tmp_path):
    fields = dict(checkpoint_sha256="a" * 64, fit_sha256="b" * 64, layer=1, code="c" * 40)
    key = jacobian_key(**fields)
    assert len(key) == 64
    for name, other in [("checkpoint_sha256", "e" * 64), ("fit_sha256", "e" * 64), ("layer", 2), ("code", "e" * 40)]:
        assert jacobian_key(**{**fields, name: other}) != key, name
    J = np.random.default_rng(3).standard_normal((16, 16)).astype(np.float32)
    save_jacobian(tmp_path, key, J)
    loaded = load_jacobian(tmp_path, key)
    assert loaded.dtype == np.float32 and np.array_equal(loaded, J)
    save_jacobian(tmp_path, key, J)  # the same content again is fine
    with pytest.raises(ValueError):
        save_jacobian(tmp_path, key, J + 1)  # a key never changes meaning
    assert load_jacobian(tmp_path, "d" * 64) is None
    blob = subprocess.run(["git", "hash-object", str(REPO_ROOT / "src/nablaframe/views/jlens.py")],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert code_identity() == blob, "D18: a cache's code identity is its module's git blob hash, not the repo's commit"
