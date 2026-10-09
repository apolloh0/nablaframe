"""Phase 4 package 3: the J-lens coordinate swap (D33) and zeroing one head, held to D25.

Spec: docs/specs/phase-4.md, D33. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit, spec_hash  # noqa: E402
from nablaframe.trace.schema import tensor_sha256  # noqa: E402
from interventions_fixtures import MODEL, PARAMS, PARAMS_AT, STORE, TOKENS, base_logits  # noqa: E402
from phase2_fixtures import TINY  # noqa: E402

J = np.random.default_rng(20).standard_normal((TINY.d_model, TINY.d_model)).astype(np.float32)
SMALL = np.eye(8, dtype=np.float32)
JH = tensor_sha256(J)
STORE_J = {**STORE, JH: J, tensor_sha256(SMALL): SMALL}
RESID = ("blocks.0.hook_resid_pre", "blocks.0.hook_resid_mid", "blocks.1.hook_resid_post")


def _run(*edits, taps=()):
    logits, recorded = run_intervened(MODEL, PARAMS, TOKENS, InterventionSpec(tuple(edits)), n_layers=TINY.n_layers,
                                      resolve=STORE_J.__getitem__, params_at=PARAMS_AT.__getitem__, taps=taps)
    return np.asarray(logits), {k: np.asarray(getattr(v, "array", v)) for k, v in recorded.items()}


def _tap(tap):
    v = MODEL.instrumented(PARAMS, TOKENS, [tap])[1][tap]
    return np.asarray(getattr(v, "array", v))


def test_jswap_and_head_arguments():
    edit(RESID[0], "all", "jswap", J=JH, a=3, b=5)
    for bad in (dict(J=JH, a=3), dict(J=JH, a=-1, b=5), dict(J=[0.0], a=3, b=5), dict(J=JH, a=3, b=1.5)):
        with pytest.raises(ValueError):
            edit(RESID[0], "all", "jswap", **bad)
    edit("blocks.1.attn.hook_z", [2], "zero", head=1)
    for op, args in (("zero", dict(head=-1)), ("zero", dict(head=1, extra=2)), ("mean", dict(reference="a" * 64, head=0))):
        with pytest.raises(ValueError):
            edit("blocks.1.attn.hook_z", "all", op, **args)
    assert spec_hash(InterventionSpec((edit("blocks.1.attn.hook_z", "all", "zero"),))) != \
        spec_hash(InterventionSpec((edit("blocks.1.attn.hook_z", "all", "zero", head=0),)))


def test_jswap_of_a_token_with_itself_is_exact():
    base = base_logits()
    for tap in RESID:
        assert np.array_equal(_run(edit(tap, "all", "jswap", J=JH, a=4, b=4))[0], base), tap
    assert np.array_equal(_run(edit("blocks.1.attn.hook_z", [], "zero", head=1))[0], base), "a head zeroed nowhere"


def test_jswap_exchanges_the_coordinates():
    tap, positions = "blocks.0.hook_resid_post", [2, 5]
    _, recorded = _run(edit(tap, positions, "jswap", J=JH, a=3, b=9), taps=[tap])
    before, after = _tap(tap).astype(np.float64), recorded[tap].astype(np.float64)
    U = J.T.astype(np.float64) @ np.asarray(PARAMS.wte, np.float64)[[3, 9]].T  # columns u_a, u_b
    coef = np.linalg.lstsq(U, before.reshape(-1, TINY.d_model).T, rcond=None)[0].T.reshape(*before.shape[:2], 2)
    expected = before + (coef[..., 1] - coef[..., 0])[..., None] * (U[:, 0] - U[:, 1])
    assert np.allclose(after[:, positions], expected[:, positions], rtol=1e-4, atol=1e-4)
    swapped = np.linalg.lstsq(U, after[:, positions].reshape(-1, TINY.d_model).T, rcond=None)[0].T
    assert np.allclose(swapped, coef[:, positions].reshape(-1, 2)[:, ::-1], rtol=1e-3, atol=1e-3), "the coordinates trade places"
    others = [p for p in range(TOKENS.shape[1]) if p not in positions]
    assert np.array_equal(recorded[tap][:, others], _tap(tap)[:, others]), "other positions untouched"


def test_jswap_and_head_zero_where_they_apply():
    for e in (edit("blocks.1.attn.hook_z", "all", "jswap", J=JH, a=3, b=5),
              edit(RESID[0], "all", "jswap", J=JH, a=3, b=TINY.vocab),
              edit(RESID[0], "all", "jswap", J=tensor_sha256(SMALL), a=3, b=5),
              edit(RESID[0], "all", "zero", head=0),
              edit("blocks.1.attn.hook_z", "all", "zero", head=TINY.n_heads)):
        with pytest.raises(ValueError):
            _run(e)
    z = "blocks.1.attn.hook_z"
    _, recorded = _run(edit(z, [3], "zero", head=1), taps=[z])
    before = _tap(z)
    assert (recorded[z][:, 3, 1] == 0).all(), "head 1 zeroed at position 3"
    assert np.array_equal(recorded[z][:, 3, 0], before[:, 3, 0]), "head 0 untouched"
    assert np.array_equal(np.delete(recorded[z], 3, axis=1), np.delete(before, 3, axis=1)), "other positions untouched"
