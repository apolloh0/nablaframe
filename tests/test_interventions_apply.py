"""Phase 3 package 2: applying edits, held to D25's exactness identities bit for bit (D23, D25, D26).

Spec: docs/specs/phase-3.md. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit  # noqa: E402
from nablaframe.trace.schema import tensor_sha256  # noqa: E402
from interventions_fixtures import (HEADS, MODEL, OTHER, PARAMS, PARAMS_AT, REFERENCE, RESID, SOURCE, STORE,  # noqa: E402
                                    TOKENS, base_logits)
from phase2_fixtures import TINY  # noqa: E402

TAPS = ["hook_embed", "blocks.0.hook_resid_post", "blocks.1.attn.hook_z", "blocks.0.hook_attn_out",
        "blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out", "blocks.1.hook_resid_mid"]


def _run(*edits, params=PARAMS, tokens=TOKENS, taps=(), store=STORE):
    logits, recorded = run_intervened(MODEL, params, tokens, InterventionSpec(tuple(edits)), n_layers=TINY.n_layers,
                                      resolve=store.__getitem__, params_at=PARAMS_AT.__getitem__, taps=taps)
    return np.asarray(logits), {k: np.asarray(getattr(v, "array", v)) for k, v in recorded.items()}


def test_null_edit_is_exact():
    base = base_logits()
    for tap in TAPS:
        assert np.array_equal(_run(edit(tap, [], "zero"))[0], base), f"an edit at no positions changes nothing: {tap}"
    assert np.array_equal(_run()[0], base), "an empty spec is the unintervened pass"


def test_zero_alpha_steering_is_exact():
    base = base_logits()
    for tap, vector in [("blocks.0.hook_resid_post", RESID), ("blocks.1.attn.hook_z", HEADS), ("hook_embed", RESID)]:
        spec = edit(tap, "all", "add", vector=tensor_sha256(vector), alpha=0.0)
        assert np.array_equal(_run(spec)[0], base), tap
    moved = _run(edit("blocks.0.hook_resid_post", "all", "add", vector=tensor_sha256(RESID), alpha=1.0))[0]
    assert not np.array_equal(moved, base), "steering with alpha 1 does something"


def test_self_swap_is_exact():
    base = base_logits()
    for component in ("embeddings", "ln_final", "blocks.0", "blocks.1.attn", "blocks.1.mlp"):
        assert np.array_equal(_run(edit(component, "all", "swap", step=0))[0], base), component


def test_full_patch_reproduces_the_source():
    last = f"blocks.{TINY.n_layers - 1}.hook_resid_post"
    source_logits = np.asarray(MODEL.instrumented(PARAMS, SOURCE, [last])[0])
    assert np.array_equal(_run(edit(last, "all", "patch", source=batch_hash(SOURCE)))[0], source_logits)
    with pytest.raises(ValueError):  # the resolver must return exactly what the hash names
        _run(edit(last, "all", "patch", source=batch_hash(SOURCE)), store={**STORE, batch_hash(SOURCE): TOKENS})


def test_full_swap_reproduces_the_other_checkpoint():
    every = [edit(c, "all", "swap", step=4) for c in ("embeddings", "blocks.0", "blocks.1", "ln_final")]
    assert np.array_equal(_run(*every)[0], base_logits(OTHER))
    assert not np.array_equal(_run(edit("blocks.1.mlp", "all", "swap", step=4))[0], base_logits()), "a swap does something"


def test_edits_only_at_existing_taps():
    for tap in ("blocks.0.mlp.hook_post", "blocks.0.attn.hook_pattern", "blocks.5.hook_resid_post", "logits"):
        with pytest.raises(ValueError):
            _run(edit(tap, "all", "zero"))
    with pytest.raises(ValueError):
        _run(edit("blocks.9", "all", "swap", step=4))


def test_edits_respect_positions_and_causality():
    base = base_logits()
    edited = _run(edit("blocks.0.hook_resid_post", [3], "zero"))[0]
    assert np.allclose(edited[:, :3], base[:, :3], rtol=0, atol=1e-6), "earlier positions can't see position 3"
    assert not np.allclose(edited[:, 3:], base[:, 3:], atol=1e-3), "position 3 and later do change"


def test_mean_and_resample_take_the_reference_values():
    tap = "blocks.0.hook_resid_post"
    reference = np.asarray(getattr(v := MODEL.instrumented(PARAMS, REFERENCE, [tap])[1][tap], "array", v))
    _, recorded = _run(edit(tap, "all", "mean", reference=batch_hash(REFERENCE)), taps=[tap])
    mean = reference.astype(np.float64).mean(axis=0).astype(np.float32)
    assert np.array_equal(recorded[tap], np.broadcast_to(mean, recorded[tap].shape)), "per-position mean of the reference"
    _, recorded = _run(edit(tap, [1, 4], "resample", reference=batch_hash(REFERENCE), seed=3), taps=[tap])
    rows = np.random.default_rng(3).integers(0, len(REFERENCE), size=len(TOKENS))
    assert np.array_equal(recorded[tap][:, [1, 4]], reference[rows][:, [1, 4]]), "a seeded draw of reference windows"
