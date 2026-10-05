"""Phase 3 package 3: effects against the unintervened pass, their meter, and their cache (D24, D25).

Spec: docs/specs/phase-3.md, "Effects and their meters".
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from conftest import REPO_ROOT
from nablaframe.interventions.effects import (IDENTITIES, effect_payload, effects, intervention_key, load_result,
                                              save_result)
from nablaframe.interventions.spec import InterventionSpec, edit, spec_hash, spec_json
from nablaframe.views.export import export_view

TOKENS = np.array([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]], np.int32)
def _spec():
    return InterventionSpec((edit("blocks.0.hook_resid_post", [2], "zero"),))


def _logits():
    return np.random.default_rng(0).standard_normal((2, 5, 9)).astype(np.float32)


def test_effects_against_the_unintervened_pass():
    base = _logits()
    same = effects(base, base, TOKENS)
    assert set(same) == {"delta_logit", "delta_loss", "kl"}
    for name in ("delta_logit", "delta_loss"):
        assert np.isnan(same[name][:, -1]).all() and (same[name][:, :-1] == 0).all(), name
    assert (same["kl"] == 0).all(), "KL is defined at every position, the last included"
    moved = base.copy()
    moved[0, 2, TOKENS[0, 3]] += 1.5
    out = effects(base, moved, TOKENS)
    assert out["delta_logit"][0, 2] == pytest.approx(1.5, abs=1e-6)
    assert (out["delta_logit"][1, :-1] == 0).all() and (out["kl"][1] == 0).all(), "the other sequence is untouched"
    b, m = base[0, 2].astype(np.float64), moved[0, 2].astype(np.float64)
    lb, lm = b - np.log(np.exp(b - b.max()).sum()) - b.max(), m - np.log(np.exp(m - m.max()).sum()) - m.max()
    assert out["delta_loss"][0, 2] == pytest.approx(-(lm[TOKENS[0, 3]] - lb[TOKENS[0, 3]]), rel=1e-9)
    assert out["kl"][0, 2] == pytest.approx(float((np.exp(lb) * (lb - lm)).sum()), rel=1e-9), "KL(unintervened || intervened)"


def test_effect_meter_names_the_identities_checked(tmp_path):
    base = _logits()
    spec = _spec()
    payload = effect_payload(spec, base, base + 0.5, TOKENS)
    assert payload["spec_hash"] == spec_hash(spec) and payload["spec"] == spec_json(spec)
    assert payload["meter"] == {"kind": "exactness", "identities": list(IDENTITIES), "checked_by": "G3.1"}
    assert IDENTITIES == ("null edit", "steering at zero", "self-swap", "full patch", "full swap")
    assert payload["delta_logit"][0][-1] is None and type(payload["kl_mean"]) is float
    export_view(tmp_path, "intervention", payload)  # JSON-ready, with its meter


def test_intervention_cache_key_covers_spec_inputs_and_code(tmp_path):
    fields = dict(checkpoint_sha256="a" * 64, batch_hash="b" * 64, spec_hash="c" * 64, code="d" * 40)
    key = intervention_key(**fields)
    assert len(key) == 64
    for name in fields:
        assert intervention_key(**{**fields, name: "e" * len(fields[name])}) != key, name
    logits = _logits()
    save_result(tmp_path, key, logits)
    assert np.array_equal(load_result(tmp_path, key), logits) and load_result(tmp_path, "f" * 64) is None
    save_result(tmp_path, key, logits)
    with pytest.raises(ValueError):
        save_result(tmp_path, key, logits + 1)
    from nablaframe.interventions.apply import code_identity
    blob = subprocess.run(["git", "hash-object", str(REPO_ROOT / "src/nablaframe/interventions/apply.py")],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert code_identity() == blob, "D18's code identity, for the module that computes results"
