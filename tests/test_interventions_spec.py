"""Phase 3 package 1: intervention specs, their positions and their hashes (D22).

Spec: docs/specs/phase-3.md, D22. Needs no JAX.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nablaframe.interventions.spec import InterventionSpec, edit, position_index, spec_from_json, spec_hash, spec_json
from nablaframe.trace.schema import canonical_json

H1, H2 = "a" * 64, "b" * 64
TAP = "blocks.0.hook_resid_post"


def test_spec_hash_is_canonical():
    a = InterventionSpec((edit(TAP, "all", "add", vector=H1, alpha=0.5),))
    b = InterventionSpec((edit(TAP, "all", "add", alpha=0.5, vector=H1),))
    assert spec_hash(a) == spec_hash(b), "argument order doesn't change the identity"
    assert len(spec_hash(a)) == 64 and all(c in "0123456789abcdef" for c in spec_hash(a))
    assert spec_hash(spec_from_json(json.loads(canonical_json(spec_json(a))))) == spec_hash(a), "round-trips through JSON"
    assert spec_hash(InterventionSpec((edit(TAP, "all", "add", vector=H1, alpha=0.25),))) != spec_hash(a)
    assert spec_hash(InterventionSpec((edit(TAP, {"range": [0, 4]}, "add", vector=H1, alpha=0.5),))) != spec_hash(a)
    e1, e2 = edit(TAP, [1], "zero"), edit(TAP, "all", "mean", reference=H2)
    assert spec_hash(InterventionSpec((e1, e2))) != spec_hash(InterventionSpec((e2, e1))), "edits apply in order"


def test_spec_refers_to_vectors_and_batches_by_hash():
    for bad in ([0.1, 0.2], "A" * 64, "a" * 63, np.zeros(4)):
        with pytest.raises(ValueError):
            edit(TAP, "all", "add", vector=bad, alpha=1.0)  # by content hash, never by value
    with pytest.raises(ValueError):
        edit(TAP, "all", "explode")
    with pytest.raises(ValueError):
        edit(TAP, "all", "add", vector=H1)  # alpha missing
    with pytest.raises(ValueError):
        edit(TAP, "all", "zero", extra=1)
    with pytest.raises(ValueError):
        edit(TAP, "all", "resample", reference=H2)  # seed missing
    with pytest.raises(ValueError):
        edit("blocks.0", [1], "swap", step=4)  # a swap replaces a whole component, at every position
    with pytest.raises(ValueError):
        edit("blocks.0.attn.hook_z", "all", "swap", step=4)  # not a component
    for component in ("embeddings", "ln_final", "blocks.0", "blocks.3.attn", "blocks.1.mlp"):
        edit(component, "all", "swap", step=4)
    edit(TAP, "all", "mean", reference=H2)
    edit(TAP, [0, 2], "resample", reference=H2, seed=0)
    edit(TAP, "all", "patch", source=H1)


def test_positions_language():
    assert position_index("all", 6).tolist() == [0, 1, 2, 3, 4, 5]
    assert position_index([4, 1], 6).tolist() == [4, 1]
    assert position_index({"range": [2, 5]}, 6).tolist() == [2, 3, 4]
    assert position_index([], 6).tolist() == []
    for bad in ([6], {"range": [4, 7]}, {"range": [3, 2]}, "last", [-1]):
        with pytest.raises(ValueError):
            position_index(bad, 6)
