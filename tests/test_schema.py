"""Schema tests: RunManifest identity, TapSpec JSON round trip.

Spec: docs/specs/phase-0.md, "Trace schema v0" and "Deliverables and tests".
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping

import pytest

from conftest import make_manifest
from nablaframe.trace.registry import tap_names, tap_spec
from nablaframe.trace.schema import Axis, RunManifest, Seeds, TapSpec

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_manifest_run_id_excludes_timestamp():
    a = make_manifest(created_at="2026-09-27T12:00:00Z")
    b = make_manifest(created_at="2031-01-01T00:00:00Z")

    assert HEX64.match(a.run_id)
    assert a.run_id == b.run_id
    # Deterministic across separately built instances.
    assert make_manifest().run_id == make_manifest().run_id
    # The timestamp is still part of the record, just not of its identity.
    assert a.to_dict()["created_at"] != b.to_dict()["created_at"]
    assert "run_id" not in {f.name for f in dataclasses.fields(RunManifest)}


def _mutated(value):
    """A different value of the same kind. Extend this when RunManifest gains a field type."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, tuple):
        return value + ((max(value) + 1) if value else 1,)
    if isinstance(value, Mapping):
        return {**value, "_mutated": 1}
    pytest.fail(f"no mutation rule for {type(value).__name__}; extend _mutated")


_TOP_LEVEL = [f.name for f in dataclasses.fields(RunManifest) if f.name not in ("created_at", "seeds")]
_SEED_FIELDS = [f"seeds.{f.name}" for f in dataclasses.fields(Seeds)]


@pytest.mark.parametrize("field", _TOP_LEVEL + _SEED_FIELDS)
def test_manifest_run_id_changes_with_each_field(field):
    base = make_manifest()
    if field.startswith("seeds."):
        seed_field = field.split(".", 1)[1]
        seeds = dataclasses.replace(base.seeds, **{seed_field: getattr(base.seeds, seed_field) + 1})
        changed = dataclasses.replace(base, seeds=seeds)
    else:
        changed = dataclasses.replace(base, **{field: _mutated(getattr(base, field))})

    assert changed.run_id != base.run_id, f"run_id ignores {field}"


def _handmade_specs() -> list[TapSpec]:
    return [
        # Axis order deliberately not alphabetical: a round trip that sorts axes fails.
        TapSpec("blocks.0.hook_resid_post", (Axis("Batch", 8), Axis("Pos", 128), Axis("Embed", 256)), "float32", "residual"),
        TapSpec("blocks.1.attn.hook_pattern", (Axis("Batch", 8), Axis("Head", 4), Axis("QPos", 128), Axis("KPos", 128)), "float32", "attention"),
        TapSpec("blocks.1.attn.hook_z", (Axis("Batch", 2), Axis("Pos", 16), Axis("Head", 4), Axis("HeadDim", 32)), "float16", "attention"),
    ]


def _registry_specs() -> list[TapSpec]:
    dims = dict(batch=8, seq=128, d_model=256, n_heads=4, d_head=64)
    return [tap_spec(name, **dims) for name in tap_names(2)]


def test_tapspec_json_roundtrip():
    for spec in _handmade_specs() + _registry_specs():
        text = spec.to_json()
        back = TapSpec.from_json(text)
        assert back == spec
        assert [a.name for a in back.axes] == [a.name for a in spec.axes]
        assert back.to_json() == text, "to_json must be canonical"

    # The registry's axes follow the spec's tap table.
    dims = dict(batch=8, seq=128, d_model=256, n_heads=4, d_head=64)
    pattern = tap_spec("blocks.0.attn.hook_pattern", **dims)
    assert [(a.name, a.size) for a in pattern.axes] == [("Batch", 8), ("Head", 4), ("QPos", 128), ("KPos", 128)]
    assert pattern.site == "attention"
    z = tap_spec("blocks.1.attn.hook_z", **dims)
    assert [(a.name, a.size) for a in z.axes] == [("Batch", 8), ("Pos", 128), ("Head", 4), ("HeadDim", 64)]
    assert tap_spec("blocks.1.ln2.hook_normalized", **dims).site == "mlp"
    assert tap_spec("ln_final.hook_normalized", **dims).site == "final_norm"
    assert tap_spec("hook_pos_embed", **dims).site == "embedding"
    with pytest.raises(KeyError):
        tap_spec("blocks.0.hook_not_a_tap", **dims)

    # Invalid specs are rejected at construction.
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis("Batch", 8),), "float32", "not_a_site")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis("Batch", 0),), "float32", "embedding")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis("Pos", 8), Axis("Pos", 8)), "float32", "embedding")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis("Batch", 8),), "not_a_dtype", "embedding")
