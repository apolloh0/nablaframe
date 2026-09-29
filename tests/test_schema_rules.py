"""Schema validation rules added in Phase 0 delta 2.

Spec: docs/deltas/phase-0-delta-2.md, "Contract changes", schema.py.
"""

from __future__ import annotations

import pytest

from conftest import make_event
from nablaframe.trace.schema import Axis, TapSpec


def test_tapspec_accepts_only_canonical_dtype_names():
    TapSpec("hook_embed", (Axis("Batch", 2),), "float32", "embedding")
    TapSpec("hook_embed", (Axis("Batch", 2),), "float16", "embedding")
    for alias in ("f4", "float", "<f4", "single"):
        with pytest.raises(ValueError):
            TapSpec("hook_embed", (Axis("Batch", 2),), alias, "embedding")


def test_tapspec_rejects_malformed_axes():
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis("", 2),), "float32", "embedding")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", (Axis(3, 2),), "float32", "embedding")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", ({"name": "Batch", "size": 2},), "float32", "embedding")
    with pytest.raises(ValueError):
        TapSpec("hook_embed", [Axis("Batch", 2)], "float32", "embedding")


def test_trace_event_rejects_unknown_kind_and_mode():
    make_event(kind="checkpoint", tap="", mode="fast")  # a tapless marker stays valid
    with pytest.raises(ValueError):
        make_event(kind="forwards")
    with pytest.raises(ValueError):
        make_event(mode="slow")
