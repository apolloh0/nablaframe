"""Registry test: canonical tap names are TransformerLens names.

Spec: docs/specs/phase-0.md, "Trace schema v0". The vendored list keeps nf-jax
free of a TransformerLens import; scripts/check_tl_names.py confirms it at
runtime in nf-torch.
"""

from __future__ import annotations

from conftest import DATA
from nablaframe.trace.registry import (
    GLOBAL_TAPS_AFTER,
    GLOBAL_TAPS_BEFORE,
    PER_LAYER_TAPS,
    tap_names,
)

import pytest


def _vendored_names() -> list[str]:
    text = (DATA / "transformerlens_legacy_hook_names.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_registry_matches_transformerlens_names():
    vendored = _vendored_names()
    names = tap_names(2)

    missing = [n for n in names if n not in vendored]
    assert not missing, f"not TransformerLens names: {missing}"

    # Exactly the spec's tap table: 2 before, 8 per block, 1 after.
    assert len(names) == len(GLOBAL_TAPS_BEFORE) + 2 * len(PER_LAYER_TAPS) + len(GLOBAL_TAPS_AFTER) == 19
    assert len(set(names)) == len(names)

    # Forward order.
    assert names[:2] == ("hook_embed", "hook_pos_embed")
    assert names[2:10] == tuple(f"blocks.0.{t}" for t in PER_LAYER_TAPS)
    assert names[10:18] == tuple(f"blocks.1.{t}" for t in PER_LAYER_TAPS)
    assert names[-1] == "ln_final.hook_normalized"

    # The registry is a strict subset: it does not simply echo the vendored list.
    assert set(names) < set(vendored)

    with pytest.raises(ValueError):
        tap_names(0)
