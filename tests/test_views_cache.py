"""Phase 2 work package 5: caches keyed by inputs and code (D11), and the export's meter rule.

Spec: docs/specs/phase-2.md, D11 and "The scrubber".
"""

from __future__ import annotations

import json
import shutil

import pytest

from nablaframe.views.cache import ViewCache, cache_key
from nablaframe.views.export import export_view

INPUTS = {"trace": "a" * 64, "checkpoint": "b" * 64}
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_cache_key_covers_inputs_and_code():
    key = cache_key(INPUTS, COMMIT)
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    assert cache_key(dict(reversed(list(INPUTS.items()))), COMMIT) == key, "input order doesn't matter"
    assert cache_key({**INPUTS, "trace": "c" * 64}, COMMIT) != key
    assert cache_key({**INPUTS, "extra": "d" * 64}, COMMIT) != key
    assert cache_key(INPUTS, "f" * 40) != key


def _payload():
    return {"meter": {"kl_mean": [0.5, 0.25, 0.0]}, "readings": [[1, 2], [3, 4]]}


def test_deleting_cache_then_rebuilding_is_identical(tmp_path):
    key = cache_key(INPUTS, COMMIT)
    first = ViewCache(tmp_path / "cache").put(key, _payload())
    written = first.read_bytes()
    assert ViewCache(tmp_path / "cache").get(key) == _payload()
    shutil.rmtree(tmp_path / "cache")
    assert ViewCache(tmp_path / "cache").get(key) is None
    assert ViewCache(tmp_path / "cache").put(key, _payload()).read_bytes() == written
    ViewCache(tmp_path / "cache").put(key, _payload())  # the same content again is fine
    with pytest.raises(ValueError):
        ViewCache(tmp_path / "cache").put(key, {**_payload(), "readings": []})  # a key never changes meaning


def test_derived_view_without_meter_is_refused(tmp_path):
    with pytest.raises(ValueError):
        export_view(tmp_path / "views", "logit_lens", {"readings": [1, 2, 3]})
    assert not (tmp_path / "views" / "logit_lens.json").exists()
    path = export_view(tmp_path / "views", "logit_lens", _payload())
    assert path.read_text() == json.dumps(_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
