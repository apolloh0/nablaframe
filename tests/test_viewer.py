"""Phase 2 work package 6: the viewer's offline rule and its in-browser hash check (D13).

Spec: docs/specs/phase-2.md, D13. Needs the jax lane for the traced fixture.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from conftest import REPO_ROOT

jax = pytest.importorskip("jax")

from nablaframe.views.shards import shard_tensor_bytes  # noqa: E402
from phase2_fixtures import traced_run  # noqa: E402


def test_viewer_has_no_external_urls():
    files = sorted((REPO_ROOT / "viewer").glob("*"))
    assert {f.name for f in files} >= {"index.html", "app.js"}
    for f in files:
        text = f.read_text()
        assert not re.search(r"https?://|src=[\"']//|import\s*\(\s*[\"']//", text), f"{f.name} reaches outside the repo"


def test_viewer_hash_check_matches_tensor_sha256(tmp_path):
    run = traced_run(tmp_path)
    for event in run.events:
        data = shard_tensor_bytes(run.root / event.tensor_ref)
        assert hashlib.sha256(data).hexdigest() == event.sha256, \
            f"{event.tap}: hashing the shard's data slice must give the event's sha256, as the page does"
