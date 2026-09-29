"""Store recovery rules added in Phase 0 delta 2.

Spec: docs/deltas/phase-0-delta-2.md, "Contract changes", store.py.

An append writes the shard first and the log line second, so the log is the commit
point. A crash in between leaves an orphan shard: a file under shards/ that no
logged event references. Its name is the event id, and the id covers the tensor's
hash, so an orphan can only legitimately hold that one event's tensor.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from safetensors.numpy import save_file

from conftest import make_event
from nablaframe.trace.schema import tensor_sha256
from nablaframe.trace.store import AppendOnlyViolation, IntegrityError, TraceStore


def _orphan(root, event, tensor, content):
    """Leave the shard an interrupted append of (event, tensor) would have written."""
    event_id = replace(event, sha256=tensor_sha256(tensor)).identity_hash()
    path = root / "shards" / f"{event_id}.safetensors"
    save_file({"tensor": content}, str(path))
    return path


def test_store_recovers_from_an_interrupted_append(tmp_path):
    store = TraceStore(tmp_path)
    tensor = np.arange(6, dtype=np.float32)
    _orphan(tmp_path, make_event(step=7), tensor, tensor)

    recovered = store.append(make_event(step=7), tensor)
    assert np.array_equal(store.load_tensor(recovered), tensor)
    assert [e.event_id for e in TraceStore(tmp_path).events()] == [recovered.event_id]

    # Once logged, the same append is a rewrite again.
    with pytest.raises(AppendOnlyViolation):
        store.append(make_event(step=7), tensor)


def test_store_rejects_an_orphan_with_other_content(tmp_path):
    store = TraceStore(tmp_path)
    tensor = np.arange(6, dtype=np.float32)
    path = _orphan(tmp_path, make_event(step=7), tensor, tensor + 1)
    log_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(IntegrityError):
        store.append(make_event(step=7), tensor)
    assert (tmp_path / "events.jsonl").read_bytes() == log_before
    assert path.exists(), "the evidence stays on disk for inspection"
