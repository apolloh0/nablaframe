"""Store tests: append-only history, superseding corrections, tensor integrity.

Spec: docs/specs/phase-0.md, "Trace schema v0", last paragraph.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from safetensors.numpy import save_file

from conftest import make_event
from nablaframe.trace.schema import tensor_sha256
from nablaframe.trace.store import AppendOnlyViolation, IntegrityError, TraceStore


def _tensor(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((2, 3, 4)).astype(np.float32)


def test_store_rejects_rewrite(tmp_path):
    store = TraceStore(tmp_path)
    log = tmp_path / "events.jsonl"

    first = store.append(make_event(step=1), _tensor(1))
    after_first = log.read_bytes()
    store.append(make_event(step=2), _tensor(2))
    after_second = log.read_bytes()
    assert after_second.startswith(after_first), "the log may only grow"
    assert after_second.count(b"\n") == 2

    # Appending the same content again is a rewrite, and leaves no trace.
    with pytest.raises(AppendOnlyViolation):
        store.append(make_event(step=1), _tensor(1))
    assert log.read_bytes() == after_second
    assert len(list((tmp_path / "shards").iterdir())) == 2

    # The same holds for events without a tensor, where no shard can collide.
    marker = make_event(step=500, kind="checkpoint", tap="", mode="fast")
    store.append(marker)
    after_marker = log.read_bytes()
    with pytest.raises(AppendOnlyViolation):
        store.append(marker)
    assert log.read_bytes() == after_marker

    # Callers cannot pre-fill what only the store may set.
    for field in ("event_id", "tensor_ref", "supersedes"):
        with pytest.raises(ValueError):
            store.append(make_event(step=3, **{field: "x" * 64}), _tensor(3))
    assert log.read_bytes() == after_marker

    # There is no way to edit history through the API.
    for name in ("update", "delete", "remove", "replace", "rewrite"):
        assert not hasattr(store, name)

    # Reopening sees the same history.
    assert [e.event_id for e in TraceStore(tmp_path).events()] == [e.event_id for e in store.events()]
    assert store.get(first.event_id) == first


def test_store_supersede_links_original(tmp_path):
    store = TraceStore(tmp_path)
    original = store.append(make_event(step=10), _tensor(10))
    other = store.append(make_event(step=11), _tensor(11))

    correction = store.supersede(original.event_id, make_event(step=10), _tensor(99))
    assert correction.supersedes == original.event_id
    assert correction.event_id != original.event_id

    # Nothing is removed; the original drops out of the current view only.
    assert [e.event_id for e in store.events()] == [original.event_id, other.event_id, correction.event_id]
    assert [e.event_id for e in store.current()] == [other.event_id, correction.event_id]
    assert store.get(original.event_id) == original

    # One original, one correction: a second correction must chain instead.
    with pytest.raises(AppendOnlyViolation):
        store.supersede(original.event_id, make_event(step=10), _tensor(98))
    second = store.supersede(correction.event_id, make_event(step=10), _tensor(97))
    assert [e.event_id for e in store.current()] == [other.event_id, second.event_id]

    with pytest.raises(KeyError):
        store.supersede("0" * 64, make_event(step=12), _tensor(12))

    # The links survive a reopen.
    reopened = TraceStore(tmp_path)
    assert reopened.get(correction.event_id).supersedes == original.event_id
    assert [e.event_id for e in reopened.current()] == [other.event_id, second.event_id]


def test_event_sha256_matches_tensor_bytes(tmp_path):
    store = TraceStore(tmp_path)
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    assert tensor_sha256(array) == expected
    # Layout and byte order do not change the hash; values, dtype and shape do.
    transposed = array.transpose(2, 1, 0)
    assert tensor_sha256(transposed) == hashlib.sha256(np.ascontiguousarray(transposed).tobytes()).hexdigest()
    assert tensor_sha256(array.astype(array.dtype.newbyteorder(">"))) == expected
    assert tensor_sha256(array.astype(np.float64)) != expected

    event = store.append(make_event(), array)
    assert event.sha256 == expected
    assert event.tensor_ref == f"shards/{event.event_id}.safetensors"

    loaded = store.load_tensor(event)
    assert loaded.dtype == array.dtype and loaded.shape == array.shape
    assert np.array_equal(loaded, array)
    assert tensor_sha256(loaded) == expected

    # A caller-supplied hash must agree with the tensor.
    with pytest.raises(IntegrityError):
        store.append(make_event(step=7, sha256="0" * 64), _tensor(7))

    # A shard altered on disk is caught on load.
    save_file({"tensor": array + 1}, str(tmp_path / event.tensor_ref))
    with pytest.raises(IntegrityError):
        store.load_tensor(event)
