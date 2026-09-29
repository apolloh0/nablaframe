"""Append-only trace store: events in JSONL, tensors in safetensors shards.

Spec: docs/specs/phase-0.md, "Trace schema v0", last paragraph.

Layout under ``root``::

    events.jsonl                  one canonical-JSON TraceEvent per line, append only
    shards/<event_id>.safetensors one tensor per event, under the key "tensor"

Nothing is edited in place. A correction is a new event that supersedes the old
one by id; the old event stays in the log. There is deliberately no update,
delete, remove, replace or rewrite method.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from .schema import TraceEvent, canonical_json, tensor_sha256


class StoreError(Exception):
    """Base class for trace store errors."""


class AppendOnlyViolation(StoreError):
    """An append would rewrite history: a duplicate event id, an existing shard,
    or a second event superseding the same original."""


class IntegrityError(StoreError):
    """A tensor's bytes do not match the hash recorded for it."""


class TraceStore:
    def __init__(self, root: str | Path) -> None:
        """Open the store at ``root``, creating the directory, ``events.jsonl`` and
        ``shards/`` if absent. Events already on disk are visible immediately."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._shards_dir = self.root / "shards"
        self._shards_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.root / "events.jsonl"
        self._log_path.touch(exist_ok=True)

        self._events: list[TraceEvent] = []
        self._by_id: dict[str, TraceEvent] = {}
        self._superseded_ids: set[str] = set()

        with self._log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = TraceEvent.from_dict(json.loads(line))
                self._events.append(event)
                self._by_id[event.event_id] = event
                if event.supersedes is not None:
                    self._superseded_ids.add(event.supersedes)

    def _record(self, event: TraceEvent, tensor: np.ndarray | None, supersedes: str | None) -> TraceEvent:
        for field_name in ("event_id", "tensor_ref", "supersedes"):
            if getattr(event, field_name) is not None:
                raise ValueError(f"{field_name} is set by the store; leave it as None")

        if tensor is not None:
            computed_sha256 = tensor_sha256(tensor)
            if event.sha256 is not None and event.sha256 != computed_sha256:
                raise IntegrityError(
                    f"event.sha256 {event.sha256!r} does not match the tensor's hash {computed_sha256!r}"
                )
            final_sha256 = computed_sha256
        else:
            if event.sha256 is not None:
                raise ValueError("event.sha256 must be None when no tensor is given")
            final_sha256 = None

        candidate = replace(event, sha256=final_sha256, supersedes=supersedes, event_id=None, tensor_ref=None)
        event_id = candidate.identity_hash()
        if event_id in self._by_id:
            raise AppendOnlyViolation(f"event {event_id} is already recorded")

        tensor_ref = f"shards/{event_id}.safetensors" if tensor is not None else None
        shard_path = self.root / tensor_ref if tensor_ref is not None else None
        if shard_path is not None and shard_path.exists():
            raise AppendOnlyViolation(f"shard {tensor_ref} already exists")

        final_event = replace(candidate, event_id=event_id, tensor_ref=tensor_ref)

        if tensor is not None:
            save_file({"tensor": np.ascontiguousarray(tensor)}, str(shard_path))

        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(final_event.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self._events.append(final_event)
        self._by_id[event_id] = final_event
        if final_event.supersedes is not None:
            self._superseded_ids.add(final_event.supersedes)
        return final_event

    def append(self, event: TraceEvent, tensor: np.ndarray | None = None) -> TraceEvent:
        """Record ``event``, and ``tensor`` if given, and return the completed event.

        * ``event.event_id``, ``event.tensor_ref`` and ``event.supersedes`` must be
          None; otherwise ValueError. Use ``supersede`` for corrections.
        * With a tensor: its hash is computed with ``tensor_sha256``. If
          ``event.sha256`` is already set and differs, raise IntegrityError. The
          shard is written to ``shards/<event_id>.safetensors``; if that file
          exists, raise AppendOnlyViolation.
        * Without a tensor: ``event.sha256`` must be None; otherwise ValueError.
        * ``event_id`` is the completed event's ``identity_hash()``. If an event
          with that id is already in the store, raise AppendOnlyViolation.
        * The event is appended as one line of canonical JSON, flushed and fsynced
          before returning. A rejected append leaves ``events.jsonl`` byte-for-byte
          unchanged and writes no shard.
        """
        return self._record(event, tensor, supersedes=None)

    def supersede(
        self, original_id: str, event: TraceEvent, tensor: np.ndarray | None = None
    ) -> TraceEvent:
        """Append ``event`` as the correction of ``original_id``.

        Sets ``supersedes=original_id`` and otherwise follows ``append``. Raises
        KeyError if ``original_id`` is unknown, and AppendOnlyViolation if another
        event already supersedes it. Corrections chain: to correct a correction,
        supersede the correction.
        """
        if original_id not in self._by_id:
            raise KeyError(original_id)
        if original_id in self._superseded_ids:
            raise AppendOnlyViolation(f"{original_id} is already superseded")
        return self._record(event, tensor, supersedes=original_id)

    def get(self, event_id: str) -> TraceEvent:
        """The event with ``event_id``, superseded or not. KeyError if unknown."""
        return self._by_id[event_id]

    def events(self) -> tuple[TraceEvent, ...]:
        """Every event ever appended, in append order, including superseded ones."""
        return tuple(self._events)

    def current(self) -> tuple[TraceEvent, ...]:
        """Events not superseded by any other event, in append order."""
        return tuple(e for e in self._events if e.event_id not in self._superseded_ids)

    def load_tensor(self, event: TraceEvent) -> np.ndarray:
        """Load the event's tensor and check it against ``event.sha256``.

        Raises ValueError if the event has no tensor, and IntegrityError if the
        loaded bytes do not hash to ``event.sha256``.
        """
        if event.tensor_ref is None:
            raise ValueError("event has no tensor")
        tensors = load_file(str(self.root / event.tensor_ref))
        array = tensors["tensor"]
        if tensor_sha256(array) != event.sha256:
            raise IntegrityError(f"tensor at {event.tensor_ref} does not match event.sha256")
        return array
