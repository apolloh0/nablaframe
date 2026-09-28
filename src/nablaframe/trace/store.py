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

from pathlib import Path

import numpy as np

from .schema import TraceEvent


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
        raise NotImplementedError

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
        raise NotImplementedError

    def supersede(
        self, original_id: str, event: TraceEvent, tensor: np.ndarray | None = None
    ) -> TraceEvent:
        """Append ``event`` as the correction of ``original_id``.

        Sets ``supersedes=original_id`` and otherwise follows ``append``. Raises
        KeyError if ``original_id`` is unknown, and AppendOnlyViolation if another
        event already supersedes it. Corrections chain: to correct a correction,
        supersede the correction.
        """
        raise NotImplementedError

    def get(self, event_id: str) -> TraceEvent:
        """The event with ``event_id``, superseded or not. KeyError if unknown."""
        raise NotImplementedError

    def events(self) -> tuple[TraceEvent, ...]:
        """Every event ever appended, in append order, including superseded ones."""
        raise NotImplementedError

    def current(self) -> tuple[TraceEvent, ...]:
        """Events not superseded by any other event, in append order."""
        raise NotImplementedError

    def load_tensor(self, event: TraceEvent) -> np.ndarray:
        """Load the event's tensor and check it against ``event.sha256``.

        Raises ValueError if the event has no tensor, and IntegrityError if the
        loaded bytes do not hash to ``event.sha256``.
        """
        raise NotImplementedError
