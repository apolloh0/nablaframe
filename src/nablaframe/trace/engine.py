"""The trace engine: records every requested tap at a trace checkpoint into the store.

Spec: docs/specs/phase-1.md, "Trace engine".

* ``TraceRun.open(root, manifest, model, checkpoint_dir, probe, like)`` opens the
  store at ``root`` as its one writer and writes ``manifest.json`` (canonical JSON
  of ``manifest.to_dict()``) the first time. Reopening with an identical manifest
  leaves the file untouched; a different one raises ``ValueError``.
* ``record(step, taps)`` loads the trace checkpoint ``step-<step>.safetensors``
  from ``checkpoint_dir`` with ``load_params``, so what is traced is exactly what is
  on disk. It first runs ``check_mode_equality`` on the probe batch under
  ``ORACLE_POLICY`` (D6), and raises ``OracleFailure`` before writing anything if it
  fails. Then it appends one event per tap: kind ``forward``, mode
  ``instrumented``, the tap name, the step, and ``batch_hash(probe)``. Taps already
  recorded with identical content are skipped. It returns only the new events and
  never supersedes on its own.
* ``record`` and ``rematerialize`` both run the instrumented pass with every
  canonical tap (``registry.tap_names(model.config.n_layers)``), whatever subset
  is stored. JAX compiles a different program for each set of outputs, and only
  the same program is guaranteed to reproduce a tensor bit for bit.
* ``rematerialize(event)`` reloads that step's trace checkpoint, reruns the
  instrumented forward pass on the probe batch, and returns the tap's tensor.
  ``verify(event)`` is whether that tensor's hash equals ``event.sha256``.
* ``close()`` releases the store; ``TraceRun`` is also a context manager.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..oracle.mode_equality import check_mode_equality
from ..train.checkpoint import load_params
from .registry import tap_names
from .schema import RunManifest, TraceEvent, tensor_sha256
from .store import TraceStore

ORACLE_POLICY = "bitwise"  # D6

MANIFEST_NAME = "manifest.json"


class OracleFailure(Exception):
    """Instrumentation changed what the model computes; nothing was recorded."""


class TraceRun:
    def __init__(self, store: TraceStore, manifest: RunManifest, model: Any, checkpoint_dir: Path,
                 probe: np.ndarray, like: Any) -> None:
        """Use ``TraceRun.open``."""
        self._store = store
        self.manifest = manifest
        self.run_id = manifest.run_id
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.probe = probe
        self.batch_hash = batch_hash(probe)
        self._like = like
        self._all_taps = tap_names(model.config.n_layers)

    @classmethod
    def open(cls, root: str | Path, manifest: RunManifest, model: Any, *, checkpoint_dir: str | Path,
             probe: np.ndarray, like: Any) -> TraceRun:
        probe = np.array(probe, copy=True)
        probe.setflags(write=False)
        store = TraceStore(root)
        try:
            # Be the one writer before touching the manifest, not only at the first event.
            store._acquire_lock()
            _write_manifest_once(store.root / MANIFEST_NAME, manifest)
            return cls(store, manifest, model, Path(checkpoint_dir), probe, like)
        except BaseException:
            store.close()
            raise

    def record(self, *, step: int, taps: Sequence[str]) -> tuple[TraceEvent, ...]:
        if isinstance(taps, str):
            raise TypeError("taps must be a sequence of tap names, not one string")
        requested = tuple(dict.fromkeys(taps))
        unknown = sorted(set(requested) - set(self._all_taps))
        if unknown:
            raise KeyError(f"not canonical tap names for a {self.model.config.n_layers}-layer model: {unknown}")

        step = _step(step)
        params = self._load(step)
        # The oracle's own instrumented pass supplies the tensors, so what is stored is
        # exactly what the oracle checked.
        capture = _Capture(self.model)
        result = check_mode_equality(capture, params, self.probe, self._all_taps, ORACLE_POLICY)
        if not result.passed:
            raise OracleFailure(f"step {step}: {result.reason}")

        recorded_ids = {event.event_id for event in self._store.events()}
        new: list[TraceEvent] = []
        for tap in requested:
            tensor = np.asarray(capture.recorded[tap])
            event = TraceEvent(run_id=self.run_id, step=step, kind="forward", tap=tap,
                               batch_hash=self.batch_hash, mode="instrumented", sha256=tensor_sha256(tensor))
            # The store's event id is this content's identity hash, so an equal id is the same fact.
            if event.identity_hash() in recorded_ids:
                continue
            new.append(self._store.append(event, tensor))
        return tuple(new)

    def rematerialize(self, event: TraceEvent) -> np.ndarray:
        if event.run_id != self.run_id:
            raise ValueError(f"event belongs to run {event.run_id}, not {self.run_id}")
        if event.batch_hash != self.batch_hash:
            raise ValueError(f"event was recorded on batch {event.batch_hash}, not this run's probe {self.batch_hash}")
        if (event.kind, event.mode) != ("forward", "instrumented"):
            raise ValueError(f"only forward instrumented events can be recomputed, got {event.kind} {event.mode}")
        if event.tap not in self._all_taps:
            raise KeyError(event.tap)
        _, recorded = self.model.instrumented(self._load(event.step), self.probe, self._all_taps)
        return np.array(recorded[event.tap])

    def verify(self, event: TraceEvent) -> bool:
        return tensor_sha256(self.rematerialize(event)) == event.sha256

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> TraceRun:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _load(self, step: int) -> Any:
        """The trace checkpoint for ``step``, exactly as it is on disk."""
        return load_params(self.checkpoint_dir / f"step-{_step(step)}.safetensors", like=self._like)


def _step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    return int(step)


class _Capture:
    """Passes the oracle's calls through to ``model`` and keeps the instrumented taps."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.recorded: dict[str, Any] = {}

    def fast(self, params: Any, tokens: Any) -> Any:
        return self.model.fast(params, tokens)

    def instrumented(self, params: Any, tokens: Any, taps: Sequence[str]) -> tuple[Any, dict[str, Any]]:
        logits, self.recorded = self.model.instrumented(params, tokens, taps)
        return logits, self.recorded


def _write_manifest_once(path: Path, manifest: RunManifest) -> None:
    """Write the manifest if absent; otherwise require it to be byte-identical."""
    data = manifest.to_json().encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} holds a different manifest; one trace store holds one run")
        return
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
