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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import RunManifest, TraceEvent

ORACLE_POLICY = "bitwise"  # D6


class OracleFailure(Exception):
    """Instrumentation changed what the model computes; nothing was recorded."""


class TraceRun:
    @classmethod
    def open(cls, root: str | Path, manifest: RunManifest, model: Any, *, checkpoint_dir: str | Path,
             probe: np.ndarray, like: Any) -> TraceRun:
        raise NotImplementedError

    def record(self, *, step: int, taps: Sequence[str]) -> tuple[TraceEvent, ...]:
        raise NotImplementedError

    def rematerialize(self, event: TraceEvent) -> np.ndarray:
        raise NotImplementedError

    def verify(self, event: TraceEvent) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> TraceRun:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
