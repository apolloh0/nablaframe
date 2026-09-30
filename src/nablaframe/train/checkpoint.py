"""Checkpoints: one safetensors file keyed by pytree path, plus a JSON sidecar.

Spec: docs/specs/phase-1.md, "Trainer and checkpoints".

* ``save_checkpoint(state, directory, run_id)`` writes ``step-<n>.safetensors``,
  where n is ``state.step``: the state after exactly n updates. It holds params,
  optimizer state and step. ``save_params(params, directory, step, run_id)`` writes
  a trace checkpoint with params only, under the same name pattern.
* Each tensor file is written through the module-level ``save_file`` to a temporary
  name in the same directory, then renamed into place. The sidecar
  ``step-<n>.json`` (step, run_id, state_sha256) is written the same way, after the
  tensor file. A crash at any point leaves neither file under its final name.
* ``load_checkpoint(path, like)`` and ``load_params(path, like)`` rebuild the pytree
  using ``like`` as the structure template, and raise ``ValueError`` if the loaded
  state's hash differs from the sidecar's.
* safetensors stores zero-dimensional arrays with shape ``(1,)``. Loading reshapes
  every tensor to its template leaf's shape, so scalars such as the step and
  optax's update counts come back as scalars. A resumed optimizer with a ``(1,)``
  count would silently compute a different learning-rate schedule.
* ``state_hash(tree)``: SHA-256 over lines ``"<pytree path>:<tensor_sha256>\n"``
  for every leaf, sorted by path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors.numpy import save_file  # noqa: F401  (module-level on purpose: tests patch it)

from .loop import TrainState


def state_hash(tree: Any) -> str:
    raise NotImplementedError


def save_checkpoint(state: TrainState, directory: str | Path, *, run_id: str) -> Path:
    raise NotImplementedError


def load_checkpoint(path: str | Path, *, like: TrainState) -> TrainState:
    raise NotImplementedError


def save_params(params: Any, directory: str | Path, *, step: int, run_id: str) -> Path:
    raise NotImplementedError


def load_params(path: str | Path, *, like: Any) -> Any:
    raise NotImplementedError
