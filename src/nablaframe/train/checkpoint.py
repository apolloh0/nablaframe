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

import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file
from safetensors.numpy import save_file  # noqa: F401  (module-level on purpose: tests patch it)

from ..trace.schema import canonical_json, sha256_hex, tensor_sha256
from .loop import TrainState


def state_hash(tree: Any) -> str:
    lines = sorted(f"{path}:{tensor_sha256(array)}\n" for path, array in _flatten(tree).items())
    return sha256_hex("".join(lines).encode("utf-8"))


def save_checkpoint(state: TrainState, directory: str | Path, *, run_id: str | None) -> Path:
    return _save(state, Path(directory), step=_step(state.step), run_id=run_id)


def load_checkpoint(path: str | Path, *, like: TrainState) -> TrainState:
    state, sidecar = _load(Path(path), like)
    if sidecar["step"] != state.step:
        raise ValueError(f"{path} holds step {state.step}, but its sidecar says {sidecar['step']}")
    return state


def save_params(params: Any, directory: str | Path, *, step: int, run_id: str | None) -> Path:
    return _save(params, Path(directory), step=_step(step), run_id=run_id)


def load_params(path: str | Path, *, like: Any) -> Any:
    return _load(Path(path), like)[0]


def _step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    return int(step)


def _flatten(tree: Any) -> dict[str, np.ndarray]:
    """Every leaf as a numpy array, keyed by its pytree path."""
    flat: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = jax.tree_util.keystr(path)
        if key in flat:
            raise ValueError(f"two leaves share the pytree path {key!r}")
        flat[key] = np.asarray(leaf)
    return flat


def _save(tree: Any, directory: Path, *, step: int, run_id: str | None) -> Path:
    """Tensor file and sidecar, each written to a temporary name, synced, then renamed.

    The sidecar is renamed last, and any old sidecar under the final name is removed
    before the tensor file is replaced, so a sidecar under its final name always
    describes the tensor file beside it. A failure removes whatever this call wrote.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {key: np.ascontiguousarray(array) for key, array in _flatten(tree).items()}
    sidecar = canonical_json({"run_id": run_id, "state_sha256": state_hash(tree), "step": step})
    final = directory / f"step-{step}.safetensors"
    final_sidecar = final.with_suffix(".json")
    tmp = final.with_name(final.name + ".tmp")
    tmp_sidecar = final_sidecar.with_name(final_sidecar.name + ".tmp")
    renamed = False
    try:
        save_file(tensors, str(tmp))
        _fsync(tmp)
        tmp_sidecar.write_text(sidecar, encoding="utf-8")
        _fsync(tmp_sidecar)
        final_sidecar.unlink(missing_ok=True)
        os.replace(tmp, final)
        renamed = True
        os.replace(tmp_sidecar, final_sidecar)
        _fsync(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp_sidecar.unlink(missing_ok=True)
        if renamed and not final_sidecar.exists():
            final.unlink(missing_ok=True)
        raise
    return final


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load(path: Path, like: Any) -> tuple[Any, dict[str, Any]]:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError(f"{path} has no sidecar {sidecar_path.name}, so it is not a complete checkpoint")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tensors = load_file(str(path))
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(like)
    expected = {jax.tree_util.keystr(key_path) for key_path, _ in leaves_with_paths}
    missing, unexpected = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} does not match the template: missing {missing}, unexpected {unexpected}")
    leaves = [_restore(tensors[jax.tree_util.keystr(key_path)], template, jax.tree_util.keystr(key_path), path)
              for key_path, template in leaves_with_paths]
    tree = jax.tree_util.tree_unflatten(treedef, leaves)
    if state_hash(tree) != sidecar["state_sha256"]:
        raise ValueError(f"{path} does not hash to the state_sha256 in {sidecar_path.name}")
    return tree, sidecar


def _restore(array: np.ndarray, template: Any, key: str, path: Path) -> Any:
    """One loaded tensor, given back the template leaf's shape, dtype and type."""
    like = np.asarray(template)
    if array.dtype != like.dtype or array.size != like.size:
        raise ValueError(f"{key} in {path} is {array.dtype}{list(array.shape)},"
                         f" but the template has {like.dtype}{list(like.shape)}")
    # safetensors may store a zero-dimensional array as shape (1,).
    array = array.reshape(like.shape)
    if isinstance(template, jax.Array):
        return jnp.asarray(array)
    if isinstance(template, np.ndarray):
        return array
    return array.item()
