"""The summed Jacobian of D17, exactly as anthropics/jacobian-lens computes it (jlens/fitting.py at 581d398).

Spec: docs/specs/phase-2.md, "The J-lens"; ruling: D17. Caches follow D11, with D18's code identity.

* ``SKIP_FIRST = 16``. ``valid_positions(seq_len, skip_first=SKIP_FIRST)``: positions
  ``skip_first`` to ``seq_len - 2`` as an int array, as the reference's
  ``valid_position_mask``. Raises ValueError if none remain.
* ``block_map(model, params, layer, n_layers)``: a pure JAX function from one sequence's
  ``blocks.<layer>.hook_resid_post`` (T, d) to the last block's ``hook_resid_post`` (T, d),
  through the model's intervention hook, which replaces that tap's value. The later blocks
  see only the residual stream, so one function serves every sequence. For the last
  layer it is the identity.
* ``summed_jacobian(f, xs, skip_first=SKIP_FIRST)``: for each sequence in ``xs`` (N, T, d)
  and each output dimension i, one reverse-mode pass with a one-hot cotangent on dimension
  i at every valid position at once. The gradient at valid source position p is then the
  sum over valid p' >= p of d f(x)[p'] / d x[p], dimension i; row i of that sequence's J is
  its mean over the valid source positions. J is the mean over sequences, each weighted
  equally, as the reference's ``fit`` averages prompts. Gradients in float32, averages in
  float64, returned as float32 (d, d). Deterministic.
* ``code_identity()``: the git blob hash of this module's source file, exactly as
  ``git hash-object`` prints it (D18), so an unrelated commit doesn't invalidate a cache.
* ``jacobian_key(*, checkpoint_sha256, fit_sha256, layer, code)``:
  ``cache_key({"checkpoint": ..., "fit_sample": ..., "layer": str(layer)}, code)``.
* ``save_jacobian(root, key, J)`` writes ``<root>/<key>.npy`` with ``numpy.save``,
  through a temporary file and a rename, and returns the path. Saving identical content
  again is a no-op; different content under an existing key raises ValueError.
  ``load_jacobian(root, key)`` returns the float32 array, or None.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.views.cache import _check_key, _write_atomic, cache_key

SKIP_FIRST = 16


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST) -> np.ndarray:
    """Positions ``skip_first`` to ``seq_len - 2``, as the reference's ``valid_position_mask``.

    Early positions act as attention sinks and the last has no next-token target.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    positions = np.arange(skip_first, seq_len - 1, dtype=np.int64)
    if positions.size == 0:
        raise ValueError(f"sequence too short: seq_len={seq_len}, need > {skip_first + 1} positions")
    return positions


def block_map(model: Any, params: Any, layer: int, n_layers: int) -> Callable[[Any], Any]:
    """``blocks.<layer>.hook_resid_post`` (T, d) to the last block's ``hook_resid_post`` (T, d)."""
    import jax
    import jax.numpy as jnp

    if n_layers != model.config.n_layers:
        raise ValueError(f"n_layers={n_layers} but the model has {model.config.n_layers} layers")
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} is outside a {n_layers}-layer model")
    if layer == n_layers - 1:
        return jnp.asarray
    source, target = f"blocks.{layer}.hook_resid_post", f"blocks.{n_layers - 1}.hook_resid_post"

    @jax.jit
    def mapped(params: Any, h: Any) -> Any:
        # The tokens only feed blocks 0 to layer, whose output the intervention replaces.
        tokens = jnp.zeros((1, h.shape[0]), dtype=jnp.int32)
        _, recorded = model.instrumented(params, tokens, [target], {source: lambda _value: h[None]})
        return recorded[target][0]

    return partial(mapped, params)


def summed_jacobian(f: Callable[[Any], Any], xs: np.ndarray, skip_first: int = SKIP_FIRST) -> np.ndarray:
    """J summed over valid target positions and averaged over valid source positions, then sequences."""
    import jax
    import jax.numpy as jnp

    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"xs must be sequences of shape (N, T, d), got {xs.shape}")
    n, t, _ = xs.shape
    if n == 0:
        raise ValueError("xs holds no sequences")
    valid = valid_positions(t, skip_first)
    mask = np.zeros(t, dtype=np.float32)
    mask[valid] = 1.0

    @jax.jit
    def per_sequence(x: Any, mask: Any) -> Any:
        # One forward pass, then one backward pass per output dimension i, with a one-hot
        # cotangent on dimension i at every valid target position at once.
        out, pull = jax.vjp(f, x)

        def row(i: Any) -> Any:
            cotangent = jnp.zeros_like(out).at[:, i].set(mask.astype(out.dtype))
            return pull(cotangent)[0]

        return jax.lax.map(row, jnp.arange(out.shape[-1]))  # (out dim, source position, in dim)

    mask_device = jnp.asarray(mask)
    total = None
    for x in xs:
        grads = np.asarray(per_sequence(jnp.asarray(x, dtype=jnp.float32), mask_device))
        rows = grads[:, valid, :].astype(np.float64).mean(axis=1)
        total = rows if total is None else total + rows
    return (total / n).astype(np.float32)


def code_identity() -> str:
    """The git blob hash of this module's source file, as ``git hash-object`` prints it (D18)."""
    data = Path(__file__).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def jacobian_key(*, checkpoint_sha256: str, fit_sha256: str, layer: int, code: str) -> str:
    return cache_key({"checkpoint": checkpoint_sha256, "fit_sample": fit_sha256, "layer": str(layer)}, code)


def _path(root: str | Path, key: str) -> Path:
    _check_key(key)
    return Path(root) / f"{key}.npy"


def save_jacobian(root: str | Path, key: str, J: np.ndarray) -> Path:
    """Store J under ``key`` and return its path; a key never changes meaning."""
    J = np.asarray(J)
    if J.dtype != np.float32 or J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"J must be a square float32 matrix, got {J.dtype} {J.shape}")
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(J), allow_pickle=False)
    data = buffer.getvalue()
    path = _path(root, key)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} holds a different Jacobian; a cache key never changes meaning")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, data)
    return path


def load_jacobian(root: str | Path, key: str) -> np.ndarray | None:
    path = _path(root, key)
    if not path.exists():
        return None
    J = np.load(path, allow_pickle=False)
    if J.dtype != np.float32:
        raise ValueError(f"{path} holds {J.dtype}, not float32")
    return J
