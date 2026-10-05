"""Applying an intervention spec to a forward pass (D23, D25, D26).

Spec: docs/specs/phase-3.md. JAX is imported inside the functions that need it, so this
module imports without it.

* ``editable_taps(n_layers)``: ``tap_names(n_layers)`` without the ``attn.hook_pattern``
  taps, whose axes are queries and keys rather than positions (D23).
* ``swap_component(params, other, component)``: ``params`` with one component replaced by
  ``other``'s: ``embeddings`` (``wte`` and ``wpe``), ``ln_final`` (``ln_f``), ``blocks.<l>``,
  ``blocks.<l>.attn`` or ``blocks.<l>.mlp``. ValueError for any other name or layer.
* ``run_intervened(model, params, tokens, spec, *, n_layers, resolve, params_at=None,
  taps=())``: ``(logits, recorded)`` from ``model.instrumented`` with the spec applied.

  - Swaps apply first, in order, each taking its component from ``params_at(step)``.
  - Every other edit applies at its tap, through the model's intervention hook, in order,
    to the positions it selects along the Pos axis; a tap outside ``editable_taps`` raises
    ValueError before anything runs.
  - ``zero`` writes zeros. ``add`` adds ``alpha`` times the vector. ``patch`` writes the
    source batch's value at that tap and those positions. ``mean`` writes the reference
    batch's mean at that tap, per position, taken in float64 and cast back. ``resample``
    writes, for each sequence, the value of a reference window drawn with
    ``numpy.random.default_rng(seed).integers(0, len(reference), size=batch)``.
  - Values for ``patch``, ``mean`` and ``resample`` come from ``model.instrumented`` on the
    source or reference tokens with the post-swap parameters, requesting only that tap.
  - ``resolve(hash)`` returns the array a hash names; ``run_intervened`` recomputes the hash
    (``batch_hash`` for token batches, ``tensor_sha256`` for vectors) and raises ValueError
    if it differs.
  - ``recorded`` holds the requested taps' values after the edits.
* ``code_identity()``: this module's git blob hash (D18, D24).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..trace.registry import tap_names
from ..trace.schema import tensor_sha256
from .spec import InterventionSpec, position_index, spec_from_json, spec_json

_COMPONENT = re.compile(r"embeddings|ln_final|blocks\.(0|[1-9][0-9]*)(\.attn|\.mlp)?")
_TOKEN_ARGS = {"patch": "source", "mean": "reference", "resample": "reference"}


def editable_taps(n_layers: int) -> tuple[str, ...]:
    return tuple(name for name in tap_names(n_layers) if not name.endswith(".attn.hook_pattern"))


def _component(component: Any, n_layers: int) -> tuple[str, int | None, str | None]:
    """``(kind, layer, part)`` for a swappable component name, or ValueError."""
    match = _COMPONENT.fullmatch(component) if isinstance(component, str) else None
    if match is None:
        raise ValueError(f"not a component (embeddings, ln_final, blocks.<l>[.attn|.mlp]): {component!r}")
    if match.group(1) is None:
        return component, None, None
    layer = int(match.group(1))
    if layer >= n_layers:
        raise ValueError(f"{component}: a {n_layers}-layer model has no block {layer}")
    return "blocks", layer, (match.group(2) or "")[1:] or None


def swap_component(params: Any, other: Any, component: str) -> Any:
    kind, layer, part = _component(component, min(len(params.blocks), len(other.blocks)))
    if kind == "embeddings":
        return dataclasses.replace(params, wte=other.wte, wpe=other.wpe)
    if kind == "ln_final":
        return dataclasses.replace(params, ln_f=other.ln_f)
    block = other.blocks[layer] if part is None else dataclasses.replace(
        params.blocks[layer], **{part: getattr(other.blocks[layer], part)})
    blocks = params.blocks[:layer] + (block,) + params.blocks[layer + 1:]
    return dataclasses.replace(params, blocks=blocks)


def _resolved(resolve: Callable[[str], np.ndarray], digest: str, hasher: Callable[[np.ndarray], str]) -> np.ndarray:
    value = np.asarray(resolve(digest))
    if hasher(value) != digest:
        raise ValueError(f"resolve({digest[:12]}...) returned an array whose hash is {hasher(value)[:12]}...")
    return value


def _tap_value(model: Any, params: Any, tokens: np.ndarray, tap: str) -> np.ndarray:
    _, recorded = model.instrumented(params, tokens, [tap])
    value = recorded[tap]
    return np.asarray(getattr(value, "array", value))


def run_intervened(model: Any, params: Any, tokens: np.ndarray, spec: InterventionSpec, *, n_layers: int,
                   resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None = None,
                   taps: Sequence[str] = ()) -> tuple[Any, dict[str, Any]]:
    edits = spec_from_json(spec_json(spec)).edits  # validates hand-built edits too
    if np.ndim(tokens) != 2:
        raise ValueError(f"tokens must be (batch, pos), got shape {np.shape(tokens)}")
    batch, seq = np.shape(tokens)

    # Everything is checked before anything runs.
    editable = set(editable_taps(n_layers))
    selected: list[np.ndarray | None] = []
    for e in edits:
        if e.op == "swap":
            _component(e.tap, n_layers)
            if params_at is None:
                raise ValueError(f"swapping {e.tap} needs params_at")
            selected.append(None)
        elif e.tap not in editable:
            raise ValueError(f"{e.tap!r} is not an editable tap for a {n_layers}-layer model (D23)")
        else:
            selected.append(position_index(e.positions, seq))
    resolved: dict[str, np.ndarray] = {}
    for e in edits:
        if e.op in _TOKEN_ARGS:
            digest = e.args[_TOKEN_ARGS[e.op]]
            other = resolved.setdefault(digest, _resolved(resolve, digest, batch_hash))
            if np.ndim(other) != 2 or (e.op == "patch" and np.shape(other)[0] != batch) or np.shape(other)[1] != seq:
                raise ValueError(f"{e.op} at {e.tap}: its {_TOKEN_ARGS[e.op]} batch is shaped {np.shape(other)},"
                                 f" which doesn't line up with tokens shaped {np.shape(tokens)}")
        elif e.op == "add":
            resolved.setdefault(e.args["vector"], _resolved(resolve, e.args["vector"], tensor_sha256))

    for e in edits:
        if e.op == "swap":
            params = swap_component(params, params_at(e.args["step"]), e.tap)
    if all(e.op == "swap" for e in edits):
        return model.instrumented(params, tokens, taps)

    # Each edit's operand: zeros, the scaled vector, or the source or reference values at its tap, per sequence.
    operations: list[tuple[str, str, np.ndarray, Any]] = []
    for e, index in zip(edits, selected):
        if e.op == "swap":
            continue
        mask = np.zeros(seq, dtype=bool)
        mask[index] = True
        if e.op == "zero":
            operand = None
        elif e.op == "add":
            vector = resolved[e.args["vector"]]
            operand = (np.float64(e.args["alpha"]) * vector.astype(np.float64)).astype(np.float32)
        elif e.op == "patch":
            operand = _tap_value(model, params, resolved[e.args["source"]], e.tap)
        else:
            reference = _tap_value(model, params, resolved[e.args["reference"]], e.tap)
            if e.op == "mean":
                operand = reference.astype(np.float64).mean(axis=0).astype(reference.dtype)[None]
            else:
                rows = np.random.default_rng(e.args["seed"]).integers(0, len(reference), size=batch)
                operand = reference[rows]
        operations.append((e.tap, e.op, mask, operand))

    import jax
    import jax.numpy as jnp

    def forward(params: Any, tokens: Any, operands: list[Any]) -> tuple[Any, dict[str, Any]]:
        # Operands are arguments, not constants closed over: baked in as constants, XLA folds
        # the layer norm after a full patch and the patched logits stop matching the source's bit for bit.
        hooks: dict[str, list[Callable[[Any], Any]]] = {}
        for (tap, op, mask, _), operand in zip(operations, operands):
            hooks.setdefault(tap, []).append(_hook(jnp, op, mask, operand))

        def chain(fns: list[Callable[[Any], Any]]) -> Callable[[Any], Any]:
            def apply(value: Any) -> Any:
                for fn in fns:
                    value = fn(value)
                return value
            return apply

        return model.instrumented(params, tokens, taps, {tap: chain(fns) for tap, fns in hooks.items()})

    return jax.jit(forward)(params, tokens, [operand for *_, operand in operations])


def _hook(jnp: Any, op: str, mask: np.ndarray, operand: Any) -> Callable[[Any], Any]:
    """The edit as a pure function of the tap's value, whose axes are (Batch, Pos, ...)."""

    def apply(value: Any) -> Any:
        where = mask.reshape((1, -1) + (1,) * (value.ndim - 2))
        if op == "zero":
            return jnp.where(where, jnp.zeros((), value.dtype), value)
        expected = value.shape[2:] if op == "add" else ((1,) if op == "mean" else value.shape[:1]) + value.shape[1:]
        if tuple(operand.shape) != tuple(expected) or operand.dtype != value.dtype:
            raise ValueError(f"{op}: operand is {operand.dtype}{list(operand.shape)}, the tap needs"
                             f" {value.dtype}{list(expected)}")
        if op == "add":
            return jnp.where(where, value + operand, value)
        return jnp.where(where, operand, value)

    return apply


def code_identity() -> str:
    data = Path(__file__).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
