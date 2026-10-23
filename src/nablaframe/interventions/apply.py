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
  - ``jswap`` applies at residual-stream taps only, ``hook_resid_pre``, ``hook_resid_mid`` or
    ``hook_resid_post``. J must be shaped (d_model, d_model), and both tokens must be below the
    vocabulary size. Anything else raises ValueError before anything runs. With u = Jᵀw for each
    token's unembedding row w, it adds (β − α)(u_a − u_b) at the selected positions, where
    α u_a + β u_b is the residual's least-squares component in their span. Its operand enters
    the compiled program as an argument, as every operand does. With a == b it adds no operation
    at all, so it reproduces the unintervened pass bit for bit (D25).
  - A ``zero`` with ``head`` applies only at ``attn.hook_z``. It zeroes that head's slice at the
    selected positions and leaves every other head untouched. A head at or beyond the tap's head
    count raises ValueError.
  - ``dpatch`` (D45) applies at one of its dictionary's output taps, which picks the site. For its
    ``dictionary``, ``resolve`` returns a ``Dictionary``, checked with ``content_hash``. Each of
    these raises ValueError before anything runs: any other tap, a transcoder's input tap among
    them; a resolved value that isn't a ``Dictionary`` or doesn't hash to its name; a ``source``
    batch shaped differently from the tokens; a latent at or beyond the site's ``active_width``.
    The codes ``f_target`` and ``f_source`` are ``encode`` of the target and source batches'
    values at the site's input tap, one row per sequence and position, from ``model.instrumented``
    with the post-swap parameters and none of the spec's other edits. At the selected positions it
    adds ``decode_delta`` of ``f_source − f_target``, with every latent not in ``latents`` zeroed
    when ``latents`` is given, so the target's error term is kept. A tap whose values aren't
    shaped (batch, pos, the dictionary's width there) raises ValueError. Where the change is zero
    at every selected position, as with a source equal to the target or an empty ``latents``, it
    adds no operation at all, so it reproduces the unintervened pass bit for bit (D25).
  - ``dsteer`` (D48) applies at one of its dictionary's output taps, which picks the site, as
    ``dpatch``'s does, and ``resolve`` returns its ``dictionary`` as for ``dpatch``. ``resolve``
    returns its three arrays too, each checked with ``array_hash``: ``remove`` and ``add``, int32
    arrays shaped (n, 3) whose rows are ``[sequence, position, latent]``, and ``amounts``, float32
    shaped (len(add),). Each of these raises ValueError before anything runs, beside ``dpatch``'s
    refusals of the tap and the dictionary: an array of another dtype or shape; a row whose
    sequence isn't in ``range(batch)``, whose position isn't one the edit selects, or whose latent
    isn't in ``range(active_width(site))``; a row repeated within ``remove`` or within ``add``; an
    amount that isn't finite. The codes ``f`` are ``encode`` of the target batch's values at the
    site's input tap, read as ``dpatch`` reads ``f_target``, and a tap whose values aren't shaped
    (batch, pos, the dictionary's width there) raises ValueError. The change starts at zero for
    every sequence, position and latent; each ``remove`` row subtracts ``f`` there, and each
    ``add`` row adds its amount there, so a latent both removed and added ends at its amount. At
    the selected positions it adds ``decode_delta`` of the change, so the target's error term is
    kept. Where the change is zero at every selected position, as with empty ``remove`` and
    ``add``, removing only latents whose code is zero, or adding only zero amounts, it adds no
    operation at all, so it reproduces the unintervened pass bit for bit (D25).
  - ``set`` (D55) applies at every editable tap whose values are shaped (batch, pos, d_model): every
    one but ``attn.hook_z``, which raises ValueError before anything runs. ``resolve`` returns its
    ``values``, checked with ``array_hash``: a float32 array shaped (batch, the number of positions
    the edit selects, d_model), ``d_model`` being ``params.wte``'s second dimension, with every
    value finite. Anything else raises ValueError before anything runs. At the ``i``-th position the
    edit selects, in ``position_index``'s order, each sequence's value at the tap becomes
    ``values[:, i]``. Its operand enters the compiled program as an argument. Where the edit selects
    no positions it adds no operation at all, and values equal to the tap's own values on the
    unintervened pass reproduce that pass bit for bit (D25).
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
from ..dictionaries.core import Dictionary, content_hash, decode_delta, encode
from ..trace.registry import tap_names
from ..trace.schema import tensor_sha256
from .spec import InterventionSpec, array_hash, position_index, spec_from_json, spec_json

_COMPONENT = re.compile(r"embeddings|ln_final|blocks\.(0|[1-9][0-9]*)(\.attn|\.mlp)?")
_TOKEN_ARGS = {"patch": "source", "mean": "reference", "resample": "reference"}
_RESIDUAL = ("hook_resid_pre", "hook_resid_mid", "hook_resid_post")


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
        elif e.op == "jswap" and e.tap.rsplit(".", 1)[-1] not in _RESIDUAL:
            raise ValueError(f"jswap applies at residual-stream taps ({', '.join(_RESIDUAL)}), not {e.tap!r}")
        elif "head" in e.args and not e.tap.endswith(".attn.hook_z"):
            raise ValueError(f"zeroing one head applies only at attn.hook_z, not {e.tap!r}")
        else:
            selected.append(position_index(e.positions, seq))
    resolved: dict[str, np.ndarray] = {}
    dictionaries: dict[str, Dictionary] = {}
    for e, index in zip(edits, selected):
        if e.op in _TOKEN_ARGS:
            digest = e.args[_TOKEN_ARGS[e.op]]
            other = resolved.setdefault(digest, _resolved(resolve, digest, batch_hash))
            if np.ndim(other) != 2 or (e.op == "patch" and np.shape(other)[0] != batch) or np.shape(other)[1] != seq:
                raise ValueError(f"{e.op} at {e.tap}: its {_TOKEN_ARGS[e.op]} batch is shaped {np.shape(other)},"
                                 f" which doesn't line up with tokens shaped {np.shape(tokens)}")
        elif e.op == "add":
            resolved.setdefault(e.args["vector"], _resolved(resolve, e.args["vector"], tensor_sha256))
        elif e.op == "jswap":
            vocab, d_model = np.shape(params.wte)
            J = resolved.setdefault(e.args["J"], _resolved(resolve, e.args["J"], tensor_sha256))
            if np.shape(J) != (d_model, d_model):
                raise ValueError(f"jswap at {e.tap}: J is shaped {np.shape(J)}, not (d_model, d_model) = "
                                 f"{(d_model, d_model)}")
            if max(e.args["a"], e.args["b"]) >= vocab:
                raise ValueError(f"jswap at {e.tap}: tokens {e.args['a']} and {e.args['b']} must be below the"
                                 f" vocabulary size {vocab}")
        elif e.op == "dpatch":
            d = dictionaries.setdefault(e.args["dictionary"], _dictionary(resolve, e.args["dictionary"]))
            site = _dpatch_site(d, e.tap)
            source = resolved.setdefault(e.args["source"], _resolved(resolve, e.args["source"], batch_hash))
            if np.shape(source) != np.shape(tokens):
                raise ValueError(f"dpatch at {e.tap}: its source batch is shaped {np.shape(source)}, not like the"
                                 f" tokens, {np.shape(tokens)}")
            latents = e.args.get("latents")
            if latents and latents[-1] >= d.active_width(site):  # stored sorted, so the last is the largest
                raise ValueError(f"dpatch at {e.tap}: latent {latents[-1]} is at or beyond the site's active width"
                                 f" {d.active_width(site)}")
        elif e.op == "dsteer":
            d = dictionaries.setdefault(e.args["dictionary"], _dictionary(resolve, e.args["dictionary"]))
            site = _dpatch_site(d, e.tap, "dsteer")
            arrays = {name: resolved.setdefault(e.args[name], _resolved(resolve, e.args[name], array_hash))
                      for name in ("remove", "add", "amounts")}
            _check_dsteer(e.tap, arrays, batch, index, d.active_width(site))
    heads = sorted({e.tap for e in edits if "head" in e.args})
    steered = [(e.tap, dictionaries[e.args["dictionary"]]) for e in edits if e.op == "dsteer"]
    if heads or steered:
        import jax

        shaped = heads + [name for tap, d in steered for name in (d.sites[_dpatch_site(d, tap)][0], tap)]
        shapes = jax.eval_shape(lambda p, t: model.instrumented(p, t, list(dict.fromkeys(shaped)))[1], params, tokens)
        for e in (e for e in edits if "head" in e.args):
            count = getattr(shapes[e.tap], "array", shapes[e.tap]).shape[2]  # (Batch, Pos, Head, HeadDim)
            if e.args["head"] >= count:
                raise ValueError(f"zero at {e.tap}: head {e.args['head']} is at or beyond the tap's {count} heads")
        for tap, d in steered:
            reads = d.sites[_dpatch_site(d, tap)][0]
            for name, width in ((reads, d.d_in), (tap, d.d_out)):
                shape = tuple(getattr(shapes[name], "array", shapes[name]).shape)
                if shape != (batch, seq, width):
                    raise ValueError(f"dsteer at {tap}: {name}'s values are shaped {shape}, not (batch, pos, the"
                                     f" dictionary's width there) = {(batch, seq, width)}")

    for e in edits:
        if e.op == "swap":
            params = swap_component(params, params_at(e.args["step"]), e.tap)

    # Each edit's operand: zeros, the scaled vector, the source or reference values at its tap, per
    # sequence, jswap's coefficient and direction differences, or dpatch's and dsteer's change in the output.
    operations: list[tuple[str, str, np.ndarray, int | None, Any]] = []
    for e, index in zip(edits, selected):
        if e.op == "swap" or (e.op == "jswap" and e.args["a"] == e.args["b"]):
            continue  # a self-swap adds no operation at all (D25)
        mask = np.zeros(seq, dtype=bool)
        mask[index] = True
        if e.op == "zero":
            operand = None
        elif e.op == "add":
            vector = resolved[e.args["vector"]]
            operand = (np.float64(e.args["alpha"]) * vector.astype(np.float64)).astype(np.float32)
        elif e.op == "jswap":
            operand = _jswap_operand(resolved[e.args["J"]], params.wte, e.args["a"], e.args["b"])
        elif e.op == "dpatch":
            operand = _dpatch_operand(model, params, tokens, resolved[e.args["source"]],
                                      dictionaries[e.args["dictionary"]], e.tap, e.args.get("latents"))
            if not np.any(operand[:, index]):
                continue  # no change at any selected position adds no operation at all (D25)
        elif e.op == "dsteer":
            d = dictionaries[e.args["dictionary"]]
            change = _dsteer_change(model, params, tokens, d, e.tap, resolved[e.args["remove"]],
                                    resolved[e.args["add"]], resolved[e.args["amounts"]])
            if not np.any(change[:, index]):
                continue  # no change at any selected position adds no operation at all (D25)
            site = _dpatch_site(d, e.tap)
            operand = decode_delta(d, site, change.reshape(-1, d.settings.width)).reshape(batch, seq, d.d_out)
        elif e.op == "patch":
            operand = _tap_value(model, params, resolved[e.args["source"]], e.tap)
        else:
            reference = _tap_value(model, params, resolved[e.args["reference"]], e.tap)
            if e.op == "mean":
                operand = reference.astype(np.float64).mean(axis=0).astype(reference.dtype)[None]
            else:
                rows = np.random.default_rng(e.args["seed"]).integers(0, len(reference), size=batch)
                operand = reference[rows]
        operations.append((e.tap, e.op, mask, e.args.get("head"), operand))
    if not operations:
        return model.instrumented(params, tokens, taps)

    import jax
    import jax.numpy as jnp

    def forward(params: Any, tokens: Any, operands: list[Any]) -> tuple[Any, dict[str, Any]]:
        # Operands are arguments, not constants closed over: baked in as constants, XLA folds
        # the layer norm after a full patch and the patched logits stop matching the source's bit for bit.
        hooks: dict[str, list[Callable[[Any], Any]]] = {}
        for (tap, op, mask, head, _), operand in zip(operations, operands):
            hooks.setdefault(tap, []).append(_hook(jnp, op, mask, head, operand))

        def chain(fns: list[Callable[[Any], Any]]) -> Callable[[Any], Any]:
            def apply(value: Any) -> Any:
                for fn in fns:
                    value = fn(value)
                return value
            return apply

        return model.instrumented(params, tokens, taps, {tap: chain(fns) for tap, fns in hooks.items()})

    return jax.jit(forward)(params, tokens, [operand for *_, operand in operations])


def _jswap_operand(J: np.ndarray, wte: Any, a: int, b: int) -> np.ndarray:
    """``[q, d]``, float32: ``r · q`` is β − α, the difference of r's least-squares coefficients
    on ``u_a`` and ``u_b``, and ``d = u_a − u_b``, with ``u = Jᵀw``. Computed in float64."""
    U = np.asarray(J, np.float64).T @ np.asarray(wte, np.float64)[[a, b]].T  # columns u_a, u_b
    P = np.linalg.pinv(U)  # rows map r to α and β
    return np.stack([P[1] - P[0], U[:, 0] - U[:, 1]]).astype(np.float32)


def _dictionary(resolve: Callable[[str], Any], digest: str) -> Dictionary:
    value = resolve(digest)
    if not isinstance(value, Dictionary):
        raise ValueError(f"resolve({digest[:12]}...) returned a {type(value).__name__}, not a Dictionary")
    if content_hash(value) != digest:
        raise ValueError(f"resolve({digest[:12]}...) returned a dictionary whose hash is {content_hash(value)[:12]}...")
    return value


def _dpatch_site(dictionary: Dictionary, tap: str, op: str = "dpatch") -> int:
    """The site whose output tap is ``tap``, or ValueError."""
    for site, (_, output) in enumerate(dictionary.sites):
        if output == tap:
            return site
    raise ValueError(f"{op} applies at its dictionary's output taps "
                     f"({', '.join(output for _, output in dictionary.sites)}), not {tap!r}")


def _dpatch_operand(model: Any, params: Any, tokens: np.ndarray, source: np.ndarray, dictionary: Dictionary,
                    tap: str, latents: list[int] | None) -> np.ndarray:
    """``decode_delta`` of ``f_source − f_target``, shaped like the tap's values, (batch, pos, d_out): the codes
    are read at the site's input tap with no other edits, and only ``latents`` change when given."""
    site = _dpatch_site(dictionary, tap)
    reads = dictionary.sites[site][0]
    batch, seq = np.shape(tokens)
    _, recorded = model.instrumented(params, tokens, list(dict.fromkeys([reads, tap])))
    x_target, output = (np.asarray(getattr(recorded[name], "array", recorded[name])) for name in (reads, tap))
    x_source = _tap_value(model, params, source, reads)
    for name, value, width in ((reads, x_target, dictionary.d_in), (reads, x_source, dictionary.d_in),
                               (tap, output, dictionary.d_out)):
        if value.shape != (batch, seq, width):
            raise ValueError(f"dpatch at {tap}: {name}'s values are shaped {value.shape}, not (batch, pos, the"
                             f" dictionary's width there) = {(batch, seq, width)}")
    f_target = encode(dictionary, site, x_target.reshape(-1, dictionary.d_in))
    f_source = encode(dictionary, site, x_source.reshape(-1, dictionary.d_in))
    change = f_source - f_target
    if latents is not None:
        others = np.ones(dictionary.settings.width, dtype=bool)
        others[latents] = False
        change[:, others] = 0.0
    return decode_delta(dictionary, site, change).reshape(batch, seq, dictionary.d_out)


def _check_dsteer(tap: str, arrays: dict[str, np.ndarray], batch: int, index: np.ndarray, width: int) -> None:
    """ValueError unless ``remove`` and ``add`` are int32 (n, 3) rows inside the batch, the selected positions and
    the site's active width, none repeated, and ``amounts`` is float32 (len(add),) and finite."""
    for name in ("remove", "add"):
        array = arrays[name]
        if array.dtype.name != "int32" or array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"dsteer at {tap}: {name} must be int32 rows [sequence, position, latent] shaped (n, 3),"
                             f" not {array.dtype.name}{list(array.shape)}")
        rows = array.astype(np.int64)
        for column, what, inside in ((0, "sequence", (rows[:, 0] >= 0) & (rows[:, 0] < batch)),
                                     (1, "position", np.isin(rows[:, 1], index)),
                                     (2, "latent", (rows[:, 2] >= 0) & (rows[:, 2] < width))):
            if not np.all(inside):
                bad = rows[np.argmin(inside)]
                allowed = {"sequence": f"range({batch})", "position": "the positions the edit selects",
                           "latent": f"range({width}), the site's active width"}[what]
                raise ValueError(f"dsteer at {tap}: {name} row {bad.tolist()} has a {what} outside {allowed}")
        if len(np.unique(rows, axis=0)) != len(rows):
            raise ValueError(f"dsteer at {tap}: {name} repeats a row")
    amounts, add = arrays["amounts"], arrays["add"]
    if amounts.dtype.name != "float32" or amounts.shape != (len(add),):
        raise ValueError(f"dsteer at {tap}: amounts must be float32 shaped (len(add),) = ({len(add)},), not"
                         f" {amounts.dtype.name}{list(amounts.shape)}")
    if not np.all(np.isfinite(amounts)):
        raise ValueError(f"dsteer at {tap}: every amount must be finite")


def _dsteer_change(model: Any, params: Any, tokens: np.ndarray, dictionary: Dictionary, tap: str, remove: np.ndarray,
                   add: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    """The change in the code, (batch, pos, width), float32: zero, minus ``f`` at each ``remove`` row, then plus its
    amount at each ``add`` row, with ``f`` read at the site's input tap with no other edits, as ``dpatch`` reads it."""
    site = _dpatch_site(dictionary, tap)
    batch, seq = np.shape(tokens)
    x = _tap_value(model, params, tokens, dictionary.sites[site][0])
    f = encode(dictionary, site, x.reshape(-1, dictionary.d_in)).reshape(batch, seq, dictionary.settings.width)
    change = np.zeros_like(f)
    s, p, l = remove.astype(np.int64).T  # no row repeats, so each index is written once
    change[s, p, l] -= f[s, p, l]
    s, p, l = add.astype(np.int64).T
    change[s, p, l] += amounts.astype(np.float32)
    return change


def _hook(jnp: Any, op: str, mask: np.ndarray, head: int | None, operand: Any) -> Callable[[Any], Any]:
    """The edit as a pure function of the tap's value, whose axes are (Batch, Pos, ...)."""

    def apply(value: Any) -> Any:
        where = mask.reshape((1, -1) + (1,) * (value.ndim - 2))
        if op == "zero":
            if head is not None:  # attn.hook_z: (Batch, Pos, Head, HeadDim)
                where = where & (np.arange(value.shape[2]) == head).reshape(1, 1, -1, 1)
            return jnp.where(where, jnp.zeros((), value.dtype), value)
        expected = {"add": value.shape[2:], "jswap": (2,) + value.shape[2:], "mean": (1,) + value.shape[1:]}.get(
            op, value.shape)
        if tuple(operand.shape) != tuple(expected) or operand.dtype != value.dtype:
            raise ValueError(f"{op}: operand is {operand.dtype}{list(operand.shape)}, the tap needs"
                             f" {value.dtype}{list(expected)}")
        if op in ("add", "dpatch", "dsteer"):
            return jnp.where(where, value + operand, value)
        if op == "jswap":
            q, d = operand[0], operand[1]
            return jnp.where(where, value + jnp.sum(value * q, axis=-1, keepdims=True) * d, value)
        return jnp.where(where, operand, value)

    return apply


def code_identity() -> str:
    data = Path(__file__).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
