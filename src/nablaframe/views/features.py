"""The features view: one site of a dictionary, its latents on the probe and on held-out windows (D45).

Spec: docs/specs/phase-5.md, D44 and D45; contract: docs/deltas/phase-5-delta-5.md. Every number
here is computed from the model, the dictionary and the windows given, except the meters, which
are copied from a gate report and cite it (D38, D44).

* ``SHIPPED = ("topk", "staircase", "transcoder")``: the kinds a features view is built for. The
  Jacobian SAE is recorded, not shipped (D43).
* ``TOP_CONTEXTS = 8`` and ``EFFECT_TOKENS = 5``.
* ``EFFECT_LABEL = "Direct effect on the logits through the unembedding: the latent's decoder row,
  times the site's scale, centred and multiplied by the final layer norm's gain, then unembedded,
  with the layer norm's divisor held at 1. It is not a name for the latent."``
* ``METER_GATES = ("G5.2", "G5.3", "G5.5")``: the gates whose reports record dictionaries' meters.
* ``features_payload(model, params, dictionary, site, probe, held_out, *, step, meters,
  variables=None, chunk=256)``: the payload for site ``site`` of ``dictionary``, on the model with
  ``params``, the parameters at checkpoint ``step``. ``probe`` and ``held_out`` are the tokens the
  model reads, (Batch, Pos) and (N, T), integers.

  The codes are ``core.encode(dictionary, site, x)`` of the site's input tap, from
  ``measure.activations(model, params, tokens, taps, chunk=chunk)``, at every position. A latent is
  active at a position where its code is above 0. Held-out positions are numbered window-major:
  window, then position. The site's latents are ``0`` to ``dictionary.active_width(site) - 1``;
  the per-latent lists below hold one entry for each, in order. The payload, plain JSON:

  - ``"step"``, ``"dictionary"`` (``core.content_hash(dictionary)``), ``"kind"``, ``"site"``,
    ``"taps"`` (``[input tap, output tap]``), ``"width"`` (the site's active width) and ``"k"``.
  - ``"probe"``: ``{"latents", "values"}``, nested lists (Batch, Pos, ·): at each probe position,
    its active latents, largest code first, ties to the lower latent, and their codes.
  - ``"held_out"``: ``held_out`` as nested lists of ints, and ``"variables"``: ``variables`` as
    ``{name: list of ints}``, ``{}`` without it.
  - ``"frequency"``: per latent, the share of held-out positions at which it is active.
  - ``"contexts"``: per latent, up to ``TOP_CONTEXTS`` ``[window, position, code]``: the held-out
    positions with its largest codes above 0, largest first, ties to the lower window and then the
    lower position. Empty for a latent that never fires.
  - ``"effect"``: ``{"label": EFFECT_LABEL, "top", "bottom"}``. Per latent, ``top`` holds the
    ``EFFECT_TOKENS`` tokens with the largest effect, largest first, and ``bottom`` the smallest,
    smallest first, each as ``[token, effect]``, ties to the lower token. Latent ``i``'s effect, in
    float64: ``e = decode_delta(dictionary, site, one-hot i)``, the change the latent writes at the
    output tap per unit of code; then ``((e - mean(e)) * ln_f.weight) @ wte.T``, ``wte`` being the
    unembedding. Both output taps, ``hook_resid_post`` and ``hook_mlp_out``, add into the
    residual stream the final layer norm reads, so the same formula serves every shipped kind.
  - ``"meters"``: ``meters``, copied: ``{"gate", "path", "values"}``, the dictionary's meters at
    ``path`` in that gate's report's measurements.
  - ``"meter"``: ``{"kind": "reconstruction", "fvu", "positions"}``: ``measure.fvu`` of the site
    over every probe position, from its input tap to its output tap, and the number of positions.

  Codes and effects are written as plain floats; every code is the float32 value ``encode``
  returns. Two calls with the same inputs return equal payloads.

  ValueError, naming the problem, for: a kind not in ``SHIPPED``; a site the dictionary doesn't
  have; a ``probe`` or ``held_out`` that isn't a 2-D integer array with at least one row and one
  position, or holds a token outside the vocabulary (``params.wte``'s rows); a ``step`` that isn't
  a non-negative int; ``meters`` that aren't exactly ``{"gate", "path", "values"}``, with ``gate``
  in ``METER_GATES``, ``path`` a non-empty list of strings and ints, and ``values`` a mapping
  from strings to finite numbers or None; ``variables`` that aren't a mapping from non-empty
  strings to 1-D integer arrays of one value per held-out window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..dictionaries import core, measure

SHIPPED = ("topk", "staircase", "transcoder")
TOP_CONTEXTS = 8
EFFECT_TOKENS = 5
EFFECT_LABEL = ("Direct effect on the logits through the unembedding: the latent's decoder row, times the site's "
                "scale, centred and multiplied by the final layer norm's gain, then unembedded, with the layer "
                "norm's divisor held at 1. It is not a name for the latent.")
METER_GATES = ("G5.2", "G5.3", "G5.5")


_ROWS = 65536  # rows per call into the dictionary's kernels: bounds memory, and each row is independent
_BLOCK = 1024  # latents per block when computing direct effects


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _tokens(name: str, value: Any, vocab: int) -> np.ndarray:
    """``value`` as a (rows, positions) int32 array of tokens inside the vocabulary, or ValueError."""
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a 2-D integer array with at least one row and one position, "
                         f"got {array.dtype}{list(array.shape)}")
    if array.min() < 0 or array.max() >= vocab:
        raise ValueError(f"{name} holds a token outside the vocabulary [0, {vocab})")
    return array.astype(np.int32)


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"a meter must be a finite number or None, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"a meter must be finite or None, got {value!r}")
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def _checked_meters(meters: Any) -> dict[str, Any]:
    """``meters`` as plain Python values in the payload's form, or ValueError."""
    if not isinstance(meters, Mapping) or set(meters) != {"gate", "path", "values"}:
        raise ValueError("meters must be a mapping with exactly the keys 'gate', 'path' and 'values'")
    gate, path, values = meters["gate"], meters["path"], meters["values"]
    if gate not in METER_GATES:
        raise ValueError(f"meters' gate must be one of {METER_GATES}, got {gate!r}")
    if isinstance(path, str | bytes | Mapping) or not isinstance(path, Sequence) or len(path) < 1:
        raise ValueError("meters' path must be a non-empty list of strings and ints")
    clean_path = []
    for part in path:
        if isinstance(part, str):
            clean_path.append(part)
        elif _is_int(part):
            clean_path.append(int(part))
        else:
            raise ValueError(f"meters' path holds {part!r}, which is neither a string nor an int")
    if not isinstance(values, Mapping):
        raise ValueError("meters' values must be a mapping from strings to finite numbers or None")
    clean_values = {}
    for name, value in values.items():
        if not isinstance(name, str):
            raise ValueError(f"meters' values must be named by strings, got {name!r}")
        clean_values[name] = _number(value)
    return {"gate": gate, "path": clean_path, "values": clean_values}


def _checked_variables(variables: Any, n_windows: int) -> dict[str, list[int]]:
    if variables is None:
        return {}
    if not isinstance(variables, Mapping):
        raise ValueError("variables must be a mapping from names to one integer per held-out window")
    clean = {}
    for name, value in variables.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"a variable's name must be a non-empty string, got {name!r}")
        array = np.asarray(value)
        if array.dtype.kind not in "iu" or array.ndim != 1 or len(array) != n_windows:
            raise ValueError(f"variable {name!r} must be {n_windows} integers, one per held-out window, "
                             f"got {array.dtype}{list(array.shape)}")
        clean[name] = array.astype(np.int64).tolist()
    return clean


def _encode(dictionary: Any, site: int, x: np.ndarray, width: int) -> list[np.ndarray]:
    """The site's codes for the rows of ``x``, (rows, width) arrays of at most ``_ROWS`` rows, from ``core.encode``."""
    return [core.encode(dictionary, site, x[start:start + _ROWS])[:, :width] for start in range(0, len(x), _ROWS)]


def _sorted_active(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Active entries of ``codes`` as ``(rows, latents, values)``, ordered by row, then largest code, then lower latent."""
    rows, latents = np.nonzero(codes > 0)
    values = codes[rows, latents]
    order = np.lexsort((latents, -values, rows))
    return rows[order], latents[order], values[order]


def _probe_codes(codes: np.ndarray, shape: tuple[int, int]) -> dict[str, list]:
    rows, latents, values = _sorted_active(codes)
    ends = np.cumsum(np.bincount(rows, minlength=len(codes)))
    starts = ends - np.bincount(rows, minlength=len(codes))
    out_latents: list = []
    out_values: list = []
    for b in range(shape[0]):
        row_latents, row_values = [], []
        for pos in range(shape[1]):
            r = b * shape[1] + pos
            row_latents.append(latents[starts[r]:ends[r]].tolist())
            row_values.append(values[starts[r]:ends[r]].tolist())
        out_latents.append(row_latents)
        out_values.append(row_values)
    return {"latents": out_latents, "values": out_values}


def _held_out_statistics(codes_by_chunk: Sequence[np.ndarray], width: int, n_positions: int
                         ) -> tuple[list[float], list[list[list]]]:
    """Each latent's share of positions at which it is active, and its ``TOP_CONTEXTS`` largest codes."""
    counts = np.zeros(width, dtype=np.int64)
    rows_all, latents_all, values_all = [], [], []
    offset = 0
    for codes in codes_by_chunk:
        rows, latents = np.nonzero(codes > 0)
        counts += np.bincount(latents, minlength=width)
        rows_all.append(rows + offset)
        latents_all.append(latents)
        values_all.append(codes[rows, latents])
        offset += len(codes)
    rows, latents, values = (np.concatenate(a) for a in (rows_all, latents_all, values_all))
    order = np.lexsort((rows, -values, latents))  # by latent, then largest code, then lower window-major position
    rows, latents, values = rows[order], latents[order], values[order]
    starts = np.searchsorted(latents, np.arange(width), side="left")
    ends = np.searchsorted(latents, np.arange(width), side="right")
    contexts = []
    for i in range(width):
        stop = min(ends[i], starts[i] + TOP_CONTEXTS)
        chosen = rows[starts[i]:stop]
        contexts.append([[w, p, v] for w, p, v in zip((chosen // n_positions).tolist(), (chosen % n_positions).tolist(),
                                                      values[starts[i]:stop].tolist())])
    return (counts / offset).tolist(), contexts


def _effects(dictionary: Any, site: int, params: Any, width: int) -> dict[str, Any]:
    """Each of the site's latents' largest and smallest direct effects on the logits (see ``EFFECT_LABEL``)."""
    gain = np.asarray(params.ln_f.weight, dtype=np.float64)
    wte = np.asarray(params.wte, dtype=np.float64)
    if gain.ndim != 1 or wte.ndim != 2 or wte.shape[1] != len(gain) or len(gain) != dictionary.d_out:
        raise ValueError(f"the dictionary's output width {dictionary.d_out} must match the model's "
                         f"{wte.shape[-1]} features")
    m = dictionary.settings.width
    top, bottom = [], []
    for start in range(0, width, _BLOCK):
        n = min(_BLOCK, width - start)
        one_hot = np.zeros((n, m), dtype=np.float32)
        one_hot[np.arange(n), start + np.arange(n)] = 1.0
        e = core.decode_delta(dictionary, site, one_hot).astype(np.float64)
        e -= e.mean(axis=1, keepdims=True)
        effect = (e * gain) @ wte.T  # (n, vocab)
        for choice, kept in ((np.argsort(-effect, axis=1, kind="stable")[:, :EFFECT_TOKENS], top),
                             (np.argsort(effect, axis=1, kind="stable")[:, :EFFECT_TOKENS], bottom)):
            values = np.take_along_axis(effect, choice, axis=1)
            kept.extend([[t, v] for t, v in zip(row.tolist(), vals.tolist())] for row, vals in zip(choice, values))
    return {"label": EFFECT_LABEL, "top": top, "bottom": bottom}


def features_payload(model: Any, params: Any, dictionary: Any, site: int, probe: Any, held_out: Any, *, step: int,
                     meters: Mapping[str, Any], variables: Mapping[str, Any] | None = None,
                     chunk: int = 256) -> dict[str, Any]:
    if not isinstance(dictionary, core.Dictionary):
        raise ValueError(f"dictionary must be a Dictionary, got {type(dictionary).__name__}")
    kind = dictionary.settings.kind
    if kind not in SHIPPED:
        raise ValueError(f"a features view is built for {SHIPPED}, not for {kind!r}")
    site = dictionary._site(site)  # ValueError for a site the dictionary doesn't have
    if not _is_int(step) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    vocab = int(np.shape(params.wte)[0])
    probe = _tokens("probe", probe, vocab)
    held_out = _tokens("held_out", held_out, vocab)
    meters = _checked_meters(meters)
    variables = _checked_variables(variables, held_out.shape[0])
    width = dictionary.active_width(site)
    taps = list(dictionary.sites[site])
    in_tap, out_tap = taps

    seen = measure.activations(model, params, probe, taps, chunk=chunk)
    probe_x = seen[in_tap].reshape(-1, dictionary.d_in)
    probe_y = seen[out_tap].reshape(-1, dictionary.d_out)
    probe_codes = np.concatenate(_encode(dictionary, site, probe_x, width), axis=0)
    meter = {"kind": "reconstruction", "fvu": measure.fvu(dictionary, site, probe_x, probe_y),
             "positions": int(probe.size)}

    held = measure.activations(model, params, held_out, [in_tap], chunk=chunk)[in_tap].reshape(-1, dictionary.d_in)
    frequency, contexts = _held_out_statistics(_encode(dictionary, site, held, width), width, held_out.shape[1])

    return {
        "step": int(step),
        "dictionary": core.content_hash(dictionary),
        "kind": kind,
        "site": site,
        "taps": taps,
        "width": width,
        "k": int(dictionary.settings.k),
        "probe": _probe_codes(probe_codes, probe.shape),
        "held_out": held_out.tolist(),
        "variables": variables,
        "frequency": frequency,
        "contexts": contexts,
        "effect": _effects(dictionary, site, params, width),
        "meters": meters,
        "meter": meter,
    }
