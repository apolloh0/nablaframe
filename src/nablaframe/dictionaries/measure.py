"""Measuring a dictionary on a model: its meters, its selectivity and its patches (D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo". JAX is imported inside the
functions that need it, so this module imports without it. Nothing here caches or judges: the
gates choose windows, sites and variables, and hold the results to the bars.

* ``activations(model, params, tokens, taps, *, chunk=256)``: each tap's values on the
  unintervened pass, as float32 numpy arrays shaped like the tap, (batch, pos, ...).
  ``model.instrumented`` runs on ``chunk`` sequences at a time, in order, the last chunk
  possibly shorter. ValueError for a ``chunk`` that isn't a positive int.
* ``fvu(dictionary, site, x, y)``: the fraction of variance unexplained, ``Σ(y − ŷ)² / Σ(y − ȳ)²``
  with ``ŷ = reconstruct(dictionary, site, x)`` and ``ȳ`` the mean row of ``y``, summed in
  float64 over every row and feature, as a plain float. ``x`` and ``y`` are (N, d) with one N.
  ValueError for rows that don't line up, or a ``y`` with no variance.
* ``dead_share(dictionary, site, x)``: the share of the site's ``active_width`` latents that
  fire on no row of ``x``, as a plain float.
* ``readout_loss(logits, windows, positions)``: ``(loss, accuracy)``, plain floats, over every
  window and every position ``p`` in ``positions``, the target being ``windows[:, p + 1]``. The
  loss is the mean negative log probability of the target in nats, from a log-softmax in
  float64; the accuracy is how often the argmax is the target, ties to the lower id. ``logits``
  is (batch, pos, vocab) and ``windows`` (batch, at least max(positions) + 2). ValueError
  otherwise, or for an empty ``positions``.
* ``spliced_logits(model, params, tokens, dictionary, site)``: the logits, float32 (batch, pos,
  vocab), of the pass in which the site's output tap holds, at every position,
  ``reconstruct`` of the site's input tap's unintervened value. The replacement enters the
  compiled program as an argument (PR #38).
* ``ablated_logits(model, params, tokens, tap, *, n_layers)``: the logits, float32, of
  ``run_intervened`` with one ``mean`` edit at ``tap``, every position, whose reference batch is
  ``tokens`` itself: a per-position mean ablation over the windows being measured.
* ``f1_table(codes, values, n_values, *, min_count=5)``: ``(f1, value_ids)``. ``codes`` is (N,
  m) and ``values`` (N,) ints in ``[0, n_values)``, one per row. ``value_ids`` holds, ascending,
  the values that occur at least ``min_count`` times. ``f1`` is float64 (m, len(value_ids)):
  for latent ``i`` and value ``v``, ``2·|i active ∧ value = v| / (|i active| + |value = v|)``,
  a latent being active where its code is above zero. ValueError for mismatched or
  out-of-range inputs, or if no value occurs ``min_count`` times.
* ``mean_best_f1(f1)``: the mean over values (columns) of the best latent's F1, as a plain float.
* ``assigned(tables, name, *, floor=0.5)``: the latents assigned to the variable ``name``, as a
  sorted list of ints. ``tables`` maps each variable at one position to its ``f1`` from
  ``f1_table``, every table with the same number of latents. A latent is assigned to the
  variable holding its best F1 over every variable and value, the first variable in ``tables``'s
  order on a tie, if that F1 is at least ``floor``. ValueError for a ``name`` not in ``tables``.
* ``probe_accuracy(features, values, n_values, *, ridge=1e-2)``: how well a linear probe reads
  the variable, as a plain float. In float64: rows at even indices fit, rows at odd indices
  test. Features are centred on the fitting rows' mean. With ``G`` the fitting rows' Gram
  matrix, the weights solve ``(G + ridge · max(trace G, 1e-12) / n_features · I) W = Xᵀ Y``,
  ``Y`` the one-hot values. The accuracy is how often the argmax of ``X W`` on a test row is its
  value, ties to the lower id. ValueError for fewer than two rows or mismatched inputs.
* ``patch_predictions(model, params, tokens, sources, *, tap, positions, read, n_layers,
  dictionary=None, latents=None)``: each pass's argmax at position ``read``, int64 (batch,),
  ties to the lower id, keyed by pass, every pass through ``run_intervened``:

  - ``"residual"``: one ``patch`` edit at ``tap`` and ``positions`` from ``sources``
  - ``"code"``, with a ``dictionary``: one ``dpatch`` edit there instead
  - ``"latents"``, with a ``dictionary`` and ``latents``, a mapping from each position in
    ``positions`` to its latent ids: one ``dpatch`` edit per position, in ``positions``'s order,
    each at that position alone with that position's ``latents``

  ``resolve`` holds only ``tokens``, ``sources`` and the dictionary. ValueError for ``latents``
  without a dictionary, or one whose keys aren't exactly ``positions``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from .core import Dictionary, content_hash, encode, reconstruct

_ROWS = 65536  # rows per call into the dictionary's kernels: bounds memory, and each row is independent


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _matrix(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    return array


def _ids(name: str, value: Any, n_rows: int, n_values: int) -> np.ndarray:
    """``value`` as one int64 id per row, each in ``[0, n_values)``, or ValueError."""
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"{name} must be {n_rows} integer ids, got {array.dtype}{list(array.shape)}")
    if not _is_int(n_values) or n_values < 1:
        raise ValueError(f"n_values must be a positive int, not {n_values!r}")
    if len(array) and (array.min() < 0 or array.max() >= n_values):
        raise ValueError(f"{name} must lie in [0, {n_values})")
    return array.astype(np.int64)


def _value(recorded: Any) -> np.ndarray:
    return np.asarray(getattr(recorded, "array", recorded))


def activations(model: Any, params: Any, tokens: Any, taps: Sequence[str], *, chunk: int = 256) -> dict[str, np.ndarray]:
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, not {chunk!r}")
    taps = list(dict.fromkeys(taps))
    if np.ndim(tokens) != 2 or len(tokens) < 1:
        raise ValueError(f"tokens must be (batch, pos) with at least one sequence, got shape {np.shape(tokens)}")
    parts: dict[str, list[np.ndarray]] = {tap: [] for tap in taps}
    for start in range(0, len(tokens), int(chunk)):
        _, recorded = model.instrumented(params, tokens[start:start + int(chunk)], taps)
        for tap in taps:
            parts[tap].append(_value(recorded[tap]).astype(np.float32))
    return {tap: np.concatenate(parts[tap], axis=0) for tap in taps}


def fvu(dictionary: Dictionary, site: int, x: Any, y: Any) -> float:
    x, y = _matrix("x", x), _matrix("y", y)
    if len(x) != len(y) or y.shape[1] != dictionary.d_out:
        raise ValueError(f"x {x.shape} and y {y.shape} must have one N, and y {dictionary.d_out} features")
    y = y.astype(np.float64)
    spread = float(((y - y.mean(axis=0)) ** 2).sum())
    if not spread > 0:
        raise ValueError("y has no variance, so there is nothing to explain")
    residual = 0.0
    for start in range(0, len(x), _ROWS):
        y_hat = reconstruct(dictionary, site, x[start:start + _ROWS]).astype(np.float64)
        residual += float(((y[start:start + _ROWS] - y_hat) ** 2).sum())
    return residual / spread


def dead_share(dictionary: Dictionary, site: int, x: Any) -> float:
    active = dictionary.active_width(site)
    fired = np.zeros(dictionary.settings.width, dtype=bool)
    x = _matrix("x", x)
    for start in range(0, len(x), _ROWS):
        fired |= (encode(dictionary, site, x[start:start + _ROWS]) > 0).any(axis=0)
    return float(active - int(fired[:active].sum())) / active


def readout_loss(logits: Any, windows: Any, positions: Sequence[int]) -> tuple[float, float]:
    logits, windows = np.asarray(logits), np.asarray(windows)
    if logits.ndim != 3 or windows.ndim != 2 or windows.dtype.kind not in "iu" or len(windows) != len(logits):
        raise ValueError(f"logits (batch, pos, vocab) and integer windows (batch, pos) must line up, got "
                         f"{logits.shape} and {windows.dtype}{list(windows.shape)}")
    positions = list(positions)
    if not positions or not all(_is_int(p) for p in positions):
        raise ValueError(f"positions must be a non-empty list of ints, not {positions!r}")
    if min(positions) < 0 or max(positions) >= logits.shape[1] or max(positions) + 2 > windows.shape[1]:
        raise ValueError(f"positions {positions} need logits to reach them and windows to hold a target after each:"
                         f" logits {logits.shape}, windows {windows.shape}")
    rows = logits[:, positions].astype(np.float64)  # (batch, len(positions), vocab)
    targets = windows[:, [p + 1 for p in positions]].astype(np.int64)
    if targets.min() < 0 or targets.max() >= rows.shape[2]:
        raise ValueError(f"a target lies outside the vocabulary of {rows.shape[2]}")
    peak = rows.max(axis=-1, keepdims=True)
    log_probs = rows - peak - np.log(np.exp(rows - peak).sum(axis=-1, keepdims=True))
    picked = np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    return float(-picked.mean()), float((rows.argmax(axis=-1) == targets).mean())


def spliced_logits(model: Any, params: Any, tokens: Any, dictionary: Dictionary, site: int) -> np.ndarray:
    import jax

    site = dictionary._site(site)
    reads, writes = dictionary.sites[site]
    _, recorded = model.instrumented(params, tokens, list(dict.fromkeys([reads, writes])))
    x, y = _value(recorded[reads]), _value(recorded[writes])
    batch, seq = np.shape(tokens)
    if x.shape != (batch, seq, dictionary.d_in) or y.shape != (batch, seq, dictionary.d_out):
        raise ValueError(f"{reads} is shaped {x.shape} and {writes} {y.shape}, not (batch, pos, the dictionary's"
                         f" width there) = {(batch, seq, dictionary.d_in)} and {(batch, seq, dictionary.d_out)}")
    replacement = reconstruct(dictionary, site, x.reshape(-1, dictionary.d_in)).reshape(y.shape).astype(y.dtype)

    def forward(params: Any, tokens: Any, replacement: Any) -> Any:
        # The replacement is an argument, not a constant closed over (PR #38): as a constant XLA would
        # fold what follows it and the logits would stop matching a pass that was given the same values.
        return model.instrumented(params, tokens, [], {writes: lambda _: replacement})[0]

    return np.asarray(jax.jit(forward)(params, tokens, replacement), dtype=np.float32)


def ablated_logits(model: Any, params: Any, tokens: Any, tap: str, *, n_layers: int) -> np.ndarray:
    from ..interventions.apply import run_intervened
    from ..interventions.spec import InterventionSpec, edit

    digest = batch_hash(tokens)
    spec = InterventionSpec((edit(tap, "all", "mean", reference=digest),))
    logits, _ = run_intervened(model, params, tokens, spec, n_layers=n_layers, resolve={digest: np.asarray(tokens)}.__getitem__)
    return np.asarray(logits, dtype=np.float32)


def f1_table(codes: Any, values: Any, n_values: int, *, min_count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    codes = _matrix("codes", codes)
    values = _ids("values", values, len(codes), n_values)
    if not _is_int(min_count) or min_count < 1:
        raise ValueError(f"min_count must be a positive int, not {min_count!r}")
    counts = np.bincount(values, minlength=n_values)
    value_ids = np.flatnonzero(counts >= min_count).astype(np.int64)
    if not len(value_ids):
        raise ValueError(f"no value occurs {min_count} times")
    column = np.full(n_values, -1)
    column[value_ids] = np.arange(len(value_ids))
    both = np.zeros((codes.shape[1], len(value_ids)))
    for start in range(0, len(codes), _ROWS):
        active = (codes[start:start + _ROWS] > 0).astype(np.float64)
        one_hot = np.zeros((len(active), len(value_ids)))
        where = column[values[start:start + _ROWS]]
        one_hot[np.flatnonzero(where >= 0), where[where >= 0]] = 1.0
        both += active.T @ one_hot
    n_active = (codes > 0).sum(axis=0).astype(np.float64)
    return 2.0 * both / (n_active[:, None] + counts[value_ids][None, :]), value_ids


def _f1(name: str, value: Any) -> np.ndarray:
    f1 = _matrix(name, value).astype(np.float64)
    if f1.shape[0] < 1 or f1.shape[1] < 1:
        raise ValueError(f"{name} needs at least one latent and one value, got shape {f1.shape}")
    return f1


def mean_best_f1(f1: Any) -> float:
    return float(_f1("f1", f1).max(axis=0).mean())


def assigned(tables: Mapping[str, Any], name: str, *, floor: float = 0.5) -> list[int]:
    if name not in tables:
        raise ValueError(f"{name!r} is not one of the variables {list(tables)}")
    if len({np.shape(table)[0] for table in tables.values()}) != 1:
        raise ValueError("every table must have the same number of latents")
    best = np.stack([_f1(f"tables[{variable!r}]", table).max(axis=1) for variable, table in tables.items()], axis=1)
    owner = best.argmax(axis=1)  # the first variable on a tie
    mine = (owner == list(tables).index(name)) & (best.max(axis=1) >= floor)
    return [int(i) for i in np.flatnonzero(mine)]


def probe_accuracy(features: Any, values: Any, n_values: int, *, ridge: float = 1e-2) -> float:
    x = _matrix("features", features).astype(np.float64)
    values = _ids("values", values, len(x), n_values)
    if len(x) < 2:
        raise ValueError("a probe needs at least two rows, one to fit and one to test")
    fit = np.arange(len(x)) % 2 == 0
    x = x - x[fit].mean(axis=0)
    gram = x[fit].T @ x[fit]
    penalty = ridge * max(np.trace(gram), 1e-12) / x.shape[1]
    weights = np.linalg.solve(gram + penalty * np.eye(x.shape[1]), x[fit].T @ np.eye(n_values)[values[fit]])
    return float(((x[~fit] @ weights).argmax(axis=1) == values[~fit]).mean())


def patch_predictions(model: Any, params: Any, tokens: Any, sources: Any, *, tap: str, positions: Sequence[int],
                      read: int, n_layers: int, dictionary: Dictionary | None = None,
                      latents: Mapping[int, Sequence[int]] | None = None) -> dict[str, np.ndarray]:
    from ..interventions.apply import run_intervened
    from ..interventions.spec import InterventionSpec, edit

    positions = [int(p) if _is_int(p) else p for p in positions]
    if latents is not None and dictionary is None:
        raise ValueError("latents need a dictionary")
    if latents is not None and set(latents) != set(positions):
        raise ValueError(f"latents must be keyed by exactly the positions {positions}, got {sorted(latents)}")
    tokens, sources = np.asarray(tokens), np.asarray(sources)
    if np.ndim(tokens) != 2 or not _is_int(read) or not 0 <= read < tokens.shape[1]:
        raise ValueError(f"read must be a position of tokens shaped {np.shape(tokens)}, not {read!r}")
    store: dict[str, Any] = {batch_hash(tokens): tokens, batch_hash(sources): sources}
    source = batch_hash(sources)
    edits = {"residual": [edit(tap, positions, "patch", source=source)]}
    if dictionary is not None:
        digest = content_hash(dictionary)
        store[digest] = dictionary
        edits["code"] = [edit(tap, positions, "dpatch", dictionary=digest, source=source)]
        if latents is not None:
            edits["latents"] = [edit(tap, [p], "dpatch", dictionary=digest, source=source,
                                     latents=[int(i) for i in latents[p]]) for p in positions]
    predictions = {}
    for name, spec_edits in edits.items():
        logits, _ = run_intervened(model, params, tokens, InterventionSpec(tuple(spec_edits)), n_layers=n_layers,
                                   resolve=store.__getitem__)
        predictions[name] = np.asarray(logits, dtype=np.float32)[:, read].argmax(axis=-1).astype(np.int64)
    return predictions
