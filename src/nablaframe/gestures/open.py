"""open-this-up (D51): which upstream parts make a target, ranked by absolute attribution, and how many reproduce it.

Spec: docs/specs/phase-6.md, D51. Built on ``sets.py``, with nothing specific to a task (D29). JAX is
imported inside the functions that need it, so this module imports without it. Nothing here caches
or judges: the gates ask for open-this-up, and a record holds what ``open_up`` returns (D47).

* ``PARTS = (1, 2, 4, 8, 16, 32, 64, 128)``, ``IG_STEPS = 5``, ``ERROR = 0.1``, ``NO_SPAN = 1e-6``,
  ``TOP_PARTS = 8`` and ``ORDERS = ("absolute", "signed", "random")``.
* ``Target``: a frozen dataclass ``(window, position, latents)``: a held-out window's index, the
  position read there, and the downstream latents whose pre-activations it sums, a tuple of ints.
* ``class_of(declaration, position)``: the class a position's parts are named at: the smallest class
  the run declares that holds the position, ties to the one declared first. ValueError where none
  does.
* ``pre_activation(dictionary, site, y, latents)``: float64 (n,), the summed pre-activations of
  ``latents`` for inputs ``y`` (n, d_in) at the site, before the top ``k`` are kept: with
  ``u = dictionary.unit(site)``, ``Σ_l ((y / scale_in[site] − b_pre[site]) · W_enc[u][:, l] +
  b_enc[site, l])``, in float64. ValueError for no latents, or a latent outside
  ``range(dictionary.active_width(site))``.
* ``evaluate(up, down, targets, codes, *, chunk=32)``: float64 (n,), each target's summed
  pre-activation, as ``pre_activation`` defines it at ``down``'s site for ``target.latents``, when
  ``up``'s output tap holds ``decode(codes) + err`` at every position. ``up`` and ``down`` are
  ``sets.SiteReading``s; ``codes`` is (n, seq, up.width), the upstream codes target ``k``'s pass
  uses, latents beyond ``up.width`` being zero; ``err`` is the upstream error term on the
  unintervened pass, the output tap's value less ``core.decode`` of ``up.codes``, held fixed as the
  edge meter holds it (D42). The downstream input tap is read at each target's position. The pass is
  ``model.instrumented`` with that one intervention, compiled with ``jax.jit`` with the codes, tokens
  and error as arguments and computed in float32, ``chunk`` targets at a time. ValueError, before
  anything runs, for readings ``open_up`` refuses, no targets, a target whose window, position or
  latents fall outside the windows, the sequence or ``down``'s active width, codes of another
  shape, or a ``chunk`` that isn't a positive int.
* ``attribute(up, down, targets, *, chunk=32)``: ``Attribution(codes, attributions, full, none)``:
  ``codes``, float32 (n, seq, up.width), ``up.codes`` of each target's window; ``attributions``,
  float64 (n, seq, up.width), integrated gradients from zero, ``codes`` times the mean over ``α`` in
  ``(s + 0.5) / IG_STEPS``, ``s`` in ``range(IG_STEPS)``, of the gradient of ``evaluate``'s
  pre-activation with respect to the codes at ``α · codes``, taken with ``jax.grad`` in float32 and
  averaged in float64, and zero at every position after the target's; ``full``, ``evaluate`` with
  ``codes``; ``none``, ``evaluate`` with every code zero. ValueError as ``evaluate`` raises it, codes aside.
* ``order(attributions, codes, kind, rng=None)``: one target's parts in order, an int64 array of
  ``position * width + latent``, for ``attributions`` and ``codes`` shaped (positions, width), the
  positions up to and including the target's: every part there, each once.

  - ``"absolute"``: by absolute attribution, largest first, ties to the lower position and then
    the lower latent
  - ``"signed"``: by attribution, largest first, with the same ties
  - ``"random"``: the parts whose code is above zero, ascending, in the order
    ``rng.permutation`` gives them; then every other part, ascending

  ValueError for another ``kind``, mismatched shapes, or ``"random"`` without ``rng``.
* ``open_up(up, down, cls, *, instances, seed, latent=None, description=None, value=None,
  chunk=32)``: D51 for one target, as a plain JSON mapping that ``records.GestureRecord`` takes as
  its results. The target is ``down``'s ``latent``, or the name ``description = value``'s set at the
  class, ``down.table(cls).set_for(description, value)``: exactly one of the two forms.

  - **Instances:** every (window, position), window by window and each window's positions in the
    class's order, where the target fires: the latent's code, or the set's summed codes, above zero
    in ``down.codes``, and for a name, ``description`` is ``value`` there. With
    ``rng = numpy.random.default_rng(seed)``, more than ``instances`` are cut to ``instances`` with
    ``sorted(rng.choice(len(candidates), size=instances, replace=False))``. Then each instance, in
    order, draws its random order's shuffle, ``rng.permutation`` of its parts whose code is above
    zero, ascending, as ``order(..., "random", rng)`` draws it.
  - ``attribute``, with ``chunk``, gives each instance's attributions, ``full`` and ``none``.
    Instances whose ``|full − none|`` is below ``NO_SPAN`` are left out of everything below, and
    counted.
  - **Each order, each instance:** its parts in that order (``order`` of its attributions and codes
    up to its position). Keeping the first ``k`` for each ``k`` in ``PARTS``, those parts' codes stay
    and every other upstream code, at every position, is zero; the error is
    ``|evaluate(kept) − full| / |full − none|``. Its reading is the fewest ``k`` whose error is at most
    ``ERROR``, None for "not within 128".

  The mapping:

  - ``"class"``; ``"target"``, ``{"latents", "latent", "description", "value"}``, the unused form's
    keys None; ``"up"`` and ``"down"``, each ``{"site", "tap", "block"}`` with its output tap
  - ``"candidates"`` (where the target fires, before the cut), ``"instances"`` (drawn), ``"no_span"``
    (left out) and ``"measured"``
  - ``"steps"`` (``IG_STEPS``) and ``"parts"`` (``PARTS`` as a list)
  - ``"windows"``: ``[window, position]`` for each measured instance
  - ``"orders"``: for each of ``ORDERS``, ``{"error_median", "readings", "reading_median",
    "within"}``: the median error over measured instances at each ``k``; each measured instance's
    reading; their median, a reading of None counted as larger than any ``k``, and None where the
    median is such a reading; and the share of measured instances with a reading. With no measured
    instance, each is None and ``readings`` empty.
  - ``"top_parts"``: for each measured instance, its first ``TOP_PARTS`` parts by absolute
    attribution, each ``{"position", "latent", "attribution", "name"}``. ``name`` is steps 1 and 2 of
    D49 for that latent at ``class_of(declaration, position)`` on ``up``: its top candidate,
    ``{"description", "value", "f1", "label"}``, labelled as ``name.NAMED`` or ``name.WEAK``; None
    where it has no candidate.

  ValueError unless both readings share their model, params and declaration, as the same objects,
  and have equal windows; unless ``up``'s output tap comes before ``down``'s input tap in
  ``trace.registry.tap_names`` order; for a class the run doesn't declare; for both target forms or
  neither, a latent outside ``range(down.width)``, or a name that doesn't count at the class; or for
  ``instances`` or ``seed`` that aren't a positive int and a non-negative int.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..dictionaries import core, measure
from ..trace.registry import tap_names
from . import name as naming
from . import sets

PARTS = (1, 2, 4, 8, 16, 32, 64, 128)
IG_STEPS = 5
ERROR = 0.1
NO_SPAN = 1e-6
TOP_PARTS = 8
ORDERS = ("absolute", "signed", "random")


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


@dataclass(frozen=True)
class Target:
    window: int
    position: int
    latents: tuple[int, ...]


@dataclass(frozen=True)
class Attribution:
    codes: np.ndarray
    attributions: np.ndarray
    full: np.ndarray
    none: np.ndarray


def class_of(declaration: Any, position: int) -> str:
    if not _is_int(position):
        raise ValueError(f"position must be an int, got {position!r}")
    best: tuple[str, int] | None = None
    for cls, positions in declaration.classes.items():
        held = [int(p) for p in positions]
        if int(position) in held and (best is None or len(held) < best[1]):  # strictly smaller: ties to the first
            best = (cls, len(held))
    if best is None:
        raise ValueError(f"run {declaration.run!r} declares no class holding position {position}")
    return best[0]


def _latents(dictionary: Any, site: int, latents: Any) -> list[int]:
    width = dictionary.active_width(site)
    latents = list(latents)
    if not latents or not all(_is_int(i) and 0 <= i < width for i in latents):
        raise ValueError(f"latents must be one or more ints in range({width}), the site's active width, got "
                         f"{latents!r}")
    return [int(i) for i in latents]


def pre_activation(dictionary: Any, site: int, y: Any, latents: Any) -> np.ndarray:
    site = dictionary._site(site)
    latents = _latents(dictionary, site, latents)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != dictionary.d_in:
        raise ValueError(f"y must be shaped (n, {dictionary.d_in}), got {y.shape}")
    a, u = dictionary.arrays, dictionary.unit(site)
    W = a["W_enc"][u][:, latents].astype(np.float64)
    z = (y / np.float64(a["scale_in"][site]) - a["b_pre"][site].astype(np.float64)) @ W
    return (z + a["b_enc"][site, latents].astype(np.float64)).sum(axis=1)


def _check(up: Any, down: Any) -> None:
    if up.model is not down.model or up.params is not down.params or up.declaration is not down.declaration:
        raise ValueError("both readings must share their model, params and declaration, as the same objects")
    if up.windows.shape != down.windows.shape or not np.array_equal(up.windows, down.windows):
        raise ValueError(f"both readings must read the same windows, got {up.windows.shape} and "
                         f"{down.windows.shape} windows that differ")
    order = tap_names(up.n_layers)
    for tap in (up.output_tap, down.input_tap):
        if tap not in order:
            raise ValueError(f"{tap!r} isn't a tap of a {up.n_layers}-layer model")
    if order.index(up.output_tap) >= order.index(down.input_tap):
        raise ValueError(f"the upstream output tap {up.output_tap!r} must come before the downstream input tap "
                         f"{down.input_tap!r}")


def _check_targets(up: Any, down: Any, targets: Sequence[Target]) -> list[Target]:
    targets = list(targets)
    if not targets:
        raise ValueError("at least one target is needed")
    batch, seq = up.tokens.shape
    for t in targets:
        if not _is_int(t.window) or not 0 <= t.window < batch:
            raise ValueError(f"a target's window must be an int in range({batch}), got {t.window!r}")
        if not _is_int(t.position) or not 0 <= t.position < seq:
            raise ValueError(f"a target's position must be an int in range({seq}), got {t.position!r}")
        _latents(down.dictionary, down.site, t.latents)
    return targets


def _compiled(model: Any, output_tap: str, input_tap: str) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """The pass and its gradient with respect to the codes. Every operand is an argument, never a constant closed
    over, so XLA folds nothing (PR #38), and one compiled program serves every chunk of every call given it."""
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST

    def pre(params, tokens, codes, err, positions, weights, up_arrays, down_arrays):
        W_dec, b_dec, scale_out = up_arrays
        W_enc, b_enc, b_pre, scale_in = down_arrays
        full = jnp.pad(codes, ((0, 0), (0, 0), (0, W_dec.shape[0] - codes.shape[2])))  # zero beyond up.width
        held = (jnp.matmul(full, W_dec, precision=highest) + b_dec) * scale_out + err
        _, recorded = model.instrumented(params, tokens, [input_tap], {output_tap: lambda value: held})
        x = recorded[input_tap]
        y = getattr(x, "array", x)[jnp.arange(tokens.shape[0]), positions]
        z = jnp.matmul(y / scale_in - b_pre, W_enc, precision=highest) + b_enc
        return jnp.sum(z * weights, axis=-1)

    def total(*args):
        return jnp.sum(pre(*args))  # each target's codes reach only its own pre-activation

    return jax.jit(pre), jax.jit(jax.grad(total, argnums=2))


class _Pass:
    """The compiled pass over a list of targets, ``chunk`` at a time, with the upstream error held at its
    unintervened value for each target's window. A short last chunk is padded with its first row, and the padding's
    results dropped, so a chunk size compiles once."""

    def __init__(self, up: Any, down: Any, targets: Sequence[Target], chunk: int,
                 compiled: tuple[Callable[..., Any], Callable[..., Any]] | None = None) -> None:
        import jax.numpy as jnp

        self.pre, self.grad = compiled or _compiled(up.model, up.output_tap, down.input_tap)
        self.params, self.tokens = up.params, up.tokens
        self.targets = list(targets)
        self.chunk = min(int(chunk), len(self.targets))
        a, b = up.dictionary.arrays, down.dictionary.arrays
        su, sd = up.site, down.site
        uu, ud = up.dictionary.unit(su), down.dictionary.unit(sd)
        self.up_arrays = tuple(jnp.asarray(v) for v in (a["W_dec"][uu], a["b_dec"][su], a["scale_out"][su]))
        self.down_arrays = tuple(jnp.asarray(v) for v in (b["W_enc"][ud], b["b_enc"][sd], b["b_pre"][sd],
                                                          b["scale_in"][sd]))
        self.m_down = down.dictionary.settings.width
        self.window = np.array([t.window for t in self.targets], dtype=np.int64)
        self.position = np.array([t.position for t in self.targets], dtype=np.int32)
        windows = np.unique(self.window)
        self.slot = np.searchsorted(windows, self.window)
        # The upstream error on the unintervened pass: the output tap's value less the decoded codes.
        x = measure.activations(up.model, up.params, up.tokens[windows], [up.output_tap], chunk=int(chunk))
        x = x[up.output_tap]
        n, seq = len(windows), up.tokens.shape[1]
        padded = np.zeros((n * seq, up.dictionary.settings.width), dtype=np.float32)
        padded[:, :up.width] = up.codes[windows].reshape(n * seq, up.width)
        self.err = (x - core.decode(up.dictionary, su, padded).reshape(x.shape)).astype(np.float32)

    def batches(self):
        n = len(self.targets)
        for start in range(0, n, self.chunk):
            rows = np.arange(start, min(start + self.chunk, n))
            yield rows, np.concatenate([rows, np.full(self.chunk - len(rows), rows[0])])

    def args(self, padded: np.ndarray, codes: np.ndarray) -> tuple[Any, ...]:
        weights = np.zeros((len(padded), self.m_down), dtype=np.float32)
        for r, k in enumerate(padded):
            for latent in self.targets[k].latents:
                weights[r, int(latent)] += 1.0
        return (self.params, np.ascontiguousarray(self.tokens[self.window[padded]]), codes,
                self.err[self.slot[padded]], self.position[padded], weights, self.up_arrays, self.down_arrays)

    def evaluate(self, codes_for: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        out = np.empty(len(self.targets), dtype=np.float64)
        for rows, padded in self.batches():
            codes = np.asarray(codes_for(padded), dtype=np.float32)
            out[rows] = np.asarray(self.pre(*self.args(padded, codes)), dtype=np.float64)[:len(rows)]
        return out

    def integrated(self, codes: np.ndarray) -> np.ndarray:
        """The mean over the midpoints of the gradient at ``α · codes``, float64, each step in float32."""
        total = np.zeros(codes.shape, dtype=np.float64)
        for rows, padded in self.batches():
            for s in range(IG_STEPS):
                at = codes[padded] * np.float32((s + 0.5) / IG_STEPS)
                total[rows] += np.asarray(self.grad(*self.args(padded, at)), dtype=np.float64)[:len(rows)]
        return total / IG_STEPS


def _codes(up: Any, targets: Sequence[Target], codes: Any) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.float32)
    shape = (len(targets), up.tokens.shape[1], up.width)
    if codes.shape != shape:
        raise ValueError(f"codes must be shaped (targets, seq, up.width) = {shape}, got {codes.shape}")
    return codes


def evaluate(up: Any, down: Any, targets: Sequence[Target], codes: Any, *, chunk: int = 32) -> np.ndarray:
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    _check(up, down)
    targets = _check_targets(up, down, targets)
    codes = _codes(up, targets, codes)
    return _Pass(up, down, targets, chunk).evaluate(codes.__getitem__)


def _attribute(up: Any, down: Any, targets: Sequence[Target], chunk: int,
               compiled: tuple[Callable[..., Any], Callable[..., Any]] | None = None) -> Attribution:
    run = _Pass(up, down, targets, chunk, compiled)
    codes = np.ascontiguousarray(up.codes[[t.window for t in targets]], dtype=np.float32)
    attributions = codes.astype(np.float64) * run.integrated(codes)
    for k, t in enumerate(targets):
        attributions[k, t.position + 1:] = 0.0
    full = run.evaluate(codes.__getitem__)
    none = run.evaluate(lambda padded: np.zeros((len(padded),) + codes.shape[1:], dtype=np.float32))
    return Attribution(codes, attributions, full, none)


def attribute(up: Any, down: Any, targets: Sequence[Target], *, chunk: int = 32) -> Attribution:
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    _check(up, down)
    return _attribute(up, down, _check_targets(up, down, targets), int(chunk))


def _shuffle(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = codes.reshape(-1)
    active = np.flatnonzero(flat > 0)
    rest = np.flatnonzero(~(flat > 0))
    return np.concatenate([np.asarray(rng.permutation(active), dtype=np.int64), rest]).astype(np.int64)


def order(attributions: Any, codes: Any, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
    attributions, codes = np.asarray(attributions, dtype=np.float64), np.asarray(codes)
    if attributions.ndim != 2 or attributions.shape != codes.shape:
        raise ValueError(f"attributions and codes must both be shaped (positions, width), got {attributions.shape} "
                         f"and {codes.shape}")
    flat = attributions.reshape(-1)
    if kind == "absolute":
        return np.argsort(-np.abs(flat), kind="stable").astype(np.int64)  # ties to the lower part
    if kind == "signed":
        return np.argsort(-flat, kind="stable").astype(np.int64)
    if kind == "random":
        if rng is None:
            raise ValueError("a random order needs rng")
        return _shuffle(codes, rng)
    raise ValueError(f"kind must be one of {ORDERS}, got {kind!r}")


def _median_reading(readings: Sequence[int | None]) -> float | None:
    median = float(np.median([np.inf if r is None else r for r in readings]))
    return None if np.isinf(median) else median


def _named(up: Any, position: int, latent: int, cache: dict[tuple[str, int], Any]) -> dict[str, Any] | None:
    """Steps 1 and 2 of D49 for an upstream latent at its position's class: its top candidate, labelled."""
    key = (class_of(up.declaration, position), latent)
    if key not in cache:
        found = up.table(key[0]).candidates(latent)
        if not found:
            cache[key] = None
        else:
            d, v, f1 = found[0]
            cache[key] = {"description": d, "value": int(v), "f1": float(f1),
                          "label": naming.NAMED if f1 >= sets.FLOOR else naming.WEAK}
    return None if cache[key] is None else dict(cache[key])


def _place(reading: Any) -> dict[str, Any]:
    return {"site": int(reading.site), "tap": reading.output_tap, "block": reading.block}


def open_up(up: Any, down: Any, cls: str, *, instances: int, seed: int, latent: int | None = None,
            description: str | None = None, value: int | None = None, chunk: int = 32) -> dict[str, Any]:
    if not _is_int(instances) or instances < 1:
        raise ValueError(f"instances must be a positive int, got {instances!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    _check(up, down)
    table = down.table(cls)  # ValueError for a class the run doesn't declare
    named = description is not None or value is not None
    if (latent is not None) == named:
        raise ValueError("give the target as exactly one of latent, or description and value")
    if latent is not None:
        if not _is_int(latent) or not 0 <= latent < down.width:
            raise ValueError(f"latent must be an int in range({down.width}), got {latent!r}")
        latents: tuple[int, ...] = (int(latent),)
    else:
        if description is None or value is None or not table.has(description, value):
            raise ValueError(f"{description!r} = {value!r} isn't a name that counts at class {cls!r}")
        latents = table.set_for(description, value)

    positions = [int(p) for p in up.declaration.classes[cls]]
    fires = down.codes[:, positions][:, :, list(latents)].astype(np.float64).sum(axis=-1) > 0
    if named:
        fires &= down.described[description][:, positions] == int(value)
    candidates = [(int(i), positions[j]) for i, j in zip(*np.nonzero(fires))]  # window by window, class order
    rng = np.random.default_rng(int(seed))
    chosen = candidates
    if len(candidates) > instances:
        chosen = [candidates[k] for k in sorted(rng.choice(len(candidates), size=int(instances), replace=False))]
    shuffles = [_shuffle(up.codes[i, :p + 1], rng) for i, p in chosen]

    results: dict[str, Any] = {
        "class": cls,
        "target": {"latents": list(latents), "latent": None if named else int(latent),
                   "description": description if named else None, "value": int(value) if named else None},
        "up": _place(up), "down": _place(down),
        "candidates": len(candidates), "instances": len(chosen), "no_span": 0, "measured": 0,
        "steps": IG_STEPS, "parts": list(PARTS), "windows": [],
        "orders": {kind: {"error_median": None, "readings": [], "reading_median": None, "within": None}
                   for kind in ORDERS},
        "top_parts": []}
    if not chosen:
        return results

    targets = [Target(i, p, latents) for i, p in chosen]
    compiled = _compiled(up.model, up.output_tap, down.input_tap)  # the attribution and every reading share it
    A = _attribute(up, down, targets, int(chunk), compiled)
    span = np.abs(A.full - A.none)
    measured = [k for k in range(len(chosen)) if not span[k] < NO_SPAN]
    results["no_span"] = len(chosen) - len(measured)
    results["measured"] = len(measured)
    if not measured:
        return results
    results["windows"] = [[chosen[k][0], chosen[k][1]] for k in measured]

    width = up.width
    ranked = {k: [order(A.attributions[k, :chosen[k][1] + 1], A.codes[k, :chosen[k][1] + 1], kind)
                  if kind != "random" else shuffles[k] for kind in ORDERS] for k in measured}
    # Every kept-parts reading: each measured instance, each order, each k, built a chunk at a time.
    rows = [(k, o, size) for k in measured for o in range(len(ORDERS)) for size in PARTS]

    def kept(padded: np.ndarray) -> np.ndarray:
        out = np.zeros((len(padded),) + A.codes.shape[1:], dtype=np.float32)
        for r, j in enumerate(padded):
            k, o, size = rows[j]
            parts = ranked[k][o][:size]
            out[r].reshape(-1)[parts] = A.codes[k].reshape(-1)[parts]
        return out

    pre = _Pass(up, down, [targets[k] for k, _, _ in rows], int(chunk), compiled).evaluate(kept)
    full = np.array([A.full[k] for k, _, _ in rows])
    errors = (np.abs(pre - full) / np.array([span[k] for k, _, _ in rows])).reshape(len(measured), len(ORDERS),
                                                                                    len(PARTS))
    for o, kind in enumerate(ORDERS):
        readings = [next((size for size, e in zip(PARTS, errors[m, o]) if e <= ERROR), None)
                    for m in range(len(measured))]
        results["orders"][kind] = {"error_median": [float(np.median(errors[:, o, s])) for s in range(len(PARTS))],
                                   "readings": readings, "reading_median": _median_reading(readings),
                                   "within": float(np.mean([r is not None for r in readings]))}

    names: dict[tuple[str, int], Any] = {}
    for k in measured:
        top = []
        for part in ranked[k][0][:TOP_PARTS]:
            position, latent_ = divmod(int(part), width)
            top.append({"position": position, "latent": latent_,
                        "attribution": float(A.attributions[k, position, latent_]),
                        "name": _named(up, position, latent_, names)})
        results["top_parts"].append(top)
    return results
