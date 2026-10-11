"""The edge-sparsity meter between two dictionaries, and the Jacobian across an MLP (D42, D43).

Spec: docs/specs/phase-5.md, D42, D43 and "The edge-sparsity meter". JAX is imported inside the
functions that need it, so this module imports without it. Nothing here caches or judges.

A side is a ``(dictionary, site)`` pair. Two kinds of pair are measured:

* **Across an MLP:** the upstream side reads and reconstructs ``blocks.<l>.ln2.hook_normalized``
  and the downstream side ``blocks.<l>.hook_mlp_out`` of the same block, with ``weights`` that
  block's MLP from ``train.mlp_weights``. Two TopK SAEs, or a Jacobian SAE's sites 0 and 1.
* **Across a block:** the upstream side reads and reconstructs ``blocks.<l>.hook_resid_post`` and
  the downstream side ``blocks.<l + 1>.hook_resid_post``. Two TopK SAEs, or a Staircase SAE's
  sites ``l`` and ``l + 1``.

Any other pair, or sides whose widths don't meet, raises ValueError. In both, ``u`` is the
upstream side's ``encode`` of its tap's values on the unintervened pass over ``tokens``, every
position, and "the downstream code" is the downstream side's encoding, computed as ``encode``
computes it, of what its tap would hold. Every matrix product in a derivative or an ablation runs
at float32 with the highest precision.

* ``EDGE_POINTS = 14``, ``IG_STEPS = 5``, ``SETTLE = 0.1``.
* ``edge_counts(n_edges, points=EDGE_POINTS)``: the ablation curve's edge counts: the distinct
  values of ``round(v)`` for ``v`` in ``numpy.logspace(0, log10(n_edges), points)``, with
  ``n_edges`` itself, ascending. ValueError unless both are positive ints.
* ``kl(clean_logits, logits)``: the mean, over every window and position, of the KL divergence
  from the clean distribution to the other, ``Σ p (log p − log q)``, from log-softmaxes in
  float64, as a plain float. ValueError unless the shapes match.
* ``mlp_jacobian(up, down, weights, codes)``: per row of upstream ``codes`` (T, m_up):
  ``(jac, idx_in, idx_out)``. ``idx_in`` (T, k_up) holds the row's ``k`` largest codes' latent ids,
  ties to the lower id. The downstream pre-activations are computed, as ``encode`` computes them,
  on ``mlp(weights, decode(up, codes))``, and ``idx_out`` (T, k_down) holds the ``k`` that
  ``encode`` would keep. ``jac`` (T, k_up, k_down), float32, is the derivative of the downstream
  pre-activation at ``idx_out[t, j]`` with respect to the upstream code at ``idx_in[t, i]``, at
  ``jac[t, i, j]``, and zero where that pre-activation is not positive. For a Jacobian SAE's sites
  0 and 1 it equals ``train.pair_jacobian``.
* ``jacobian_stats(jac)``: ``{"entries_for_90", "share_above_0.01", "mean_abs"}`` over the rows
  of ``jac`` (T, a, b), as plain floats: the median over rows of the fewest entries whose
  absolute values sum to at least 90% of the row's total (0 for a row of zeros); the mean over
  rows of the share of entries whose absolute value is above 0.01; the mean absolute entry.
* ``mlp_edge_scores(model, params, tokens, up, down, weights, *, ig_steps=IG_STEPS)``: float64
  (m_up, m_down). Each edge ``(i, j)`` is scored by integrated gradients from the origin, per
  position of every window: with ``α`` at the midpoints ``(s + 0.5) / ig_steps``, its attribution
  is ``u[i]`` times the mean over ``α`` of the derivative of downstream code ``j`` with respect to
  upstream code ``i`` at ``α · u``, the derivative ``mlp_jacobian`` gives (zero for a downstream
  latent ``encode`` wouldn't keep there). The score is the root mean square of the attribution
  over every position of every window.
* ``block_edge_scores(model, params, tokens, up, down, *, ig_steps=IG_STEPS)``: float64 (m_up,
  m_down). Per window, with the upstream tap holding ``decode(up, α · u)`` at every position, let
  ``f_j`` be downstream code ``j`` summed over positions. Edge ``(i, j)``'s attribution is the sum
  over positions ``p`` of ``u[p, i]`` times the mean over ``α`` of ``∂f_j / ∂u[p, i]`` at ``α · u``.
  The score is the root mean square of the attribution over windows.
* ``mlp_curve(model, params, tokens, up, down, weights, scores, *, points=EDGE_POINTS)`` and
  ``block_curve(model, params, tokens, up, down, scores, *, points=EDGE_POINTS)``:
  ``{"curve", "pair_kl", "scored_edges"}``. ``scored_edges`` counts the scores above zero, and
  ``curve`` is a list of ``[n, kl]`` for each ``n`` in ``edge_counts(scored_edges, points)``:

  - keep the ``n`` highest-scoring edges, ties to the lower upstream id, then the lower
    downstream id
  - every downstream latent with a kept edge takes, at every position, its downstream code when
    the upstream code is ``u`` with every latent not on one of its kept edges zeroed: across an
    MLP, through ``mlp`` at each position; across a block, through the model with the upstream
    tap holding that code's ``decode`` at every position. Every other downstream latent is zero
  - the downstream side's ``decode`` of those codes replaces its tap at every position, and
    ``kl`` is taken from the clean logits

  ``pair_kl`` is ``kl`` with the pair spliced in whole: the downstream tap holds the downstream
  side's ``reconstruct`` of what it holds when the upstream tap holds ``reconstruct`` of ``u``'s
  input (across an MLP, ``mlp`` of the upstream reconstruction). Every replacement enters the
  compiled program as an argument (PR #38). ValueError for ``scores`` not shaped (m_up, m_down),
  or with no score above zero.
* ``meter(curve, pair_kl)``: the edges the computation needs: the fewest ``n`` in ``curve`` from
  which every later point's KL, its own included, is at most ``floor + SETTLE · (kl₁ − floor)``,
  where ``kl₁`` is the first point's KL and ``floor`` is the lower of ``pair_kl`` and the last
  point's KL, the term ``kl₁ − floor`` taken as 0 when negative. None if no point qualifies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np

from .core import _MLP_IN, _MLP_OUT, _RESID, Dictionary, content_hash, decode, encode
from .train import _check_mlp, mlp

EDGE_POINTS, IG_STEPS, SETTLE = 14, 5, 0.1

_BUDGET = 1 << 22  # floats per chunk of a batched pass: bounds memory without changing any result


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _value(recorded: Any) -> Any:
    return getattr(recorded, "array", recorded)


def edge_counts(n_edges: int, points: int = EDGE_POINTS) -> list[int]:
    for name, value in (("n_edges", n_edges), ("points", points)):
        if not _is_int(value) or value < 1:
            raise ValueError(f"{name} must be a positive int, not {value!r}")
    values = np.logspace(0, math.log10(int(n_edges)), int(points))
    return sorted({int(round(float(v))) for v in values} | {int(n_edges)})


def _log_softmax(x: np.ndarray) -> np.ndarray:
    peak = x.max(axis=-1, keepdims=True)
    return x - peak - np.log(np.exp(x - peak).sum(axis=-1, keepdims=True))


def kl(clean_logits: Any, logits: Any) -> float:
    p, q = np.asarray(clean_logits, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if p.shape != q.shape or p.ndim < 1:
        raise ValueError(f"the logits must have one shape, got {p.shape} and {q.shape}")
    p, q = _log_softmax(p), _log_softmax(q)
    return float((np.exp(p) * (p - q)).sum(axis=-1).mean())


def meter(curve: Sequence[Sequence[float]], pair_kl: float) -> int | None:
    if isinstance(curve, (str, bytes)) or not isinstance(curve, Sequence) or not curve:
        raise ValueError("the curve must be a non-empty list of [edges, kl] points")
    points = []
    for point in curve:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"a point of the curve is [edges, kl], got {point!r}")
        points.append((int(point[0]), float(point[1])))
    floor = min(float(pair_kl), points[-1][1])
    bound = floor + SETTLE * max(points[0][1] - floor, 0.0)
    needed = None
    for n, value in reversed(points):  # the earliest point from which every later point stays within the bound
        if value > bound:
            break
        needed = n
    return needed


def jacobian_stats(jac: Any) -> dict[str, float]:
    jac = np.asarray(jac, dtype=np.float64)
    if jac.ndim != 3 or 0 in jac.shape:
        raise ValueError(f"jac must be a non-empty (T, a, b) array, got shape {jac.shape}")
    mass = np.abs(jac).reshape(len(jac), -1)
    held = np.cumsum(-np.sort(-mass, axis=1), axis=1)
    total = held[:, -1]
    needed = np.where(total > 0, (held < 0.9 * total[:, None]).sum(axis=1) + 1, 0)
    return {"entries_for_90": float(np.median(needed)), "share_above_0.01": float((mass > 0.01).mean(axis=1).mean()),
            "mean_abs": float(mass.mean())}


# Pairs.

def _side(side: Any, name: str) -> tuple[Dictionary, int, str]:
    """``(dictionary, site, tap)`` of one side, whose site reads and reconstructs ``tap``."""
    if isinstance(side, (str, bytes)) or not isinstance(side, Sequence) or len(side) != 2 \
            or not isinstance(side[0], Dictionary):
        raise ValueError(f"the {name} side must be a (dictionary, site) pair")
    dictionary, site = side[0], side[0]._site(side[1])
    reads, writes = dictionary.sites[site]
    if reads != writes:
        raise ValueError(f"the {name} side must read and reconstruct one tap, got {(reads, writes)!r}")
    return dictionary, site, reads


def _same(a: Dictionary, b: Dictionary) -> bool:
    return a is b or content_hash(a) == content_hash(b)


def _widths(up: Dictionary, down: Dictionary, model: Any = None) -> None:
    if up.d_out != down.d_in:
        raise ValueError(f"the upstream side writes width {up.d_out} and the downstream side reads {down.d_in}")
    d_model = getattr(getattr(model, "config", None), "d_model", None)
    if d_model is not None and {up.d_in, up.d_out, down.d_in, down.d_out} != {d_model}:
        raise ValueError(f"every side must read and write the model's width {d_model}")


def _layers(model: Any) -> int | None:
    return getattr(getattr(model, "config", None), "n_layers", None)


def _mlp_pair(up: Any, down: Any, model: Any = None) -> tuple[tuple[Dictionary, int], tuple[Dictionary, int], int]:
    (du, su, tu), (dd, sd, td) = _side(up, "upstream"), _side(down, "downstream")
    a, b = _MLP_IN.fullmatch(tu), _MLP_OUT.fullmatch(td)
    if a is None or b is None or a.group(1) != b.group(1):
        raise ValueError(f"across an MLP the sides read blocks.<l>.ln2.hook_normalized and blocks.<l>.hook_mlp_out, "
                         f"got {tu!r} and {td!r}")
    kinds = (du.settings.kind, dd.settings.kind)
    if kinds != ("topk", "topk") and not (kinds == ("jsae", "jsae") and _same(du, dd) and (su, sd) == (0, 1)):
        raise ValueError(f"across an MLP the pair is two TopK SAEs or a Jacobian SAE's sites 0 and 1, got {kinds}")
    layer, n_layers = int(a.group(1)), _layers(model)
    if n_layers is not None and layer >= n_layers:
        raise ValueError(f"the model has no block {layer}")
    _widths(du, dd, model)
    return (du, su), (dd, sd), layer


def _block_pair(up: Any, down: Any, model: Any) -> tuple[tuple[Dictionary, int], tuple[Dictionary, int], int]:
    (du, su, tu), (dd, sd, td) = _side(up, "upstream"), _side(down, "downstream")
    a, b = _RESID.fullmatch(tu), _RESID.fullmatch(td)
    if a is None or b is None or int(b.group(1)) != int(a.group(1)) + 1:
        raise ValueError(f"across a block the sides read blocks.<l>.hook_resid_post and blocks.<l + 1>.hook_resid_post, "
                         f"got {tu!r} and {td!r}")
    kinds = (du.settings.kind, dd.settings.kind)
    if kinds != ("topk", "topk") and not (kinds == ("staircase", "staircase") and _same(du, dd) and sd == su + 1):
        raise ValueError(f"across a block the pair is two TopK SAEs or a Staircase SAE's sites l and l + 1, got {kinds}")
    layer, n_layers = int(a.group(1)), _layers(model)
    if n_layers is not None and layer + 1 >= n_layers:
        raise ValueError(f"the model has no block {layer + 1}")
    _widths(du, dd, model)
    return (du, su), (dd, sd), layer


def _scores(scores: Any, up: tuple[Dictionary, int], down: tuple[Dictionary, int]) -> np.ndarray:
    shape = (up[0].settings.width, down[0].settings.width)
    try:
        scores = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("scores must be a numeric array") from error
    if scores.shape != shape:
        raise ValueError(f"scores must be shaped (m_up, m_down) = {shape}, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not (scores > 0).any():
        raise ValueError("no score is above zero, so there are no edges to keep")
    return scores


def _ig_steps(ig_steps: Any) -> int:
    if not _is_int(ig_steps) or ig_steps < 1:
        raise ValueError(f"ig_steps must be a positive int, not {ig_steps!r}")
    return int(ig_steps)


def _decoder(dictionary: Dictionary, site: int) -> tuple[Any, ...]:
    a, u = dictionary.arrays, dictionary.unit(site)
    return a["W_dec"][u], a["b_dec"][site], a["scale_out"][site]


def _encoder(dictionary: Dictionary, site: int) -> tuple[Any, ...]:
    a, u = dictionary.arrays, dictionary.unit(site)
    return a["W_enc"][u], a["b_pre"][site], a["b_enc"][site], a["scale_in"][site]


def _clean(model: Any, params: Any, tokens: Any, tap: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tokens, clean logits, the tap's values)`` on the unintervened pass."""
    tokens = np.asarray(tokens)
    logits, recorded = model.instrumented(params, tokens, [tap])
    return tokens, np.asarray(logits), np.asarray(_value(recorded[tap]), dtype=np.float32)


def _top(scores: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` highest-scoring edges as a (m_up, m_down) mask, ties to the lower upstream id, then downstream id."""
    keep = np.zeros(scores.size, dtype=bool)
    keep[np.argsort(-scores, axis=None, kind="stable")[:n]] = True
    return keep.reshape(scores.shape)


# Kernels. Every matrix product runs at float32 with the highest precision.

@cache
def _kernels() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST

    def gelu(h):
        return jax.nn.gelu(h, approximate=True)

    def encode_rows(y, W_enc, b_pre, b_enc, scale_in, *, k, active):
        """``encode`` of rows ``y`` (..., d), as core computes it."""
        flat = y.reshape(-1, y.shape[-1])
        pre = jnp.matmul(flat / scale_in - b_pre, W_enc, precision=highest) + b_enc
        pre = jnp.where(jnp.arange(pre.shape[1]) < active, pre, -jnp.inf)
        values, index = jax.lax.top_k(pre, k)
        rows = jnp.arange(pre.shape[0])[:, None]
        codes = jnp.zeros(pre.shape, jnp.float32).at[rows, index].set(jnp.maximum(values, 0.0))
        return codes.reshape(y.shape[:-1] + (pre.shape[1],))

    def jacobian(w, codes, idx_in, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in, *, k_out, active):
        """The derivative of the downstream pre-activations ``encode`` keeps, through the MLP, with respect to the
        upstream codes at ``idx_in``, as ``train.pair_jacobian`` computes it."""
        x_hat = (jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out
        h = jnp.matmul(x_hat, w["W_fc"], precision=highest) + w["b_fc"]
        y = jnp.matmul(gelu(h), w["W_proj"], precision=highest) + w["b_proj"]
        pre = jnp.matmul(y / scale_in - b_pre, W_enc, precision=highest) + b_enc
        pre = jnp.where(jnp.arange(pre.shape[1]) < active, pre, -jnp.inf)
        values, idx_out = jax.lax.top_k(pre, k_out)  # ties go to the lower index
        slope = jax.jvp(gelu, (h,), (jnp.ones_like(h),))[1]
        rows = jnp.matmul(W_dec[idx_in] * scale_out, w["W_fc"], precision=highest) * slope[:, None, :]
        rows = jnp.matmul(rows, w["W_proj"], precision=highest)
        cols = jnp.moveaxis(W_enc[:, idx_out], 0, 1) / scale_in
        jac = jnp.matmul(rows, cols, precision=highest)
        return jnp.where((values > 0)[:, None, :], jac, 0.0), idx_out

    def mlp_jacobian_fn(w, codes, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in, *, k_in, k_out, active):
        _, idx_in = jax.lax.top_k(codes, k_in)
        jac, idx_out = jacobian(w, codes, idx_in, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in,
                                k_out=k_out, active=active)
        return jac, idx_in, idx_out

    def mlp_attribution_fn(w, u, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in, *, k_in, k_out, active,
                           ig_steps):
        """Per row of ``u``: the attribution of each of its ``k_in`` largest latents to every downstream latent,
        (T, k_in, m_down), and those latents' ids."""
        slots, idx_in = jax.lax.top_k(u, k_in)
        total = jnp.zeros((u.shape[0], k_in, W_enc.shape[1]), jnp.float32)
        rows, cols = jnp.arange(u.shape[0])[:, None, None], jnp.arange(k_in)[None, :, None]
        for s in range(ig_steps):
            jac, idx_out = jacobian(w, u * ((s + 0.5) / ig_steps), idx_in, W_dec, b_dec, scale_out, W_enc, b_pre,
                                    b_enc, scale_in, k_out=k_out, active=active)
            total = total.at[rows, cols, idx_out[:, None, :]].add(jac)
        return slots[:, :, None] * (total / ig_steps), idx_in

    def mlp_codes_fn(w, codes, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in, *, k, active):
        """The downstream codes of the MLP on the upstream ``decode`` of each row of ``codes``."""
        x_hat = (jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out
        hidden = gelu(jnp.matmul(x_hat, w["W_fc"], precision=highest) + w["b_fc"])
        y = jnp.matmul(hidden, w["W_proj"], precision=highest) + w["b_proj"]
        return encode_rows(y, W_enc, b_pre, b_enc, scale_in, k=k, active=active)

    statics = ("k_in", "k_out", "active", "ig_steps", "k")
    return {"encode_rows": encode_rows,
            "mlp_jacobian": jax.jit(mlp_jacobian_fn, static_argnames=statics[:3]),
            "mlp_attribution": jax.jit(mlp_attribution_fn, static_argnames=statics[:4]),
            "mlp_codes": jax.jit(mlp_codes_fn, static_argnames=("k", "active"))}


def _spliced(model: Any, tap: str) -> Any:
    """A compiled pass whose ``tap`` holds an argument at every position: its logits. The replacement is an argument,
    not a constant closed over (PR #38)."""
    import jax

    def forward(params: Any, tokens: Any, replacement: Any) -> Any:
        return model.instrumented(params, tokens, [], {tap: lambda _: replacement})[0]

    return jax.jit(forward)


def _chunk(per_row: int, rows: int) -> int:
    return max(1, min(rows, _BUDGET // max(per_row, 1)))


# Across an MLP.

def mlp_jacobian(up: Any, down: Any, weights: Mapping[str, Any], codes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up, down, _ = _mlp_pair(up, down)
    w = _check_mlp(weights, up[0].d_out, down[0].d_in)
    m_up = up[0].settings.width
    try:
        codes = np.asarray(codes, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("codes must be a numeric array") from error
    if codes.ndim != 2 or codes.shape[1] != m_up:
        raise ValueError(f"codes must be shaped (T, {m_up}), got {codes.shape}")
    jac, idx_in, idx_out = _kernels()["mlp_jacobian"](
        w, codes, *_decoder(*up), *_encoder(*down), k_in=up[0].settings.k, k_out=down[0].settings.k,
        active=down[0].active_width(down[1]))
    return np.array(jac, dtype=np.float32), np.array(idx_in), np.array(idx_out)


def _upstream_codes(model: Any, params: Any, tokens: Any, up: tuple[Dictionary, int]) -> tuple[np.ndarray, ...]:
    """``(tokens, clean logits, u)``, ``u`` shaped (batch, pos, m_up)."""
    tap = up[0].sites[up[1]][0]
    tokens, logits, x = _clean(model, params, tokens, tap)
    u = encode(*up, x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1] + (up[0].settings.width,))
    return tokens, logits, u


def mlp_edge_scores(model: Any, params: Any, tokens: Any, up: Any, down: Any, weights: Mapping[str, Any], *,
                    ig_steps: int = IG_STEPS) -> np.ndarray:
    up, down, _ = _mlp_pair(up, down, model)
    ig_steps = _ig_steps(ig_steps)
    w = _check_mlp(weights, up[0].d_out, down[0].d_in)
    _, _, u = _upstream_codes(model, params, tokens, up)
    u = u.reshape(-1, u.shape[-1])
    m_up, m_down, k_in = up[0].settings.width, down[0].settings.width, up[0].settings.k
    rows = _chunk(k_in * m_down, len(u))  # one shape for every call, so the attribution compiles once
    attribute = _kernels()["mlp_attribution"]
    arrays = (*_decoder(*up), *_encoder(*down))
    total = np.zeros((m_up, m_down))
    for start in range(0, len(u), rows):
        part = np.zeros((rows, m_up), np.float32)
        part[:len(u[start:start + rows])] = u[start:start + rows]  # padding rows are zero and attribute nothing
        attribution, idx_in = attribute(w, part, *arrays, k_in=k_in, k_out=down[0].settings.k,
                                        active=down[0].active_width(down[1]), ig_steps=ig_steps)
        attribution = np.asarray(attribution, dtype=np.float64)
        np.add.at(total, np.asarray(idx_in).ravel(), (attribution ** 2).reshape(-1, m_down))
    return np.sqrt(total / len(u))


def _patterns(u: np.ndarray, idx_in: np.ndarray, keep: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Per position and downstream latent in ``columns``, which of the position's active upstream latents it keeps,
    as bytes: (N, len(columns), ceil(k / 8)) uint8."""
    live = np.take_along_axis(u, idx_in, axis=1) > 0  # (N, k)
    kept = keep[idx_in][:, :, columns] & live[:, :, None]  # (N, k, J)
    return np.packbits(kept, axis=1).transpose(0, 2, 1)


def mlp_curve(model: Any, params: Any, tokens: Any, up: Any, down: Any, weights: Mapping[str, Any], scores: Any, *,
              points: int = EDGE_POINTS) -> dict[str, Any]:
    up, down, _ = _mlp_pair(up, down, model)
    w = _check_mlp(weights, up[0].d_out, down[0].d_in)
    scores = _scores(scores, up, down)
    scored = int((scores > 0).sum())
    counts = edge_counts(scored, points)
    tap = down[0].sites[down[1]][0]
    tokens, clean, u = _upstream_codes(model, params, tokens, up)
    shape = u.shape[:-1]
    u = u.reshape(-1, u.shape[-1])
    n_rows, m_up, m_down, k_in = len(u), up[0].settings.width, down[0].settings.width, up[0].settings.k
    idx_in = np.argsort(-u, axis=1, kind="stable")[:, :k_in]  # the row's k largest codes, ties to the lower id
    splice = _spliced(model, tap)

    def spliced_kl(codes: np.ndarray) -> float:
        replacement = decode(*down, codes).reshape(shape + (down[0].d_out,))
        return kl(clean, splice(params, tokens, replacement))

    # A downstream latent's code at a position depends only on which of the position's active upstream latents it
    # keeps, so at each point every distinct (position, kept latents) is computed once.
    compute, arrays = _kernels()["mlp_codes"], (*_decoder(*up), *_encoder(*down))
    k, active = down[0].settings.k, down[0].active_width(down[1])
    chunk = _chunk(max(m_up, m_down, w["W_fc"].shape[1]), n_rows * m_down)  # one shape, so the codes compile once
    slots = np.take_along_axis(u, idx_in, axis=1)
    curve = []
    for n in counts:
        keep = _top(scores, n)
        columns = np.flatnonzero(keep.any(axis=0))
        pattern = _patterns(u, idx_in, keep, columns)  # (N, J, bytes)
        row = np.broadcast_to(np.arange(n_rows, dtype=">i4")[:, None, None].view(np.uint8), pattern.shape[:2] + (4,))
        key = np.ascontiguousarray(np.concatenate([row, pattern], axis=2))
        key = key.view(np.dtype((np.void, key.shape[2])))[..., 0].ravel()
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        where = first // len(columns)  # each distinct combination's position
        bits = np.unpackbits(pattern.reshape(-1, pattern.shape[2])[first], axis=1, count=k_in).astype(bool)
        order = np.argsort(inverse, kind="stable")  # rows grouped by their combination
        bounds = np.searchsorted(inverse[order], np.arange(0, len(first) + chunk, chunk))
        codes = np.zeros((n_rows, m_down), np.float32)
        for c, start in enumerate(range(0, len(first), chunk)):
            stop = min(start + chunk, len(first))
            part = np.zeros((chunk, m_up), np.float32)
            np.put_along_axis(part[:stop - start], idx_in[where[start:stop]], slots[where[start:stop]] * bits[start:stop],
                              axis=1)
            out = np.asarray(compute(w, part, *arrays, k=k, active=active))
            rows = order[bounds[c]:bounds[c + 1]]
            t, j = rows // len(columns), columns[rows % len(columns)]
            codes[t, j] = out[inverse[rows] - start, j]
        curve.append([n, spliced_kl(codes)])
    whole = encode(*down, mlp(w, decode(*up, u)))
    return {"curve": curve, "pair_kl": spliced_kl(whole), "scored_edges": scored}


# Across a block.

def _through(model: Any, up_tap: str, down_tap: str, down: tuple[Dictionary, int]) -> Any:
    """A compiled pass that puts the upstream decode of each code in ``codes`` (C, batch, pos, m_up) at the upstream
    tap, every position, and returns the downstream codes, (C, batch, pos, m_down). The C passes run as one batch."""
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST
    encode_rows = _kernels()["encode_rows"]
    k, active = down[0].settings.k, down[0].active_width(down[1])

    def forward(params, tokens, u, masks, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in):
        codes = u[None] * masks[:, None, None, :]
        x_hat = (jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out
        c, batch, pos, d = x_hat.shape
        replacement = x_hat.reshape(c * batch, pos, d)
        _, recorded = model.instrumented(params, jnp.tile(tokens, (c, 1)), [down_tap], {up_tap: lambda _: replacement})
        y = _value(recorded[down_tap]).reshape(c, batch, pos, -1)
        return encode_rows(y, W_enc, b_pre, b_enc, scale_in, k=k, active=active)

    return jax.jit(forward)


def _attribution(model: Any, up_tap: str, down_tap: str, up: tuple[Dictionary, int], down: tuple[Dictionary, int]) -> Any:
    """A compiled pass over one window with the upstream tap holding the decode of ``codes`` (pos, m_up): for each
    upstream latent in ``ids`` (C,), the derivative of every downstream code, summed over positions, along ``u`` at
    that latent, weighted by ``live``. Forward mode, one tangent per upstream latent."""
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST
    k, active, m_down = down[0].settings.k, down[0].active_width(down[1]), down[0].settings.width

    def forward(params, window, codes, u, ids, live, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in):
        def downstream(x):
            _, recorded = model.instrumented(params, window, [down_tap], {up_tap: lambda _: x})
            return _value(recorded[down_tap])

        x_hat = ((jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out)[None]
        y, along = jax.linearize(downstream, x_hat)
        pre = jnp.matmul(y[0] / scale_in - b_pre, W_enc, precision=highest) + b_enc
        pre = jnp.where(jnp.arange(pre.shape[1]) < active, pre, -jnp.inf)
        values, idx_out = jax.lax.top_k(pre, k)  # (pos, k): the latents encode keeps; they move where positive
        tangents = (u[:, ids] * live).T[:, :, None] * (W_dec[ids] * scale_out)[:, None, :]  # (C, pos, d)
        dy = jax.vmap(along)(tangents[:, None])[:, 0] / scale_in  # (C, pos, d)
        moved = jnp.einsum("cpd,dpk->cpk", dy, W_enc[:, idx_out], precision=highest)
        moved = jnp.where(values[None] > 0, moved, 0.0)
        return jnp.zeros((len(ids), m_down), jnp.float32).at[:, idx_out.ravel()].add(moved.reshape(len(ids), -1))

    return jax.jit(forward)


def block_edge_scores(model: Any, params: Any, tokens: Any, up: Any, down: Any, *, ig_steps: int = IG_STEPS) -> np.ndarray:
    up, down, _ = _block_pair(up, down, model)
    ig_steps = _ig_steps(ig_steps)
    up_tap, down_tap = up[0].sites[up[1]][0], down[0].sites[down[1]][0]
    tokens, _, u = _upstream_codes(model, params, tokens, up)
    m_up, m_down = up[0].settings.width, down[0].settings.width
    batch, pos = tokens.shape
    d_mlp = getattr(getattr(model, "config", None), "d_mlp", 4 * up[0].d_out)
    most = min(up[0].active_width(up[1]), pos * up[0].settings.k)  # latents active in one window, at most
    chunk = _chunk(pos * max(d_mlp, m_down, up[0].d_out), most)  # one shape, so the pass compiles once
    attribute = _attribution(model, up_tap, down_tap, up, down)
    arrays = (*_decoder(*up), *_encoder(*down))
    total = np.zeros((m_up, m_down))
    for b in range(batch):
        active = np.flatnonzero((u[b] > 0).any(axis=0))
        window = tokens[b:b + 1]
        attribution = np.zeros((m_up, m_down))
        for s in range(ig_steps):
            codes = u[b] * np.float32((s + 0.5) / ig_steps)
            for start in range(0, len(active), chunk):
                ids = np.zeros(chunk, np.int32)
                live = np.zeros(chunk, np.float32)
                part = active[start:start + chunk]
                ids[:len(part)], live[:len(part)] = part, 1.0
                moved = np.asarray(attribute(params, window, codes, u[b], ids, live, *arrays), dtype=np.float64)
                attribution[part] += moved[:len(part)]
        total += (attribution / ig_steps) ** 2
    return np.sqrt(total / batch)


def block_curve(model: Any, params: Any, tokens: Any, up: Any, down: Any, scores: Any, *,
                points: int = EDGE_POINTS) -> dict[str, Any]:
    up, down, _ = _block_pair(up, down, model)
    scores = _scores(scores, up, down)
    scored = int((scores > 0).sum())
    counts = edge_counts(scored, points)
    up_tap, down_tap = up[0].sites[up[1]][0], down[0].sites[down[1]][0]
    tokens, clean, u = _upstream_codes(model, params, tokens, up)
    m_up, m_down = up[0].settings.width, down[0].settings.width
    batch, pos = tokens.shape
    splice = _spliced(model, down_tap)

    def spliced_kl(codes: np.ndarray) -> float:
        replacement = decode(*down, codes.reshape(-1, m_down)).reshape(batch, pos, down[0].d_out)
        return kl(clean, splice(params, tokens, replacement))

    # A downstream latent's codes depend only on which upstream latents active in these windows it keeps, so each
    # distinct set is passed through the model once, across every point of the curve and the pair spliced in whole.
    live = (u > 0).any(axis=(0, 1))
    plan, masks = [], [live]
    for n in counts:
        keep = _top(scores, n)
        columns = np.flatnonzero(keep.any(axis=0))
        plan.append(columns)
        masks.extend(keep[:, columns].T & live[None])
    unique, inverse = np.unique(np.array(masks), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    d_mlp = getattr(getattr(model, "config", None), "d_mlp", 4 * up[0].d_out)
    chunk = _chunk(batch * pos * max(m_up, m_down, d_mlp), len(unique))  # one shape, so the pass compiles once
    through = _through(model, up_tap, down_tap, down)
    arrays = (*_decoder(*up), *_encoder(*down))
    wanted = [[] for _ in unique]  # per distinct set: (point, downstream latent) pairs that take their codes from it
    wanted[inverse[0]].append((-1, -1))
    offset = 1
    for p, columns in enumerate(plan):
        for j, c in zip(columns, inverse[offset:offset + len(columns)]):
            wanted[c].append((p, j))
        offset += len(columns)
    codes = np.zeros((len(counts), batch, pos, m_down), np.float32)
    whole = None
    for start in range(0, len(unique), chunk):
        part = np.zeros((chunk, m_up), np.float32)
        stop = min(start + chunk, len(unique))
        part[:stop - start] = unique[start:stop]
        out = np.asarray(through(params, tokens, u, part, *arrays))
        for c in range(start, stop):
            for p, j in wanted[c]:
                if p < 0:
                    whole = out[c - start]
                else:
                    codes[p, ..., j] = out[c - start, ..., j]
    curve = [[n, spliced_kl(codes[p])] for p, n in enumerate(counts)]
    return {"curve": curve, "pair_kl": spliced_kl(whole), "scored_edges": scored}
