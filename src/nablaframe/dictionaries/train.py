"""Training a dictionary, and the Jacobian SAE's Jacobian through an MLP (D40, D43).

Spec: docs/specs/phase-5.md, D40 and D43. JAX and optax are imported inside the functions that
need them, so this module imports without them.

* ``mlp_weights(params, layer)``: block ``layer``'s MLP as ``{"W_fc" (d_model, d_mlp), "b_fc",
  "W_proj" (d_mlp, d_model), "b_proj"}``, float32 numpy arrays. ValueError for a layer the model
  doesn't have.
* ``mlp(weights, x)``: the MLP the model applies, ``gelu(x @ W_fc + b_fc) @ W_proj + b_proj`` with
  the tanh approximation, in float32.
* ``pair_jacobian(dictionary, codes, weights)``: for a ``jsae``, per row of the input site's
  ``codes`` (B, m): ``(jac, idx_in, idx_out)``. ``idx_in`` (B, k) holds the ``k`` largest input
  codes' latent ids, ties to the lower id. The output site's pre-activations are computed, as
  ``encode`` computes them, on ``mlp(weights, decode(dictionary, 0, codes))``; ``idx_out`` (B, k)
  holds their ``k`` largest, as ``encode`` keeps them. ``jac`` (B, k, k), float32, is the
  derivative of the output's pre-activation at ``idx_out[b, j]`` with respect to the input code at
  ``idx_in[b, i]``, at ``jac[b, i, j]``, and zero for an output latent whose pre-activation is not
  positive. ValueError for any other kind.
* ``train(settings, sites, data, *, mlp_weights=None, key=None, progress=None)``: a trained
  ``Dictionary``. ``data`` holds, per site in order, ``(inputs, outputs)``: float32 arrays shaped
  (N, d_in) and (N, d_out), with one N for every site; for a ``topk``, ``staircase`` or ``jsae``
  site they must be equal. A ``jsae`` needs ``mlp_weights`` of its block, and nothing else takes
  them. Anything else raises ValueError before anything trains.

  - **Scales:** each site's ``scale_in`` and ``scale_out`` are the root mean square norm of its
    inputs and outputs over ``√d``, computed in float64.
  - **Initialisation:** with ``keys = jax.random.split(jax.random.PRNGKey(settings.seed), U)``,
    unit ``u``'s ``W_dec`` is ``jax.random.normal(keys[u], (m, d_out))`` with its rows scaled to
    unit norm, and its ``W_enc`` is that transposed, times 0.1 for a ``transcoder``. ``b_enc``
    starts at zero, and ``b_pre`` and ``b_dec`` at the mean of the site's scaled inputs and outputs.
  - **Batches:** step ``s`` draws ``batch`` rows with
    ``numpy.random.default_rng(settings.seed).integers(0, N, batch)``, one draw per step from
    one generator, the same rows for every site.
  - **Loss:** per site, the squared error of the scaled output summed over features and averaged
    over rows, divided by ``d_out``; plus ``aux_weight`` times AuxK's: among latents that haven't
    fired for ``dead_after`` steps on that site, the ``min(4k, m // 2)`` largest pre-activations,
    clipped at zero, reconstruct the site's error, held constant. For a ``jsae`` with ``lam > 0``,
    plus ``lam`` times the mean absolute entry of ``pair_jacobian`` on the batch's first
    ``jac_batch`` rows.
  - **Optimiser:** Adam (optax defaults) at a learning rate rising linearly from 0 over
    ``min(warmup, steps // 10)`` steps, held, then falling linearly to 0 over the last
    ``int(steps * decay_fraction)`` steps. After each step every ``W_dec`` row is renormalised to
    unit length.
  - **Determinism:** the same settings, sites, data, ``mlp_weights`` and ``key`` give a dictionary
    with the same content hash on the same backend.
  - ``progress(step, steps, losses)``, when given, is called at each tenth of the steps and at the
    last, ``losses`` holding each site's reconstruction term.
  - The result carries ``key``.
* ``code_identity()``: this module's git blob hash, the code in a dictionary's cache key (D18).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .core import _HEX64, Dictionary, Settings, check_sites

MLP_NAMES = ("W_fc", "b_fc", "W_proj", "b_proj")
TRAINED = ("W_enc", "W_dec", "b_enc", "b_pre", "b_dec")


def _layer(value: Any, n_layers: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < n_layers:
        raise ValueError(f"layer must be an int in range({n_layers}), got {value!r}")
    return int(value)


def mlp_weights(params: Any, layer: int) -> dict[str, np.ndarray]:
    block = params.blocks[_layer(layer, len(params.blocks))].mlp
    values = (block.c_fc.weight, block.c_fc.bias, block.c_proj.weight, block.c_proj.bias)
    return {name: np.array(value, dtype=np.float32, copy=True) for name, value in zip(MLP_NAMES, values)}


def _check_mlp(weights: Any, d_in: int | None = None, d_out: int | None = None) -> dict[str, np.ndarray]:
    """The MLP's weights as float32 arrays, checked against each other and, when given, the widths
    it reads (``d_in``) and writes (``d_out``)."""
    if not isinstance(weights, Mapping) or set(weights) != set(MLP_NAMES):
        got = sorted(weights) if isinstance(weights, Mapping) else weights
        raise ValueError(f"MLP weights must hold exactly {MLP_NAMES}, got {got!r}")
    try:
        w = {name: np.asarray(weights[name], dtype=np.float32) for name in MLP_NAMES}
    except (TypeError, ValueError) as error:
        raise ValueError("MLP weights must be numeric arrays") from error
    if w["W_fc"].ndim != 2:
        raise ValueError(f"W_fc must be two-dimensional, got shape {w['W_fc'].shape}")
    d, h = w["W_fc"].shape
    shapes = {"W_fc": (d, h), "b_fc": (h,), "W_proj": (h, d), "b_proj": (d,)}
    for name in MLP_NAMES:
        if w[name].shape != shapes[name]:
            raise ValueError(f"{name} must have shape {shapes[name]}, got {w[name].shape}")
        if not np.isfinite(w[name]).all():
            raise ValueError(f"{name} holds a value that is not finite")
    for width in (d_in, d_out):
        if width is not None and width != d:
            raise ValueError(f"the MLP reads and writes width {d}, not the dictionary's {width}")
    return w


@cache
def _kernels() -> dict[str, Any]:
    """The jitted MLP and pair Jacobian, built on first use."""
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST

    def gelu(h):
        return jax.nn.gelu(h, approximate=True)

    def mlp_fn(w, x):
        hidden = gelu(jnp.matmul(x, w["W_fc"], precision=highest) + w["b_fc"])
        return jnp.matmul(hidden, w["W_proj"], precision=highest) + w["b_proj"]

    def jacobian_fn(w, codes, W_dec, b_dec, scale_out, W_enc, b_pre, b_enc, scale_in, *, k):
        """``pair_jacobian`` from the input site's arrays (decoder side) and the output site's
        (encoder side); differentiable in all of them."""
        _, idx_in = jax.lax.top_k(codes, k)  # ties go to the lower index
        x_hat = (jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out
        h = jnp.matmul(x_hat, w["W_fc"], precision=highest) + w["b_fc"]
        y = jnp.matmul(gelu(h), w["W_proj"], precision=highest) + w["b_proj"]
        pre = jnp.matmul(y / scale_in - b_pre, W_enc, precision=highest) + b_enc
        values, idx_out = jax.lax.top_k(pre, k)
        slope = jax.jvp(gelu, (h,), (jnp.ones_like(h),))[1]  # GELU's derivative at each hidden unit
        # d pre[j] / d codes[i] = scale_out / scale_in * W_dec[i] @ W_fc @ diag(slope) @ W_proj @ W_enc[:, j]
        rows = jnp.matmul(W_dec[idx_in] * scale_out, w["W_fc"], precision=highest) * slope[:, None, :]
        rows = jnp.matmul(rows, w["W_proj"], precision=highest)  # (B, k in, d)
        cols = jnp.moveaxis(W_enc[:, idx_out], 0, 1) / scale_in  # (B, d, k out)
        jac = jnp.matmul(rows, cols, precision=highest)
        return jnp.where((values > 0)[:, None, :], jac, 0.0), idx_in, idx_out

    return {"mlp": jax.jit(mlp_fn), "jacobian": jax.jit(jacobian_fn, static_argnames=("k",)),
            "jacobian_raw": jacobian_fn}


def mlp(weights: Mapping[str, Any], x: Any) -> np.ndarray:
    w = _check_mlp(weights)
    try:
        x = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("x must be a numeric array") from error
    if x.ndim < 1 or x.shape[-1] != w["W_fc"].shape[0]:
        raise ValueError(f"x must end in an axis of width {w['W_fc'].shape[0]}, got shape {x.shape}")
    return np.array(_kernels()["mlp"](w, x), dtype=np.float32)


def pair_jacobian(dictionary: Dictionary, codes: Any, weights: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(dictionary, Dictionary) or dictionary.settings.kind != "jsae":
        kind = dictionary.settings.kind if isinstance(dictionary, Dictionary) else type(dictionary).__name__
        raise ValueError(f"the pair Jacobian is a jsae's, not a {kind}'s")
    m, k = dictionary.settings.width, dictionary.settings.k
    try:
        codes = np.asarray(codes, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("codes must be a numeric array") from error
    if codes.ndim != 2 or codes.shape[1] != m:
        raise ValueError(f"codes must be shaped (B, {m}), got {codes.shape}")
    w = _check_mlp(weights, dictionary.d_out, dictionary.d_in)
    a = dictionary.arrays
    jac, idx_in, idx_out = _kernels()["jacobian"](
        w, codes, a["W_dec"][0], a["b_dec"][0], a["scale_out"][0],
        a["W_enc"][1], a["b_pre"][1], a["b_enc"][1], a["scale_in"][1], k=k)
    return np.array(jac, dtype=np.float32), np.array(idx_in), np.array(idx_out)


def _site_data(data: Any, sites: tuple[tuple[str, str], ...], kind: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or len(data) != len(sites):
        raise ValueError(f"data must hold one (inputs, outputs) pair per site, {len(sites)} in all")
    pairs = []
    for s, pair in enumerate(data):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"site {s}'s data must be an (inputs, outputs) pair")
        arrays = []
        for name, value in zip(("inputs", "outputs"), pair):
            array = np.asarray(value)
            if array.dtype != np.float32 or array.ndim != 2 or 0 in array.shape:
                raise ValueError(f"site {s}'s {name} must be a non-empty float32 array shaped (N, d), "
                                 f"got {array.dtype} {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"site {s}'s {name} hold a value that is not finite")
            arrays.append(array)
        x, y = arrays
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"site {s}'s inputs and outputs must have the same rows")
        if kind != "transcoder" and not (x is y or np.array_equal(x, y)):
            raise ValueError(f"a {kind} reconstructs its input: site {s}'s inputs and outputs must be equal")
        pairs.append((x, y))
    n, d_in, d_out = pairs[0][0].shape[0], pairs[0][0].shape[1], pairs[0][1].shape[1]
    for s, (x, y) in enumerate(pairs):
        if x.shape[0] != n:
            raise ValueError(f"every site needs the same rows: site {s} has {x.shape[0]}, site 0 {n}")
        if x.shape[1] != d_in or y.shape[1] != d_out:
            raise ValueError(f"every site must have the same widths: site {s} has {x.shape[1]} and {y.shape[1]}")
    if d_in != d_out:  # the encoder starts as the decoder transposed
        raise ValueError(f"inputs and outputs must have the same width, got {d_in} and {d_out}")
    return pairs


def _scale(name: str, rows: np.ndarray) -> np.float32:
    """The root mean square norm of ``rows`` over the square root of their width, in float64."""
    rows = rows.astype(np.float64)
    value = np.float32(math.sqrt(float(np.einsum("nd,nd->", rows, rows)) / rows.shape[0] / rows.shape[1]))
    if not value > 0:
        raise ValueError(f"{name} are all zero, so they have no scale")
    return value


def _mean(rows: np.ndarray, scale: np.float32) -> np.ndarray:
    """The mean of the scaled rows, in float64."""
    return (rows.astype(np.float64) / np.float64(scale)).mean(0)


def _schedule(settings: Settings) -> np.ndarray:
    """The learning rate at each step: linear from 0 over the warmup, held, then linear to 0 over
    the decay."""
    steps = settings.steps
    warmup, decay = min(settings.warmup, steps // 10), int(steps * settings.decay_fraction)
    step = np.arange(steps, dtype=np.float64)
    rise = np.minimum(step / warmup, 1.0) if warmup else np.ones(steps)
    fall = np.minimum((steps - step) / decay, 1.0) if decay else np.ones(steps)
    return (settings.lr * np.minimum(rise, fall)).astype(np.float32)


def _reported(steps: int) -> frozenset[int]:
    """The steps, counted from 1, that end each tenth of the run; the last among them."""
    return frozenset(-(-i * steps // 10) for i in range(1, 11))


@cache
def _step(kind: str, n_sites: int, width: int, k: int, dead_after: int, aux_weight: float, lam: float,
          jac_rows: int, same: bool) -> Any:
    """One jitted training step for these static settings."""
    import jax
    import jax.numpy as jnp
    import optax

    highest = jax.lax.Precision.HIGHEST
    adam = optax.scale_by_adam()
    jacobian = _kernels()["jacobian_raw"]
    units = 2 if kind == "jsae" else 1
    active = [(s + 1) * width // n_sites if kind == "staircase" else width for s in range(n_sites)]
    k_aux = min(4 * k, width // 2)

    def keep(pre, count):
        """The ``count`` largest pre-activations clipped at zero, every other latent zero."""
        values, index = jax.lax.top_k(pre, count)  # ties go to the lower index
        rows = jnp.arange(pre.shape[0])[:, None]
        return jnp.zeros(pre.shape, jnp.float32).at[rows, index].set(jnp.maximum(values, 0.0))

    def loss_fn(params, consts, since, xs, ys):
        total, recon, fired, codes_in = 0.0, [], [], None
        for s in range(n_sites):
            u = s if units == 2 else 0
            x = xs[s] / consts["scale_in"][s]
            y = x if same else ys[s] / consts["scale_out"][s]
            pre = jnp.matmul(x - params["b_pre"][s], params["W_enc"][u], precision=highest) + params["b_enc"][s]
            pre = jnp.where(jnp.arange(width) < active[s], pre, -jnp.inf)
            codes = keep(pre, k)
            y_hat = jnp.matmul(codes, params["W_dec"][u], precision=highest) + params["b_dec"][s]
            error = y - y_hat
            mse = jnp.mean(jnp.sum(error ** 2, axis=1)) / y.shape[1]
            total = total + mse
            if k_aux > 0:  # AuxK: the dead latents' largest pre-activations reconstruct the error, held constant
                dead = since[s] >= dead_after
                aux_codes = keep(jnp.where(dead, pre, -jnp.inf), k_aux)
                aux_hat = jnp.matmul(aux_codes, params["W_dec"][u], precision=highest)
                aux = jnp.mean(jnp.sum((jax.lax.stop_gradient(error) - aux_hat) ** 2, axis=1)) / y.shape[1]
                total = total + aux_weight * aux
            recon.append(mse)
            fired.append(jnp.any(codes > 0, axis=0))
            if s == 0:
                codes_in = codes
        if lam > 0:
            jac, _, _ = jacobian(consts["mlp"], codes_in[:jac_rows], params["W_dec"][0], params["b_dec"][0],
                                 consts["scale_out"][0], params["W_enc"][1], params["b_pre"][1],
                                 params["b_enc"][1], consts["scale_in"][1], k=k)
            total = total + lam * jnp.mean(jnp.abs(jac))
        return total, (jnp.stack(recon), jnp.stack(fired))

    def step(params, state, consts, since, xs, ys, lr):
        grads, (recon, fired) = jax.grad(loss_fn, has_aux=True)(params, consts, since, xs, ys)
        updates, state = adam.update(grads, state, params)
        params = {name: params[name] - lr * updates[name] for name in TRAINED}
        W_dec = params["W_dec"]
        params["W_dec"] = W_dec / jnp.linalg.norm(W_dec, axis=2, keepdims=True)
        since = jnp.where(fired, 0, since + 1)
        return params, state, since, recon

    return adam, jax.jit(step)


def train(settings: Settings, sites: Sequence[Sequence[str]], data: Sequence[tuple[Any, Any]], *,
          mlp_weights: Mapping[str, Any] | None = None, key: str | None = None,
          progress: Callable[[int, int, list[float]], None] | None = None) -> Dictionary:
    if not isinstance(settings, Settings):
        raise ValueError(f"settings must be a Settings, got {type(settings).__name__}")
    kind, m = settings.kind, settings.width
    sites = check_sites(kind, sites)
    n_sites, units = len(sites), 2 if kind == "jsae" else 1
    if kind == "staircase" and m % n_sites:
        raise ValueError(f"width {m} must be divisible by the {n_sites} staircase sites")
    if key is not None and (not isinstance(key, str) or _HEX64.fullmatch(key) is None):
        raise ValueError(f"key must be None or 64 lowercase hex characters, got {key!r}")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    pairs = _site_data(data, sites, kind)
    n, d_in, d_out = pairs[0][0].shape[0], pairs[0][0].shape[1], pairs[0][1].shape[1]
    if kind == "jsae":
        if mlp_weights is None:
            raise ValueError("a jsae needs its block's mlp_weights")
        weights = _check_mlp(mlp_weights, d_out, d_in)
    elif mlp_weights is not None:
        raise ValueError(f"only a jsae takes mlp_weights, not a {kind}")
    else:
        weights = None

    import jax
    import jax.numpy as jnp

    scale_in = np.array([_scale(f"site {s}'s inputs", x) for s, (x, _) in enumerate(pairs)], dtype=np.float32)
    scale_out = np.array([_scale(f"site {s}'s outputs", y) for s, (_, y) in enumerate(pairs)], dtype=np.float32)
    keys = jax.random.split(jax.random.PRNGKey(settings.seed), units)
    W_dec = np.empty((units, m, d_out), dtype=np.float32)
    for u in range(units):
        w = np.asarray(jax.random.normal(keys[u], (m, d_out)), dtype=np.float64)
        W_dec[u] = w / np.linalg.norm(w, axis=1, keepdims=True)
    W_enc = np.ascontiguousarray(W_dec.transpose(0, 2, 1))
    if kind == "transcoder":
        W_enc = W_enc * np.float32(0.1)
    params = {"W_enc": W_enc, "W_dec": W_dec, "b_enc": np.zeros((n_sites, m), np.float32),
              "b_pre": np.array([_mean(x, scale_in[s]) for s, (x, _) in enumerate(pairs)], dtype=np.float32),
              "b_dec": np.array([_mean(y, scale_out[s]) for s, (_, y) in enumerate(pairs)], dtype=np.float32)}
    consts = {"scale_in": scale_in, "scale_out": scale_out}
    if weights is not None:
        consts["mlp"] = weights

    same = kind != "transcoder"
    adam, step = _step(kind, n_sites, m, settings.k, settings.dead_after, settings.aux_weight, settings.lam,
                       min(settings.jac_batch, settings.batch), same)
    params = {name: jnp.asarray(value) for name, value in params.items()}
    consts = jax.tree_util.tree_map(jnp.asarray, consts)
    state = adam.init(params)
    since = jnp.zeros((n_sites, m), jnp.int32)
    rates, reported = _schedule(settings), _reported(settings.steps)
    rng = np.random.default_rng(settings.seed)
    for s in range(settings.steps):
        rows = rng.integers(0, n, settings.batch)
        xs = np.stack([x[rows] for x, _ in pairs])
        ys = None if same else np.stack([y[rows] for _, y in pairs])
        params, state, since, recon = step(params, state, consts, since, xs, ys, rates[s])
        if progress is not None and s + 1 in reported:
            progress(s + 1, settings.steps, [float(v) for v in np.asarray(recon)])

    arrays = {name: np.array(value, dtype=np.float32) for name, value in params.items()}
    arrays |= {"scale_in": scale_in, "scale_out": scale_out}
    return Dictionary(settings, sites, arrays, key)


def code_identity() -> str:
    """The git blob hash of this module's source file, as ``git hash-object`` prints it (D18)."""
    data = Path(__file__).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
