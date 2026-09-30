"""GPT-2, matching Levanter's architecture, with every canonical tap built in.

Spec: docs/specs/phase-1.md, "The reference model".

Architecture, pinned by G0.3's recorded parameter counts: tied input and output
embeddings, learned position embeddings, biases on every linear layer, pre-norm
blocks with two layer norms each, and a final layer norm. The GELU variant,
layer-norm epsilon and attention scaling follow Levanter's GPT-2 at the pinned
version; G1.2 verifies them against Levanter itself. float32 throughout. The
implementation uses Haliax named axes internally but returns plain arrays.

``params`` is a pytree of float32 arrays; ``init_params`` is deterministic in its
seed. ``GPT2Model`` satisfies the oracle's ``InstrumentedModel`` protocol:

* ``fast(params, tokens)`` returns logits ``(Batch, Pos, Vocab)`` from a
  ``jit``-compiled function.
* ``instrumented(params, tokens, taps, interventions=None)`` returns the logits and
  a dict with exactly the requested taps, shaped as ``registry.tap_spec`` says, from
  a separately compiled function that outputs them directly, never through host
  callbacks. A name outside ``registry.tap_names(n_layers)`` raises KeyError.
* ``interventions`` maps tap names to pure JAX functions, each applied at its tap
  point. The recorded tap is the value after the intervention, which is what flows
  downstream. It exists for oracle discrimination tests now and for Phase 3's
  interventions later.

``load_hf_gpt2(path, config)`` loads a Hugging Face GPT-2 export into our params, for G1.2.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.attention import causal_mask

from ..trace.registry import tap_names, tap_spec

Intervention = Callable[[Any], Any]

# What Levanter's GPT-2 does at the pinned version (marin-levanter
# 0.2.129.dev36313914471, levanter/models/gpt2.py, with haliax/nn/normalization.py
# and levanter/layers/attention.py), and what this file copies:
#
# * GELU: Gpt2Config.activation_function defaults to ``gelu_new``, which maps to
#   ``hnn.gelu(approximate=True)``, i.e. ``jax.nn.gelu`` with the tanh approximation.
# * Layer norm: ``layer_norm_epsilon = 1e-5``. Haliax computes
#   ``(x - mean) * rsqrt(var + eps)`` with the biased variance, then
#   ``weight * out + bias``.
# * Attention scaling: ``dot_product_attention`` multiplies the query by
#   ``1 / sqrt(head_size)`` before the QK dot product. Scaling by inverse layer
#   index and upcasting are off by default. Masked scores become -1e9 before the
#   softmax (-1e10 in the blockwise path Levanter picks on CPU; either underflows
#   to an exact zero probability).
#
# Parameters are stored the way a Hugging Face GPT-2 export stores them: linear
# weights as (in, out), every bias and layer-norm vector one-dimensional, fused
# q, k, v in one c_attn. That keeps load_hf_gpt2 a rename, and leaves exactly the
# weight matrices and embeddings with two or more dimensions, which is how the
# trainer picks what to decay. The forward pass wraps them in Haliax named axes.
#
# Taps follow TransformerLens: hook_pos_embed is broadcast over the batch,
# hook_pattern holds the attention probabilities after the causal mask and
# softmax, and the hook_normalized taps are layer-norm outputs after scale and
# bias. Both modes trace the same function, _forward; fast() returns only the
# logits and lets XLA drop the rest.

LN_EPS = 1e-5
MASKED_SCORE = -1e9
EMBED_INIT_STD = 0.02  # Levanter's initializer_range; position embeddings get half of it

# Haliax axis names for the tap-spec axis names in trace/registry.py.
_AXIS_NAMES = {"Batch": "batch", "Pos": "position", "QPos": "position", "KPos": "key_position",
               "Embed": "embed", "Head": "heads", "HeadDim": "head_size"}


@dataclass(frozen=True)
class GPT2Config:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    seq: int

    def __post_init__(self) -> None:
        for name in ("vocab", "d_model", "n_layers", "n_heads", "d_mlp", "seq"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Exact parameter count for this architecture, including tied embeddings once."""
        d, m = self.d_model, self.d_mlp
        per_block = (
            2 * 2 * d  # ln_1 and ln_2, weight and bias
            + d * 3 * d + 3 * d  # attn.c_attn
            + d * d + d  # attn.c_proj
            + d * m + m  # mlp.c_fc
            + m * d + d  # mlp.c_proj
        )
        return self.vocab * d + self.seq * d + self.n_layers * per_block + 2 * d

    @classmethod
    def from_size(cls, size: str, vocab: int) -> GPT2Config:
        """Size "S": 4 layers, d_model 256, 4 heads, d_mlp 1024, seq 256.
        Size "M": 4 layers, d_model 512, 8 heads, d_mlp 2048, seq 256."""
        sizes = {
            "S": dict(d_model=256, n_layers=4, n_heads=4, d_mlp=1024, seq=256),
            "M": dict(d_model=512, n_layers=4, n_heads=8, d_mlp=2048, seq=256),
        }
        if size not in sizes:
            raise ValueError(f"size must be one of {sorted(sizes)}, got {size!r}")
        return cls(vocab=vocab, **sizes[size])


class Linear(eqx.Module):
    weight: jax.Array  # (in, out)
    bias: jax.Array  # (out,)


class LayerNorm(eqx.Module):
    weight: jax.Array  # (d_model,)
    bias: jax.Array  # (d_model,)


class Attention(eqx.Module):
    c_attn: Linear  # (d_model, 3 * d_model): q, k, v, each laid out heads-major
    c_proj: Linear  # (d_model, d_model), input laid out heads-major


class Mlp(eqx.Module):
    c_fc: Linear  # (d_model, d_mlp)
    c_proj: Linear  # (d_mlp, d_model)


class Block(eqx.Module):
    ln_1: LayerNorm
    attn: Attention
    ln_2: LayerNorm
    mlp: Mlp


class GPT2Params(eqx.Module):
    """Every parameter, float32. ``wte`` is also the unembedding."""

    wte: jax.Array  # (vocab, d_model)
    wpe: jax.Array  # (seq, d_model)
    blocks: tuple[Block, ...]
    ln_f: LayerNorm


def init_params(config: GPT2Config, seed: int) -> Any:
    """Levanter's initialization: truncated normals cut at 3 standard deviations,
    ``1 / sqrt(fan_in)`` for linear weights, 0.02 for token and 0.01 for position
    embeddings; zero biases, unit layer-norm weights."""
    d, m = config.d_model, config.d_mlp
    keys = iter(jax.random.split(jax.random.key(seed), 2 + 4 * config.n_layers))

    def normal(std: float, *shape: int) -> jax.Array:
        return jax.random.truncated_normal(next(keys), -3.0, 3.0, shape, dtype=jnp.float32) * jnp.float32(std)

    def linear(n_in: int, n_out: int) -> Linear:
        return Linear(normal(1 / math.sqrt(n_in), n_in, n_out), jnp.zeros((n_out,), jnp.float32))

    def layer_norm() -> LayerNorm:
        return LayerNorm(jnp.ones((d,), jnp.float32), jnp.zeros((d,), jnp.float32))

    wte = normal(EMBED_INIT_STD, config.vocab, d)
    wpe = normal(EMBED_INIT_STD / 2, config.seq, d)
    blocks = tuple(
        Block(
            ln_1=layer_norm(),
            attn=Attention(c_attn=linear(d, 3 * d), c_proj=linear(d, d)),
            ln_2=layer_norm(),
            mlp=Mlp(c_fc=linear(d, m), c_proj=linear(m, d)),
        )
        for _ in range(config.n_layers)
    )
    return GPT2Params(wte=wte, wpe=wpe, blocks=blocks, ln_f=layer_norm())


class GPT2Model:
    config: GPT2Config

    def __init__(self, config: GPT2Config) -> None:
        self.config = config
        self._tap_names = tap_names(config.n_layers)
        self._fast = jax.jit(partial(_fast, config))
        self._instrumented = jax.jit(partial(_instrumented, config), static_argnums=(2, 3))

    def fast(self, params: Any, tokens: Any) -> Any:
        return self._fast(params, self._check_tokens(tokens))

    def instrumented(self, params: Any, tokens: Any, taps: Sequence[str],
                     interventions: Mapping[str, Intervention] | None = None) -> tuple[Any, dict[str, Any]]:
        requested = tuple(dict.fromkeys(taps))
        interventions = dict(interventions or {})
        unknown = sorted((set(requested) | set(interventions)) - set(self._tap_names))
        if unknown:
            raise KeyError(f"not canonical tap names for a {self.config.n_layers}-layer model: {unknown}")
        # Functions hash by identity, so each new intervention compiles its own program.
        return self._instrumented(params, self._check_tokens(tokens), requested,
                                  tuple(sorted(interventions.items(), key=lambda item: item[0])))

    def _check_tokens(self, tokens: Any) -> Any:
        """Shape and dtype only, so this also works on tracers. An out-of-range id
        embeds as NaN (see _forward), which no loss or oracle lets through."""
        shape, dtype = np.shape(tokens), getattr(tokens, "dtype", None)
        if len(shape) != 2 or not 1 <= shape[1] <= self.config.seq:
            raise ValueError(f"tokens must be (batch, pos) with 1 <= pos <= {self.config.seq}, got {shape}")
        if dtype is None or not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"tokens must be integer ids, got dtype {dtype}")
        return tokens


def _fast(config: GPT2Config, params: GPT2Params, tokens: Any) -> jax.Array:
    return _forward(config, params, tokens, {})[0]


def _instrumented(config: GPT2Config, params: GPT2Params, tokens: Any, taps: tuple[str, ...],
                  interventions: tuple[tuple[str, Intervention], ...]) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits, recorded = _forward(config, params, tokens, dict(interventions))
    return logits, {name: recorded[name] for name in taps}


def _forward(config: GPT2Config, params: GPT2Params, tokens: Any,
             interventions: Mapping[str, Intervention]) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Logits ``(Batch, Pos, Vocab)`` and every canonical tap, in one pure function."""
    batch, pos = tokens.shape
    Batch, Pos, KPos = hax.Axis("batch", batch), hax.Axis("position", pos), hax.Axis("key_position", pos)
    Embed, Vocab = hax.Axis("embed", config.d_model), hax.Axis("vocab", config.vocab)
    Heads, HeadSize = hax.Axis("heads", config.n_heads), hax.Axis("head_size", config.d_head)
    Qkv, Mlp_ = hax.Axis("qkv", 3), hax.Axis("mlp", config.d_mlp)
    recorded: dict[str, jax.Array] = {}

    def tap(name: str, x: hax.NamedArray) -> hax.NamedArray:
        spec = tap_spec(name, batch=batch, seq=pos, d_model=config.d_model, n_heads=config.n_heads,
                        d_head=config.d_head)
        axes = tuple(_AXIS_NAMES[axis.name] for axis in spec.axes)
        value = x.rearrange(axes).array
        if name in interventions:
            changed = interventions[name](value)
            if jnp.shape(changed) != value.shape or jnp.result_type(changed) != value.dtype:
                raise ValueError(f"intervention at {name} must keep shape {value.shape} and dtype {value.dtype},"
                                 f" got {jnp.shape(changed)} {jnp.result_type(changed)}")
            value = jnp.asarray(changed)
            x = hax.named(value, tuple(x.resolve_axis(axis) for axis in axes))
        recorded[name] = value
        return x

    def linear(x: hax.NamedArray, layer: Linear, In: tuple[hax.Axis, ...], Out: tuple[hax.Axis, ...]) -> hax.NamedArray:
        weight = hax.named(layer.weight.reshape(tuple(a.size for a in In + Out)), In + Out)
        bias = hax.named(layer.bias.reshape(tuple(a.size for a in Out)), Out)
        return hax.dot(x, weight, axis=In) + bias

    def layer_norm(x: hax.NamedArray, ln: LayerNorm) -> hax.NamedArray:
        mean = x.mean(Embed)
        var = x.var(Embed)
        out = (x - mean) * hax.rsqrt(var + LN_EPS)
        return hax.named(ln.weight, Embed) * out + hax.named(ln.bias, Embed)

    # mode="fill" turns an out-of-range id into a NaN row rather than a silent clamp.
    ids = jnp.asarray(tokens)
    embed = tap("hook_embed", hax.named(jnp.take(params.wte, ids, axis=0, mode="fill", fill_value=jnp.nan),
                                        (Batch, Pos, Embed)))
    pos_embed = tap("hook_pos_embed", hax.broadcast_to(hax.named(params.wpe[:pos], (Pos, Embed)),
                                                       (Batch, Pos, Embed)))
    resid = embed + pos_embed

    mask = causal_mask(Pos, KPos)
    scale = 1 / math.sqrt(config.d_head)
    for layer, block in enumerate(params.blocks):
        prefix = f"blocks.{layer}."
        resid = tap(prefix + "hook_resid_pre", resid)

        qkv = linear(layer_norm(resid, block.ln_1), block.attn.c_attn, (Embed,), (Qkv, Heads, HeadSize))
        q, k, v = qkv.unbind(Qkv)
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        scores = hax.dot(q * scale, k, axis=HeadSize)
        scores = hax.where(mask, scores, MASKED_SCORE)
        pattern = tap(prefix + "attn.hook_pattern", hnn.softmax(scores, axis=KPos))
        z = tap(prefix + "attn.hook_z", hax.dot(pattern, v, axis=KPos))
        attn_out = tap(prefix + "hook_attn_out", linear(z, block.attn.c_proj, (Heads, HeadSize), (Embed,)))
        resid = tap(prefix + "hook_resid_mid", resid + attn_out)

        normalized = tap(prefix + "ln2.hook_normalized", layer_norm(resid, block.ln_2))
        hidden = hnn.gelu(linear(normalized, block.mlp.c_fc, (Embed,), (Mlp_,)), approximate=True)
        mlp_out = tap(prefix + "hook_mlp_out", linear(hidden, block.mlp.c_proj, (Mlp_,), (Embed,)))
        resid = tap(prefix + "hook_resid_post", resid + mlp_out)

    final = tap("ln_final.hook_normalized", layer_norm(resid, params.ln_f))
    logits = hax.dot(final, hax.named(params.wte, (Vocab, Embed)), axis=Embed)
    return logits.rearrange((Batch, Pos, Vocab)).array, recorded


def load_hf_gpt2(path: str | Path, config: GPT2Config) -> Any:
    """Params from a Hugging Face GPT-2 export: a ``model.safetensors`` file, or a
    directory holding one or a sharded ``model.safetensors.index.json``.

    Keys may carry a ``transformer.`` prefix. Older exports' causal-mask buffers
    (``h.N.attn.bias`` shaped as a mask, ``h.N.attn.masked_bias``) are ignored, and
    ``lm_head.weight``, if present, must equal ``wte.weight``. Any other missing or
    unexpected key, any shape that differs from ``config``, and any dtype other
    than float32 raises ValueError: G1.2 compares exact parameter counts and
    float32 logits, so nothing here pads, truncates or casts.
    """
    from safetensors.numpy import load_file

    tensors: dict[str, np.ndarray] = {}
    for file in _safetensors_files(Path(path)):
        for key, value in load_file(file).items():
            name = key.removeprefix("transformer.")
            if name in tensors:
                raise ValueError(f"{name} appears twice in {path}")
            tensors[name] = value

    d, m, n_ctx = config.d_model, config.d_mlp, config.seq
    expected: dict[str, tuple[int, ...]] = {"wte.weight": (config.vocab, d), "wpe.weight": (n_ctx, d),
                                            "ln_f.weight": (d,), "ln_f.bias": (d,)}
    for i in range(config.n_layers):
        h = f"h.{i}."
        expected |= {
            h + "ln_1.weight": (d,), h + "ln_1.bias": (d,),
            h + "attn.c_attn.weight": (d, 3 * d), h + "attn.c_attn.bias": (3 * d,),
            h + "attn.c_proj.weight": (d, d), h + "attn.c_proj.bias": (d,),
            h + "ln_2.weight": (d,), h + "ln_2.bias": (d,),
            h + "mlp.c_fc.weight": (d, m), h + "mlp.c_fc.bias": (m,),
            h + "mlp.c_proj.weight": (m, d), h + "mlp.c_proj.bias": (d,),
        }

    lm_head = tensors.pop("lm_head.weight", None)
    for i in range(config.n_layers):
        tensors.pop(f"h.{i}.attn.masked_bias", None)
        buffer = tensors.get(f"h.{i}.attn.bias")
        if buffer is not None and buffer.ndim == 4:  # (1, 1, n_positions, n_positions) mask, not a parameter
            del tensors[f"h.{i}.attn.bias"]
    missing, unexpected = sorted(set(expected) - set(tensors)), sorted(set(tensors) - set(expected))
    if missing or unexpected:
        raise ValueError(f"{path} does not match a GPT-2 of {config}: missing {missing}, unexpected {unexpected}")
    for name, shape in expected.items():
        value = tensors[name]
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"{name} is {value.dtype}{list(value.shape)}, expected float32{list(shape)}")
    if lm_head is not None and not np.array_equal(lm_head, tensors["wte.weight"]):
        raise ValueError(f"lm_head.weight differs from wte.weight in {path}; this model ties them")

    def get(name: str) -> jax.Array:
        return jnp.asarray(tensors[name])

    def linear(prefix: str) -> Linear:
        return Linear(get(prefix + ".weight"), get(prefix + ".bias"))

    def layer_norm(prefix: str) -> LayerNorm:
        return LayerNorm(get(prefix + ".weight"), get(prefix + ".bias"))

    blocks = tuple(
        Block(
            ln_1=layer_norm(f"h.{i}.ln_1"),
            attn=Attention(c_attn=linear(f"h.{i}.attn.c_attn"), c_proj=linear(f"h.{i}.attn.c_proj")),
            ln_2=layer_norm(f"h.{i}.ln_2"),
            mlp=Mlp(c_fc=linear(f"h.{i}.mlp.c_fc"), c_proj=linear(f"h.{i}.mlp.c_proj")),
        )
        for i in range(config.n_layers)
    )
    return GPT2Params(wte=get("wte.weight"), wpe=get("wpe.weight"), blocks=blocks, ln_f=layer_norm("ln_f"))


def _safetensors_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [path / name for name in sorted(set(weight_map.values()))]
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {path}")
