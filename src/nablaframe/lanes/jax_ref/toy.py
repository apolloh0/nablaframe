"""G0.6 toy model: a 2-layer, d=128 GPT used to measure mode equality under jit.

Spec: docs/specs/phase-0.md, G0.6 row and "Mode-equality oracle".

Implement ``ToyGPT`` so it satisfies ``nablaframe.oracle.mode_equality.InstrumentedModel``:

* ``fast(params, tokens)`` returns logits only, from a ``jax.jit``-compiled function.
* ``instrumented(params, tokens, taps)`` returns ``(logits, {tap: tensor})`` from a
  separately ``jit``-compiled function that also outputs the requested taps. Taps
  are recorded as extra outputs, never by host callbacks or side effects.
* Tap names come from ``nablaframe.trace.registry.tap_names(2)``; requesting any
  other name raises KeyError.
* Pre-norm GPT-2 block: LayerNorm, causal attention, residual add, LayerNorm, MLP
  with GELU, residual add; learned positional embeddings; untied unembedding.
* float32 throughout; parameters come only from ``init(seed)``, deterministically.

G0.6 then runs both functions on 3 seeds and records whether the logits are
bitwise equal, and the max abs difference if not.
"""

# Both modes trace the same function, _forward, which computes every tap as a
# named intermediate. fast() returns only the logits, so XLA is free to drop or
# fuse the intermediates; instrumented() also returns the requested ones, which
# pins them as outputs. That difference in what XLA may optimize is exactly what
# G0.6 measures.
#
# Tap values follow TransformerLens: hook_pos_embed is broadcast over the batch,
# hook_pattern holds the attention probabilities after the causal mask and softmax,
# and ln2.hook_normalized and ln_final.hook_normalized are the LayerNorm outputs
# after scale and bias. tokens are int token ids shaped (batch, pos) with
# pos <= SEQ; logits are float32 shaped (batch, pos, VOCAB).

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nablaframe.trace.registry import tap_names

N_LAYERS = 2
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
VOCAB = 256
SEQ = 64

LN_EPS = 1e-5
INIT_STD = 0.02
TAPS: tuple[str, ...] = tap_names(N_LAYERS)


class ToyGPT:
    def __init__(self) -> None:
        import jax

        self._fast = jax.jit(lambda params, tokens: _forward(params, tokens)[0])
        self._instrumented = jax.jit(_instrumented, static_argnums=2)

    def init(self, seed: int) -> Any:
        """Parameters as a pytree, deterministic in ``seed``."""
        import jax
        import jax.numpy as jnp

        keys = iter(jax.random.split(jax.random.PRNGKey(seed), 3 + 6 * N_LAYERS))

        def normal(*shape: int) -> Any:
            return INIT_STD * jax.random.normal(next(keys), shape, dtype=jnp.float32)

        def zeros(*shape: int) -> Any:
            return jnp.zeros(shape, dtype=jnp.float32)

        def layer_norm() -> dict[str, Any]:
            return {"w": jnp.ones((D_MODEL,), dtype=jnp.float32), "b": zeros(D_MODEL)}

        blocks = [
            {
                "ln1": layer_norm(),
                "W_Q": normal(N_HEADS, D_MODEL, D_HEAD),
                "W_K": normal(N_HEADS, D_MODEL, D_HEAD),
                "W_V": normal(N_HEADS, D_MODEL, D_HEAD),
                "b_Q": zeros(N_HEADS, D_HEAD),
                "b_K": zeros(N_HEADS, D_HEAD),
                "b_V": zeros(N_HEADS, D_HEAD),
                "W_O": normal(N_HEADS, D_HEAD, D_MODEL),
                "b_O": zeros(D_MODEL),
                "ln2": layer_norm(),
                "W_in": normal(D_MODEL, D_MLP),
                "b_in": zeros(D_MLP),
                "W_out": normal(D_MLP, D_MODEL),
                "b_out": zeros(D_MODEL),
            }
            for _ in range(N_LAYERS)
        ]
        return {
            "W_E": normal(VOCAB, D_MODEL),
            "W_pos": normal(SEQ, D_MODEL),
            "blocks": blocks,
            "ln_final": layer_norm(),
            "W_U": normal(D_MODEL, VOCAB),
            "b_U": zeros(VOCAB),
        }

    def fast(self, params: Any, tokens: Any) -> Any:
        return self._fast(params, tokens)

    def instrumented(self, params: Any, tokens: Any, taps: Sequence[str]) -> tuple[Any, dict[str, Any]]:
        requested = tuple(taps)
        unknown = sorted(set(requested) - set(TAPS))
        if unknown:
            raise KeyError(f"not canonical tap names for a {N_LAYERS}-layer model: {unknown}")
        return self._instrumented(params, tokens, tuple(dict.fromkeys(requested)))


def _instrumented(params: Any, tokens: Any, taps: tuple[str, ...]) -> tuple[Any, dict[str, Any]]:
    logits, recorded = _forward(params, tokens)
    return logits, {name: recorded[name] for name in taps}


def _layer_norm(x: Any, ln: dict[str, Any]) -> Any:
    import jax.numpy as jnp

    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered / jnp.sqrt(var + LN_EPS) * ln["w"] + ln["b"]


def _forward(params: Any, tokens: Any) -> tuple[Any, dict[str, Any]]:
    """Logits and every canonical tap, in one pure function."""
    import jax
    import jax.numpy as jnp

    batch, pos = tokens.shape
    taps: dict[str, Any] = {}

    embed = params["W_E"][tokens]
    pos_embed = jnp.broadcast_to(params["W_pos"][:pos], (batch, pos, D_MODEL))
    taps["hook_embed"] = embed
    taps["hook_pos_embed"] = pos_embed
    resid = embed + pos_embed

    causal = jnp.tril(jnp.ones((pos, pos), dtype=bool))
    for layer, block in enumerate(params["blocks"]):
        prefix = f"blocks.{layer}."
        taps[prefix + "hook_resid_pre"] = resid

        x = _layer_norm(resid, block["ln1"])
        q = jnp.einsum("bpd,hde->bphe", x, block["W_Q"]) + block["b_Q"]
        k = jnp.einsum("bpd,hde->bphe", x, block["W_K"]) + block["b_K"]
        v = jnp.einsum("bpd,hde->bphe", x, block["W_V"]) + block["b_V"]
        scores = jnp.einsum("bqhe,bkhe->bhqk", q, k) / jnp.sqrt(jnp.float32(D_HEAD))
        scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)
        pattern = jax.nn.softmax(scores, axis=-1)
        z = jnp.einsum("bhqk,bkhe->bqhe", pattern, v)
        attn_out = jnp.einsum("bphe,hed->bpd", z, block["W_O"]) + block["b_O"]
        taps[prefix + "attn.hook_pattern"] = pattern
        taps[prefix + "attn.hook_z"] = z
        taps[prefix + "hook_attn_out"] = attn_out

        resid = resid + attn_out
        taps[prefix + "hook_resid_mid"] = resid

        normalized = _layer_norm(resid, block["ln2"])
        taps[prefix + "ln2.hook_normalized"] = normalized
        hidden = jax.nn.gelu(normalized @ block["W_in"] + block["b_in"], approximate=True)
        mlp_out = hidden @ block["W_out"] + block["b_out"]
        taps[prefix + "hook_mlp_out"] = mlp_out

        resid = resid + mlp_out
        taps[prefix + "hook_resid_post"] = resid

    final = _layer_norm(resid, params["ln_final"])
    taps["ln_final.hook_normalized"] = final
    logits = final @ params["W_U"] + params["b_U"]
    return logits, taps
