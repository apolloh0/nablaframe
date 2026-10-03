"""Canonical tap names and their specs.

Spec: docs/specs/phase-0.md, "Trace schema v0", tap table.

Names are TransformerLens hook names. TransformerLens 4.0 removed the legacy
``HookedTransformer``; these names survive as aliases on ``TransformerBridge``
for pre-norm GPT-2-style models, so traces cross-check against TransformerLens
without a translation table. The vendored list in
``tests/data/transformerlens_legacy_hook_names.txt`` records where each name was
verified, and ``scripts/check_tl_names.py`` confirms them at runtime in nf-torch.

This module never imports TransformerLens.

Phase 2 delta 1 (docs/deltas/phase-2-delta-1.md):

* `LOGITS = "logits"`: the model's output, recorded by the engine (D12). It is the
  one canonical tap that isn't a TransformerLens hook, so `tap_names` does not
  include it; those stay TransformerLens names only.
* `tap_spec` gains a keyword `vocab: int | None = None`: For `LOGITS`, the axes are
  Batch, Pos and Vocab, and the site is `unembed`; `vocab` is required, and
  omitting it raises ValueError. Other names ignore it.
"""

from __future__ import annotations

from .schema import Axis, TapSpec

# Taps before the first block, in forward order.
GLOBAL_TAPS_BEFORE: tuple[str, ...] = ("hook_embed", "hook_pos_embed")

# Taps inside block L, in forward order. Each is prefixed with "blocks.{L}.".
PER_LAYER_TAPS: tuple[str, ...] = (
    "hook_resid_pre",
    "attn.hook_pattern",
    "attn.hook_z",
    "hook_attn_out",
    "hook_resid_mid",
    "ln2.hook_normalized",
    "hook_mlp_out",
    "hook_resid_post",
)

# Taps after the last block, in forward order.
GLOBAL_TAPS_AFTER: tuple[str, ...] = ("ln_final.hook_normalized",)

# The model's output, recorded by the engine (D12). Not a TransformerLens hook, so
# not in tap_names.
LOGITS = "logits"


def tap_names(n_layers: int) -> tuple[str, ...]:
    """Every canonical tap name for a model with ``n_layers`` blocks, in forward order.

    Order: ``GLOBAL_TAPS_BEFORE``, then ``PER_LAYER_TAPS`` for block 0, block 1, and
    so on, then ``GLOBAL_TAPS_AFTER``. Raises ValueError if ``n_layers < 1``.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be at least 1, got {n_layers}")
    per_layer = tuple(f"blocks.{layer}.{tap}" for layer in range(n_layers) for tap in PER_LAYER_TAPS)
    return GLOBAL_TAPS_BEFORE + per_layer + GLOBAL_TAPS_AFTER


def tap_spec(
    name: str,
    *,
    batch: int,
    seq: int,
    d_model: int,
    n_heads: int,
    d_head: int,
    dtype: str = "float32",
    vocab: int | None = None,
) -> TapSpec:
    """The TapSpec for one canonical tap name.

    Axes and sites, per the spec's tap table:

    ======================================  ===================================  ===========
    Tap                                     Axes                                 Site
    ======================================  ===================================  ===========
    hook_embed, hook_pos_embed              Batch, Pos, Embed                    embedding
    blocks.L.hook_resid_pre / _mid / _post  Batch, Pos, Embed                    residual
    blocks.L.attn.hook_pattern              Batch, Head, QPos, KPos              attention
    blocks.L.attn.hook_z                    Batch, Pos, Head, HeadDim            attention
    blocks.L.hook_attn_out                  Batch, Pos, Embed                    attention
    blocks.L.ln2.hook_normalized            Batch, Pos, Embed                    mlp
    blocks.L.hook_mlp_out                   Batch, Pos, Embed                    mlp
    ln_final.hook_normalized                Batch, Pos, Embed                    final_norm
    logits (D12)                            Batch, Pos, Vocab                    unembed
    ======================================  ===================================  ===========

    Sizes: Batch=batch, Pos=QPos=KPos=seq, Embed=d_model, Head=n_heads,
    HeadDim=d_head, Vocab=vocab. Raises KeyError for a name that is not canonical.

    For ``LOGITS``, the axes are Batch, Pos and Vocab, and the site is ``unembed``;
    ``vocab`` is required, and omitting it raises ValueError. Other names ignore it.
    """
    if name == LOGITS:
        if vocab is None:
            raise ValueError(f"tap_spec({LOGITS!r}) needs vocab")
        return TapSpec(name, (Axis("Batch", batch), Axis("Pos", seq), Axis("Vocab", vocab)), dtype, "unembed")

    tap = name
    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) != 3:
            raise KeyError(name)
        _, layer, tap = parts
        if not layer.isdigit() or layer != str(int(layer)) or tap not in PER_LAYER_TAPS:
            raise KeyError(name)
    elif name not in GLOBAL_TAPS_BEFORE + GLOBAL_TAPS_AFTER:
        raise KeyError(name)

    sizes = {"Batch": batch, "Pos": seq, "QPos": seq, "KPos": seq, "Embed": d_model, "Head": n_heads, "HeadDim": d_head}
    axis_names, site = _TAP_TABLE[tap]
    return TapSpec(name, tuple(Axis(axis, sizes[axis]) for axis in axis_names), dtype, site)


_STREAM = ("Batch", "Pos", "Embed")

# Unprefixed tap name -> (axis names, site), per the spec's tap table.
_TAP_TABLE: dict[str, tuple[tuple[str, ...], str]] = {
    "hook_embed": (_STREAM, "embedding"),
    "hook_pos_embed": (_STREAM, "embedding"),
    "hook_resid_pre": (_STREAM, "residual"),
    "attn.hook_pattern": (("Batch", "Head", "QPos", "KPos"), "attention"),
    "attn.hook_z": (("Batch", "Pos", "Head", "HeadDim"), "attention"),
    "hook_attn_out": (_STREAM, "attention"),
    "hook_resid_mid": (_STREAM, "residual"),
    "ln2.hook_normalized": (_STREAM, "mlp"),
    "hook_mlp_out": (_STREAM, "mlp"),
    "hook_resid_post": (_STREAM, "residual"),
    "ln_final.hook_normalized": (_STREAM, "final_norm"),
}
