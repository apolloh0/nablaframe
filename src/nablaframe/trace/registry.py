"""Canonical tap names and their specs.

Spec: docs/specs/phase-0.md, "Trace schema v0", tap table.

Names are TransformerLens hook names. TransformerLens 4.0 removed the legacy
``HookedTransformer``; these names survive as aliases on ``TransformerBridge``
for pre-norm GPT-2-style models, so traces cross-check against TransformerLens
without a translation table. The vendored list in
``tests/data/transformerlens_legacy_hook_names.txt`` records where each name was
verified, and ``scripts/check_tl_names.py`` confirms them at runtime in nf-torch.

This module never imports TransformerLens.
"""

from __future__ import annotations

from .schema import TapSpec

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


def tap_names(n_layers: int) -> tuple[str, ...]:
    """Every canonical tap name for a model with ``n_layers`` blocks, in forward order.

    Order: ``GLOBAL_TAPS_BEFORE``, then ``PER_LAYER_TAPS`` for block 0, block 1, and
    so on, then ``GLOBAL_TAPS_AFTER``. Raises ValueError if ``n_layers < 1``.
    """
    raise NotImplementedError


def tap_spec(
    name: str,
    *,
    batch: int,
    seq: int,
    d_model: int,
    n_heads: int,
    d_head: int,
    dtype: str = "float32",
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
    ======================================  ===================================  ===========

    Sizes: Batch=batch, Pos=QPos=KPos=seq, Embed=d_model, Head=n_heads,
    HeadDim=d_head. Raises KeyError for a name that is not canonical.
    """
    raise NotImplementedError
