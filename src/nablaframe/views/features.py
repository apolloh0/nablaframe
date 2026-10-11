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

from collections.abc import Mapping
from typing import Any

import numpy as np

SHIPPED = ("topk", "staircase", "transcoder")
TOP_CONTEXTS = 8
EFFECT_TOKENS = 5
EFFECT_LABEL = ("Direct effect on the logits through the unembedding: the latent's decoder row, times the site's "
                "scale, centred and multiplied by the final layer norm's gain, then unembedded, with the layer "
                "norm's divisor held at 1. It is not a name for the latent.")
METER_GATES = ("G5.2", "G5.3", "G5.5")


def features_payload(model: Any, params: Any, dictionary: Any, site: int, probe: Any, held_out: Any, *, step: int,
                     meters: Mapping[str, Any], variables: Mapping[str, Any] | None = None,
                     chunk: int = 256) -> dict[str, Any]:
    raise NotImplementedError
