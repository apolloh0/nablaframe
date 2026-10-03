"""Direct logit attribution: an exact decomposition of the target token's logit.

Spec: docs/specs/phase-2.md, "Views", and D14.

* ``attribution(parts, final_resid, ln_weight, ln_bias, wte, targets)``: for each
  part (B, T, d), its contribution to the logit of ``targets`` (B, T): the part,
  centered by its own mean over Embed, times the final layer norm's scale at that
  position (computed from ``final_resid``, epsilon ``LN_EPS``), times ``ln_weight``,
  dotted with ``wte[target]``. An extra entry ``"ln_bias"`` holds
  ``ln_bias . wte[target]``. Because the parts sum to ``final_resid``, the
  contributions sum to the lens logit of the target, exactly in exact arithmetic.
* ``parts_from_trace(reader, step, n_layers)``: ``(parts, final_resid)``. The parts
  are ``hook_embed``, ``hook_pos_embed`` and each block's ``hook_attn_out`` and
  ``hook_mlp_out``, keyed by tap name; ``final_resid`` is the last block's
  ``hook_resid_post``.
* ``attribution_from_trace(reader, step, params, probe, n_layers)``:
  ``(contributions, error)``. The target at position t is ``probe[:, t + 1]``; the
  last position has none, and its contributions are NaN. ``error`` is the largest
  absolute gap, over the other positions, between the summed contributions and the
  traced logit of the target: the decomposition error, a plain Python float.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .lens import LN_EPS  # noqa: F401


def attribution(parts: Mapping[str, np.ndarray], final_resid: np.ndarray, ln_weight: np.ndarray,
                ln_bias: np.ndarray, wte: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    raise NotImplementedError


def parts_from_trace(reader: Any, step: int, n_layers: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raise NotImplementedError


def attribution_from_trace(reader: Any, step: int, params: Any, probe: np.ndarray,
                           n_layers: int) -> tuple[dict[str, np.ndarray], float]:
    raise NotImplementedError
