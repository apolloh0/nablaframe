"""The J-lens view: D17's summed Jacobian read through the final layer norm and unembedding.

Spec: docs/specs/phase-2.md, "The J-lens"; rulings D17 and D20. The J's come from G2.9's cache.

* ``JLENS_BADGE``: the sentence stating which discrimination tests the view is held to and
  which gate checks them (D14).
* ``jlens_lens(resid, J, ln_weight, ln_bias, wte)``: ``logit_lens(resid @ J.T, ...)``, in
  float32: J carries the residual stream to a predicted final residual stream, which the
  logit lens then reads. With J the identity, it equals ``logit_lens(resid, ...)`` exactly.
* ``jlens_readings(reader, step, params, jacobians, n_layers)``: (n_layers, Batch, Pos,
  Vocab). Reading l is ``jlens_lens`` of ``blocks.<l>.hook_resid_post`` with
  ``jacobians[l]``, for l below the last block; the last reading is the logit lens of the
  last block's output, whose J is the identity. A missing J raises KeyError.
* ``lens_divergence(a, b)``: one dict per reading, in order: ``{"layer": i, "js_mean": float,
  "top1_agreement": float}``. ``js_mean`` is the mean over positions of the Jensen-Shannon
  divergence between the two readings' softmax distributions, in nats, computed in float64;
  ``top1_agreement`` is the fraction of positions whose top-1 tokens agree. Plain floats.
* ``target_ranks(readings, probe)``: ``(ranks, probs)``, each nested lists (reading, Batch,
  Pos). At position t the target is ``probe[:, t + 1]``; its rank is its 0-based place by
  descending softmax probability, ties going to the lower token id (as the payloads' top
  tokens are ordered), and its probability is computed in float64. The last position has
  no target: both are null there.
* ``jlens_payload(reader, step, params, jacobians, probe, n_layers)``: raises ValueError
  unless ``batch_hash(probe)`` equals the step's events' batch hash. Otherwise
  ``{"step", "layers", "top_tokens", "top_probs", "target_rank", "target_prob", "meter",
  "comparison", "badge"}``: ``layers`` names each reading's residual stream; the top tokens
  and probabilities follow the logit lens payload's rules; ``meter`` is
  ``lens_meter(readings, traced logits)``; ``comparison`` is ``lens_divergence`` of the
  readings against the logit lens readings of the same block outputs; ``badge`` is
  ``JLENS_BADGE``. JSON-ready: plain Python values, null for NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

JLENS_BADGE = ("Discrimination tests: recovers planted structure at decoder level (a J that carries planted "
               "directions to tokens), and is empty on random weights, checked by G2.10. Its definition matches "
               "anthropics/jacobian-lens to 1.1e-7 (G2.8). Planted structure at model level waits for Phase 4's zoo.")


def jlens_lens(resid: np.ndarray, J: np.ndarray, ln_weight: np.ndarray, ln_bias: np.ndarray,
               wte: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def jlens_readings(reader: Any, step: int, params: Any, jacobians: Mapping[int, np.ndarray],
                   n_layers: int) -> np.ndarray:
    raise NotImplementedError


def lens_divergence(a: np.ndarray, b: np.ndarray) -> list[dict[str, float]]:
    raise NotImplementedError


def target_ranks(readings: np.ndarray, probe: np.ndarray) -> tuple[list, list]:
    raise NotImplementedError


def jlens_payload(reader: Any, step: int, params: Any, jacobians: Mapping[int, np.ndarray], probe: np.ndarray,
                  n_layers: int) -> dict[str, Any]:
    raise NotImplementedError
