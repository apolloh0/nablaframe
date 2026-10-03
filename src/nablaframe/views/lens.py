"""The logit lens and its meter.

Spec: docs/specs/phase-2.md, "Views", and D14 for its discrimination tests.

* ``lens_params(params)``: ``(ln_f.weight, ln_f.bias, wte)`` as float32 numpy arrays.
* ``logit_lens(resid, ln_weight, ln_bias, wte)``: applies the final layer norm over
  the last axis (mean and biased variance, epsilon ``LN_EPS``, then weight and bias,
  in float32), then multiplies by ``wte.T``, the tied unembedding. Shape
  ``resid.shape[:-1] + (vocab,)``, float32.
* ``lens_readings(reader, step, params, n_layers)``: (n_layers + 1, Batch, Pos,
  Vocab). Reading 0 is the lens of ``blocks.0.hook_resid_pre``, the residual stream
  entering the first block; reading l + 1 is the lens of ``blocks.<l>.hook_resid_post``.
* ``lens_meter(readings, final_logits)``: one dict per reading, in order:
  ``{"layer": i, "kl_mean": float, "top1_agreement": float}``. ``kl_mean`` is the
  mean over positions of KL(final || lens) in nats, computed in float64.
  ``top1_agreement`` is the fraction of positions whose top-1 token matches the
  model's. Values are plain Python floats.
* ``next_token_accuracy(lens_logits, tokens)``: the fraction of positions t below
  the last whose top-1 lens token equals ``tokens[:, t + 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LN_EPS = 1e-5


def lens_params(params: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError


def logit_lens(resid: np.ndarray, ln_weight: np.ndarray, ln_bias: np.ndarray, wte: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def lens_readings(reader: Any, step: int, params: Any, n_layers: int) -> np.ndarray:
    raise NotImplementedError


def lens_meter(readings: np.ndarray, final_logits: np.ndarray) -> list[dict[str, float]]:
    raise NotImplementedError


def next_token_accuracy(lens_logits: np.ndarray, tokens: np.ndarray) -> float:
    raise NotImplementedError
