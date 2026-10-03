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
    return (
        np.asarray(params.ln_f.weight, dtype=np.float32),
        np.asarray(params.ln_f.bias, dtype=np.float32),
        np.asarray(params.wte, dtype=np.float32),
    )


def logit_lens(resid: np.ndarray, ln_weight: np.ndarray, ln_bias: np.ndarray, wte: np.ndarray) -> np.ndarray:
    resid = np.asarray(resid, dtype=np.float32)
    centered = resid - resid.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = (centered * centered).mean(axis=-1, keepdims=True, dtype=np.float32)
    normed = centered / np.sqrt(variance + np.float32(LN_EPS), dtype=np.float32)
    normed = normed * np.asarray(ln_weight, dtype=np.float32) + np.asarray(ln_bias, dtype=np.float32)
    return (normed @ np.asarray(wte, dtype=np.float32).T).astype(np.float32, copy=False)


def lens_readings(reader: Any, step: int, params: Any, n_layers: int) -> np.ndarray:
    ln_w, ln_b, wte = lens_params(params)
    taps = ["blocks.0.hook_resid_pre"] + [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    return np.stack([logit_lens(reader.tensor(step, tap), ln_w, ln_b, wte) for tap in taps])


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64)
    shifted -= shifted.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def lens_meter(readings: np.ndarray, final_logits: np.ndarray) -> list[dict[str, float]]:
    log_p = _log_softmax(final_logits)
    p = np.exp(log_p)
    final_top1 = final_logits.argmax(axis=-1)
    meter = []
    for layer, reading in enumerate(readings):
        log_q = _log_softmax(reading)
        kl = (p * (log_p - log_q)).sum(axis=-1)
        meter.append({
            "layer": layer,
            "kl_mean": float(kl.mean()),
            "top1_agreement": float((reading.argmax(axis=-1) == final_top1).mean()),
        })
    return meter


def next_token_accuracy(lens_logits: np.ndarray, tokens: np.ndarray) -> float:
    predicted = lens_logits[:, :-1].argmax(axis=-1)
    return float((predicted == np.asarray(tokens)[:, 1:]).mean())
