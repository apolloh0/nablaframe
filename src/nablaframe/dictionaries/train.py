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

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .core import Dictionary, Settings

MLP_NAMES = ("W_fc", "b_fc", "W_proj", "b_proj")


def mlp_weights(params: Any, layer: int) -> dict[str, np.ndarray]:
    raise NotImplementedError


def mlp(weights: Mapping[str, Any], x: Any) -> np.ndarray:
    raise NotImplementedError


def pair_jacobian(dictionary: Dictionary, codes: Any, weights: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError


def train(settings: Settings, sites: Sequence[Sequence[str]], data: Sequence[tuple[Any, Any]], *,
          mlp_weights: Mapping[str, Any] | None = None, key: str | None = None,
          progress: Callable[[int, int, list[float]], None] | None = None) -> Dictionary:
    raise NotImplementedError


def code_identity() -> str:
    raise NotImplementedError
