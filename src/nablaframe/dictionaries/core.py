"""Dictionaries: the record, encoding and decoding, content hashes, cache keys and files (D40).

Spec: docs/specs/phase-5.md, D40. JAX is imported inside the functions that need it, so this
module imports without it.

* ``KINDS = ("topk", "staircase", "transcoder", "jsae")``.
* ``Settings``: a frozen dataclass of how a dictionary trains: ``kind``, ``width``, ``k``,
  ``steps``, ``batch``, and, with these defaults, ``lr=5e-4``, ``seed=0``, ``lam=0.0``,
  ``jac_batch=1024``, ``warmup=100``, ``decay_fraction=0.3``, ``aux_weight=1/32`` and
  ``dead_after=100``. ``kind`` is one of ``KINDS``; ``width``, ``k``, ``steps``, ``batch``,
  ``jac_batch``, ``warmup`` and ``dead_after`` are positive ints, ``seed`` a non-negative int, with
  ``k <= width``; ``lr`` and ``aux_weight`` are positive finite floats, ``decay_fraction`` a float
  in [0, 1], ``lam`` a non-negative finite float that only a ``jsae`` may set above zero. Anything
  else raises ValueError. ``settings_json(settings)``: its fields as a plain JSON object, ints as
  ints and floats as floats.
* Sites. A dictionary reads and reconstructs at its sites, each ``(input tap, output tap)``:

  - ``topk``: one site, its two taps equal.
  - ``staircase``: one site per block, in block order, each ``hook_resid_post`` read and
    reconstructed; ``width`` divisible by the number of sites.
  - ``transcoder``: one site, ``(blocks.<l>.ln2.hook_normalized, blocks.<l>.hook_mlp_out)``.
  - ``jsae``: two sites of one block, ``(blocks.<l>.ln2.hook_normalized,)`` and
    ``(blocks.<l>.hook_mlp_out,)``, each with its taps equal.

  ``check_sites(kind, sites)`` returns the sites as a tuple of 2-tuples of strings, or raises
  ValueError for anything else.
* ``Dictionary(settings, sites, arrays, key=None)``: a frozen dataclass. ``arrays`` maps exactly
  these names to float32 numpy arrays, with ``m = settings.width``, ``S`` sites and ``U`` weight
  units (two for ``jsae``, one otherwise):

  - ``W_enc`` (U, d_in, m) and ``W_dec`` (U, m, d_out)
  - ``b_enc`` (S, m), ``b_pre`` (S, d_in), ``b_dec`` (S, d_out)
  - ``scale_in`` (S,) and ``scale_out`` (S,), each positive

  ``d_in`` and ``d_out`` are the same for every site. Construction checks the sites, the names,
  dtypes and shapes, and that every value is finite, and raises ValueError otherwise; the arrays
  are copied and made read-only. ``key`` is the cache key it was trained under, or None.
  ``unit(site)`` is the site's weight unit: the site itself for a ``jsae``, otherwise 0.
  ``active_width(site)`` is ``(site + 1) * m // S`` for a ``staircase``, otherwise ``m``.
* ``encode(dictionary, site, x)``: the codes of raw activations ``x``, shaped (N, d_in), as
  float32 (N, m). With ``u = unit(site)``, the pre-activations are
  ``(x / scale_in[site] - b_pre[site]) @ W_enc[u] + b_enc[site]``; latents at or beyond
  ``active_width(site)`` never fire; the ``k`` largest pre-activations are kept, ties to the lower
  latent id, and clipped at zero; every other latent is zero. Computed with JAX in float32.
* ``decode(dictionary, site, codes)``: ``(codes @ W_dec[u] + b_dec[site]) * scale_out[site]``,
  float32 (N, d_out). ``decode_delta(dictionary, site, codes)``: the same without ``b_dec``, the
  change in the output for a change in the codes. ``reconstruct(dictionary, site, x)``:
  ``decode(dictionary, site, encode(dictionary, site, x))``. A site that isn't an int in
  ``range(S)``, or an input of the wrong width, raises ValueError.
* ``content_hash(dictionary)``: SHA-256 of the canonical JSON of ``{"kind", "settings":
  settings_json, "sites", "key", "arrays": {name: tensor_sha256}}``. Any change to any of them
  changes it; equal dictionaries hash equal.
* ``dictionary_key(*, state_sha256, sites, settings, batch_hashes, code)``: the cache key under D11
  and D18, ``views.cache.cache_key`` of ``{"state": state_sha256, "sites": the SHA-256 of the
  sites' canonical JSON, "settings": the SHA-256 of settings_json's canonical JSON, "batches": the
  SHA-256 of the batch hashes' canonical JSON, in order}`` and ``code``.
* ``save_dictionary(dictionary, folder)``: writes ``<content_hash>.safetensors`` (the arrays) and
  ``<content_hash>.json`` (canonical JSON of the kind, settings, sites, key and array hashes), each
  through a temporary file and a rename, creating the folder, and returns the safetensors path.
  Saving an identical dictionary again is a no-op. ``load_dictionary(path)``: the dictionary
  saved there; ValueError if either file is missing or its content hash isn't the file's name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("topk", "staircase", "transcoder", "jsae")
ARRAY_NAMES = ("W_enc", "W_dec", "b_enc", "b_pre", "b_dec", "scale_in", "scale_out")


@dataclass(frozen=True)
class Settings:
    kind: str
    width: int
    k: int
    steps: int
    batch: int
    lr: float = 5e-4
    seed: int = 0
    lam: float = 0.0
    jac_batch: int = 1024
    warmup: int = 100
    decay_fraction: float = 0.3
    aux_weight: float = 1 / 32
    dead_after: int = 100

    def __post_init__(self) -> None:
        raise NotImplementedError


def settings_json(settings: Settings) -> dict[str, Any]:
    raise NotImplementedError


def check_sites(kind: str, sites: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    raise NotImplementedError


@dataclass(frozen=True, eq=False)
class Dictionary:
    settings: Settings
    sites: tuple[tuple[str, str], ...]
    arrays: Mapping[str, np.ndarray]
    key: str | None = None

    def __post_init__(self) -> None:
        raise NotImplementedError

    @property
    def d_in(self) -> int:
        raise NotImplementedError

    @property
    def d_out(self) -> int:
        raise NotImplementedError

    def unit(self, site: int) -> int:
        raise NotImplementedError

    def active_width(self, site: int) -> int:
        raise NotImplementedError


def encode(dictionary: Dictionary, site: int, x: Any) -> np.ndarray:
    raise NotImplementedError


def decode(dictionary: Dictionary, site: int, codes: Any) -> np.ndarray:
    raise NotImplementedError


def decode_delta(dictionary: Dictionary, site: int, codes: Any) -> np.ndarray:
    raise NotImplementedError


def reconstruct(dictionary: Dictionary, site: int, x: Any) -> np.ndarray:
    raise NotImplementedError


def content_hash(dictionary: Dictionary) -> str:
    raise NotImplementedError


def dictionary_key(*, state_sha256: str, sites: Sequence[Sequence[str]], settings: Settings,
                   batch_hashes: Sequence[str], code: str) -> str:
    raise NotImplementedError


def save_dictionary(dictionary: Dictionary, folder: str | Path) -> Path:
    raise NotImplementedError


def load_dictionary(path: str | Path) -> Dictionary:
    raise NotImplementedError
