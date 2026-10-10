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

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from nablaframe.trace.schema import canonical_json, sha256_hex, tensor_sha256
from nablaframe.views.cache import cache_key

KINDS = ("topk", "staircase", "transcoder", "jsae")
ARRAY_NAMES = ("W_enc", "W_dec", "b_enc", "b_pre", "b_dec", "scale_in", "scale_out")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_INDEX = r"(0|[1-9][0-9]*)"
_RESID = re.compile(rf"blocks\.{_INDEX}\.hook_resid_post")
_MLP_IN = re.compile(rf"blocks\.{_INDEX}\.ln2\.hook_normalized")
_MLP_OUT = re.compile(rf"blocks\.{_INDEX}\.hook_mlp_out")


def _int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def _float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a float, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


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
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        ints = {"width": 1, "k": 1, "steps": 1, "batch": 1, "seed": 0, "jac_batch": 1, "warmup": 1, "dead_after": 1}
        for name, minimum in ints.items():
            object.__setattr__(self, name, _int(name, getattr(self, name), minimum))
        if self.k > self.width:
            raise ValueError(f"k ({self.k}) must not exceed width ({self.width})")
        for name in ("lr", "aux_weight"):
            value = _float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            object.__setattr__(self, name, value)
        decay = _float("decay_fraction", self.decay_fraction)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay_fraction must be in [0, 1], got {decay!r}")
        object.__setattr__(self, "decay_fraction", decay)
        lam = _float("lam", self.lam)
        if lam < 0:
            raise ValueError(f"lam must not be negative, got {lam!r}")
        if lam > 0 and self.kind != "jsae":
            raise ValueError(f"only a jsae may set lam above zero, not a {self.kind}")
        object.__setattr__(self, "lam", lam)


def settings_json(settings: Settings) -> dict[str, Any]:
    return {f.name: getattr(settings, f.name) for f in fields(settings)}


def _taps(site: Any) -> tuple[str, str]:
    if isinstance(site, (str, bytes)) or not isinstance(site, Sequence) or len(site) != 2:
        raise ValueError(f"a site is a pair of tap names, got {site!r}")
    if not all(isinstance(tap, str) and tap for tap in site):
        raise ValueError(f"a site's taps are non-empty strings, got {site!r}")
    return (site[0], site[1])


def check_sites(kind: str, sites: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence):
        raise ValueError(f"sites must be a sequence of pairs of tap names, got {sites!r}")
    pairs = tuple(_taps(site) for site in sites)
    if kind == "staircase":
        if not pairs:
            raise ValueError("a staircase needs at least one site")
        for layer, pair in enumerate(pairs):
            if pair != (f"blocks.{layer}.hook_resid_post",) * 2:
                raise ValueError(f"staircase site {layer} must read and reconstruct blocks.{layer}.hook_resid_post, got {pair!r}")
        return pairs
    if kind == "topk":
        if len(pairs) != 1 or pairs[0][0] != pairs[0][1]:
            raise ValueError(f"a topk has one site with equal taps, got {pairs!r}")
        return pairs
    if kind == "transcoder":
        if len(pairs) != 1:
            raise ValueError(f"a transcoder has one site, got {len(pairs)}")
        reads, writes = pairs[0]
        a, b = _MLP_IN.fullmatch(reads), _MLP_OUT.fullmatch(writes)
        if a is None or b is None or a.group(1) != b.group(1):
            raise ValueError(f"a transcoder reads blocks.<l>.ln2.hook_normalized and reconstructs blocks.<l>.hook_mlp_out, got {pairs[0]!r}")
        return pairs
    if len(pairs) != 2 or pairs[0][0] != pairs[0][1] or pairs[1][0] != pairs[1][1]:
        raise ValueError(f"a jsae has two sites, each with equal taps, got {pairs!r}")
    a, b = _MLP_IN.fullmatch(pairs[0][0]), _MLP_OUT.fullmatch(pairs[1][0])
    if a is None or b is None or a.group(1) != b.group(1):
        raise ValueError(f"a jsae reads blocks.<l>.ln2.hook_normalized then blocks.<l>.hook_mlp_out, got {pairs!r}")
    return pairs


@dataclass(frozen=True, eq=False)
class Dictionary:
    settings: Settings
    sites: tuple[tuple[str, str], ...]
    arrays: Mapping[str, np.ndarray]
    key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.settings, Settings):
            raise ValueError(f"settings must be a Settings, got {type(self.settings).__name__}")
        sites = check_sites(self.settings.kind, self.sites)
        n_sites = len(sites)
        units = 2 if self.settings.kind == "jsae" else 1
        m = self.settings.width
        if self.settings.kind == "staircase" and m % n_sites:
            raise ValueError(f"width {m} must be divisible by the {n_sites} staircase sites")
        if self.key is not None and (not isinstance(self.key, str) or _HEX64.fullmatch(self.key) is None):
            raise ValueError(f"key must be None or 64 lowercase hex characters, got {self.key!r}")
        if not isinstance(self.arrays, Mapping) or set(self.arrays) != set(ARRAY_NAMES):
            got = sorted(self.arrays) if isinstance(self.arrays, Mapping) else self.arrays
            raise ValueError(f"arrays must hold exactly {ARRAY_NAMES}, got {got!r}")
        for name in ARRAY_NAMES:
            value = self.arrays[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise ValueError(f"{name} must be a float32 numpy array")
        w_enc, w_dec = self.arrays["W_enc"], self.arrays["W_dec"]
        d_in, d_out = (w_enc.shape[1], w_dec.shape[2]) if w_enc.ndim == 3 and w_dec.ndim == 3 else (0, 0)
        shapes = {"W_enc": (units, d_in, m), "W_dec": (units, m, d_out), "b_enc": (n_sites, m),
                  "b_pre": (n_sites, d_in), "b_dec": (n_sites, d_out), "scale_in": (n_sites,), "scale_out": (n_sites,)}
        if d_in < 1 or d_out < 1:
            raise ValueError("W_enc and W_dec must be three-dimensional with non-empty feature axes")
        copies = {}
        for name in ARRAY_NAMES:
            value = self.arrays[name]
            if value.shape != shapes[name]:
                raise ValueError(f"{name} must have shape {shapes[name]}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} holds a value that is not finite")
            copy = np.array(value, dtype=np.float32, order="C", copy=True)
            copy.setflags(write=False)
            copies[name] = copy
        for name in ("scale_in", "scale_out"):
            if not (copies[name] > 0).all():
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "arrays", MappingProxyType(copies))

    @property
    def d_in(self) -> int:
        return int(self.arrays["W_enc"].shape[1])

    @property
    def d_out(self) -> int:
        return int(self.arrays["W_dec"].shape[2])

    def _site(self, site: Any) -> int:
        if isinstance(site, (bool, np.bool_)) or not isinstance(site, (int, np.integer)) or not 0 <= site < len(self.sites):
            raise ValueError(f"site must be an int in range({len(self.sites)}), got {site!r}")
        return int(site)

    def unit(self, site: int) -> int:
        site = self._site(site)
        return site if self.settings.kind == "jsae" else 0

    def active_width(self, site: int) -> int:
        site = self._site(site)
        m = self.settings.width
        return (site + 1) * m // len(self.sites) if self.settings.kind == "staircase" else m


@cache
def _kernels() -> tuple[Any, Any, Any]:
    """The jitted encode, decode and decode-delta, built on first use."""
    import jax
    import jax.numpy as jnp

    highest = jax.lax.Precision.HIGHEST

    def encode_fn(x, W_enc, b_enc, b_pre, scale_in, *, k, active):
        pre = jnp.matmul(x / scale_in - b_pre, W_enc, precision=highest) + b_enc
        pre = jnp.where(jnp.arange(pre.shape[1]) < active, pre, -jnp.inf)
        values, index = jax.lax.top_k(pre, k)  # ties go to the lower index
        rows = jnp.arange(pre.shape[0])[:, None]
        return jnp.zeros(pre.shape, jnp.float32).at[rows, index].set(jnp.maximum(values, 0.0))

    def decode_fn(codes, W_dec, b_dec, scale_out):
        return (jnp.matmul(codes, W_dec, precision=highest) + b_dec) * scale_out

    def delta_fn(codes, W_dec, scale_out):
        return jnp.matmul(codes, W_dec, precision=highest) * scale_out

    return (jax.jit(encode_fn, static_argnames=("k", "active")), jax.jit(decode_fn), jax.jit(delta_fn))


def _rows(name: str, value: Any, width: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must be shaped (N, {width}), got {array.shape}")
    return array


def encode(dictionary: Dictionary, site: int, x: Any) -> np.ndarray:
    u, active = dictionary.unit(site), dictionary.active_width(site)
    x = _rows("x", x, dictionary.d_in)
    a = dictionary.arrays
    out = _kernels()[0](x, a["W_enc"][u], a["b_enc"][site], a["b_pre"][site], a["scale_in"][site],
                        k=dictionary.settings.k, active=active)
    return np.array(out, dtype=np.float32)


def decode(dictionary: Dictionary, site: int, codes: Any) -> np.ndarray:
    u = dictionary.unit(site)
    codes = _rows("codes", codes, dictionary.settings.width)
    a = dictionary.arrays
    return np.array(_kernels()[1](codes, a["W_dec"][u], a["b_dec"][site], a["scale_out"][site]), dtype=np.float32)


def decode_delta(dictionary: Dictionary, site: int, codes: Any) -> np.ndarray:
    u = dictionary.unit(site)
    codes = _rows("codes", codes, dictionary.settings.width)
    a = dictionary.arrays
    return np.array(_kernels()[2](codes, a["W_dec"][u], a["scale_out"][site]), dtype=np.float32)


def reconstruct(dictionary: Dictionary, site: int, x: Any) -> np.ndarray:
    return decode(dictionary, site, encode(dictionary, site, x))


def _digest(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("ascii"))


def _record(dictionary: Dictionary) -> dict[str, Any]:
    """The JSON form saved beside the arrays, and the object the content hash covers."""
    return {"kind": dictionary.settings.kind, "settings": settings_json(dictionary.settings),
            "sites": [list(site) for site in dictionary.sites], "key": dictionary.key,
            "arrays": {name: tensor_sha256(dictionary.arrays[name]) for name in ARRAY_NAMES}}


def content_hash(dictionary: Dictionary) -> str:
    return _digest(_record(dictionary))


def dictionary_key(*, state_sha256: str, sites: Sequence[Sequence[str]], settings: Settings,
                   batch_hashes: Sequence[str], code: str) -> str:
    if not isinstance(settings, Settings):
        raise ValueError(f"settings must be a Settings, got {type(settings).__name__}")
    if not isinstance(state_sha256, str) or not isinstance(code, str):
        raise ValueError("state_sha256 and code must be strings")
    if isinstance(batch_hashes, (str, bytes)) or not all(isinstance(h, str) for h in batch_hashes):
        raise ValueError("batch_hashes must be a sequence of strings")
    checked = check_sites(settings.kind, sites)
    return cache_key({"state": state_sha256, "sites": _digest([list(site) for site in checked]),
                      "settings": _digest(settings_json(settings)), "batches": _digest(list(batch_hashes))}, code)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and a rename."""
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_dictionary(dictionary: Dictionary, folder: str | Path) -> Path:
    from safetensors.numpy import save

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record = _record(dictionary)
    name = _digest(record)
    weights, sidecar = folder / f"{name}.safetensors", folder / f"{name}.json"
    if not weights.exists():
        _write_atomic(weights, save({n: np.ascontiguousarray(dictionary.arrays[n]) for n in ARRAY_NAMES}))
    if not sidecar.exists():  # written last, so a sidecar means the weights are complete
        _write_atomic(sidecar, canonical_json(record).encode("ascii"))
    return weights


def load_dictionary(path: str | Path) -> Dictionary:
    from safetensors.numpy import load

    weights = Path(path)
    sidecar = weights.with_suffix(".json")
    if weights.suffix != ".safetensors":
        raise ValueError(f"a dictionary is a .safetensors file, got {weights.name!r}")
    if not weights.is_file() or not sidecar.is_file():
        raise ValueError(f"{weights.name} and {sidecar.name} must both exist")
    try:
        record = json.loads(sidecar.read_text(encoding="ascii"))
        arrays = load(weights.read_bytes())
        dictionary = Dictionary(Settings(**record["settings"]), [tuple(site) for site in record["sites"]],
                                arrays, record["key"])
    except ValueError:
        raise
    except Exception as error:  # a malformed sidecar or safetensors file
        raise ValueError(f"{weights.name} is not a readable dictionary: {error}") from error
    if record != _record(dictionary):
        raise ValueError(f"{sidecar.name} does not describe the arrays in {weights.name}")
    if content_hash(dictionary) != weights.stem:
        raise ValueError(f"the content hash of {weights.name} is not its name")
    return dictionary
