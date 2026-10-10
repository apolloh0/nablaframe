"""Phase 5 delta 1: a dictionary's record, encoding and decoding, content hash, cache key and files (D40).

Contract: src/nablaframe/dictionaries/core.py. Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.trace.schema import canonical_json, sha256_hex  # noqa: E402
from nablaframe.views.cache import cache_key  # noqa: E402

SITES = {
    "topk": [("blocks.0.hook_resid_post",) * 2],
    "staircase": [(f"blocks.{layer}.hook_resid_post",) * 2 for layer in range(2)],
    "transcoder": [("blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out")],
    "jsae": [("blocks.1.ln2.hook_normalized",) * 2, ("blocks.1.hook_mlp_out",) * 2],
}


def arrays(kind, d=5, m=12, seed=0):
    rng = np.random.default_rng(seed)
    S, U = len(SITES[kind]), (2 if kind == "jsae" else 1)
    W_dec = rng.standard_normal((U, m, d))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    out = {"W_enc": rng.standard_normal((U, d, m)), "W_dec": W_dec, "b_enc": rng.normal(0, 0.1, (S, m)),
           "b_pre": rng.normal(0, 0.1, (S, d)), "b_dec": rng.normal(0, 0.1, (S, d)),
           "scale_in": rng.uniform(0.5, 2, S), "scale_out": rng.uniform(0.5, 2, S)}
    return {name: value.astype(np.float32) for name, value in out.items()}


def make(kind, m=12, k=3, d=5, seed=0, key=None, **settings):
    return core.Dictionary(core.Settings(kind, m, k, steps=10, batch=4, **settings), SITES[kind], arrays(kind, d, m, seed), key)


def expected_codes(dictionary, site, x):
    """The contract's encode, in float64 numpy: shift, encode, keep the k largest (lower id on ties), clip."""
    a, u = dictionary.arrays, dictionary.unit(site)
    pre = (x.astype(np.float64) / a["scale_in"][site] - a["b_pre"][site]) @ a["W_enc"][u] + a["b_enc"][site]
    pre[:, dictionary.active_width(site):] = -np.inf
    order = np.argsort(-pre, axis=1, kind="stable")[:, :dictionary.settings.k]
    codes = np.zeros_like(pre)
    np.put_along_axis(codes, order, np.maximum(np.take_along_axis(pre, order, axis=1), 0), axis=1)
    return codes


def test_settings_are_checked():
    s = core.Settings("jsae", 64, 8, steps=100, batch=32, lam=1)
    assert core.settings_json(s) == {"kind": "jsae", "width": 64, "k": 8, "steps": 100, "batch": 32, "lr": 5e-4,
                                     "seed": 0, "lam": 1.0, "jac_batch": 1024, "warmup": 100, "decay_fraction": 0.3,
                                     "aux_weight": 1 / 32, "dead_after": 100}
    assert isinstance(core.settings_json(s)["lam"], float), "floats stay floats in the JSON form"
    bad = [dict(kind="sparse"), dict(width=0), dict(k=65), dict(steps=True), dict(batch=2.0), dict(lr=0.0),
           dict(lr=float("nan")), dict(seed=-1), dict(lam=-0.5), dict(decay_fraction=1.5), dict(aux_weight=0),
           dict(dead_after=0), dict(jac_batch=0), dict(warmup=0)]
    for change in bad:
        with pytest.raises(ValueError):
            core.Settings(**{**dict(kind="topk", width=64, k=8, steps=100, batch=32), **change})
    with pytest.raises(ValueError, match="jsae"):
        core.Settings("topk", 64, 8, steps=100, batch=32, lam=0.1)


def test_sites_are_checked():
    for kind, sites in SITES.items():
        assert core.check_sites(kind, sites) == tuple(tuple(s) for s in sites)
    bad = {
        "topk": [[("blocks.0.hook_resid_post", "blocks.0.hook_mlp_out")], SITES["topk"] * 2, []],
        "staircase": [[("blocks.1.hook_resid_post",) * 2], [("blocks.0.hook_mlp_out",) * 2], []],
        "transcoder": [[("blocks.1.ln2.hook_normalized", "blocks.0.hook_mlp_out")], [("blocks.1.hook_mlp_out",) * 2]],
        "jsae": [SITES["jsae"][::-1], [("blocks.1.ln2.hook_normalized",) * 2, ("blocks.2.hook_mlp_out",) * 2],
                 SITES["jsae"][:1]],
    }
    for kind, cases in bad.items():
        for sites in cases:
            with pytest.raises(ValueError):
                core.check_sites(kind, sites)


def test_a_dictionary_checks_its_arrays():
    for kind in core.KINDS:
        d = make(kind)
        assert (d.d_in, d.d_out) == (5, 5) and d.key is None
        assert [d.unit(s) for s in range(len(d.sites))] == ([0, 1] if kind == "jsae" else [0] * len(d.sites))
    assert [make("staircase").active_width(s) for s in (0, 1)] == [6, 12]
    good = arrays("topk")
    settings = core.Settings("topk", 12, 3, steps=10, batch=4)
    for name, value in [("W_enc", good["W_enc"].astype(np.float64)), ("b_dec", good["b_dec"][:, :4]),
                        ("scale_in", -good["scale_in"]), ("b_enc", np.full_like(good["b_enc"], np.nan)),
                        ("W_dec", good["W_dec"][:, :11])]:
        with pytest.raises(ValueError):
            core.Dictionary(settings, SITES["topk"], {**good, name: value})
    with pytest.raises(ValueError):
        core.Dictionary(settings, SITES["topk"], {n: v for n, v in good.items() if n != "b_pre"})
    with pytest.raises(ValueError):
        core.Dictionary(settings, SITES["topk"], good, key="not a hash")
    with pytest.raises(ValueError):  # 13 latents can't split between two blocks
        core.Dictionary(core.Settings("staircase", 13, 3, steps=10, batch=4), SITES["staircase"], arrays("staircase", m=13))
    source = arrays("topk")
    d = core.Dictionary(settings, SITES["topk"], source)
    source["W_enc"][:] = 0
    assert np.abs(d.arrays["W_enc"]).sum() > 0, "the dictionary keeps its own copy"
    with pytest.raises(ValueError):
        d.arrays["W_enc"][0, 0, 0] = 1.0


def test_encode_keeps_the_k_largest_pre_activations():
    rng = np.random.default_rng(1)
    for kind in core.KINDS:
        d = make(kind, seed=2)
        for site in range(len(d.sites)):
            x = rng.standard_normal((40, 5)).astype(np.float32)
            got = core.encode(d, site, x)
            want = expected_codes(d, site, x)
            assert got.dtype == np.float32 and got.shape == (40, 12)
            assert ((got > 0) == (want > 0)).all(), f"{kind} site {site}: the same latents fire"
            assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
            assert ((got > 0).sum(1) <= 3).all()
    # Ties go to the lower latent id, and nothing below zero survives.
    d = make("topk", m=6, k=2, d=1)
    tied = {**d.arrays, "W_enc": np.array([[[1, 3, 3, 0, 3, -1]]], np.float32), "b_enc": np.zeros((1, 6), np.float32),
            "b_pre": np.zeros((1, 1), np.float32), "scale_in": np.ones(1, np.float32)}
    d = core.Dictionary(d.settings, d.sites, tied)
    assert np.flatnonzero(core.encode(d, 0, np.ones((1, 1), np.float32))[0]).tolist() == [1, 2]
    # With x = -1 the two largest are latent 5 (1.0) and latent 3 (0.0); the zero stays zero.
    assert np.flatnonzero(core.encode(d, 0, -np.ones((1, 1), np.float32))[0]).tolist() == [5]


def test_a_staircase_site_uses_only_its_share():
    d = make("staircase", m=12, k=3)
    loud = {**d.arrays, "b_enc": d.arrays["b_enc"].copy()}
    loud["b_enc"][:, 6:] = 50.0  # the later half would win everywhere if it were allowed
    d = core.Dictionary(d.settings, d.sites, loud)
    x = np.random.default_rng(3).standard_normal((30, 5)).astype(np.float32)
    assert not core.encode(d, 0, x)[:, 6:].any(), "block 0 uses the first half only"
    assert (core.encode(d, 1, x)[:, 6:] > 0).sum(1).min() == 3, "block 1 may use all of them"


def test_decode_delta_and_reconstruct():
    rng = np.random.default_rng(4)
    for kind in core.KINDS:
        d = make(kind, seed=5)
        a = d.arrays
        for site in range(len(d.sites)):
            u = d.unit(site)
            z = np.abs(rng.standard_normal((7, 12))).astype(np.float32)
            assert np.allclose(core.decode(d, site, z), (z @ a["W_dec"][u] + a["b_dec"][site]) * a["scale_out"][site],
                               rtol=1e-5, atol=1e-5)
            assert np.allclose(core.decode_delta(d, site, z), z @ a["W_dec"][u] * a["scale_out"][site],
                               rtol=1e-5, atol=1e-5)
            x = rng.standard_normal((7, 5)).astype(np.float32)
            assert np.array_equal(core.reconstruct(d, site, x), core.decode(d, site, core.encode(d, site, x)))
    d = make("topk")
    for call in (lambda: core.encode(d, 1, np.zeros((2, 5), np.float32)), lambda: core.encode(d, -1, np.zeros((2, 5), np.float32)),
                 lambda: core.encode(d, 0, np.zeros((2, 4), np.float32)), lambda: core.decode(d, 0, np.zeros((2, 11), np.float32)),
                 lambda: core.decode_delta(d, True, np.zeros((2, 12), np.float32))):
        with pytest.raises(ValueError):
            call()


def test_the_content_hash_covers_everything():
    base = make("topk")
    same = make("topk")
    assert core.content_hash(base) == core.content_hash(same)
    nudged = {**base.arrays, "W_dec": base.arrays["W_dec"].copy()}
    nudged["W_dec"][0, 0, 0] = np.nextafter(nudged["W_dec"][0, 0, 0], np.float32(np.inf))
    variants = [
        core.Dictionary(base.settings, base.sites, nudged),
        make("topk", key="a" * 64),
        make("topk", lr=1e-3),
        core.Dictionary(base.settings, [("blocks.1.hook_resid_post",) * 2], base.arrays),
    ]
    hashes = {core.content_hash(base)} | {core.content_hash(v) for v in variants}
    assert len(hashes) == 5, "a weight, the key, a setting or a site each change the hash"


def test_the_cache_key():
    settings = core.Settings("topk", 64, 8, steps=100, batch=32)
    args = dict(state_sha256="1" * 64, sites=SITES["topk"], settings=settings, batch_hashes=["2" * 64, "3" * 64],
                code="4" * 40)
    key = core.dictionary_key(**args)
    digest = lambda obj: sha256_hex(canonical_json(obj).encode("ascii"))  # noqa: E731
    assert key == cache_key({"state": "1" * 64, "sites": digest([list(s) for s in SITES["topk"]]),
                             "settings": digest(core.settings_json(settings)), "batches": digest(["2" * 64, "3" * 64])},
                            "4" * 40)
    changed = [dict(state_sha256="5" * 64), dict(sites=[("blocks.1.hook_resid_post",) * 2]),
               dict(settings=core.Settings("topk", 64, 8, steps=100, batch=32, seed=1)),
               dict(batch_hashes=["3" * 64, "2" * 64]), dict(code="6" * 40)]
    keys = {key} | {core.dictionary_key(**{**args, **change}) for change in changed}
    assert len(keys) == 6, "every input, the batches' order included, changes the key"


def test_files_round_trip_and_refuse_tampering(tmp_path):
    d = make("jsae", key="b" * 64)
    path = core.save_dictionary(d, tmp_path / "dicts")
    name = core.content_hash(d)
    assert path == tmp_path / "dicts" / f"{name}.safetensors" and (tmp_path / "dicts" / f"{name}.json").exists()
    back = core.load_dictionary(path)
    assert core.content_hash(back) == name and back.key == "b" * 64 and back.sites == d.sites
    assert all(np.array_equal(back.arrays[n], d.arrays[n]) for n in core.ARRAY_NAMES)
    before = path.read_bytes()
    assert core.save_dictionary(d, tmp_path / "dicts") == path and path.read_bytes() == before
    sidecar = json.loads(path.with_suffix(".json").read_text())
    sidecar["key"] = "c" * 64
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        core.load_dictionary(path)
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError):
        core.load_dictionary(path)
