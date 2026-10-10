"""Phase 5 delta 1: ``dpatch``, a patch made through a dictionary's code (D45).

Contracts: src/nablaframe/interventions/spec.py and apply.py, the dpatch bullets. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.interventions.apply import run_intervened, swap_component  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit, spec_from_json, spec_hash, spec_json  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

M, K = 24, 4
L = TINY.n_layers - 1  # the last block


def dictionary(kind, sites, seed=0, d=TINY.d_model):
    rng = np.random.default_rng(seed)
    S, U = len(sites), (2 if kind == "jsae" else 1)
    W_dec = rng.standard_normal((U, M, d))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    arrays = {"W_enc": rng.standard_normal((U, d, M)), "W_dec": W_dec, "b_enc": rng.normal(0, 0.1, (S, M)),
              "b_pre": rng.normal(0, 0.1, (S, d)), "b_dec": rng.normal(0, 0.1, (S, d)),
              "scale_in": rng.uniform(0.5, 2, S), "scale_out": rng.uniform(0.5, 2, S)}
    return core.Dictionary(core.Settings(kind, M, K, steps=10, batch=4), sites,
                           {n: v.astype(np.float32) for n, v in arrays.items()})


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    tokens = rng.integers(0, TINY.vocab, (3, TINY.seq)).astype(np.int32)
    source = rng.integers(0, TINY.vocab, (3, TINY.seq)).astype(np.int32)
    dicts = {
        "topk": dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 1),
        "transcoder": dictionary("transcoder", [(f"blocks.{L}.ln2.hook_normalized", f"blocks.{L}.hook_mlp_out")], 2),
        "staircase": dictionary("staircase", [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(TINY.n_layers)], 3),
    }
    store = {batch_hash(tokens): tokens, batch_hash(source): source}
    store.update({core.content_hash(d): d for d in dicts.values()})
    return model, params, tokens, source, dicts, store


def value(model, params, tokens, tap):
    _, rec = model.instrumented(params, tokens, [tap])
    return np.asarray(getattr(rec[tap], "array", rec[tap]))


def run(world, *edits, taps=(), params_at=None):
    model, params, tokens, _, _, store = world
    return run_intervened(model, params, tokens, InterventionSpec(tuple(edits)), n_layers=TINY.n_layers,
                          resolve=store.__getitem__, taps=taps, params_at=params_at)


def expected(world, d, site, tap, positions, latents=None, params=None):
    """The tap's value after the patch, from the contract: the code difference, decoded without its bias."""
    model, world_params, tokens, source, _, _ = world
    params = world_params if params is None else params
    in_tap = d.sites[site][0]
    f_t = core.encode(d, site, value(model, params, tokens, in_tap).reshape(-1, TINY.d_model))
    f_s = core.encode(d, site, value(model, params, source, in_tap).reshape(-1, TINY.d_model))
    diff = f_s - f_t
    if latents is not None:
        keep = np.zeros(M, bool)
        keep[list(latents)] = True
        diff = diff * keep
    delta = core.decode_delta(d, site, diff).reshape(tokens.shape + (TINY.d_model,))
    out = value(model, params, tokens, tap).copy()
    out[:, positions] += delta[:, positions]
    return out


def test_dpatch_arguments():
    h1, h2 = "1" * 64, "2" * 64
    e = edit("blocks.0.hook_resid_post", [1, 2], "dpatch", dictionary=h1, source=h2)
    assert dict(e.args) == {"dictionary": h1, "source": h2}
    e = edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h1, source=h2, latents=[5, 0, 3])
    assert e.args["latents"] == [0, 3, 5], "latents are kept sorted"
    assert edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h1, source=h2, latents=[]).args["latents"] == []
    one = InterventionSpec((edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h1, source=h2, latents=[3, 1]),))
    two = InterventionSpec((edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h1, source=h2, latents=[1, 3]),))
    assert spec_hash(one) == spec_hash(two) and spec_from_json(spec_json(one)) == two
    for bad in (dict(dictionary=h1), dict(dictionary=h1, source=h2, latents=[1, 1]), dict(dictionary=h1, source=h2, latents=[-1]),
                dict(dictionary=h1, source=h2, latents=[1.0]), dict(dictionary=h1, source=h2, latents=3),
                dict(dictionary="x", source=h2), dict(dictionary=h1, source=h2, head=0)):
        with pytest.raises(ValueError):
            edit("blocks.0.hook_resid_post", "all", "dpatch", **bad)


class Spy:
    """The model, noting the edits each forward pass is given."""

    def __init__(self, model):
        self.model, self.hooked = model, []

    def instrumented(self, params, tokens, taps, interventions=None):
        self.hooked.append(sorted(interventions or {}))
        return self.model.instrumented(params, tokens, taps, interventions)


def test_dpatch_identities_are_exact(world):
    model, params, tokens, source, dicts, store = world
    base, _ = model.instrumented(params, tokens, [])
    base = np.asarray(base)
    for kind, d in dicts.items():
        h = core.content_hash(d)
        for _, tap in d.sites:
            for src, latents in ((tokens, None), (source, [])):
                spy = Spy(model)
                args = dict(dictionary=h, source=batch_hash(src)) | ({} if latents is None else {"latents": latents})
                logits, _ = run_intervened(spy, params, tokens, InterventionSpec((edit(tap, "all", "dpatch", **args),)),
                                           n_layers=TINY.n_layers, resolve=store.__getitem__)
                what = "a source equal to the target" if latents is None else "no latents"
                assert np.array_equal(np.asarray(logits), base), f"{kind} at {tap}: {what} changes nothing"
                assert spy.hooked[-1] == [], f"{kind} at {tap}: {what} adds no operation at all (D25)"


def test_dpatch_moves_the_code(world):
    model, params, tokens, source, dicts, store = world
    d = dicts["topk"]
    tap = "blocks.0.hook_resid_post"
    edit_ = edit(tap, [2, 5], "dpatch", dictionary=core.content_hash(d), source=batch_hash(source))
    spy = Spy(model)
    _, rec = run_intervened(spy, params, tokens, InterventionSpec((edit_,)), n_layers=TINY.n_layers,
                            resolve=store.__getitem__, taps=[tap])
    assert spy.hooked[-1] == [tap], "the edit runs through the model's intervention hook"
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    want = expected(world, d, 0, tap, [2, 5])
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    untouched = [p for p in range(TINY.seq) if p not in (2, 5)]
    assert np.array_equal(got[:, untouched], want[:, untouched]), "other positions are exactly as they were"
    assert not np.array_equal(got, value(world[0], world[1], world[2], tap)), "the patch did something"


def test_dpatch_latents_move_only_theirs(world):
    _, _, _, source, dicts, _ = world
    d = dicts["topk"]
    tap = "blocks.0.hook_resid_post"
    latents = [0, 5, 7, 11, 19]
    _, rec = run(world, edit(tap, "all", "dpatch", dictionary=core.content_hash(d), source=batch_hash(source),
                             latents=latents), taps=[tap])
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    assert np.allclose(got, expected(world, d, 0, tap, list(range(TINY.seq)), latents), rtol=1e-5, atol=1e-5)
    assert not np.allclose(got, expected(world, d, 0, tap, list(range(TINY.seq))), rtol=1e-5, atol=1e-5)


def test_dpatch_on_a_transcoder_and_a_staircase(world):
    _, _, _, source, dicts, _ = world
    d = dicts["transcoder"]
    tap = f"blocks.{L}.hook_mlp_out"
    _, rec = run(world, edit(tap, "all", "dpatch", dictionary=core.content_hash(d), source=batch_hash(source)), taps=[tap])
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    assert np.allclose(got, expected(world, d, 0, tap, list(range(TINY.seq))), rtol=1e-5, atol=1e-5), \
        "a transcoder's codes come from ln2.hook_normalized; the edit lands on hook_mlp_out"
    d = dicts["staircase"]
    for site in range(TINY.n_layers):
        tap = f"blocks.{site}.hook_resid_post"
        _, rec = run(world, edit(tap, [0, 3], "dpatch", dictionary=core.content_hash(d), source=batch_hash(source)), taps=[tap])
        got = np.asarray(getattr(rec[tap], "array", rec[tap]))
        assert np.allclose(got, expected(world, d, site, tap, [0, 3]), rtol=1e-5, atol=1e-5), f"staircase block {site}"


def test_dpatch_reads_the_codes_after_a_swap(world):
    model, params, tokens, source, dicts, _ = world
    other = _perturbed(init_params(TINY, seed=1))
    d = dicts["transcoder"]
    tap = f"blocks.{L}.hook_mlp_out"
    swap = edit("blocks.0", "all", "swap", step=7)
    _, rec = run(world, swap, edit(tap, "all", "dpatch", dictionary=core.content_hash(d), source=batch_hash(source)),
                 taps=[tap], params_at={7: other}.__getitem__)
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    swapped = swap_component(params, other, "blocks.0")
    assert np.allclose(got, expected(world, d, 0, tap, list(range(TINY.seq)), params=swapped), rtol=1e-5, atol=1e-5)
    assert not np.allclose(got, expected(world, d, 0, tap, list(range(TINY.seq))), rtol=1e-5, atol=1e-5)


def test_dpatch_refusals(world):
    model, params, tokens, source, dicts, store = world
    src = batch_hash(source)
    h = {kind: core.content_hash(d) for kind, d in dicts.items()}
    short = source[:2]
    impostor = dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 9)
    narrow = dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 10, d=TINY.d_model // 2)
    bad_store = {**store, batch_hash(short): short, "f" * 64: impostor, core.content_hash(narrow): narrow}
    before = [  # each refused before anything runs
        edit(f"blocks.{L}.hook_resid_post", "all", "dpatch", dictionary=h["topk"], source=src),  # not its site
        edit(f"blocks.{L}.ln2.hook_normalized", "all", "dpatch", dictionary=h["transcoder"], source=src),  # an input tap
        edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary="f" * 64, source=src),  # the hash doesn't match
        edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=src, source=src),  # a batch, not a dictionary
        edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h["staircase"], source=src, latents=[M // 2]),
        edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h["topk"], source=src, latents=[M]),
        edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=h["topk"], source=batch_hash(short)),
    ]
    for case in before + [edit("blocks.0.hook_resid_post", "all", "dpatch", dictionary=core.content_hash(narrow), source=src)]:
        spy = Spy(model)
        with pytest.raises(ValueError):
            run_intervened(spy, params, tokens, InterventionSpec((case,)), n_layers=TINY.n_layers,
                           resolve=bad_store.__getitem__)
        if case in before:
            assert spy.hooked == [], f"{dict(case.args)} at {case.tap} is refused before anything runs"
