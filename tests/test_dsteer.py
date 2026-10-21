"""Phase 6 delta 1: ``dsteer``, moving chosen latents of a dictionary's code to chosen values (D48).

Contracts: src/nablaframe/interventions/spec.py and apply.py, the dsteer bullets. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.interventions.apply import run_intervened, swap_component  # noqa: E402
from nablaframe.interventions import spec  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit, spec_from_json, spec_hash, spec_json  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from nablaframe.trace.schema import tensor_sha256  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

M, K = 24, 4
L = TINY.n_layers - 1  # the last block
B = 3


def dictionary(kind, sites, seed=0, d=TINY.d_model):
    rng = np.random.default_rng(seed)
    S = len(sites)
    W_dec = rng.standard_normal((1, M, d))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    arrays = {"W_enc": rng.standard_normal((1, d, M)), "W_dec": W_dec, "b_enc": rng.normal(0, 0.1, (S, M)),
              "b_pre": rng.normal(0, 0.1, (S, d)), "b_dec": rng.normal(0, 0.1, (S, d)),
              "scale_in": rng.uniform(0.5, 2, S), "scale_out": rng.uniform(0.5, 2, S)}
    return core.Dictionary(core.Settings(kind, M, K, steps=10, batch=4), sites,
                           {n: v.astype(np.float32) for n, v in arrays.items()})


def rows(*triples):
    return np.array(triples, dtype=np.int32).reshape(-1, 3)


def amounts(*values):
    return np.array(values, dtype=np.float32)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    tokens = rng.integers(0, TINY.vocab, (B, TINY.seq)).astype(np.int32)
    dicts = {
        "topk": dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 1),
        "transcoder": dictionary("transcoder", [(f"blocks.{L}.ln2.hook_normalized", f"blocks.{L}.hook_mlp_out")], 2),
        "staircase": dictionary("staircase", [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(TINY.n_layers)], 3),
    }
    store = {batch_hash(tokens): tokens}
    store.update({core.content_hash(d): d for d in dicts.values()})
    return model, params, tokens, dicts, store


def value(model, params, tokens, tap):
    _, rec = model.instrumented(params, tokens, [tap])
    return np.asarray(getattr(rec[tap], "array", rec[tap]))


def put(store, array):
    """Store an array under its array hash, and return the hash."""
    h = spec.array_hash(array)
    store[h] = array
    return h


def steer(store, d, tap, positions, remove, add, amts):
    return edit(tap, positions, "dsteer", dictionary=core.content_hash(d), remove=put(store, remove),
                add=put(store, add), amounts=put(store, amts))


class Spy:
    """The model, noting the edits each forward pass is given."""

    def __init__(self, model):
        self.model, self.hooked = model, []

    def instrumented(self, params, tokens, taps, interventions=None):
        self.hooked.append(sorted(interventions or {}))
        return self.model.instrumented(params, tokens, taps, interventions)


def run(world, *edits, taps=(), params_at=None, store=None, model=None):
    world_model, params, tokens, _, world_store = world
    return run_intervened(model or world_model, params, tokens, InterventionSpec(tuple(edits)), n_layers=TINY.n_layers,
                          resolve=(store or world_store).__getitem__, taps=taps, params_at=params_at)


def codes(world, d, site, params=None):
    model, world_params, tokens, _, _ = world
    x = value(model, world_params if params is None else params, tokens, d.sites[site][0])
    return core.encode(d, site, x.reshape(-1, TINY.d_model)).reshape(B, TINY.seq, M)


def expected(world, d, site, tap, remove, add, amts, params=None):
    """The tap's value after the edit, from the contract: the code change, decoded without its bias."""
    model, world_params, tokens, _, _ = world
    params = world_params if params is None else params
    f = codes(world, d, site, params)
    change = np.zeros_like(f)
    for s, p, l in remove:
        change[s, p, l] -= f[s, p, l]
    for (s, p, l), a in zip(add, amts):
        change[s, p, l] += a
    delta = core.decode_delta(d, site, change.reshape(-1, M)).reshape(B, TINY.seq, -1)
    return value(model, params, tokens, tap) + delta


def test_dsteer_arguments():
    h = ["1" * 64, "2" * 64, "3" * 64, "4" * 64]
    e = edit("blocks.0.hook_resid_post", [1, 2], "dsteer", dictionary=h[0], remove=h[1], add=h[2], amounts=h[3])
    assert dict(e.args) == {"dictionary": h[0], "remove": h[1], "add": h[2], "amounts": h[3]}
    one = InterventionSpec((e,))
    assert spec_from_json(spec_json(one)) == one and spec_hash(one) == spec_hash(spec_from_json(spec_json(one)))
    full = dict(dictionary=h[0], remove=h[1], add=h[2], amounts=h[3])
    for bad in ({k: v for k, v in full.items() if k != "amounts"}, {k: v for k, v in full.items() if k != "remove"},
                full | {"latents": [1]}, full | {"source": h[1]}, full | {"add": "x"}, full | {"amounts": 3}):
        with pytest.raises(ValueError):
            edit("blocks.0.hook_resid_post", "all", "dsteer", **bad)

    # array_hash covers dtype and shape as well as bytes, so empty arrays of each role differ
    empties = [np.zeros((0, 3), np.int32), np.zeros((0,), np.float32), np.zeros((0, 3), np.int64), np.zeros((3, 0), np.int32)]
    assert len({spec.array_hash(a) for a in empties}) == 4
    assert len({tensor_sha256(a) for a in empties}) == 1, "the byte hash alone can't tell them apart"
    a = rows((0, 1, 2), (1, 2, 3))
    assert spec.array_hash(a) == spec.array_hash(a.copy()) == spec.array_hash(a.astype(">i4")), "equal values hash equal"
    assert spec.array_hash(a) != spec.array_hash(a.reshape(3, 2)) != spec.array_hash(a.astype(np.float32))
    assert spec.array_hash(amounts(1.5)) != spec.array_hash(amounts(2.5))
    for bad in (np.array([True]), np.array(["x"]), np.array([object()], dtype=object)):
        with pytest.raises(ValueError):
            spec.array_hash(bad)


def test_dsteer_identities_are_exact(world):
    model, params, tokens, dicts, store = world
    base, _ = model.instrumented(params, tokens, [])
    base = np.asarray(base)
    for kind, d in dicts.items():
        for site, (_, tap) in enumerate(d.sites):
            f = codes(world, d, site)
            width = d.active_width(site)
            zeros = np.argwhere(f[:, :, :width] == 0)[:5]  # latents that don't fire there
            fired = np.argwhere(f[:, :, :width] > 0)[:3]
            assert len(zeros) == 5 and len(fired) == 3
            cases = {
                "empty remove and add": (rows(), rows(), amounts()),
                "removing only latents whose code is zero": (rows(*map(tuple, zeros)), rows(), amounts()),
                "adding only zero amounts": (rows(), rows(*map(tuple, fired)), amounts(0, 0, -0.0)),
                "both at once": (rows(*map(tuple, zeros)), rows(*map(tuple, fired)), amounts(0, 0, 0)),
            }
            for what, (remove, add, amts) in cases.items():
                spy = Spy(model)
                logits, _ = run(world, steer(store, d, tap, "all", remove, add, amts), model=spy)
                assert np.array_equal(np.asarray(logits), base), f"{kind} at {tap}: {what} changes nothing"
                assert spy.hooked[-1] == [], f"{kind} at {tap}: {what} adds no operation at all (D25)"


def test_dsteer_moves_only_its_rows(world):
    model, params, tokens, dicts, store = world
    d = dicts["topk"]
    tap = "blocks.0.hook_resid_post"
    f = codes(world, d, 0)
    on = [tuple(int(x) for x in r) for r in np.argwhere(f > 0) if r[1] in (1, 4)]
    off = [tuple(int(x) for x in r) for r in np.argwhere(f == 0) if r[1] in (1, 4)]
    remove = rows(on[0], on[-1])
    add = rows(off[0], off[-1], on[0])  # on[0] is removed and added: its code becomes the amount
    amts = amounts(1.5, -0.5, 2.0)
    spy = Spy(model)
    _, rec = run(world, steer(store, d, tap, [1, 4], remove, add, amts), taps=[tap], model=spy)
    assert spy.hooked[-1] == [tap], "the edit runs through the model's intervention hook"
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    want = expected(world, d, 0, tap, remove, add, amts)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5)
    before = value(model, params, tokens, tap)
    touched = {(s, p) for s, p, _ in np.concatenate([remove, add])}
    for s in range(B):
        for p in range(TINY.seq):
            if (s, p) not in touched:
                assert np.array_equal(got[s, p], before[s, p]), f"sequence {s}, position {p} is exactly as it was"
            else:
                assert not np.allclose(got[s, p], before[s, p]), f"sequence {s}, position {p} moved"

    # Each row's change is its own: removing a latent moves the tap by minus its code times its decoder row, scaled.
    s, p, l = on[-1]
    one = rows(on[-1])
    _, rec = run(world, steer(store, d, tap, [p], one, rows(), amounts()), taps=[tap])
    moved = np.asarray(getattr(rec[tap], "array", rec[tap]))[s, p] - before[s, p]
    row = d.arrays["W_dec"][0][l] * d.arrays["scale_out"][0]
    assert np.allclose(moved, -f[s, p, l] * row, rtol=1e-5, atol=1e-6)
    _, rec = run(world, steer(store, d, tap, [p], rows(), one, amounts(0.75)), taps=[tap])
    moved = np.asarray(getattr(rec[tap], "array", rec[tap]))[s, p] - before[s, p]
    assert np.allclose(moved, 0.75 * row, rtol=1e-5, atol=1e-6), "an added amount moves the tap by amount times the row"


def test_dsteer_on_a_transcoder_and_a_staircase(world):
    model, params, tokens, dicts, store = world
    d = dicts["transcoder"]
    tap = f"blocks.{L}.hook_mlp_out"
    f = codes(world, d, 0)
    remove = rows(*[tuple(int(x) for x in r) for r in np.argwhere(f > 0)[:4]])
    add = rows((0, 2, 5), (2, 7, 9))
    amts = amounts(1.0, 3.0)
    _, rec = run(world, steer(store, d, tap, "all", remove, add, amts), taps=[tap])
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    assert np.allclose(got, expected(world, d, 0, tap, remove, add, amts), rtol=1e-5, atol=1e-5), \
        "a transcoder's codes come from ln2.hook_normalized; the edit lands on hook_mlp_out"

    d = dicts["staircase"]
    for site in range(TINY.n_layers):
        tap = f"blocks.{site}.hook_resid_post"
        width = d.active_width(site)
        f = codes(world, d, site)
        remove = rows(*[tuple(int(x) for x in r) for r in np.argwhere(f[:, :, :width] > 0) if r[1] in (0, 3)][:3])
        add = rows((1, 3, width - 1))
        amts = amounts(2.0)
        _, rec = run(world, steer(store, d, tap, [0, 3], remove, add, amts), taps=[tap])
        got = np.asarray(getattr(rec[tap], "array", rec[tap]))
        assert np.allclose(got, expected(world, d, site, tap, remove, add, amts), rtol=1e-5, atol=1e-5), \
            f"staircase block {site}"

    # The codes are read after the spec's swaps, as dpatch reads them.
    other = _perturbed(init_params(TINY, seed=1))
    d = dicts["transcoder"]
    tap = f"blocks.{L}.hook_mlp_out"
    swapped = swap_component(params, other, "blocks.0")
    f = codes(world, d, 0, params=swapped)
    remove = rows(*[tuple(int(x) for x in r) for r in np.argwhere(f > 0)[:6]])
    _, rec = run(world, edit("blocks.0", "all", "swap", step=7), steer(store, d, tap, "all", remove, rows(), amounts()),
                 taps=[tap], params_at={7: other}.__getitem__)
    got = np.asarray(getattr(rec[tap], "array", rec[tap]))
    assert np.allclose(got, expected(world, d, 0, tap, remove, rows(), amounts(), params=swapped), rtol=1e-5, atol=1e-5)
    assert not np.allclose(got, value(model, swapped, tokens, tap), rtol=1e-5, atol=1e-5)


def test_dsteer_refusals(world):
    model, params, tokens, dicts, store = world
    store = dict(store)
    h = {kind: core.content_hash(d) for kind, d in dicts.items()}
    tap = "blocks.0.hook_resid_post"
    ok = (rows((0, 1, 2)), rows((1, 4, 3)), amounts(1.0))

    def case(remove=None, add=None, amts=None, *, at=tap, positions=(1, 4), dictionary=None, raw=None):
        r, a, m = ok
        args = dict(dictionary=dictionary or h["topk"],
                    remove=put(store, r if remove is None else remove),
                    add=put(store, a if add is None else add),
                    amounts=put(store, m if amts is None else amts))
        args.update(raw or {})
        return edit(at, list(positions), "dsteer", **args)

    impostor = dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 9)
    store["f" * 64] = impostor
    wrong = rows((0, 1, 5))
    store["e" * 64] = wrong  # an array stored under a hash that isn't its own
    narrow = dictionary("topk", [("blocks.0.hook_resid_post",) * 2], 10, d=TINY.d_model // 2)
    store[core.content_hash(narrow)] = narrow
    before = {  # each refused before anything runs
        "not its site": case(at=f"blocks.{L}.hook_resid_post"),
        "a transcoder's input tap": case(at=f"blocks.{L}.ln2.hook_normalized", dictionary=h["transcoder"]),
        "a dictionary whose hash doesn't match": case(dictionary="f" * 64),
        "a batch, not a dictionary": case(dictionary=batch_hash(tokens)),
        "an array whose hash doesn't match": case(raw={"remove": "e" * 64}),
        "remove as int64": case(remove=rows((0, 1, 2)).astype(np.int64)),
        "remove with two columns": case(remove=np.array([[0, 1]], np.int32)),
        "add as floats": case(add=rows((1, 4, 3)).astype(np.float32)),
        "amounts as float64": case(amts=amounts(1.0).astype(np.float64)),
        "amounts of the wrong length": case(amts=amounts(1.0, 2.0)),
        "amounts shaped (1, 1)": case(amts=amounts(1.0).reshape(1, 1)),
        "a sequence past the batch": case(remove=rows((B, 1, 2))),
        "a negative sequence": case(add=rows((-1, 4, 3))),
        "a position the edit doesn't select": case(remove=rows((0, 2, 2))),
        "a negative position": case(remove=rows((0, -1, 2))),
        "a latent past the width": case(add=rows((1, 4, M))),
        "a negative latent": case(remove=rows((0, 1, -1))),
        "a latent past a staircase site's share": case(at="blocks.0.hook_resid_post", dictionary=h["staircase"],
                                                       add=rows((1, 4, M // TINY.n_layers))),
        "a repeated remove row": case(remove=rows((0, 1, 2), (0, 1, 2))),
        "a repeated add row": case(add=rows((1, 4, 3), (1, 4, 3)), amts=amounts(1.0, 1.0)),
        "a NaN amount": case(amts=amounts(np.nan)),
        "an infinite amount": case(amts=amounts(np.inf)),
    }
    for what, e in before.items():
        spy = Spy(model)
        with pytest.raises(ValueError):
            run(world, e, store=store, model=spy)
        assert spy.hooked == [], f"{what}: refused before anything runs"
    with pytest.raises(ValueError):  # a dictionary whose width doesn't fit the tap
        run(world, case(dictionary=core.content_hash(narrow)), store=store)
    run(world, case(), store=store)  # and the case they all vary is fine
