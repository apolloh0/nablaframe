"""Phase 7 delta 1: ``set``, writing given vectors into a tap (D55).

Contracts: src/nablaframe/interventions/spec.py and apply.py, the set bullets. Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.interventions import spec  # noqa: E402
from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit, spec_from_json, spec_hash, spec_json  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from nablaframe.trace.schema import tensor_sha256  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

B, D, L = 3, TINY.d_model, TINY.n_layers - 1
TAPS = ("hook_embed", "hook_pos_embed", "blocks.0.hook_resid_pre", "blocks.0.hook_attn_out", "blocks.0.hook_resid_mid",
        "blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out", f"blocks.{L}.hook_resid_post", "ln_final.hook_normalized")


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    tokens = rng.integers(0, TINY.vocab, (B, TINY.seq)).astype(np.int32)
    return model, params, tokens, {batch_hash(tokens): tokens}


class Spy:
    """The model, noting the edits each forward pass is given."""

    def __init__(self, model):
        self.model, self.hooked = model, []

    def instrumented(self, params, tokens, taps, interventions=None):
        self.hooked.append(sorted(interventions or {}))
        return self.model.instrumented(params, tokens, taps, interventions)


def value(model, params, tokens, tap):
    _, rec = model.instrumented(params, tokens, [tap])
    return np.asarray(getattr(rec[tap], "array", rec[tap]))


def put(store, array):
    h = spec.array_hash(array)
    store[h] = array
    return h


def setting(store, tap, positions, values):
    return edit(tap, positions, "set", values=put(store, values))


def run(world, *edits, taps=(), store=None, model=None):
    world_model, params, tokens, world_store = world
    return run_intervened(model or world_model, params, tokens, InterventionSpec(tuple(edits)), n_layers=TINY.n_layers,
                          resolve=(store if store is not None else world_store).__getitem__, taps=taps)


def test_set_arguments():
    assert "set" in spec.OPERATIONS
    h = "1" * 64
    e = edit("blocks.0.hook_resid_post", [1, 2], "set", values=h)
    assert dict(e.args) == {"values": h}
    one = InterventionSpec((e,))
    assert spec_from_json(spec_json(one)) == one and spec_hash(one) == spec_hash(spec_from_json(spec_json(one)))
    assert spec_hash(one) != spec_hash(InterventionSpec((edit("blocks.0.hook_resid_post", [1, 2], "set", values="2" * 64),)))
    for bad in ({}, {"values": "x"}, {"values": 3}, {"values": h, "alpha": 1.0}, {"vector": h}, {"source": h}):
        with pytest.raises(ValueError):
            edit("blocks.0.hook_resid_post", "all", "set", **bad)
    a = np.ones((B, 2, D), np.float32)
    assert spec.array_hash(a) != spec.array_hash(a.reshape(B, 2 * D)) != tensor_sha256(a)


def test_set_identities_are_exact(world):
    model, params, tokens, store = world
    store = dict(store)
    base, base_rec = model.instrumented(params, tokens, list(TAPS))
    base = np.asarray(base)
    for tap in TAPS:
        own = value(model, params, tokens, tap)
        for positions, index in (("all", np.arange(TINY.seq)), ([3, 1], np.array([3, 1])), ({"range": [2, 5]}, np.arange(2, 5))):
            logits, rec = run(world, setting(store, tap, positions, own[:, index]), taps=list(TAPS), store=store)
            assert np.array_equal(np.asarray(logits), base), f"{tap} at {positions}: its own values change nothing"
            for name in TAPS:
                got = np.asarray(getattr(rec[name], "array", rec[name]))
                assert np.array_equal(got, np.asarray(getattr(base_rec[name], "array", base_rec[name]))), name
        spy = Spy(model)
        logits, _ = run(world, setting(store, tap, [], np.zeros((B, 0, D), np.float32)), store=store, model=spy)
        assert np.array_equal(np.asarray(logits), base), f"{tap}: a set at no positions changes nothing"
        assert spy.hooked[-1] == [], f"{tap}: a set at no positions adds no operation at all (D25)"


def test_set_moves_only_its_positions(world):
    model, params, tokens, store = world
    store = dict(store)
    rng = np.random.default_rng(4)
    base = np.asarray(model.instrumented(params, tokens, [])[0])
    for tap in ("blocks.0.hook_resid_post", "blocks.0.hook_mlp_out", "blocks.0.hook_attn_out", "hook_embed"):
        before = value(model, params, tokens, tap)
        values = rng.standard_normal((B, 2, D)).astype(np.float32)
        spy = Spy(model)
        logits, rec = run(world, setting(store, tap, [5, 2], values), taps=[tap], store=store, model=spy)
        assert spy.hooked[-1] == [tap], "the edit runs through the model's intervention hook"
        got = np.asarray(getattr(rec[tap], "array", rec[tap]))
        assert np.array_equal(got[:, 5], values[:, 0]) and np.array_equal(got[:, 2], values[:, 1]), \
            "values[:, i] lands at the i-th position the edit selects, in the order given"
        others = [p for p in range(TINY.seq) if p not in (2, 5)]
        assert np.array_equal(got[:, others], before[:, others]), "every other position is exactly as it was"
        logits = np.asarray(logits)
        assert np.array_equal(logits[:, :2], base[:, :2]), "positions before the first one set can't see it"
        assert not np.allclose(logits[:, 2], base[:, 2]) and not np.allclose(logits[:, 5], base[:, 5])

    # The operand is an argument: new values reuse the compiled program and give new results.
    tap = "blocks.0.hook_resid_post"
    first = rng.standard_normal((B, 1, D)).astype(np.float32)
    second = rng.standard_normal((B, 1, D)).astype(np.float32)
    one, _ = run(world, setting(store, tap, [4], first), store=store)
    two, _ = run(world, setting(store, tap, [4], second), store=store)
    assert not np.allclose(np.asarray(one), np.asarray(two))

    # In order with other edits at the same tap: a set after an add overwrites it, an add after a set adds to it.
    vector = rng.standard_normal(D).astype(np.float32)
    store[tensor_sha256(vector)] = vector
    add = edit(tap, [4], "add", vector=tensor_sha256(vector), alpha=1.0)
    _, rec = run(world, add, setting(store, tap, [4], first), taps=[tap], store=store)
    assert np.array_equal(np.asarray(getattr(rec[tap], "array", rec[tap]))[:, 4], first[:, 0])
    _, rec = run(world, setting(store, tap, [4], first), add, taps=[tap], store=store)
    assert np.allclose(np.asarray(getattr(rec[tap], "array", rec[tap]))[:, 4], first[:, 0] + vector, atol=1e-6)


def test_set_refusals(world):
    model, params, tokens, store = world
    store = dict(store)
    ok = np.zeros((B, 2, D), np.float32)
    wrong = np.ones((B, 2, D), np.float32)
    store["e" * 64] = wrong  # an array stored under a hash that isn't its own
    before = {  # each refused before anything runs
        "a head's tap": setting(store, "blocks.0.attn.hook_z", [1, 2], ok),
        "the attention pattern": setting(store, "blocks.0.attn.hook_pattern", [1, 2], ok),
        "an array whose hash doesn't match": edit("blocks.0.hook_resid_post", [1, 2], "set", values="e" * 64),
        "float64 values": setting(store, "blocks.0.hook_resid_post", [1, 2], ok.astype(np.float64)),
        "int32 values": setting(store, "blocks.0.hook_resid_post", [1, 2], ok.astype(np.int32)),
        "values for three positions": setting(store, "blocks.0.hook_resid_post", [1, 2], np.zeros((B, 3, D), np.float32)),
        "values for another batch": setting(store, "blocks.0.hook_resid_post", [1, 2], np.zeros((B + 1, 2, D), np.float32)),
        "values of another width": setting(store, "blocks.0.hook_resid_post", [1, 2], np.zeros((B, 2, D + 1), np.float32)),
        "values without a position axis": setting(store, "blocks.0.hook_resid_post", [1], np.zeros((B, D), np.float32)),
        "a NaN": setting(store, "blocks.0.hook_resid_post", [1, 2], np.where(np.arange(D) == 3, np.nan, 0.0).astype(np.float32)
                         * np.ones((B, 2, 1), np.float32)),
        "an infinity": setting(store, "blocks.0.hook_resid_post", [1, 2], np.full((B, 2, D), np.inf, np.float32)),
        "a position past the sequence": setting(store, "blocks.0.hook_resid_post", [TINY.seq], np.zeros((B, 1, D), np.float32)),
    }
    for what, e in before.items():
        spy = Spy(model)
        with pytest.raises(ValueError):
            run(world, e, store=store, model=spy)
        assert spy.hooked == [], f"{what}: refused before anything runs"
    run(world, setting(store, "blocks.0.hook_resid_post", [1, 2], ok), store=store)  # the case they vary is fine
