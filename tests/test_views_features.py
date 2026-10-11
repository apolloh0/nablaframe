"""Phase 5 delta 5: the features view's payload (D45).

Contract: src/nablaframe/views/features.py. Each expected value is computed here from the model's
activations and the dictionary's own encode, with the ordering rules written out. Needs the jax lane.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import equinox as eqx  # noqa: E402

from nablaframe.dictionaries import core, measure  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from nablaframe.trace.schema import canonical_json  # noqa: E402
from nablaframe.views import features as f  # noqa: E402
from nablaframe.views.export import export_view  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

D, M, K = TINY.d_model, 24, 4
RESID = [f"blocks.{l}.hook_resid_post" for l in range(TINY.n_layers)]
LN2_0, MLP_OUT_0 = "blocks.0.ln2.hook_normalized", "blocks.0.hook_mlp_out"
SILENT = 23  # a latent whose encoder bias keeps it from ever firing
METERS = {"gate": "G5.2", "path": ["meters", "topk blocks.1.hook_resid_post"],
          "values": {"fvu": 0.031, "dead_share": 0.0, "twin_fvu": 0.15, "recovered": 0.995, "cost": None}}


def _dictionary(kind, sites, seed, m=M):
    rng = np.random.default_rng(seed)
    U, S = (2 if kind == "jsae" else 1), len(sites)
    W_dec = rng.standard_normal((U, m, D))
    W_dec /= np.linalg.norm(W_dec, axis=2, keepdims=True)
    b_enc = rng.normal(0, 0.2, (S, m))
    b_enc[:, SILENT] = -1e4
    arrays = {"W_enc": rng.standard_normal((U, D, m)) * 0.5, "W_dec": W_dec, "b_enc": b_enc,
              "b_pre": rng.normal(0, 0.1, (S, D)), "b_dec": rng.normal(0, 0.1, (S, D)),
              "scale_in": rng.uniform(0.5, 1.5, S), "scale_out": rng.uniform(0.5, 1.5, S)}
    return core.Dictionary(core.Settings(kind, m, K, steps=10, batch=4), sites,
                           {n: np.asarray(v, np.float32) for n, v in arrays.items()})


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    model, params = GPT2Model(TINY), _perturbed(init_params(TINY, seed=0))
    probe = rng.integers(0, TINY.vocab, (2, TINY.seq)).astype(np.int32)
    windows = rng.integers(0, TINY.vocab, (5, TINY.seq)).astype(np.int32)
    held_out = np.concatenate([windows, windows[:2]])  # windows 5 and 6 repeat 0 and 1: equal codes, tied
    topk = _dictionary("topk", [(RESID[1], RESID[1])], 1)
    stair = _dictionary("staircase", [(t, t) for t in RESID], 2)
    transcoder = _dictionary("transcoder", [(LN2_0, MLP_OUT_0)], 3)
    return model, params, probe, held_out, topk, stair, transcoder


def _payload(world, dictionary=None, site=0, **kw):
    model, params, probe, held_out, topk, _, _ = world
    kw = {"step": 7, "meters": METERS, **kw}
    return f.features_payload(model, params, topk if dictionary is None else dictionary, site, kw.pop("probe", probe),
                              kw.pop("held_out", held_out), **kw)


def _codes(model, params, dictionary, site, tokens):
    tap = dictionary.sites[site][0]
    _, rec = model.instrumented(params, tokens, [tap])
    x = np.asarray(getattr(rec[tap], "array", rec[tap]), np.float32)
    return core.encode(dictionary, site, x.reshape(-1, D)).reshape(*tokens.shape, -1)


def _active(row):
    ids = [int(i) for i in np.flatnonzero(row > 0)]
    ids.sort(key=lambda i: (-row[i], i))
    return ids, [float(row[i]) for i in ids]


def test_features_constants():
    assert f.SHIPPED == ("topk", "staircase", "transcoder")
    assert f.TOP_CONTEXTS == 8 and f.EFFECT_TOKENS == 5 and f.METER_GATES == ("G5.2", "G5.3", "G5.5")
    assert f.EFFECT_LABEL == ("Direct effect on the logits through the unembedding: the latent's decoder row, times the "
                              "site's scale, centred and multiplied by the final layer norm's gain, then unembedded, with "
                              "the layer norm's divisor held at 1. It is not a name for the latent.")


def test_the_payload_describes_its_site_and_the_probe(world):
    model, params, probe, held_out, topk, _, _ = world
    p = _payload(world, variables={"b": np.arange(7) % 3})
    assert set(p) == {"step", "dictionary", "kind", "site", "taps", "width", "k", "probe", "held_out", "variables",
                      "frequency", "contexts", "effect", "meters", "meter"}
    assert (p["step"], p["dictionary"], p["kind"], p["site"]) == (7, core.content_hash(topk), "topk", 0)
    assert (p["taps"], p["width"], p["k"]) == ([RESID[1], RESID[1]], M, K)
    assert p["held_out"] == held_out.tolist() and p["variables"] == {"b": [0, 1, 2, 0, 1, 2, 0]}
    codes = _codes(model, params, topk, 0, probe)
    for b in range(probe.shape[0]):
        for pos in range(probe.shape[1]):
            ids, values = _active(codes[b, pos])
            assert p["probe"]["latents"][b][pos] == ids, "active latents, largest code first, ties to the lower id"
            assert p["probe"]["values"][b][pos] == values
    assert _payload(world)["variables"] == {}


def test_frequency_and_top_contexts(world):
    model, params, _, held_out, topk, _, _ = world
    p = _payload(world)
    codes = _codes(model, params, topk, 0, held_out).reshape(-1, M)  # window-major rows
    T = held_out.shape[1]
    assert len(p["frequency"]) == len(p["contexts"]) == M
    tied = 0
    for i in range(M):
        assert p["frequency"][i] == float((codes[:, i] > 0).mean()), f"latent {i}'s share of held-out positions"
        rows = [int(r) for r in np.flatnonzero(codes[:, i] > 0)]
        rows.sort(key=lambda r: (-codes[r, i], r))
        want = [[r // T, r % T, float(codes[r, i])] for r in rows[:f.TOP_CONTEXTS]]
        assert p["contexts"][i] == want, f"latent {i}: largest codes first, ties to the lower window, then position"
        tied += sum(a[2] == b[2] for a, b in zip(want, want[1:]))
    assert tied > 0, "the repeated windows give tied codes, so the tie rule is exercised"
    assert p["frequency"][SILENT] == 0.0 and p["contexts"][SILENT] == [], "a latent that never fires"


def test_the_direct_effect(world):
    model, params, _, _, topk, stair, _ = world
    for dictionary, site in ((topk, 0), (stair, 1)):
        p = _payload(world, dictionary, site)
        assert p["effect"]["label"] == f.EFFECT_LABEL
        a = dictionary.arrays
        rows = a["W_dec"][0][:dictionary.active_width(site)] * a["scale_out"][site]  # decode_delta of each one-hot
        rows = rows.astype(np.float64)
        rows -= rows.mean(axis=1, keepdims=True)
        effect = (rows * np.asarray(params.ln_f.weight, np.float64)) @ np.asarray(params.wte, np.float64).T
        for i, e in enumerate(effect):
            top = sorted(range(TINY.vocab), key=lambda t: (-e[t], t))[:f.EFFECT_TOKENS]
            bottom = sorted(range(TINY.vocab), key=lambda t: (e[t], t))[:f.EFFECT_TOKENS]
            assert [t for t, _ in p["effect"]["top"][i]] == top, f"latent {i}'s largest effects, largest first"
            assert [t for t, _ in p["effect"]["bottom"][i]] == bottom, f"latent {i}'s smallest effects, smallest first"
            assert [v for _, v in p["effect"]["top"][i]] == pytest.approx([e[t] for t in top], rel=1e-9, abs=1e-12)
            assert [v for _, v in p["effect"]["bottom"][i]] == pytest.approx([e[t] for t in bottom], rel=1e-9, abs=1e-12)
    gainless = eqx.tree_at(lambda q: q.ln_f.weight, params, np.zeros(D, np.float32))
    p = f.features_payload(model, gainless, topk, 0, world[2], world[3], step=7, meters=METERS)
    assert all(row == [[t, 0.0] for t in range(f.EFFECT_TOKENS)] for row in p["effect"]["top"] + p["effect"]["bottom"]), \
        "with every effect equal, ties go to the lower token"


def test_staircase_and_transcoder_sites(world):
    model, params, probe, held_out, _, stair, transcoder = world
    for site, width in ((0, M // 2), (1, M)):
        p = _payload(world, stair, site)
        assert (p["site"], p["width"], p["taps"]) == (site, width, [RESID[site], RESID[site]])
        assert len(p["frequency"]) == len(p["contexts"]) == len(p["effect"]["top"]) == width, "the site's own latents"
        codes = _codes(model, params, stair, site, probe)
        assert p["probe"]["latents"][1][3] == _active(codes[1, 3])[0], "the site's own biases and scale"
        assert all(i < width for row in p["probe"]["latents"] for ids in row for i in ids)
        held = _codes(model, params, stair, site, held_out).reshape(-1, M)[:, :width]
        assert p["frequency"] == [float(v) for v in (held > 0).mean(axis=0)], "held-out codes at the site's own share"
    p = _payload(world, transcoder)
    assert p["taps"] == [LN2_0, MLP_OUT_0]
    codes = _codes(model, params, transcoder, 0, probe)
    assert p["probe"]["latents"][0][2] == _active(codes[0, 2])[0], "a transcoder's codes come from its input tap"
    held = _codes(model, params, transcoder, 0, held_out).reshape(-1, M)
    assert p["frequency"] == [float(v) for v in (held > 0).mean(axis=0)], "on held-out windows too"
    _, rec = model.instrumented(params, probe, [LN2_0, MLP_OUT_0])
    x, y = (np.asarray(getattr(rec[t], "array", rec[t]), np.float32).reshape(-1, D) for t in (LN2_0, MLP_OUT_0))
    assert p["meter"] == {"kind": "reconstruction", "fvu": measure.fvu(transcoder, 0, x, y), "positions": probe.size}


def test_meters_are_copied_and_the_meter_measures_the_probe(world):
    model, params, probe, _, topk, _, _ = world
    meters = copy.deepcopy(METERS)
    p = _payload(world, meters=meters)
    assert p["meters"] == METERS
    meters["values"]["fvu"] = 0.5
    meters["path"].append("x")
    assert p["meters"] == METERS, "a copy, not the caller's objects"
    _, rec = model.instrumented(params, probe, [RESID[1]])
    x = np.asarray(getattr(rec[RESID[1]], "array", rec[RESID[1]]), np.float32).reshape(-1, D)
    assert p["meter"] == {"kind": "reconstruction", "fvu": measure.fvu(topk, 0, x, x), "positions": probe.size}


def test_the_payload_is_plain_json_and_repeatable(world, tmp_path):
    p, q = _payload(world), _payload(world)
    assert p == q
    text = canonical_json(p)
    assert canonical_json(q) == text
    assert export_view(tmp_path, "features", p).read_text() == text
    leaves = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            leaves.append(v)

    walk(p)
    assert all(type(v) in (int, float, str, type(None)) for v in leaves), "plain Python values only"


def test_bad_inputs_are_refused(world):
    model, params, probe, held_out, topk, stair, _ = world
    jsae = _dictionary("jsae", [("blocks.1.ln2.hook_normalized",) * 2, ("blocks.1.hook_mlp_out",) * 2], 4)
    cases = {
        "a Jacobian SAE isn't shipped": lambda: _payload(world, jsae),
        "a site the dictionary doesn't have": lambda: _payload(world, stair, 2),
        "a site that isn't an int": lambda: _payload(world, topk, True),
        "a 1-D probe": lambda: _payload(world, probe=probe[0]),
        "a float probe": lambda: _payload(world, probe=probe.astype(np.float32)),
        "an empty held-out set": lambda: _payload(world, held_out=held_out[:0]),
        "a token outside the vocabulary": lambda: _payload(world, probe=np.where(probe == probe[0, 0], TINY.vocab, probe)),
        "a negative token": lambda: _payload(world, held_out=held_out - 1 - held_out.max()),
        "a negative step": lambda: _payload(world, step=-1),
        "a step that isn't an int": lambda: _payload(world, step=7.0),
        "meters from a gate that records none": lambda: _payload(world, meters={**METERS, "gate": "G5.4"}),
        "meters with another key": lambda: _payload(world, meters={**METERS, "extra": 1}),
        "meters with an empty path": lambda: _payload(world, meters={**METERS, "path": []}),
        "meters with a path of floats": lambda: _payload(world, meters={**METERS, "path": ["meters", 1.5]}),
        "a meter that isn't finite": lambda: _payload(world, meters={**METERS, "values": {"fvu": float("nan")}}),
        "a meter that isn't a number": lambda: _payload(world, meters={**METERS, "values": {"fvu": "0.03"}}),
        "variables of the wrong length": lambda: _payload(world, variables={"b": np.arange(6)}),
        "variables that aren't integers": lambda: _payload(world, variables={"b": np.arange(7) / 2}),
        "a variable without a name": lambda: _payload(world, variables={"": np.arange(7)}),
    }
    for label, case in cases.items():
        with pytest.raises(ValueError):
            case()
            pytest.fail(label)
