"""Phase 6 delta 4: the command line, and the one path every gesture record is built by (D47).

Contract: src/nablaframe/gestures/cli.py. The toy run is phase6_fixtures.py's identity model with a TopK SAE after
each of its two blocks, at two checkpoints: step 0, whose embeddings are scrambled, and step 3, the model itself. A
second dictionary seed holds each dictionary's latents under other ids, decoding to minus their tokens, so that it
reads apart from seed 0. Needs the jax lane.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import phase6_fixtures as F  # noqa: E402
from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.gestures import cli, declare, edge, generalize, name, records, sets  # noqa: E402
from nablaframe.gestures import open as opening  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config  # noqa: E402
from nablaframe.train.checkpoint import save_params, state_hash  # noqa: E402

RUN = "e" * 64
STEPS = (0, 3)


def scrambled(params):
    """Token t embeds as direction t + 1, and 11 as 1, so the final dictionaries read the wrong tokens."""
    wte = np.asarray(params.wte).copy()
    wte[1:] = np.roll(wte[1:], -1, axis=0)
    return type(params)(wte=jnp.asarray(wte), wpe=params.wpe, blocks=params.blocks, ln_f=params.ln_f)


def renumbered(d, seed):
    perm = np.random.default_rng(seed).permutation(d.settings.width)
    a = d.arrays
    arrays = dict(a, W_enc=a["W_enc"][:, :, perm], W_dec=a["W_dec"][:, perm], b_enc=a["b_enc"][:, perm])
    return core.Dictionary(d.settings, d.sites, {n: np.array(v, np.float32) for n, v in arrays.items()})


def held(*dictionaries):
    found = {core.content_hash(d): d for d in dictionaries}

    def get(digest):
        if digest not in found:
            raise ValueError(f"no dictionary {digest}")
        return found[digest]

    return get


@pytest.fixture(scope="module")
def toy():
    w = F.world("identity")
    w.at = {0: scrambled(w.params), 3: w.params}
    w.d = [w.dictionaries[0], w.dictionaries[1]]
    w.h = [core.content_hash(d) for d in w.d]
    w.seed1 = [renumbered(F.dictionary(block, decoder="negative"), 7) for block in (0, 1)]
    w.h1 = [core.content_hash(d) for d in w.seed1]
    w.context = cli.Context(run=RUN, declaration=w.declaration, model=w.model, n_layers=w.n_layers, steps=STEPS,
                            params_at=w.at.__getitem__, state_at=lambda s: state_hash(w.at[s]), windows=w.windows,
                            dictionary=held(*w.d, *w.seed1), seeds={1: dict(zip(w.h, w.h1))})
    return w


def request(w, kind, args, *, step=3, instances=8, seed=5):
    return records.GestureRequest(kind, RUN, step, args, instances, seed)


def read(w, block, *, params=None, dictionary=None):
    return sets.read_site(w.model, w.at[3] if params is None else params, n_layers=w.n_layers,
                          declaration=w.declaration, dictionary=dictionary or w.d[block], site=0, windows=w.windows,
                          chunk=cli.READ_CHUNK)


def thawed(record):
    return json.loads(json.dumps(records.record_json(record)["results"]))


def plain(results):
    return json.loads(json.dumps(results))


def test_declarations_for_runs():
    from nablaframe.zoo.tasks import InductionTask, TwoHopTask

    two_hop = cli.declaration_for("two_hop", GPT2Config(vocab=51, d_model=8, n_layers=4, n_heads=2, d_mlp=8, seq=26))
    assert two_hop.run == "two_hop" and two_hop.seq == TwoHopTask().seq == 26 and "b" in two_hop.variables
    induction = cli.declaration_for("induction", GPT2Config(vocab=64, d_model=8, n_layers=2, n_heads=2, d_mlp=8,
                                                            seq=InductionTask().seq))
    assert induction.run == "induction" and [r.name for r in induction.relations] == ["copy"]
    reference = cli.declaration_for("reference", F.config(2))
    assert reference.run == "reference" and reference.seq == F.SEQ and list(reference.classes) == ["every"]
    with pytest.raises(ValueError):
        cli.declaration_for("two_hop", F.config(2))  # the toy reads 6 tokens, two-hop 26
    with pytest.raises(ValueError):
        cli.declaration_for("modular_addition", F.config(2))
    assert cli.READ_CHUNK == 256 and cli.OPEN_CHUNK == 32 and cli.RUNS == ("induction", "reference", "two_hop")


def test_code_and_inputs(toy):
    w = toy
    common = {"nablaframe.gestures.cli", "nablaframe.gestures.declare", "nablaframe.gestures.sets",
              "nablaframe.zoo.declarations"}
    assert set(cli.MODULES["name"]) == common | {"nablaframe.gestures.name"}
    assert set(cli.MODULES["edge"]) == common | {"nablaframe.gestures.edge"}
    assert set(cli.MODULES["open"]) == common | {"nablaframe.gestures.name", "nablaframe.gestures.open"}
    assert set(cli.MODULES["generalize"]) == common | {"nablaframe.gestures.name", "nablaframe.gestures.edge",
                                                       "nablaframe.gestures.generalize"}
    import nablaframe.gestures.edge as e_mod
    import nablaframe.zoo.declarations as z_mod

    want = records.code_identity(cli, declare, sets, z_mod, e_mod)
    assert cli.code("edge") == want, "the blob hash of each module, as code_identity gives it"
    with pytest.raises(ValueError):
        cli.code("crystallize")

    named = request(w, "name", {"dictionary": w.h[0], "site": 0, "class": "every", "latent": F.latent(3)})
    assert cli.inputs(named, w.context) == {"states": {"3": state_hash(w.at[3])}, "dictionaries": [w.h[0]],
                                            "windows": [batch_hash(w.windows)]}
    opened = request(w, "open", {"dictionary": w.h[1], "site": 0, "class": "late", "upstream": w.h[0],
                                 "upstream_site": 0, "latent": 0}, step=0)
    assert cli.inputs(opened, w.context) == {"states": {"0": state_hash(w.at[0])}, "dictionaries": sorted(w.h),
                                             "windows": [batch_hash(w.windows)]}
    general = request(w, "generalize", {"relation": "carry", "up": w.h[0], "up_site": 0, "down": w.h[1],
                                        "down_site": 0})
    got = cli.inputs(general, w.context)
    assert got["states"] == {str(s): state_hash(w.at[s]) for s in STEPS}, "generalize-this reads every checkpoint"
    assert got["dictionaries"] == sorted(w.h + w.h1), "and every seed's counterparts"


def test_build_runs_each_kind(toy, monkeypatch):
    w = toy
    up, down = read(w, 0), read(w, 1)
    kinds = [
        (request(w, "name", {"dictionary": w.h[0], "site": 0, "class": "every", "latent": F.latent(3)}),
         lambda: name.name(up, "every", F.latent(3), instances=8, seed=5)),
        (request(w, "edge", {"relation": "carry", "up": w.h[0], "up_site": 0, "down": w.h[1], "down_site": 0}),
         lambda: edge.edge(up, down, "carry", instances=8, seed=5)),
        (request(w, "open", {"dictionary": w.h[1], "site": 0, "class": "late", "upstream": w.h[0], "upstream_site": 0,
                             "latent": F.latent(4)}),
         lambda: opening.open_up(up, down, "late", instances=8, seed=5, latent=F.latent(4), chunk=32)),
        (request(w, "open", {"dictionary": w.h[1], "site": 0, "class": "late", "upstream": w.h[0], "upstream_site": 0,
                             "description": "token(p)", "value": 5}),
         lambda: opening.open_up(up, down, "late", instances=8, seed=5, description="token(p)", value=5, chunk=32)),
        (request(w, "generalize", {"dictionary": w.h[0], "site": 0, "class": "every", "description": "token(p)"}),
         lambda: generalize.generalize_name(up, "every", "token(p)", instances=8, seed=5, steps=STEPS,
                                            params_at=w.at.__getitem__,
                                            seed_readings={1: read(w, 0, dictionary=w.seed1[0])})),
        (request(w, "generalize", {"relation": "carry", "up": w.h[0], "up_site": 0, "down": w.h[1], "down_site": 0}),
         lambda: generalize.generalize_edge(up, down, "carry", instances=8, seed=5, steps=STEPS,
                                            params_at=w.at.__getitem__,
                                            seed_readings={1: (read(w, 0, dictionary=w.seed1[0]),
                                                               read(w, 1, dictionary=w.seed1[1]))})),
    ]
    for req, direct in kinds:
        built = cli.build(req, w.context)
        assert isinstance(built, records.GestureRecord) and built.request == req
        assert thawed(built) == plain(direct()), f"{req.kind} {sorted(req.args)}: the gesture's own results"
        assert dict(built.code) == cli.code(req.kind) and json.loads(json.dumps(dict(built.inputs))) == \
            cli.inputs(req, w.context)
    quiet = F.declaration("silent")
    req = kinds[0][0]
    got = thawed(cli.build(req, dataclasses.replace(w.context, declaration=quiet)))
    reading = sets.read_site(w.model, w.at[3], n_layers=w.n_layers, declaration=quiet, dictionary=w.d[0], site=0,
                             windows=w.windows, chunk=cli.READ_CHUNK)
    assert got == plain(name.name(reading, "every", F.latent(3), instances=8, seed=5)), \
        "the context's declaration, and the request's seed: no prediction, so its examples show the draws"
    assert got["steering"]["causal"] is False
    seen = []
    real_read, real_open = sets.read_site, opening.open_up
    monkeypatch.setattr(sets, "read_site", lambda *a, **k: seen.append(("read", k["chunk"])) or real_read(*a, **k))
    monkeypatch.setattr(opening, "open_up", lambda *a, **k: seen.append(("open", k["chunk"])) or real_open(*a, **k))
    cli.build(kinds[2][0], w.context)
    assert seen == [("read", cli.READ_CHUNK), ("read", cli.READ_CHUNK), ("open", cli.OPEN_CHUNK)], "the fixed chunks"
    monkeypatch.undo()
    seedless = dataclasses.replace(w.context, seeds={})
    req = kinds[4][0]
    got = thawed(cli.build(req, seedless))
    assert got["seeds"] == [] and got == plain(generalize.generalize_name(
        up, "every", "token(p)", instances=8, seed=5, steps=STEPS, params_at=w.at.__getitem__)), "no seeds, none read"


def test_records_are_cached_and_repeatable(toy, tmp_path, monkeypatch):
    w = toy
    req = request(w, "open", {"dictionary": w.h[1], "site": 0, "class": "late", "upstream": w.h[0],
                              "upstream_site": 0, "description": "token(p)", "value": 5})
    first, second = cli.build(req, w.context), cli.build(req, w.context)
    a, b = (records.record_json(r) for r in (first, second))
    from nablaframe.trace.schema import canonical_json

    assert canonical_json(a) == canonical_json(b), "built twice from one request, byte for byte"
    written = cli.record(tmp_path, req, w.context)
    path = tmp_path / f"gesture-{written.key}.json"
    assert written.key == first.key and path.read_text() == canonical_json(a) + "\n"
    before = path.read_bytes()
    calls = []
    monkeypatch.setattr(cli, "results", lambda *args: calls.append(1) or {})
    again = cli.record(tmp_path, req, w.context)
    assert calls == [] and again == first and path.read_bytes() == before, "the cache answers; nothing recomputes"


def test_refusals(toy, monkeypatch):
    w = toy
    calls = []
    real = cli.results
    monkeypatch.setattr(name, "name", lambda *a, **k: calls.append(1))
    args = {"dictionary": w.h[0], "site": 0, "class": "every", "latent": 0}
    for bad in (records.GestureRequest("name", "f" * 64, 3, args, 8, 5),  # another run
                request(w, "name", args, step=2),  # not a checkpoint
                request(w, "name", dict(args, dictionary="0" * 64))):  # a dictionary the context doesn't hold
        for call in (cli.inputs, real, cli.build):
            with pytest.raises(ValueError):
                call(bad, w.context)
    partial = dataclasses.replace(w.context, seeds={1: {w.h[0]: w.h1[0]}})
    general = request(w, "generalize", {"relation": "carry", "up": w.h[0], "up_site": 0, "down": w.h[1],
                                        "down_site": 0})
    with pytest.raises(ValueError):
        cli.inputs(general, partial)  # seed 1 has no counterpart of the downstream dictionary
    assert calls == [], "refused before anything runs"


def _files(w, tmp_path, req, *, run_id=RUN, extra=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dataclasses.asdict(F.config(w.n_layers))))
    checkpoints = tmp_path / "checkpoints"
    for step in STEPS:
        save_params(w.at[step], checkpoints, step=step, run_id=run_id)
    windows = tmp_path / "windows.npy"
    np.save(windows, w.windows)
    dictionaries = tmp_path / "dictionaries"
    for k, d in enumerate(w.d + w.seed1):
        core.save_dictionary(d, dictionaries / f"key{k}")
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"1": dict(zip(w.h, w.h1))}))
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(records.request_json(req)))
    out = tmp_path / "out"
    return ["--run", "reference", "--config", str(config), "--checkpoints", str(checkpoints), "--windows", str(windows),
            "--dictionaries", str(dictionaries), "--request", str(request_path), "--out", str(out),
            "--seeds", str(seeds)] + (extra or []), out


def test_the_command_line(toy, tmp_path, capsys):
    from nablaframe.trace.schema import canonical_json

    w = toy
    reference = declare.reference(F.SEQ)
    context = dataclasses.replace(w.context, declaration=reference)
    for k, req in enumerate((
            request(w, "name", {"dictionary": w.h[0], "site": 0, "class": "every", "latent": F.latent(6)}),
            request(w, "generalize", {"dictionary": w.h[1], "site": 0, "class": "every", "description": "token(p)"},
                    step=0))):
        args, out = _files(w, tmp_path / str(k), req)
        assert cli.main(args) == 0
        want = cli.build(req, context)
        key = capsys.readouterr().out.strip()
        assert key == want.key, "it prints the record's key"
        path = out / f"gesture-{want.key}.json"
        written = path.read_bytes()
        assert written == (canonical_json(records.record_json(want)) + "\n").encode("ascii"), \
            "the command line writes the record build gives, byte for byte"
        index = json.loads((out / "index.json").read_text())
        assert [e["key"] for e in index["records"]] == [want.key]
        assert cli.main(args) == 0 and path.read_bytes() == written, "a rerun is byte-identical"
        assert capsys.readouterr().out.strip() == want.key


def test_the_command_line_refuses(toy, tmp_path, capsys):
    w = toy
    good = request(w, "name", {"dictionary": w.h[0], "site": 0, "class": "every", "latent": F.latent(6)})
    cases = {
        "the checkpoints are another run's": (good, {"run_id": "f" * 64}),
        "a dictionary that isn't there": (request(w, "name", {"dictionary": "0" * 64, "site": 0, "class": "every",
                                                              "latent": 0}), {}),
        "a step that isn't a checkpoint": (request(w, "name", {"dictionary": w.h[0], "site": 0, "class": "every",
                                                               "latent": 0}, step=1), {}),
        "a relation the reference run doesn't declare": (request(w, "edge", {"relation": "carry", "up": w.h[0],
                                                                             "up_site": 0, "down": w.h[1],
                                                                             "down_site": 0}), {}),
    }
    for k, (label, (req, options)) in enumerate(cases.items()):
        args, out = _files(w, tmp_path / str(k), req, **options)
        assert cli.main(args) == 1, label
        assert capsys.readouterr().err.startswith("error:"), label
        assert not out.exists() or not any(out.iterdir()), f"{label}: nothing written"
    args, out = _files(w, tmp_path / "twice", good)
    core.save_dictionary(w.d[0], tmp_path / "twice" / "dictionaries" / "again")
    assert cli.main(args) == 1 and not out.exists(), "two files of one dictionary are refused"
