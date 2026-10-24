"""Phase 7 delta 3: crystallize-this through the command line, the one path every gesture record is built by (D47, D56).

Contract: src/nablaframe/gestures/cli.py, the crystallize-this bullets. The toy run is phase7_fixtures.py's modular
addition at p = 11, with two checkpoints: step 0, the twin's own initialisation nudged another way, and step 3, the
fixture's params. The command line reads the task at its defaults (p = 113), so its test builds a model of that
vocabulary. Needs the jax lane.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import phase7_fixtures as F  # noqa: E402
from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.gestures import cli, crystallize, declare, records, sets  # noqa: E402
from nablaframe.gestures.name import NAMED  # noqa: E402
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params  # noqa: E402
from nablaframe.trace.schema import canonical_json  # noqa: E402
from nablaframe.train.checkpoint import save_params, state_hash  # noqa: E402
from nablaframe.zoo import declarations, functions  # noqa: E402
from nablaframe.zoo.tasks import InductionTask, ModularAdditionTask, TwoHopTask  # noqa: E402

RUN, OTHER = "e" * 64, "c" * 64
STEPS = (0, 3)
D = "d" * 64


def name_record(run, latent, value, *, label=NAMED, description="token(p+1)"):
    """A name record, made up: only its request and results matter to a citation."""
    return records.GestureRecord(
        records.GestureRequest("name", run, 3, {"dictionary": D, "site": 0, "class": "answer", "latent": latent}, 8, 0),
        {"states": {"3": "f" * 64}, "dictionaries": [D], "windows": ["b" * 64]}, {"nablaframe.gestures.name": "0" * 40},
        {"name": {"description": description, "value": value, "f1": 0.8, "label": label}})


def holding(*held):
    found = {r.key: r for r in held}

    def cited(key):
        if key not in found:
            raise ValueError(f"no record {key}")
        return found[key]

    return cited


def no_dictionary(digest):
    raise ValueError(f"no dictionary {digest}")


@pytest.fixture(scope="module")
def ma():
    w = F.world("modular_addition")
    w.at = {0: F.loud(init_params(w.config, seed=1), 9), 3: w.params}
    sums = sorted(set(w.functions.keys["sum"].values(w.held)[:, 2].tolist()))
    w.names = [name_record(RUN, i, v) for i, v in enumerate(sums[:5])]
    w.operands = [name_record(RUN, 20 + v, v, description="token(p-2)") for v in range(3)]
    return w


def context(w, **changes):
    """The toy run's context: every checkpoint, the twin, the run's functions and the made-up name records."""
    fields = dict(run=RUN, declaration=w.declaration, model=w.model, n_layers=w.n_layers, steps=STEPS,
                  params_at=w.at.__getitem__, state_at=lambda s: state_hash(w.at[s]), windows=w.held,
                  dictionary=no_dictionary, seeds={}, fit_windows=w.fit, functions=w.functions, twin=w.twin,
                  twin_state=state_hash(w.twin), cited=holding(*w.names, *w.operands))
    return cli.Context(**(fields | changes))


def request(part="write", function="sum", form="table", *, cites=(), step=3, instances=None, seed=5, w=None, **args):
    fields = {"block": 0, "part": part, "class": "answer", "function": function, "form": form,
              "cites": [r.key for r in cites]} | args
    return records.GestureRequest("crystallize", RUN, step, fields, len(w.held) if instances is None else instances,
                                  seed)


def direct(w, req, params, *, cited):
    a = req.args
    return crystallize.crystallize(w.model, params, n_layers=w.n_layers, functions=w.functions,
                                   declaration=w.declaration, block=a["block"], part=a["part"], cls=a["class"],
                                   function=a["function"], form=a["form"], fit_windows=w.fit, held_windows=w.held,
                                   seed=req.seed, twin=w.twin, steps=STEPS, params_at=w.at.__getitem__, cited=cited)


def thawed(record):
    return json.loads(json.dumps(records.record_json(record)["results"]))


def plain(results):
    return json.loads(json.dumps(results))


def test_crystallize_runs_and_code():
    assert cli.RUNS == ("induction", "modular_addition", "reference", "two_hop")
    config = GPT2Config(vocab=114, d_model=8, n_layers=1, n_heads=2, d_mlp=8, seq=3)
    got, want = cli.declaration_for("modular_addition", config), declarations.modular_addition(ModularAdditionTask())
    assert (got.run, got.seq, {k: list(v) for k, v in got.classes.items()}) == \
        (want.run, want.seq, {k: list(v) for k, v in want.classes.items()}) == ("modular_addition", 3, {
            "answer": [2], "every": [0, 1, 2]})
    with pytest.raises(ValueError):
        cli.declaration_for("modular_addition", dataclasses.replace(config, seq=26))
    for run, task in (("induction", InductionTask()), ("two_hop", TwoHopTask()), ("modular_addition",
                                                                                  ModularAdditionTask())):
        got, want = cli.functions_for(run), functions.for_run(run, task)
        assert got.run == run and got.seq == want.seq and got.functions == want.functions, run
        assert {k: v.n_values for k, v in got.keys.items()} == {k: v.n_values for k, v in want.keys.items()}, \
            "the task at its defaults, as G4.1 and G7.1 trained it"
    for bad in ("reference", "addition"):
        with pytest.raises(ValueError):
            cli.functions_for(bad)

    common = {"nablaframe.gestures.cli", "nablaframe.gestures.declare", "nablaframe.gestures.sets",
              "nablaframe.zoo.declarations"}
    assert set(cli.MODULES["crystallize"]) == common | {"nablaframe.gestures.crystallize", "nablaframe.zoo.functions"}
    assert cli.code("crystallize") == records.code_identity(cli, declare, sets, declarations, crystallize, functions)
    with pytest.raises(ValueError):
        cli.code("steer")


def test_crystallize_inputs_and_build(ma):
    w = ma
    req = request(cites=w.names, w=w)
    assert cli.inputs(req, context(w)) == {
        "states": {"0": state_hash(w.at[0]), "3": state_hash(w.at[3]), "twin": state_hash(w.twin)},
        "dictionaries": [], "windows": sorted([batch_hash(w.fit), batch_hash(w.held)])}, \
        "every checkpoint the envelope reads, and the twin; both window sets"
    same = context(w, fit_windows=w.held)
    assert cli.inputs(req, same)["windows"] == [batch_hash(w.held)], "sorted, with no repeats"

    built = cli.build(req, context(w))
    assert built.request == req and dict(built.code) == cli.code("crystallize")
    assert json.loads(json.dumps(dict(built.inputs))) == cli.inputs(req, context(w))
    assert thawed(built) == plain(direct(w, req, w.at[3], cited=sorted(w.names, key=lambda r: r.key))), \
        "the gesture's own results, at the request's checkpoint, with every checkpoint, the twin and the cited records"
    assert thawed(built)["cites"]["names"][0]["records"] == sorted(r.key for r in w.names)

    early = request(form="fourier", step=0, w=w)
    assert thawed(cli.build(early, context(w))) == plain(direct(w, early, w.at[0], cited=[])), \
        "the request's step picks the params; no cites, an empty citation"

    mlp = request("mlp", "sum and operands", "additive", cites=w.operands, w=w)
    got = thawed(cli.build(mlp, context(w)))
    assert got == plain(direct(w, mlp, w.at[3], cited=sorted(w.operands, key=lambda r: r.key)))
    assert got["counterfactual"] != plain(direct(w, request("mlp", "sum and operands", "additive", cites=w.operands,
                                                            seed=6, w=w), w.at[3], cited=w.operands))["counterfactual"], \
        "the request's seed draws the counterfactual"


def test_records_are_cached_and_repeatable(ma, tmp_path, monkeypatch):
    w = ma
    req = request(form="fourier", cites=w.names[:2], w=w)
    first, second = cli.build(req, context(w)), cli.build(req, context(w))
    a, b = (records.record_json(r) for r in (first, second))
    assert canonical_json(a) == canonical_json(b), "built twice from one request, byte for byte"
    written = cli.record(tmp_path, req, context(w))
    path = tmp_path / f"gesture-{written.key}.json"
    assert written.key == first.key and path.read_text() == canonical_json(a) + "\n"
    before = path.read_bytes()
    calls = []
    monkeypatch.setattr(crystallize, "crystallize", lambda *args, **kwargs: calls.append(1) or {})
    again = cli.record(tmp_path, req, context(w))
    assert calls == [] and again == first and path.read_bytes() == before, "the cache answers; nothing recomputes"
    other = cli.build(request(form="fourier", cites=w.names[:2], seed=6, w=w), context(w))
    assert calls == [1] and other.key != first.key, "another seed is another record"


def test_crystallize_refusals(ma, monkeypatch):
    w = ma
    calls = []
    real = cli.results
    monkeypatch.setattr(crystallize, "crystallize", lambda *args, **kwargs: calls.append(1) or {})
    good = request(cites=w.names, w=w)
    stranger = name_record(OTHER, 99, 0)
    lacking = [context(w, **{field: None})
               for field in ("fit_windows", "functions", "twin", "twin_state", "cited")]
    narrow = context(w, fit_windows=w.fit[:, :3])  # fitting windows of another width
    cases = [(req, context(w)) for req in (
        request(function="product", w=w),  # a function the run doesn't declare
        request(instances=len(w.held) - 1, w=w),  # not every held-out window
        request(cites=[*w.names, name_record(RUN, 98, 1)], w=w),  # a key the context doesn't hold
        request(step=2, w=w),  # not a checkpoint
        request(w=w, **{"class": "nowhere"}),  # a class the run doesn't declare
        records.GestureRequest("crystallize", OTHER, 3, dict(good.args), len(w.held), 5))]  # another run
    cases += [(good, ctx) for ctx in (*lacking, narrow)]
    cases.append((request(cites=[*w.names, stranger], w=w),  # a cited record of another run
                  context(w, cited=holding(*w.names, stranger))))
    for req, ctx in cases:
        for call in (cli.inputs, real, cli.build):
            with pytest.raises(ValueError):
                call(req, ctx)
    assert calls == [], "refused before anything runs"
    cli.build(good, context(w))
    assert calls == [1], "and the good request runs"


def _modadd_world(tmp_path):
    """The command line's run: ModularAdditionTask() at its defaults, a small model, and a few of its pairs."""
    task = ModularAdditionTask()
    config = GPT2Config(vocab=len(task.vocab), d_model=16, n_layers=1, n_heads=2, d_mlp=32, seq=task.seq)
    at = {0: init_params(config, seed=0), 3: F.loud(init_params(config, seed=0), 3)}
    fit, held = task.training()[::12], task.held_out()[::40]
    sums = sorted(set(((held[:, 0] + held[:, 1]) % task.p).tolist()))
    names = [name_record(RUN, i, v) for i, v in enumerate(sums[:4])]
    folders = {"out": tmp_path / "out", "first": tmp_path / "records" / "first", "second": tmp_path / "records" / "second"}
    for record, folder in zip(names, ("out", "first", "first", "second")):
        records.write_record(folders[folder], record)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config)))
    for step, params in at.items():
        save_params(params, tmp_path / "checkpoints", step=step, run_id=RUN)
    np.save(tmp_path / "held.npy", held)
    np.save(tmp_path / "fit.npy", fit)
    twin = init_params(config, seed=1)
    context = cli.Context(
        run=RUN, declaration=cli.declaration_for("modular_addition", config), model=GPT2Model(config),
        n_layers=config.n_layers, steps=STEPS, params_at=at.__getitem__, state_at=lambda s: state_hash(at[s]),
        windows=held, dictionary=no_dictionary, seeds={}, fit_windows=fit, functions=cli.functions_for("modular_addition"),
        twin=twin, twin_state=state_hash(twin), cited=holding(*names))
    return names, folders, context


def _args(tmp_path, folders, req, *, run="modular_addition", fit=True):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(records.request_json(req)))
    return (["--run", run, "--config", str(tmp_path / "config.json"), "--checkpoints", str(tmp_path / "checkpoints"),
             "--windows", str(tmp_path / "held.npy"), "--request", str(path), "--out", str(folders["out"]),
             "--records", str(folders["first"]), "--records", str(folders["second"])]
            + (["--fit-windows", str(tmp_path / "fit.npy")] if fit else []))


def test_the_command_line(tmp_path, capsys):
    names, folders, context = _modadd_world(tmp_path)
    for form, cites in (("fourier", names), ("table", names[1:3])):
        req = records.GestureRequest("crystallize", RUN, 3, {"block": 0, "part": "write", "class": "answer",
                                                             "function": "sum", "form": form,
                                                             "cites": [r.key for r in cites]}, len(context.windows), 50)
        args = _args(tmp_path, folders, req)
        assert cli.main(args) == 0, capsys.readouterr().err
        want = cli.build(req, context)
        assert capsys.readouterr().out.strip() == want.key, "it prints the record's key"
        path = folders["out"] / f"gesture-{want.key}.json"
        written = path.read_bytes()
        assert written == (canonical_json(records.record_json(want)) + "\n").encode("ascii"), \
            "the command line writes the record build gives, byte for byte: the run's functions, its twin, " \
            "every checkpoint and the records it cites, from the out folder and each --records folder"
        assert want.key in [e["key"] for e in json.loads((folders["out"] / "index.json").read_text())["records"]]
        assert cli.main(args) == 0 and path.read_bytes() == written, "a rerun is byte-identical"
        assert capsys.readouterr().out.strip() == want.key


def test_the_command_line_refuses(tmp_path, capsys):
    names, folders, context = _modadd_world(tmp_path)
    good = {"block": 0, "part": "write", "class": "answer", "function": "sum", "form": "table",
            "cites": [r.key for r in names]}
    before = {p.name: p.read_bytes() for p in folders["out"].iterdir()}
    cases = {
        "a key no folder holds": (dict(good, cites=[*good["cites"], "0" * 64]), {}),
        "no fitting windows": (good, {"fit": False}),
        "the reference run declares no functions": (dict(good, **{"class": "every"}), {"run": "reference"}),
        "another run's function": (dict(good, function="copied"), {}),
    }
    for label, (args, options) in cases.items():
        req = records.GestureRequest("crystallize", RUN, 3, args, len(context.windows), 50)
        assert cli.main(_args(tmp_path, folders, req, **options)) == 1, label
        assert capsys.readouterr().err.startswith("error:"), label
        assert {p.name: p.read_bytes() for p in folders["out"].iterdir()} == before, f"{label}: nothing written"
