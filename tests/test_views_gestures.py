"""Phase 6 delta 5: the gestures panel, each record with its meter first, and the features view's links (D47).

Contract: src/nablaframe/views/gestures.py. The records are real ones, built by ``cli.build`` on test_gestures_cli's
toy run, so each meter is read from the results its gesture actually returns; a few are then edited, to reach the
steps a toy can't. Needs the jax lane, to build them.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

jax = pytest.importorskip("jax")

import phase6_fixtures as F  # noqa: E402
from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.gestures import cli, records  # noqa: E402
from nablaframe.trace.schema import canonical_json, sha256_hex  # noqa: E402
from nablaframe.train.checkpoint import state_hash  # noqa: E402
from nablaframe.views import gestures as G  # looked up per test, so a missing name fails its test alone  # noqa: E402
from test_gestures_cli import RUN, STEPS, held, renumbered, scrambled  # noqa: E402

OTHER_RUN = "f" * 64


@pytest.fixture(scope="module")
def toy():
    """test_gestures_cli's toy run, and a record of each kind and form built from it."""
    w = F.world("identity")
    w.at = {0: scrambled(w.params), 3: w.params}
    w.d = [w.dictionaries[0], w.dictionaries[1]]
    w.h = [core.content_hash(d) for d in w.d]
    seed1 = [renumbered(F.dictionary(block, decoder="negative"), 7) for block in (0, 1)]
    w.context = cli.Context(run=RUN, declaration=w.declaration, model=w.model, n_layers=w.n_layers, steps=STEPS,
                            params_at=w.at.__getitem__, state_at=lambda s: state_hash(w.at[s]), windows=w.windows,
                            dictionary=held(*w.d, *seed1),
                            seeds={1: dict(zip(w.h, [core.content_hash(d) for d in seed1]))})
    h0, h1 = w.h
    edge_args = {"up": h0, "up_site": 0, "down": h1, "down_site": 0}
    asks = {
        "name": ("name", {"dictionary": h0, "site": 0, "class": "every", "latent": F.latent(3)}),
        "name late": ("name", {"dictionary": h0, "site": 0, "class": "late", "latent": F.latent(3)}),
        "name other": ("name", {"dictionary": h0, "site": 0, "class": "every", "latent": F.latent(5, 1)}),
        "name down": ("name", {"dictionary": h1, "site": 0, "class": "every", "latent": F.latent(3)}),
        "name dead": ("name", {"dictionary": h0, "site": 0, "class": "every", "latent": F.DEAD_LATENT}),
        "edge carry": ("edge", {"relation": "carry", **edge_args}),
        "edge next": ("edge", {"relation": "next", **edge_args}),
        "open latent": ("open", {"dictionary": h1, "site": 0, "class": "late", "upstream": h0, "upstream_site": 0,
                                 "latent": F.latent(3)}),
        "open name": ("open", {"dictionary": h1, "site": 0, "class": "late", "upstream": h0, "upstream_site": 0,
                               "description": "token(p)", "value": 5}),
        "generalize name": ("generalize", {"dictionary": h0, "site": 0, "class": "every", "description": "token(p)"}),
        "generalize edge": ("generalize", {"relation": "carry", **edge_args}),
    }
    w.records = {label: cli.build(records.GestureRequest(kind, RUN, 3, args, 8, 5), w.context)
                 for label, (kind, args) in asks.items()}
    quiet = dataclasses.replace(w.context, declaration=F.declaration("silent"))
    kind, args = asks["name"]
    w.records["name silent"] = cli.build(records.GestureRequest(kind, RUN, 3, args, 8, 5), quiet)
    return w


def results_of(record) -> dict:
    return json.loads(json.dumps(records.record_json(record)["results"]))


def edited(record, edit, *, run=None):
    """The record with its results edited (and its run changed), as a new record."""
    results = results_of(record)
    edit(results)
    request = record.request if run is None else dataclasses.replace(record.request, run=run)
    return records.GestureRecord(request, records.record_json(record)["inputs"], dict(record.code), results)


def distinct_name(r: dict) -> None:
    d = r["name"]["description"]
    r["set"]["name_f1"] = 0.5625
    r["coverage"]["latent"][d], r["coverage"]["set"][d] = 0.1875, 0.4375
    r["effect"]["set"].update(rank_from_top=3, rank_from_bottom=8)
    r["effect"]["latent"].update(rank_from_top=5, rank_from_bottom=6)
    r["steering"].update(follows=0.8125, null_follows=0.0625, unchanged=0.125, null_unchanged=0.875,
                         ablated_changed=0.5, instances=7)


def distinct_edge(r: dict) -> None:
    r.update(follows=0.6875, null_follows=0.0625, kept=0.3125, null_kept=0.9375, instances=9,
             residual_alone={"instances": 4, "blocks": [1], "follows": 0.25, "follows_unfrozen": 0.75,
                             "output_follows": 0.5},
             output={"instances": 4, "follows": 0.625, "null_follows": 0.125},
             single={"follows": 0.375, "kept": 0.8125})


def distinct_open(r: dict) -> None:
    curves = {"absolute": [0.9, 0.5, 0.25, 0.05, 0.04, 0.03, 0.02, 0.01],
              "signed": [0.8, 0.7, 0.6, 0.3, 0.2, 0.15, 0.12, 0.11],
              "random": [1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94]}
    r["orders"] = {"absolute": {"error_median": curves["absolute"], "readings": [8], "reading_median": 8.0,
                                "within": 0.875},
                   "signed": {"error_median": curves["signed"], "readings": [16], "reading_median": 16.0,
                              "within": 0.625},
                   "random": {"error_median": curves["random"], "readings": [None], "reading_median": None,
                              "within": 0.125}}


def distinct_generalize(r: dict) -> None:
    r["slices"].update(f1=[0.5, 0.75], follows=[0.8125, 0.9375])
    r["checkpoints"][0]["f1"], r["checkpoints"][-1]["f1"] = 0.0625, 0.9375


def is_plain(value) -> bool:
    """Plain JSON built of dict, list, str, int, float, bool and None alone: nothing read-only, nothing numpy."""
    if value is None or type(value) in (str, int, float, bool):
        return True
    if type(value) is list:
        return all(is_plain(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and is_plain(v) for k, v in value.items())
    return False


def expected_meter(record) -> dict:
    """The contract's meter, read from the results path by path."""
    r, kind, args = results_of(record), record.request.kind, dict(record.request.args)
    if kind == "name":
        out = dict.fromkeys(G.METER_KEYS["name"])
        if r["name"] is None:
            return out
        n, s = r["name"], r["steering"]
        out.update(label=n["label"], description=n["description"], value=n["value"], f1=n["f1"],
                   set_f1=r["set"]["name_f1"], coverage=r["coverage"]["set"][n["description"]],
                   rank_from_top=r["effect"]["set"]["rank_from_top"],
                   rank_from_bottom=r["effect"]["set"]["rank_from_bottom"])
        if s is not None:
            out.update(causal=s["causal"], instances=s["instances"])
            keys = (("follows", "null_follows", "unchanged", "null_unchanged", "ablated_changed") if s["causal"]
                    else ("kl", "null_kl"))
            out.update({k: s[k] for k in keys})
        return out
    if kind == "edge":
        alone, output = r["residual_alone"], r["output"]
        return {"relation": r["relation"], "judged": r["judged"], "instances": r["instances"], "follows": r["follows"],
                "null_follows": r["null_follows"], "kept": r["kept"], "null_kept": r["null_kept"],
                "label": r["label"], "residual_alone": None if alone is None else alone["follows"],
                "follows_unfrozen": None if alone is None else alone["follows_unfrozen"],
                "output_follows": None if output is None else output["follows"],
                "output_null_follows": None if output is None else output["null_follows"]}
    if kind == "open":
        a = r["orders"]["absolute"]
        return {"class": r["class"], "target": r["target"], "candidates": r["candidates"],
                "instances": r["instances"], "measured": r["measured"], "no_span": r["no_span"], "parts": r["parts"],
                "reading": a["reading_median"], "within": a["within"], "signed_within": r["orders"]["signed"]["within"],
                "random_within": r["orders"]["random"]["within"], "error_median": a["error_median"]}
    if "description" in args:
        return {"form": "name", "relation": None, "description": r["description"], "class": r["class"],
                "values": r["values"], "envelope": r["envelope"], "slices_f1": r["slices"]["f1"],
                "slices_follows": r["slices"]["follows"], "seeds": r["seeds"], "final": r["checkpoints"][-1]}
    return {"form": "edge", "relation": r["relation"], "description": None, "class": None, "values": None,
            "envelope": None, "slices_f1": None, "slices_follows": None, "seeds": r["seeds"],
            "final": r["checkpoints"][-1]}


def file_bytes(record) -> bytes:
    return (canonical_json(records.record_json(record)) + "\n").encode("ascii")


def folder_of(tmp_path, recs, name="run"):
    folder = tmp_path / name
    for record in recs:
        records.write_record(folder, record)
    return folder


def snapshot(folder) -> dict:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_gesture_payloads_are_plain_json_and_repeatable(toy, tmp_path):
    assert G.PANEL == "panel.json"
    assert G.METER_KEYS == {
        "name": ("label", "description", "value", "f1", "set_f1", "coverage", "causal", "instances", "follows",
                 "null_follows", "unchanged", "null_unchanged", "ablated_changed", "kl", "null_kl", "rank_from_top",
                 "rank_from_bottom"),
        "edge": ("relation", "judged", "instances", "follows", "null_follows", "kept", "null_kept", "label",
                 "residual_alone", "follows_unfrozen", "output_follows", "output_null_follows"),
        "open": ("class", "target", "candidates", "instances", "measured", "no_span", "parts", "reading", "within",
                 "signed_within", "random_within", "error_median"),
        "generalize": ("form", "relation", "description", "class", "values", "envelope", "slices_f1",
                       "slices_follows", "seeds", "final"),
        "crystallize": ("label", "kl", "ablation", "removed", "agreement", "loss_delta"),
    }
    recs = dict(toy.records)
    recs["name no instance"] = edited(recs["name"], lambda r: r.update(steering=None))
    recs["open none measured"] = edited(recs["open name"], lambda r: r.update(
        measured=0, no_span=r["instances"], windows=[], top_parts=[],
        orders={o: {"error_median": None, "readings": [], "reading_median": None, "within": None}
                for o in ("absolute", "signed", "random")}))
    # On the toy, entries the meters must tell apart often hold equal values: these copies make each one distinct.
    recs["name distinct"] = edited(recs["name"], distinct_name)
    recs["edge distinct"] = edited(recs["edge carry"], distinct_edge)
    recs["open distinct"] = edited(recs["open name"], distinct_open)
    recs["generalize distinct"] = edited(recs["generalize name"], distinct_generalize)
    # The toy reaches every branch the contract names: a check that the fixtures do what the labels say.
    got = {label: results_of(record) for label, record in recs.items()}
    assert got["name"]["steering"]["causal"] is True and got["name silent"]["steering"]["causal"] is False
    assert got["name dead"]["name"] is None and got["name no instance"]["steering"] is None
    assert got["edge carry"]["residual_alone"] is not None and got["edge carry"]["output"] is not None
    assert got["edge next"]["residual_alone"] is None and got["edge next"]["output"] is None

    for label, record in recs.items():
        payload = G.gesture_payload(record)
        assert set(payload) == {"file", "key", "sha256", "kind", "request", "meter"}, label
        assert payload["file"] == f"gesture-{record.key}.json" and payload["key"] == record.key, label
        assert payload["kind"] == record.request.kind and payload["request"] == records.request_json(record.request)
        assert payload["sha256"] == sha256_hex(file_bytes(record)), f"{label}: the SHA-256 of the record's file"
        meter = payload["meter"]
        assert list(meter) == list(G.METER_KEYS[record.request.kind]), f"{label}: the kind's keys, in order"
        assert meter == expected_meter(record), label
        assert G.gesture_meter(record) == meter, label
        assert is_plain(payload), f"{label}: plain JSON, nothing read-only"
        assert json.loads(canonical_json(payload)) == payload, label
        again = records.record_from_json(json.loads(json.dumps(records.record_json(record))))
        assert G.gesture_payload(again) == payload, f"{label}: equal records give equal payloads"
        kept = json.loads(json.dumps(payload))
        for value in [*payload["meter"].values(), payload["meter"], payload["request"]["args"]]:
            if isinstance(value, (dict, list)):
                value.clear()
        assert G.gesture_payload(record) == kept, f"{label}: no object shared between calls"

    written = records.write_record(tmp_path / "one", recs["edge carry"])
    assert G.gesture_payload(recs["edge carry"])["sha256"] == sha256_hex(written.read_bytes()), "the file's own bytes"

    assert G.gesture_meter(recs["name dead"]) == dict.fromkeys(G.METER_KEYS["name"]), "no name, nothing to show"
    silent = G.gesture_meter(recs["name silent"])
    assert silent["kl"] is not None and silent["follows"] is None and silent["ablated_changed"] is None
    causal = G.gesture_meter(recs["name"])
    assert causal["kl"] is None and causal["follows"] is not None and causal["label"] in ("named", "weak")
    none = G.gesture_meter(recs["name no instance"])
    assert none["causal"] is None and none["instances"] is None and none["f1"] is not None
    assert G.gesture_meter(recs["generalize name"])["form"] == "name"
    assert G.gesture_meter(recs["generalize edge"])["form"] == "edge"
    assert G.gesture_meter(recs["open none measured"])["reading"] is None

    for label, path in (("name", "coverage"), ("name", "effect"), ("name", "set"), ("edge carry", "residual_alone"),
                        ("edge carry", "kept"), ("open latent", "orders"), ("open latent", "parts"),
                        ("generalize name", "slices"), ("generalize edge", "checkpoints"),
                        ("generalize name", "envelope")):
        broken = edited(recs[label], lambda r, p=path: r.pop(p))
        with pytest.raises(ValueError):
            G.gesture_payload(broken)
    broken = edited(recs["name"], lambda r: r["coverage"]["set"].pop(r["name"]["description"]))
    with pytest.raises(ValueError):
        G.gesture_meter(broken)


def test_the_index_lists_every_record(toy, tmp_path):
    recs = toy.records
    chosen = [recs[label] for label in ("generalize name", "name", "edge carry", "open latent", "name late",
                                        "generalize edge", "open name", "edge next")]
    folder = folder_of(tmp_path, chosen)
    index = json.loads((folder / "index.json").read_text())["records"]
    panel = G.gesture_panel(folder)
    assert set(panel) == {"run", "records", "names"} and panel["run"] == RUN
    assert [p["key"] for p in panel["records"]] == [e["key"] for e in index], "the index's order"
    assert sorted(p["key"] for p in panel["records"]) == sorted(r.key for r in chosen), "every record, once"
    by_key = {r.key: r for r in chosen}
    assert panel["records"] == [G.gesture_payload(by_key[e["key"]]) for e in index]
    assert is_plain(panel)

    before = snapshot(folder)
    path = G.write_panel(folder)
    assert path == folder / G.PANEL
    assert path.read_bytes() == (canonical_json(panel) + "\n").encode("ascii")
    after = snapshot(folder)
    assert {k: v for k, v in after.items() if k != G.PANEL} == before, "the records and their index stay as they are"
    G.write_panel(folder)
    assert snapshot(folder) == after, "written again, the same bytes"
    assert G.gesture_panel(folder) == panel, "the panel file isn't a record"

    reordered = tmp_path / "reordered"
    shutil.copytree(folder, reordered)
    (reordered / "index.json").write_text(canonical_json({"records": index[::-1]}) + "\n")
    assert [p["key"] for p in G.gesture_panel(reordered)["records"]] == [e["key"] for e in index[::-1]]

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "index.json").write_text(canonical_json({"records": []}) + "\n")
    assert G.gesture_panel(empty) == {"run": None, "records": [], "names": {}}

    nowhere = tmp_path / "nowhere"
    for call in (G.gesture_panel, G.write_panel):
        with pytest.raises(ValueError):
            call(nowhere)
    assert not nowhere.exists(), "a folder that isn't there isn't made"
    for i, bad in enumerate(({"kept": []}, {"records": {}}, [], {"records": [{"file": "gesture-x.json"}]})):
        odd = tmp_path / f"odd{i}"
        odd.mkdir()
        (odd / "index.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            G.gesture_panel(odd)

    def refused(prepare, label):
        target = tmp_path / label.replace(" ", "_")
        shutil.copytree(folder, target)
        (target / G.PANEL).unlink()
        prepare(target)
        with pytest.raises(ValueError):
            G.gesture_panel(target)
        with pytest.raises(ValueError):
            G.write_panel(target)
        assert not (target / G.PANEL).exists(), f"{label}: nothing written"

    refused(lambda f: (f / "index.json").unlink(), "no index")
    refused(lambda f: (f / "index.json").write_text("{\"kept\": []}\n"), "not an index")
    refused(lambda f: (f / f"gesture-{'0' * 64}.json").write_bytes(file_bytes(recs["name"])), "an unlisted file")
    refused(lambda f: (f / f"gesture-{recs['name'].key}.json").write_bytes(file_bytes(recs["name"]).replace(
        b'"results":{', b'"results":{"x":1,')), "a changed record")
    refused(lambda f: (f / f"gesture-{recs['edge next'].key}.json").unlink(), "a listed file missing")
    refused(lambda f: records.write_record(f, edited(recs["name other"], lambda r: None, run=OTHER_RUN)),
            "two runs in one folder")


def test_features_link_to_names(toy, tmp_path):
    recs = toy.records
    labels = ("open latent", "name other", "name late", "generalize name", "name down", "edge carry", "name",
              "name dead")
    panel = G.gesture_panel(folder_of(tmp_path, [recs[label] for label in labels]))
    h0, h1 = toy.h
    order = [p["key"] for p in panel["records"]]

    def keys(*names):
        return sorted((recs[n].key for n in names), key=order.index)

    assert panel["names"] == {
        f"{h0}-0": {str(F.latent(3)): keys("name", "name late"), str(F.latent(5, 1)): keys("name other"),
                    str(F.DEAD_LATENT): keys("name dead")},
        f"{h1}-0": {str(F.latent(3)): keys("name down")},
    }, "each name record under its dictionary, site and latent; the open record pointed at a latent is no name"
    for place, latents in panel["names"].items():
        for latent, linked in latents.items():
            for key in linked:
                entry = next(p for p in panel["records"] if p["key"] == key)
                args = entry["request"]["args"]
                assert entry["kind"] == "name" and place == f"{args['dictionary']}-{args['site']}"
                assert latent == str(args["latent"])
    no_names = G.gesture_panel(folder_of(tmp_path, [recs["edge next"], recs["open name"]], name="none"))
    assert no_names["names"] == {}
