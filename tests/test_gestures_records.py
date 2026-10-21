"""Phase 6 delta 1: gesture requests and records (D47).

Contract: src/nablaframe/gestures/records.py. Numpy only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from nablaframe.gestures import records
from nablaframe.gestures.records import (FORMS, KINDS, GestureRecord, GestureRequest, cached_record, code_identity,
                                         read_record, record_from_json, record_json, record_key, request_from_json,
                                         request_hash, request_json, write_record)
from nablaframe.trace.schema import canonical_json

H = ["%064x" % i for i in range(1, 9)]
RUN = "a" * 64


def args_for(kind, form=0):
    values = {"dictionary": H[0], "site": 2, "class": "query", "latent": 867, "relation": "hop", "up": H[1],
              "up_site": 0, "down": H[2], "down_site": 0, "upstream": H[3], "upstream_site": 0, "description": "b",
              "value": 12}
    return {k: values[k] for k in FORMS[kind][form]}


def request(kind="name", form=0, **changes):
    fields = dict(kind=kind, run=RUN, step=20000, args=args_for(kind if kind in FORMS else "name", form),
                  instances=512, seed=31)
    fields.update(changes)
    return GestureRequest(**fields)


INPUTS = {"states": {"20000": H[4]}, "dictionaries": [H[0], H[5]], "windows": [H[6]]}
CODE = {"nablaframe.gestures.name": "1" * 40, "nablaframe.interventions.apply": "2" * 40}
RESULTS = {"follows": 0.939, "null": {"follows": 0.008}, "reading": None, "top": [[3, 0.5], [7, 0.25]], "label": "computed"}


def test_requests_round_trip_and_hash():
    assert KINDS == ("name", "edge", "open", "generalize")
    hashes = set()
    for kind in KINDS:
        for form in range(len(FORMS[kind])):
            r = request(kind, form)
            data = request_json(r)
            assert set(data) == {"kind", "run", "step", "args", "instances", "seed"}
            assert json.loads(canonical_json(data)) == data, "plain JSON"
            assert request_from_json(json.loads(json.dumps(data))) == r
            assert request_hash(r) == hashlib.sha256(canonical_json(data).encode("ascii")).hexdigest()
            hashes.add(request_hash(r))
    assert len(hashes) == sum(len(f) for f in FORMS.values()), "every form hashes differently"

    r = request()
    reordered = GestureRequest("name", RUN, 20000, dict(reversed(list(args_for("name").items()))), 512, 31)
    assert reordered == r and request_hash(reordered) == request_hash(r), "argument order doesn't matter"
    numpy_ints = request(step=np.int64(20000), instances=np.int32(512), seed=np.uint8(31),
                         args=args_for("name") | {"site": np.int64(2), "latent": np.int16(867)})
    assert numpy_ints == r and request_hash(numpy_ints) == request_hash(r), "numpy ints are ints"
    assert all(type(v) is int for v in (numpy_ints.step, numpy_ints.instances, numpy_ints.seed,
                                        numpy_ints.args["site"], numpy_ints.args["latent"]))
    with pytest.raises(TypeError):
        r.args["latent"] = 1  # read-only
    for change in (dict(step=1), dict(seed=32), dict(instances=128), dict(run="b" * 64),
                   dict(args=args_for("name") | {"latent": 879}), dict(args=args_for("name") | {"class": "every"})):
        assert request_hash(request(**change)) != request_hash(r), f"{change} changes the hash"


def test_request_refusals():
    good = args_for("name")
    bad = [
        dict(kind="crystallize"), dict(kind=None), dict(run="A" * 64), dict(run="a" * 63), dict(run=7),
        dict(step=-1), dict(step=True), dict(step=1.0), dict(instances=0), dict(instances=False), dict(seed=-1),
        dict(args={k: v for k, v in good.items() if k != "latent"}), dict(args=good | {"value": 3}),
        dict(args=good | {"site": True}), dict(args=good | {"site": 2.0}), dict(args=good | {"latent": -1}),
        dict(args=good | {"dictionary": "x"}), dict(args=good | {"class": ""}), dict(args=good | {"class": "q⇒"}),
        dict(args=good | {"class": 5}), dict(args=[("latent", 1)]),
        dict(kind="open", args=args_for("open", 0) | {"description": "b"}),  # a latent and a description at once
        dict(kind="open", args={k: v for k, v in args_for("open", 1).items() if k != "value"}),
        dict(kind="generalize", args=args_for("generalize", 0) | {"relation": "hop"}),  # two forms mixed
        dict(kind="edge", args=args_for("name")),
    ]
    for change in bad:
        with pytest.raises(ValueError):
            request(**change)
    data = request_json(request())
    for broken in (data | {"extra": 1}, {k: v for k, v in data.items() if k != "seed"}, [data]):
        with pytest.raises(ValueError):
            request_from_json(broken)


def test_the_record_key_covers_its_inputs():
    r = request()
    base = record_key(r, INPUTS, CODE)
    assert base == hashlib.sha256(canonical_json({"request": request_hash(r), "inputs": INPUTS, "code": CODE})
                                  .encode("ascii")).hexdigest()
    rec = GestureRecord(r, INPUTS, CODE, RESULTS)
    assert rec.key == base
    assert GestureRecord(r, INPUTS, CODE, {"follows": 0.1}).key == base, "results aren't in the key"
    changes = [
        dict(request=request(seed=1)),
        dict(inputs=INPUTS | {"states": {"20000": H[7]}}),
        dict(inputs=INPUTS | {"states": {"19000": H[4]}}),
        dict(inputs=INPUTS | {"states": {"20000": H[4], "0": H[5]}}),
        dict(inputs=INPUTS | {"dictionaries": [H[0]]}),
        dict(inputs=INPUTS | {"windows": [H[6], H[7]]}),
        dict(code=CODE | {"nablaframe.gestures.name": "3" * 40}),
        dict(code=CODE | {"nablaframe.gestures.edge": "4" * 40}),
    ]
    keys = {base}
    for change in changes:
        fields = dict(request=r, inputs=INPUTS, code=CODE) | change
        keys.add(record_key(**fields))
    assert len(keys) == len(changes) + 1, "every input changes the key"

    for bad in (INPUTS | {"states": {}}, INPUTS | {"states": {"020": H[4]}}, INPUTS | {"states": {20000: H[4]}},
                INPUTS | {"dictionaries": [H[5], H[0]]}, INPUTS | {"windows": [H[6], H[6]]},
                INPUTS | {"windows": ["x"]}, {k: v for k, v in INPUTS.items() if k != "windows"}, INPUTS | {"more": []}):
        with pytest.raises(ValueError):
            GestureRecord(r, bad, CODE, RESULTS)
    for bad in ({}, {"m": "1" * 39}, {"m": "G" * 40}, {"": "1" * 40}):
        with pytest.raises(ValueError):
            GestureRecord(r, INPUTS, bad, RESULTS)
    for bad in ({"x": float("nan")}, {"x": [float("inf")]}, {"x": np.float32(1.0)}, {"x": {1, 2}}, [1]):
        with pytest.raises(ValueError):
            GestureRecord(r, INPUTS, CODE, bad)

    # The code identity is each module's git blob hash (D18).
    identity = code_identity(records, json)
    assert list(identity) == sorted(identity) == ["json", "nablaframe.gestures.records"]
    data = open(records.__file__, "rb").read()
    assert identity["nablaframe.gestures.records"] == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if shutil.which("git"):
        printed = subprocess.run(["git", "hash-object", records.__file__], capture_output=True, text=True).stdout.strip()
        assert identity["nablaframe.gestures.records"] == printed
    with pytest.raises(ValueError):
        code_identity()


def test_records_round_trip_and_refuse_tampering(tmp_path):
    inputs, code, results = json.loads(json.dumps(INPUTS)), dict(CODE), json.loads(json.dumps(RESULTS))
    rec = GestureRecord(request(), inputs, code, results)
    inputs["windows"].append(H[7])
    results["follows"] = 0.0
    code["x"] = "5" * 40
    assert rec.key == record_key(request(), INPUTS, CODE) and rec.results["follows"] == 0.939, \
        "changing the caller's objects never changes a record"
    with pytest.raises(TypeError):
        rec.results["follows"] = 1.0
    data = record_json(rec)
    assert data["key"] == rec.key and record_from_json(json.loads(json.dumps(data))) == rec
    with pytest.raises(ValueError):
        record_from_json(data | {"key": "0" * 64})
    with pytest.raises(ValueError):
        record_from_json(data | {"extra": 1})

    folder = tmp_path / "gestures" / "two_hop"
    path = write_record(folder, rec)
    assert path == folder / f"gesture-{rec.key}.json"
    assert path.read_bytes() == (canonical_json(data) + "\n").encode("ascii")
    edge = GestureRecord(request("edge"), INPUTS, CODE, {"follows": 0.939, "label": "computed"})
    write_record(folder, edge)
    index = json.loads((folder / "index.json").read_text())
    assert (folder / "index.json").read_bytes() == (canonical_json(index) + "\n").encode("ascii")
    assert [e["kind"] for e in index["records"]] == ["edge", "name"], "sorted by kind, then key"
    for e, r in zip(index["records"], (edge, rec)):
        assert e == {"file": f"gesture-{r.key}.json", "key": r.key, "kind": r.request.kind,
                     "request": request_hash(r.request),
                     "sha256": hashlib.sha256((folder / e["file"]).read_bytes()).hexdigest()}
    assert read_record(folder, rec.key) == rec and read_record(folder, edge.key) == edge

    snapshot = {p.name: p.read_bytes() for p in folder.iterdir()}
    write_record(folder, rec)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == snapshot, "writing it again changes nothing"
    with pytest.raises(ValueError):
        write_record(folder, GestureRecord(request(), INPUTS, CODE, {"follows": 0.5}))  # a key never changes meaning
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == snapshot, "and the refusal wrote nothing"

    with pytest.raises(ValueError):
        read_record(folder, "0" * 64)  # not in the index
    with pytest.raises(ValueError):
        read_record(tmp_path / "nowhere", rec.key)
    tampered = path.read_text().replace("0.939", "0.999")
    path.write_text(tampered)
    with pytest.raises(ValueError):
        read_record(folder, rec.key)  # the bytes don't match the index
    path.write_bytes(snapshot[path.name])
    other = folder / f"gesture-{edge.key}.json"
    other.write_bytes(snapshot[path.name])
    index["records"][0]["sha256"] = hashlib.sha256(snapshot[path.name]).hexdigest()
    (folder / "index.json").write_text(canonical_json(index) + "\n")
    with pytest.raises(ValueError):
        read_record(folder, edge.key)  # the file under edge's name holds another record

    calls = []

    def make():
        calls.append(1)
        return {"follows": 0.9}

    fresh = tmp_path / "fresh"
    made = cached_record(fresh, request("open"), INPUTS, CODE, make)
    again = cached_record(fresh, request("open"), INPUTS, CODE, make)
    assert made == again and made.results["follows"] == 0.9 and calls == [1], "make runs once"
    assert read_record(fresh, made.key) == made
