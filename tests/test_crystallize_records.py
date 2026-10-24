"""Phase 7 delta 3: crystallize-this requests and records (D47, D56).

Contract: src/nablaframe/gestures/records.py, the crystallize form, the hashes type and the twin's state. Numpy only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from nablaframe.gestures.records import (FORMS, KINDS, GestureRecord, GestureRequest, read_record, record_from_json,
                                         record_json, record_key, request_from_json, request_hash, request_json,
                                         write_record)
from nablaframe.trace.schema import canonical_json

H = ["%064x" % i for i in range(1, 12)]
RUN = "a" * 64
ARGS = {"block": 0, "part": "write", "class": "answer", "function": "sum", "form": "fourier", "cites": [H[2], H[0]]}
INPUTS = {"states": {"0": H[3], "10000": H[4], "twin": H[5]}, "dictionaries": [], "windows": [H[6], H[7]]}
CODE = {"nablaframe.gestures.crystallize": "1" * 40, "nablaframe.zoo.functions": "2" * 40}
RESULTS = {"meter": {"label": "holds", "kl": 0.0187, "ablation": 4.143}, "twin": None, "cites": None}


def request(**changes):
    fields = dict(kind="crystallize", run=RUN, step=10000, args=dict(ARGS), instances=8938, seed=50)
    fields.update(changes)
    return GestureRequest(**fields)


def test_crystallize_requests_round_trip_and_hash():
    assert KINDS == ("name", "edge", "open", "generalize", "crystallize")
    assert FORMS["crystallize"] == ({"block": "int", "part": "name", "class": "name", "function": "name",
                                     "form": "name", "cites": "hashes"},)
    r = request()
    data = request_json(r)
    assert data["args"]["cites"] == [H[0], H[2]], "cites are stored sorted"
    assert json.loads(canonical_json(data)) == data and type(data["args"]["cites"]) is list, "plain JSON"
    assert request_from_json(json.loads(json.dumps(data))) == r
    assert request_hash(r) == hashlib.sha256(canonical_json(data).encode("ascii")).hexdigest()
    for same in (request(args=ARGS | {"cites": [H[0], H[2]]}), request(args=ARGS | {"cites": (H[2], H[0])}),
                 request(args=dict(reversed(list(ARGS.items())))), request(args=ARGS | {"block": np.int64(0)})):
        assert same == r and request_hash(same) == request_hash(r), "the same request, however it was given"
    assert type(request(args=ARGS | {"block": np.int64(0)}).args["block"]) is int
    with pytest.raises(TypeError):
        r.args["cites"].append(H[1])  # read-only
    with pytest.raises(TypeError):
        r.args["cites"][0] = H[1]
    with pytest.raises(TypeError):
        r.args["form"] = "table"
    assert request(args=ARGS | {"cites": []}).args["cites"] == [], "citing nothing is a request too"

    hashes = {request_hash(r)}
    changes = [dict(args=ARGS | {"block": 1}), dict(args=ARGS | {"part": "mlp"}), dict(args=ARGS | {"class": "every"}),
               dict(args=ARGS | {"function": "sum and operands"}), dict(args=ARGS | {"form": "table"}),
               dict(args=ARGS | {"cites": []}), dict(args=ARGS | {"cites": [H[0]]}),
               dict(args=ARGS | {"cites": [H[0], H[2], H[1]]}), dict(step=0), dict(seed=51), dict(instances=8937)]
    for change in changes:
        hashes.add(request_hash(request(**change)))
    assert len(hashes) == len(changes) + 1, "every argument changes the hash"

    for bad in (ARGS | {"cites": [H[0], H[0]]}, ARGS | {"cites": [H[0], "x"]}, ARGS | {"cites": H[0]},
                ARGS | {"cites": [H[10].upper()]}, ARGS | {"cites": {H[0]: 1}}, ARGS | {"cites": None},
                ARGS | {"block": True}, ARGS | {"block": -1}, ARGS | {"part": ""}, ARGS | {"form": 3},
                ARGS | {"function": "sum⇒"}, {k: v for k, v in ARGS.items() if k != "cites"}, ARGS | {"latent": 0}):
        with pytest.raises(ValueError):
            request(args=bad)
    with pytest.raises(ValueError):
        GestureRequest("name", RUN, 0, {"dictionary": H[0], "site": 0, "class": "answer", "latent": 0,
                                        "cites": []}, 8, 0)  # no other kind takes cites

    # The twin's state hash is an input of a crystallize-this record, and of no other kind's.
    rec = GestureRecord(r, INPUTS, CODE, RESULTS)
    assert rec.key == record_key(r, INPUTS, CODE) == hashlib.sha256(canonical_json(
        {"request": request_hash(r), "inputs": INPUTS, "code": CODE}).encode("ascii")).hexdigest()
    assert record_from_json(json.loads(json.dumps(record_json(rec)))) == rec
    twins = {record_key(r, INPUTS | {"states": dict(INPUTS["states"], twin=h)}, CODE) for h in (H[5], H[8], H[9])}
    assert len(twins) == 3, "the twin's state is in the key"
    for bad in (INPUTS | {"states": {"0": H[3], "10000": H[4]}},  # no twin
                INPUTS | {"states": {"twin": H[5]}},  # no checkpoint
                INPUTS | {"states": dict(INPUTS["states"], twin="x")}, INPUTS | {"states": dict(INPUTS["states"], Twin=H[5])}):
        with pytest.raises(ValueError):
            GestureRecord(r, bad, CODE, RESULTS)
        with pytest.raises(ValueError):
            record_key(r, bad, CODE)
    named = GestureRequest("name", RUN, 0, {"dictionary": H[0], "site": 0, "class": "answer", "latent": 0}, 8, 0)
    with pytest.raises(ValueError):
        GestureRecord(named, {"states": {"0": H[3], "twin": H[5]}, "dictionaries": [H[0]], "windows": [H[6]]}, CODE,
                      {"name": None})
    assert GestureRecord(named, {"states": {"0": H[3]}, "dictionaries": [H[0]], "windows": [H[6]]}, CODE,
                         {"name": None}).request == named, "other kinds are as they were"


def test_crystallize_records_are_written_and_read(tmp_path):
    rec = GestureRecord(request(), INPUTS, CODE, RESULTS)
    path = write_record(tmp_path, rec)
    assert path.read_bytes() == (canonical_json(record_json(rec)) + "\n").encode("ascii")
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["records"] == [{"file": path.name, "key": rec.key, "kind": "crystallize",
                                 "request": request_hash(rec.request),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    back = read_record(tmp_path, rec.key)
    assert back == rec and back.request.args["cites"] == [H[0], H[2]] and back.inputs["states"]["twin"] == H[5]
