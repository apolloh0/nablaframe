"""Phase 7 delta 4: crystallize-this in the gestures panel, its meter first (D47, D56).

Contract: src/nablaframe/views/gestures.py, the crystallize bullets. The records are real ones, built by ``cli.build``
on test_crystallize_cli's toy modular addition, so each meter is read from the results crystallize-this returns; edited
copies set apart entries the toy leaves equal. A real name record of the same run shares their folder, so the panel's
links to names are checked beside them. Needs the jax lane, to build them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.dictionaries import core  # noqa: E402
from nablaframe.gestures import cli, records  # noqa: E402
from nablaframe.trace.schema import canonical_json, sha256_hex  # noqa: E402
from nablaframe.views import gestures as G  # looked up per test, so a missing name fails its test alone  # noqa: E402
from test_crystallize_cli import RUN, context, ma, request  # noqa: E402, F401  (ma is a fixture)
from test_views_gestures import edited, file_bytes, folder_of, is_plain, results_of  # noqa: E402

KEYS = ("label", "kl", "ablation", "removed", "agreement", "loss_delta")
TAP = "blocks.0.hook_resid_post"


def toy_dictionary(d_model: int, width: int = 24, k: int = 4) -> core.Dictionary:
    """A random TopK SAE at block 0's output, so the toy run gets real name records too."""
    rng = np.random.default_rng(7)
    arrays = {"W_enc": rng.standard_normal((1, d_model, width)), "W_dec": rng.standard_normal((1, width, d_model)) / width,
              "b_enc": np.zeros((1, width)), "b_pre": np.zeros((1, d_model)), "b_dec": np.zeros((1, d_model)),
              "scale_in": np.ones(1), "scale_out": np.ones(1)}
    return core.Dictionary(core.Settings("topk", width, k, steps=1, batch=1), [(TAP, TAP)],
                           {n: v.astype(np.float32) for n, v in arrays.items()})


@pytest.fixture(scope="module")
def built(ma):
    """A crystallize-this record of each form, and two name records of the same run."""
    w = ma
    d = toy_dictionary(w.config.d_model)
    h = core.content_hash(d)

    def holds(digest):
        if digest != h:
            raise ValueError(f"no dictionary {digest}")
        return d

    ctx = context(w, dictionary=holds)
    w.crystallizations = {
        "table": cli.build(request(cites=w.names, w=w), ctx),
        "fourier": cli.build(request(form="fourier", w=w), ctx),
        "additive": cli.build(request("mlp", "sum and operands", "additive", cites=w.operands, w=w), ctx),
    }
    w.name_records = [cli.build(records.GestureRequest("name", RUN, 3, {"dictionary": h, "site": 0, "class": "answer",
                                                                        "latent": latent}, 8, 5), ctx)
                      for latent in (2, 0)]
    w.dictionary_hash = h
    return w


def distinct(r: dict) -> None:
    """Every meter entry set apart from the twin's, the wrong functions', the counterfactual's and its neighbours."""
    r["meter"] = {"loss_delta": -0.0625, "agreement": 0.8125, "removed": 0.4375, "ablation": 2.5, "kl": 0.3125,
                  "label": "doesn't hold"}
    r["accuracy"] = {"model": 0.9375, "crystallized": 0.6875}
    r["loss"] = {"model": 0.125, "crystallized": 0.1875}
    r["r2"] = 0.5625
    if r["twin"] is not None:
        r["twin"]["meter"] = {"label": "inert", "kl": 0.0078125, "ablation": 0.015625, "removed": 0.5,
                              "agreement": 0.96875, "loss_delta": 0.25}


def test_crystallize_payloads(built, tmp_path):
    w = built
    assert G.METER_KEYS["crystallize"] == KEYS
    assert set(G.METER_KEYS) == {"name", "edge", "open", "generalize", "crystallize"}

    recs = dict(w.crystallizations)
    recs["distinct"] = edited(recs["table"], distinct)
    recs["no removed share"] = edited(recs["fourier"], lambda r: r["meter"].update(ablation=0.0, removed=None,
                                                                                  label="inert"))
    got = {label: results_of(record) for label, record in recs.items()}
    assert got["additive"]["counterfactual"] is not None and got["table"]["counterfactual"] is None
    assert got["fourier"]["frequencies"] is not None and got["table"]["twin"] is not None

    for label, record in recs.items():
        r = got[label]
        payload = G.gesture_payload(record)
        assert set(payload) == {"file", "key", "sha256", "kind", "request", "meter"}, label
        assert payload["file"] == f"gesture-{record.key}.json" and payload["key"] == record.key, label
        assert payload["kind"] == "crystallize" and payload["request"] == records.request_json(record.request), label
        assert payload["sha256"] == sha256_hex(file_bytes(record)), f"{label}: the SHA-256 of the record's file"
        meter = payload["meter"]
        assert list(meter) == list(KEYS), f"{label}: the meter's keys, in order"
        assert meter == {key: r["meter"][key] for key in KEYS}, f"{label}: copied from the results' meter"
        assert G.gesture_meter(record) == meter, label
        assert is_plain(payload) and json.loads(canonical_json(payload)) == payload, f"{label}: plain JSON"
        again = records.record_from_json(json.loads(json.dumps(records.record_json(record))))
        assert G.gesture_payload(again) == payload, f"{label}: equal records give equal payloads"
        kept = json.loads(json.dumps(payload))
        for value in (payload["meter"], payload["request"]["args"]):
            value.clear()
        assert G.gesture_payload(record) == kept, f"{label}: no object shared between calls"

    assert G.gesture_meter(recs["distinct"]) == {"label": "doesn't hold", "kl": 0.3125, "ablation": 2.5,
                                                 "removed": 0.4375, "agreement": 0.8125, "loss_delta": -0.0625}, \
        "the record's own meter, not its twin's, nor one worked out from its KLs or its losses"
    assert G.gesture_meter(recs["no removed share"])["removed"] is None, "a null is copied as it is"

    for missing in ("meter", "twin", "checkpoints"):
        record = edited(recs["table"], lambda r, m=missing: r.pop(m))
        if missing == "meter":
            with pytest.raises(ValueError, match=record.key):
                G.gesture_payload(record)
        else:
            assert G.gesture_payload(record)["meter"] == G.gesture_meter(recs["table"]), \
                f"the meter reads nothing but the results' meter: no {missing} needed"
    for entry in KEYS:
        record = edited(recs["fourier"], lambda r, e=entry: r["meter"].pop(e))
        with pytest.raises(ValueError, match=entry):
            G.gesture_meter(record)

    order = [w.name_records[0], recs["additive"], recs["table"], w.name_records[1], recs["fourier"]]
    folder = folder_of(tmp_path, order, name="modular_addition")
    panel = G.gesture_panel(folder)
    index = json.loads((folder / "index.json").read_bytes())["records"]
    assert [p["key"] for p in panel["records"]] == [e["key"] for e in index], "the index's order"
    assert panel["run"] == RUN and is_plain(panel)
    for payload in panel["records"]:
        record = records.read_record(folder, payload["key"])
        assert payload == G.gesture_payload(record)
    named = {}
    for record in w.name_records:
        place = f"{w.dictionary_hash}-0"
        named.setdefault(place, {}).setdefault(str(record.request.args["latent"]), []).append(record.key)
    assert panel["names"] == named, "the name records link to their latents; crystallize-this records don't"
    path = G.write_panel(folder)
    assert path.read_bytes() == (canonical_json(panel) + "\n").encode("ascii")

    alone = folder_of(tmp_path, [recs["fourier"], recs["table"]], name="zoo_run")
    assert G.gesture_panel(alone)["names"] == {}, "a folder of crystallizations alone links no latent"
