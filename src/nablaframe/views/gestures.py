"""The gestures panel: a run's gesture records as the viewer lists them, each with its meter first (D47).

Spec: docs/specs/phase-6.md, D47 and "Records, the viewer and badges"; contract:
docs/deltas/phase-6-delta-5.md. The viewer stays a pure decoder (D13, D27): every value here is
copied from a record's results, never computed, and the viewer reads each record's own file,
checking it against the SHA-256 its entry holds, as it checks the trace's tensors. Nothing here
runs a model or imports JAX.

* ``PANEL = "panel.json"``: the file ``write_panel`` writes into a run's records folder, beside the
  ``index.json`` that ``records.write_record`` keeps there.
* ``METER_KEYS``: each kind's meter keys, in order:

  - ``"name"``: ``("label", "description", "value", "f1", "set_f1", "coverage", "causal",
    "instances", "follows", "null_follows", "unchanged", "null_unchanged", "ablated_changed",
    "kl", "null_kl", "rank_from_top", "rank_from_bottom")``
  - ``"edge"``: ``("relation", "judged", "instances", "follows", "null_follows", "kept",
    "null_kept", "label", "residual_alone", "follows_unfrozen", "output_follows",
    "output_null_follows")``
  - ``"open"``: ``("class", "target", "candidates", "instances", "measured", "no_span", "parts",
    "reading", "within", "signed_within", "random_within", "error_median")``
  - ``"generalize"``: ``("form", "relation", "description", "class", "values", "envelope",
    "slices_f1", "slices_follows", "seeds", "final")``

* ``gesture_meter(record)``: a ``records.GestureRecord``'s meter, ``{key: value}`` for its kind's
  ``METER_KEYS`` in order, each value a copy of what its results hold, as plain JSON. A value the
  results don't hold because the step behind it didn't happen is None. With ``r`` the results:

  - **name:** where ``r["name"]`` is None (the latent fires on no row of the class), every value
    is None. Otherwise, with ``d`` the name's description: ``label``, ``description``, ``value``
    and ``f1`` are ``r["name"]``'s; ``set_f1`` is ``r["set"]["name_f1"]``; ``coverage`` is
    ``r["coverage"]["set"][d]``, the share of the set's mass where ``d`` is the name's value;
    ``rank_from_top`` and ``rank_from_bottom`` are ``r["effect"]["set"]``'s. With ``s =
    r["steering"]``: where ``s`` is None (no instance), ``causal``, ``instances`` and the seven
    shares and KLs are None. Otherwise ``causal`` and ``instances`` are ``s``'s; where ``s`` is
    causal, ``follows``, ``null_follows``, ``unchanged``, ``null_unchanged`` and
    ``ablated_changed`` are ``s``'s and ``kl`` and ``null_kl`` None; where it isn't, ``kl`` and
    ``null_kl`` are ``s``'s and the five shares None.
  - **edge:** ``relation``, ``judged``, ``instances``, ``follows``, ``null_follows``, ``kept``,
    ``null_kept`` and ``label`` are ``r``'s; ``residual_alone`` and ``follows_unfrozen`` are
    ``r["residual_alone"]``'s ``follows`` and ``follows_unfrozen``, and ``output_follows`` and
    ``output_null_follows`` are ``r["output"]``'s ``follows`` and ``null_follows``, each None where
    that entry is None.
  - **open:** ``class``, ``target``, ``candidates``, ``instances``, ``measured``, ``no_span`` and
    ``parts`` are ``r``'s; ``reading``, ``within`` and ``error_median`` are
    ``r["orders"]["absolute"]``'s ``reading_median``, ``within`` and ``error_median``;
    ``signed_within`` and ``random_within`` are ``r["orders"]["signed"]["within"]`` and
    ``r["orders"]["random"]["within"]``.
  - **generalize:** ``form`` is ``"name"`` where the request's arguments hold ``description``,
    and ``"edge"`` otherwise. ``seeds`` is ``r["seeds"]`` and ``final`` the last of
    ``r["checkpoints"]``, the final checkpoint's row, both as they are. A name adds
    ``description``, ``class``, ``values`` and ``envelope``, all ``r``'s, and ``slices_f1`` and
    ``slices_follows``, ``r["slices"]``'s ``f1`` and ``follows``, with ``relation`` None. An edge
    adds ``relation``, ``r``'s, with the other five None.

  ValueError, naming the record's key and the missing entry, where the results don't hold an entry
  this reads.
* ``gesture_payload(record)``: the record as the panel lists it: ``{"file", "key", "sha256",
  "kind", "request", "meter"}``: ``gesture-<key>.json``; ``record.key``; the SHA-256 of the bytes
  ``records.write_record`` writes for the record, its canonical JSON and a newline; its kind;
  ``records.request_json`` of its request; and ``gesture_meter``. Plain JSON, sharing no object
  with the record or with another call's result; records equal as values give equal payloads.
* ``gesture_panel(folder)``: the panel of a run's records folder, as ``records.write_record``
  leaves it: ``{"run", "records", "names"}``.

  - ``"records"``: ``gesture_payload`` of each record the folder's ``index.json`` lists, in the
    index's order, each read with ``records.read_record``, which checks the file's SHA-256 and key.
  - ``"run"``: the run id every record's request names; None where the index lists no record.
  - ``"names"``: the name records by place, so the features view can link a latent to them:
    ``{"<dictionary>-<site>": {"<latent>": [key, ...]}}`` from each name record's request
    arguments, each list in ``"records"`` order. Every other kind is left out.

  ValueError, naming the problem, for a folder without an ``index.json`` in ``records.py``'s form,
  a ``gesture-*.json`` file the index doesn't list, an entry ``read_record`` refuses, or two
  records whose requests name different runs.
* ``write_panel(folder)``: writes ``<folder>/PANEL``, the canonical JSON of
  ``gesture_panel(folder)`` and a newline, ASCII, through a temporary file and a rename, and
  returns its path. It writes nothing else: the records and their index stay as they are. With the
  records unchanged, writing it again gives the same bytes. Raises as ``gesture_panel`` does,
  before anything is written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nablaframe.gestures import records
from nablaframe.trace.schema import canonical_json, sha256_hex

PANEL = "panel.json"
METER_KEYS = {
    "name": ("label", "description", "value", "f1", "set_f1", "coverage", "causal", "instances", "follows",
             "null_follows", "unchanged", "null_unchanged", "ablated_changed", "kl", "null_kl", "rank_from_top",
             "rank_from_bottom"),
    "edge": ("relation", "judged", "instances", "follows", "null_follows", "kept", "null_kept", "label",
             "residual_alone", "follows_unfrozen", "output_follows", "output_null_follows"),
    "open": ("class", "target", "candidates", "instances", "measured", "no_span", "parts", "reading", "within",
             "signed_within", "random_within", "error_median"),
    "generalize": ("form", "relation", "description", "class", "values", "envelope", "slices_f1", "slices_follows",
                   "seeds", "final"),
}
_INDEX = "index.json"


def _get(key: str, node: Any, *path: str) -> Any:
    """The entry of a record's results at ``path``, as it stands; ValueError naming the record and the entry if missing."""
    for i, step in enumerate(path):
        if not isinstance(node, dict) or step not in node:
            raise ValueError(f"record {key}: results hold no {'.'.join(path[:i + 1])!r}")
        node = node[step]
    return node


def _name_meter(key: str, r: dict) -> dict[str, Any]:
    out = dict.fromkeys(METER_KEYS["name"])
    name = _get(key, r, "name")
    if name is None:
        return out
    description = _get(key, name, "description")
    out["label"] = _get(key, name, "label")
    out["description"] = description
    out["value"] = _get(key, name, "value")
    out["f1"] = _get(key, name, "f1")
    out["set_f1"] = _get(key, r, "set", "name_f1")
    out["coverage"] = _get(key, r, "coverage", "set", description)
    out["rank_from_top"] = _get(key, r, "effect", "set", "rank_from_top")
    out["rank_from_bottom"] = _get(key, r, "effect", "set", "rank_from_bottom")
    steering = _get(key, r, "steering")
    if steering is not None:
        causal = _get(key, steering, "causal")
        out["causal"] = causal
        out["instances"] = _get(key, steering, "instances")
        shares = (("follows", "null_follows", "unchanged", "null_unchanged", "ablated_changed") if causal
                  else ("kl", "null_kl"))
        for share in shares:
            out[share] = _get(key, steering, share)
    return out


def _edge_meter(key: str, r: dict) -> dict[str, Any]:
    out = {k: _get(key, r, k) for k in ("relation", "judged", "instances", "follows", "null_follows", "kept",
                                         "null_kept", "label")}
    alone, output = _get(key, r, "residual_alone"), _get(key, r, "output")
    out["residual_alone"] = None if alone is None else _get(key, alone, "follows")
    out["follows_unfrozen"] = None if alone is None else _get(key, alone, "follows_unfrozen")
    out["output_follows"] = None if output is None else _get(key, output, "follows")
    out["output_null_follows"] = None if output is None else _get(key, output, "null_follows")
    return out


def _open_meter(key: str, r: dict) -> dict[str, Any]:
    out = {k: _get(key, r, k) for k in ("class", "target", "candidates", "instances", "measured", "no_span", "parts")}
    out["reading"] = _get(key, r, "orders", "absolute", "reading_median")
    out["within"] = _get(key, r, "orders", "absolute", "within")
    out["signed_within"] = _get(key, r, "orders", "signed", "within")
    out["random_within"] = _get(key, r, "orders", "random", "within")
    out["error_median"] = _get(key, r, "orders", "absolute", "error_median")
    return out


def _generalize_meter(key: str, r: dict, args: dict) -> dict[str, Any]:
    out = dict.fromkeys(METER_KEYS["generalize"])
    checkpoints = _get(key, r, "checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"record {key}: results hold no checkpoint row")
    if "description" in args:
        out["form"] = "name"
        for k in ("description", "class", "values", "envelope"):
            out[k] = _get(key, r, k)
        out["slices_f1"] = _get(key, r, "slices", "f1")
        out["slices_follows"] = _get(key, r, "slices", "follows")
    else:
        out["form"] = "edge"
        out["relation"] = _get(key, r, "relation")
    out["seeds"] = _get(key, r, "seeds")
    out["final"] = checkpoints[-1]
    return out


def _meter(key: str, kind: str, args: dict, results: dict) -> dict[str, Any]:
    if kind == "name":
        return _name_meter(key, results)
    if kind == "edge":
        return _edge_meter(key, results)
    if kind == "open":
        return _open_meter(key, results)
    return _generalize_meter(key, results, args)


def _payload(data: dict) -> dict[str, Any]:
    """The panel entry of a record already in its JSON form."""
    key, request = data["key"], data["request"]
    sha = sha256_hex((canonical_json(data) + "\n").encode("ascii"))
    return {"file": f"gesture-{key}.json", "key": key, "sha256": sha, "kind": request["kind"], "request": request,
            "meter": _meter(key, request["kind"], request["args"], data["results"])}


def gesture_meter(record: Any) -> dict[str, Any]:
    data = records.record_json(record)
    return _meter(data["key"], data["request"]["kind"], data["request"]["args"], data["results"])


def gesture_payload(record: Any) -> dict[str, Any]:
    return _payload(records.record_json(record))


def _listed(folder: Path) -> list[dict]:
    """The entries of the folder's ``index.json``, which must be in ``records.py``'s form."""
    path = folder / _INDEX
    if not path.is_file():
        raise ValueError(f"{folder} has no {_INDEX}")
    try:
        index = json.loads(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path} isn't JSON: {error}") from error
    entries = index.get("records") if isinstance(index, dict) else None
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path} isn't an index of gesture records")
    return entries


def gesture_panel(folder: str | Path) -> dict[str, Any]:
    folder = Path(folder)
    entries = _listed(folder)
    listed = [e.get("file") for e in entries]
    for path in sorted(folder.glob("gesture-*.json")):
        if path.name not in listed:
            raise ValueError(f"{path} is a record file the index doesn't list")
    keys = [e.get("key") for e in entries]
    if len(set(map(str, keys))) != len(keys):
        raise ValueError(f"{folder / _INDEX} lists a record twice")
    payloads, names, run = [], {}, None
    for key in keys:
        record = records.read_record(folder, key)
        if run is not None and record.request.run != run:
            raise ValueError(f"{folder} holds records of two runs, {run} and {record.request.run}")
        run = record.request.run
        payload = gesture_payload(record)
        payloads.append(payload)
        if payload["kind"] == "name":
            args = payload["request"]["args"]
            names.setdefault(f"{args['dictionary']}-{args['site']}", {}).setdefault(str(args["latent"]), []).append(key)
    return {"run": run, "records": payloads, "names": names}


def write_panel(folder: str | Path) -> Path:
    folder = Path(folder)
    data = (canonical_json(gesture_panel(folder)) + "\n").encode("ascii")
    path = folder / PANEL
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
