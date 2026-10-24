"""Gesture requests and records (D47): what was asked, of which inputs, by which code, and what was measured.

Spec: docs/specs/phase-6.md, D47. Nothing here runs a model; the gesture modules do, and hand
their results here to be recorded.

* ``KINDS = ("name", "edge", "open", "generalize", "crystallize")``.
* ``FORMS``: each kind's argument forms, as ``{kind: (form, ...)}``, each form a mapping from an
  argument's name to its type. A request's ``args`` must match one form exactly, no key missing
  and none extra:

  - ``name``: ``dictionary`` (hash), ``site`` (int), ``class`` (name), ``latent`` (int)
  - ``edge``: ``relation`` (name), ``up`` (hash), ``up_site`` (int), ``down`` (hash),
    ``down_site`` (int)
  - ``open``: ``dictionary``, ``site``, ``class``, ``upstream`` (hash), ``upstream_site`` (int),
    and either ``latent`` (int) or both ``description`` (name) and ``value`` (int)
  - ``generalize``: either ``dictionary``, ``site``, ``class`` and ``description``, a name read
    over all its values, or ``edge``'s five arguments
  - ``crystallize``: ``block`` (int), ``part`` (name), ``class`` (name), ``function`` (name),
    ``form`` (name) and ``cites`` (hashes), the keys of the records it cites (D56, D57)

  A hash is 64 lowercase hex characters. Hashes are a list or tuple of distinct hashes, possibly
  empty, stored as a sorted read-only list, so requests listing the same hashes in another order
  are equal. An int is a non-negative int, never a bool. A name is a non-empty string of
  printable ASCII characters (codes 32 to 126).
* ``GestureRequest(kind, run, step, args, instances, seed)``: a frozen dataclass. ``kind`` is
  one of ``KINDS``; ``run`` a run id (a hash); ``step`` the checkpoint the gesture reads, a
  non-negative int; ``args`` as above; ``instances`` how many instances it measures, a positive
  int; ``seed`` the seed that draws them, a non-negative int. Construction checks every field and
  raises ValueError otherwise. ``args`` is stored as a read-only mapping, so assigning into it
  raises TypeError, with ints as ``int``, so a request built from numpy ints equals one built from
  Python ints.
* ``request_json(request)``: ``{"kind", "run", "step", "args", "instances", "seed"}`` as plain
  JSON. ``request_from_json(data)`` inverts it; ValueError for any other key or anything
  ``GestureRequest`` refuses. ``request_hash(request)``: the SHA-256 of
  ``canonical_json(request_json(request))``. Requests equal as values hash equal, whatever the
  order their arguments were given in.
* ``code_identity(*modules)``: ``{module.__name__: blob hash}``, the hash ``git hash-object``
  prints for each module's file (D18), as a plain dict sorted by name. ValueError for no modules.
* ``GestureRecord(request, inputs, code, results)``: a frozen dataclass.

  - ``request``: a ``GestureRequest``.
  - ``inputs``: exactly ``{"states", "dictionaries", "windows"}``. ``states`` maps each
    checkpoint read, its step as a decimal string without leading zeros, to its state hash, and
    holds at least one; a crystallize-this record's ``states`` must also map ``"twin"`` to the
    state hash of its run's twin (D30), and no other kind's may;
    ``dictionaries`` and ``windows`` are lists of content hashes and batch hashes, sorted, with no
    repeats. Every hash is 64 lowercase hex characters.
  - ``code``: a non-empty mapping from module names, each a name as above, to 40-character
    lowercase hex blob hashes.
  - ``results``: a mapping ``canonical_json`` can encode. NaN and infinity are refused; a reading
    that didn't happen, such as "not within 128 parts", is ``None``.

  Construction checks each and raises ValueError otherwise. ``inputs``, ``code`` and
  ``results`` are stored as read-only copies taken through their JSON form: assigning into them
  raises TypeError, and changing the caller's objects afterwards never changes a record. ``record.key`` is ``record_key(request,
  inputs, code)``.
* ``record_key(request, inputs, code)``: the SHA-256 of ``canonical_json({"request":
  request_hash(request), "inputs": inputs, "code": code})``. The results aren't in the key: it
  names what was asked of which inputs by which code, and the same key always rebuilds the same
  results (D11).
* ``record_json(record)``: ``{"key", "request": request_json, "inputs", "code", "results"}`` as
  plain JSON. ``record_from_json(data)`` inverts it; ValueError for any other key, anything
  ``GestureRecord`` refuses, or a ``key`` that isn't the key of the rest.
* ``write_record(folder, record)``: writes ``gesture-<key>.json``, the canonical JSON of
  ``record_json(record)`` and a newline, through a temporary file and a rename, creating the
  folder. Then it rewrites ``index.json`` the same way, as ``{"records": [{"file", "key", "kind",
  "request", "sha256"}, ...]}``: one entry per ``gesture-*.json`` file in the folder, ``request``
  being its request hash and ``sha256`` the SHA-256 of the file's bytes, sorted by kind and then
  key. Writing an identical record again leaves every byte as it was. Writing a different record
  under a key the folder already holds raises ValueError and writes nothing, because a key never
  changes meaning. Returns the record file's path.
* ``read_record(folder, key)``: the record the folder holds under ``key``. ValueError, naming the
  problem, if there is no index or the index has no entry for ``key``, if the file's SHA-256 isn't
  the index's, or if the file isn't a record whose key is ``key``.
* ``cached_record(folder, request, inputs, code, make)``: ``read_record`` when the folder's index
  holds ``record_key(request, inputs, code)``; otherwise ``make()`` gives the results, and the
  record built from them is written with ``write_record`` and returned. ``make`` is called at most
  once, and never when the record is already there.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from nablaframe.trace.schema import canonical_json, sha256_hex

KINDS = ("name", "edge", "open", "generalize", "crystallize")
_EDGE = {"relation": "name", "up": "hash", "up_site": "int", "down": "hash", "down_site": "int"}
FORMS: dict[str, tuple[dict[str, str], ...]] = {
    "name": ({"dictionary": "hash", "site": "int", "class": "name", "latent": "int"},),
    "edge": (dict(_EDGE),),
    "open": ({"dictionary": "hash", "site": "int", "class": "name", "upstream": "hash", "upstream_site": "int",
              "latent": "int"},
             {"dictionary": "hash", "site": "int", "class": "name", "upstream": "hash", "upstream_site": "int",
              "description": "name", "value": "int"}),
    "generalize": ({"dictionary": "hash", "site": "int", "class": "name", "description": "name"}, dict(_EDGE)),
    "crystallize": ({"block": "int", "part": "name", "class": "name", "function": "name", "form": "name",
                     "cites": "hashes"},),
}


_HASH = re.compile(r"[0-9a-f]{64}")
_BLOB = re.compile(r"[0-9a-f]{40}")
_STEP = re.compile(r"0|[1-9][0-9]*")
_INDEX = "index.json"
_TWIN = "twin"


class _FrozenDict(dict):
    """A dict that refuses writes, so a record's contents can't change under its key."""

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("a gesture request or record is read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse
    update = pop = popitem = clear = setdefault = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))

    def __copy__(self) -> dict:
        return dict(self)

    def __deepcopy__(self, memo: dict) -> dict:
        return _thaw(self)


class _FrozenList(list):
    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("a gesture request or record is read-only")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = extend = insert = remove = pop = clear = sort = reverse = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (list, (list(self),))

    def __copy__(self) -> list:
        return list(self)

    def __deepcopy__(self, memo: dict) -> list:
        return _thaw(self)


def _freeze(value: Any, where: str) -> Any:
    """A read-only copy of ``value`` through its JSON form; ValueError for anything JSON can't hold exactly."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{where}: NaN and infinity can't be recorded")
        return float(value)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: keys must be strings, got {key!r}")
            out[key] = _freeze(item, f"{where}[{key!r}]")
        return _FrozenDict(out)
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(item, f"{where}[{i}]") for i, item in enumerate(value))
    raise ValueError(f"{where}: {type(value).__name__} isn't plain JSON")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _is_name(value: Any) -> bool:
    return isinstance(value, str) and value != "" and all(32 <= ord(c) <= 126 for c in value)


def _int(value: Any, where: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where} must be an int, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{where} must be at least {minimum}, got {value}")
    return value


def _check_args(kind: str, args: Any) -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise ValueError(f"args must be a mapping, got {type(args).__name__}")
    if not all(isinstance(k, str) for k in args):
        raise ValueError("args keys must be strings")
    for form in FORMS[kind]:
        if set(args) != set(form):
            continue
        out = {}
        for name in sorted(form):
            value, type_ = args[name], form[name]
            if type_ == "hash":
                if not _is_hash(value):
                    raise ValueError(f"{kind} argument {name!r} must be 64 lowercase hex characters, got {value!r}")
                out[name] = value
            elif type_ == "hashes":
                if not isinstance(value, (list, tuple)) or not all(_is_hash(h) for h in value):
                    raise ValueError(f"{kind} argument {name!r} must be a list of 64-character lowercase hex "
                                     f"hashes, got {value!r}")
                if len(set(value)) != len(value):
                    raise ValueError(f"{kind} argument {name!r} must hold distinct hashes, got {value!r}")
                out[name] = _FrozenList(sorted(value))
            elif type_ == "name":
                if not _is_name(value):
                    raise ValueError(f"{kind} argument {name!r} must be a non-empty printable ASCII string, "
                                     f"got {value!r}")
                out[name] = value
            else:
                out[name] = _int(value, f"{kind} argument {name!r}", 0)
        return out
    forms = " or ".join("{" + ", ".join(form) + "}" for form in FORMS[kind])
    raise ValueError(f"a {kind} request takes exactly the arguments {forms}, got {{{', '.join(sorted(args))}}}")


@dataclass(frozen=True, eq=False)
class GestureRequest:
    kind: str
    run: str
    step: int
    args: Mapping[str, Any]
    instances: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not _is_hash(self.run):
            raise ValueError(f"run must be 64 lowercase hex characters, got {self.run!r}")
        args = _check_args(self.kind, self.args)
        for name, minimum in (("step", 0), ("instances", 1), ("seed", 0)):
            object.__setattr__(self, name, _int(getattr(self, name), name, minimum))
        object.__setattr__(self, "args", _FrozenDict(args))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GestureRequest) and request_json(self) == request_json(other)

    def __hash__(self) -> int:
        return hash(request_hash(self))


def request_json(request: GestureRequest) -> dict[str, Any]:
    return {"kind": request.kind, "run": request.run, "step": request.step, "args": _thaw(request.args),
            "instances": request.instances, "seed": request.seed}


def request_from_json(data: Mapping[str, Any]) -> GestureRequest:
    if not isinstance(data, Mapping):
        raise ValueError(f"a request is a mapping, got {type(data).__name__}")
    fields = {"kind", "run", "step", "args", "instances", "seed"}
    if set(data) != fields:
        raise ValueError(f"a request has exactly the keys {sorted(fields)}, got {sorted(map(str, data))}")
    return GestureRequest(**{k: data[k] for k in fields})


def request_hash(request: GestureRequest) -> str:
    return sha256_hex(canonical_json(request_json(request)).encode("ascii"))


def code_identity(*modules: ModuleType) -> dict[str, str]:
    if not modules:
        raise ValueError("code_identity needs at least one module")
    out = {}
    for module in modules:
        path = getattr(module, "__file__", None)
        if path is None:
            raise ValueError(f"module {module.__name__} has no source file")
        data = Path(path).read_bytes()
        out[module.__name__] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return dict(sorted(out.items()))


def _check_inputs(inputs: Any, kind: str) -> Mapping[str, Any]:
    if not isinstance(inputs, Mapping) or set(inputs) != {"states", "dictionaries", "windows"}:
        keys = sorted(map(str, inputs)) if isinstance(inputs, Mapping) else type(inputs).__name__
        raise ValueError(f"inputs have exactly the keys states, dictionaries and windows, got {keys}")
    frozen = _freeze(inputs, "inputs")
    states = frozen["states"]
    if not isinstance(states, Mapping):
        raise ValueError("inputs['states'] must map at least one step to a state hash")
    if kind == "crystallize" and _TWIN not in states:
        raise ValueError(f"a crystallize-this record's inputs['states'] must map {_TWIN!r} to its twin's state hash")
    if not any(step != _TWIN for step in states):
        raise ValueError("inputs['states'] must map at least one step to a state hash")
    for step, state in states.items():
        if step == _TWIN and kind != "crystallize":
            raise ValueError(f"only a crystallize-this record's inputs['states'] holds {_TWIN!r}")
        if step != _TWIN and _STEP.fullmatch(step) is None:
            raise ValueError(f"inputs['states'] key {step!r} must be a decimal step without leading zeros")
        if not _is_hash(state):
            raise ValueError(f"inputs['states'][{step!r}] must be 64 lowercase hex characters, got {state!r}")
    for name in ("dictionaries", "windows"):
        hashes = frozen[name]
        if not isinstance(hashes, list) or not all(_is_hash(h) for h in hashes):
            raise ValueError(f"inputs[{name!r}] must be a list of 64-character lowercase hex hashes")
        if any(a >= b for a, b in zip(hashes, hashes[1:])):
            raise ValueError(f"inputs[{name!r}] must be sorted, with no repeats")
    return frozen


def _check_code(code: Any) -> Mapping[str, str]:
    if not isinstance(code, Mapping) or not code:
        raise ValueError("code must map at least one module name to a blob hash")
    for name, blob in code.items():
        if not _is_name(name):
            raise ValueError(f"code module name {name!r} must be a non-empty printable ASCII string")
        if not isinstance(blob, str) or _BLOB.fullmatch(blob) is None:
            raise ValueError(f"code[{name!r}] must be 40 lowercase hex characters, got {blob!r}")
    return _freeze(code, "code")


def _check_results(results: Any) -> Mapping[str, Any]:
    if not isinstance(results, Mapping):
        raise ValueError(f"results must be a mapping, got {type(results).__name__}")
    return _freeze(results, "results")


@dataclass(frozen=True, eq=False)
class GestureRecord:
    request: GestureRequest
    inputs: Mapping[str, Any]
    code: Mapping[str, str]
    results: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GestureRequest):
            raise ValueError(f"request must be a GestureRequest, got {type(self.request).__name__}")
        object.__setattr__(self, "inputs", _check_inputs(self.inputs, self.request.kind))
        object.__setattr__(self, "code", _check_code(self.code))
        object.__setattr__(self, "results", _check_results(self.results))

    @property
    def key(self) -> str:
        return record_key(self.request, self.inputs, self.code)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GestureRecord) and record_json(self) == record_json(other)

    def __hash__(self) -> int:
        return hash(self.key)


def record_key(request: GestureRequest, inputs: Mapping[str, Any], code: Mapping[str, str]) -> str:
    body = {"request": request_hash(request), "inputs": _thaw(_check_inputs(inputs, request.kind)),
            "code": _thaw(_check_code(code))}
    return sha256_hex(canonical_json(body).encode("ascii"))


def record_json(record: GestureRecord) -> dict[str, Any]:
    return {"key": record.key, "request": request_json(record.request), "inputs": _thaw(record.inputs),
            "code": _thaw(record.code), "results": _thaw(record.results)}


def record_from_json(data: Mapping[str, Any]) -> GestureRecord:
    if not isinstance(data, Mapping):
        raise ValueError(f"a record is a mapping, got {type(data).__name__}")
    fields = {"key", "request", "inputs", "code", "results"}
    if set(data) != fields:
        raise ValueError(f"a record has exactly the keys {sorted(fields)}, got {sorted(map(str, data))}")
    record = GestureRecord(request_from_json(data["request"]), data["inputs"], data["code"], data["results"])
    if data["key"] != record.key:
        raise ValueError(f"the record's key {data['key']!r} isn't the key of the rest, {record.key}")
    return record


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and a rename."""
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


def _write_if_changed(path: Path, data: bytes) -> None:
    if not path.exists() or path.read_bytes() != data:
        _write_atomic(path, data)


def _record_bytes(record: GestureRecord) -> bytes:
    return (canonical_json(record_json(record)) + "\n").encode("ascii")


def _entry(record: GestureRecord, data: bytes) -> dict[str, str]:
    return {"file": f"gesture-{record.key}.json", "key": record.key, "kind": record.request.kind,
            "request": request_hash(record.request), "sha256": sha256_hex(data)}


def write_record(folder: str | Path, record: GestureRecord) -> Path:
    folder = Path(folder)
    data = _record_bytes(record)
    path = folder / f"gesture-{record.key}.json"
    entries = {}
    for other in sorted(folder.glob("gesture-*.json")) if folder.is_dir() else []:
        if other == path:
            if other.read_bytes() != data:
                raise ValueError(f"{other} already holds a different record under key {record.key}; "
                                 "a key never changes meaning")
            continue
        other_data = other.read_bytes()
        try:
            held = record_from_json(json.loads(other_data))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{other} isn't a gesture record: {error}") from error
        if other.name != f"gesture-{held.key}.json":
            raise ValueError(f"{other} holds a record whose key is {held.key}")
        entries[other.name] = _entry(held, other_data)
    entries[path.name] = _entry(record, data)
    index = {"records": sorted(entries.values(), key=lambda e: (e["kind"], e["key"]))}
    folder.mkdir(parents=True, exist_ok=True)
    _write_if_changed(path, data)
    _write_if_changed(folder / _INDEX, (canonical_json(index) + "\n").encode("ascii"))
    return path


def _read_index(folder: Path) -> list[Mapping[str, Any]]:
    path = folder / _INDEX
    if not path.is_file():
        raise ValueError(f"{folder} has no {_INDEX}")
    try:
        index = json.loads(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path} isn't JSON: {error}") from error
    records = index.get("records") if isinstance(index, dict) else None
    if not isinstance(records, list) or not all(isinstance(e, dict) for e in records):
        raise ValueError(f"{path} isn't an index of gesture records")
    return records


def read_record(folder: str | Path, key: str) -> GestureRecord:
    folder = Path(folder)
    if not _is_hash(key):
        raise ValueError(f"a record key is 64 lowercase hex characters, got {key!r}")
    entry = next((e for e in _read_index(folder) if e.get("key") == key), None)
    if entry is None:
        raise ValueError(f"{folder / _INDEX} has no entry for {key}")
    path = folder / f"gesture-{key}.json"
    if entry.get("file") != path.name:
        raise ValueError(f"the index names {entry.get('file')!r} for {key}, not {path.name!r}")
    if not path.is_file():
        raise ValueError(f"{path} is missing")
    data = path.read_bytes()
    if sha256_hex(data) != entry.get("sha256"):
        raise ValueError(f"{path} doesn't match the SHA-256 the index holds")
    try:
        record = record_from_json(json.loads(data))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path} isn't a gesture record: {error}") from error
    if record.key != key:
        raise ValueError(f"{path} holds the record {record.key}, not {key}")
    return record


def cached_record(folder: str | Path, request: GestureRequest, inputs: Mapping[str, Any], code: Mapping[str, str],
                  make: Callable[[], Mapping[str, Any]]) -> GestureRecord:
    folder = Path(folder)
    key = record_key(request, inputs, code)
    if (folder / _INDEX).is_file() and any(e.get("key") == key for e in _read_index(folder)):
        return read_record(folder, key)
    record = GestureRecord(request, inputs, code, make())
    write_record(folder, record)
    return record
