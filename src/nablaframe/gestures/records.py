"""Gesture requests and records (D47): what was asked, of which inputs, by which code, and what was measured.

Spec: docs/specs/phase-6.md, D47. Nothing here runs a model; the gesture modules do, and hand
their results here to be recorded.

* ``KINDS = ("name", "edge", "open", "generalize")``.
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

  A hash is 64 lowercase hex characters. An int is a non-negative int, never a bool. A name is a
  non-empty string of printable ASCII characters (codes 32 to 126).
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
    holds at least one;
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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

KINDS = ("name", "edge", "open", "generalize")
_EDGE = {"relation": "name", "up": "hash", "up_site": "int", "down": "hash", "down_site": "int"}
FORMS: dict[str, tuple[dict[str, str], ...]] = {
    "name": ({"dictionary": "hash", "site": "int", "class": "name", "latent": "int"},),
    "edge": (dict(_EDGE),),
    "open": ({"dictionary": "hash", "site": "int", "class": "name", "upstream": "hash", "upstream_site": "int",
              "latent": "int"},
             {"dictionary": "hash", "site": "int", "class": "name", "upstream": "hash", "upstream_site": "int",
              "description": "name", "value": "int"}),
    "generalize": ({"dictionary": "hash", "site": "int", "class": "name", "description": "name"}, dict(_EDGE)),
}


@dataclass(frozen=True, eq=False)
class GestureRequest:
    kind: str
    run: str
    step: int
    args: Mapping[str, Any]
    instances: int
    seed: int

    def __post_init__(self) -> None:
        raise NotImplementedError


def request_json(request: GestureRequest) -> dict[str, Any]:
    raise NotImplementedError


def request_from_json(data: Mapping[str, Any]) -> GestureRequest:
    raise NotImplementedError


def request_hash(request: GestureRequest) -> str:
    raise NotImplementedError


def code_identity(*modules: ModuleType) -> dict[str, str]:
    raise NotImplementedError


@dataclass(frozen=True, eq=False)
class GestureRecord:
    request: GestureRequest
    inputs: Mapping[str, Any]
    code: Mapping[str, str]
    results: Mapping[str, Any]

    def __post_init__(self) -> None:
        raise NotImplementedError

    @property
    def key(self) -> str:
        raise NotImplementedError


def record_key(request: GestureRequest, inputs: Mapping[str, Any], code: Mapping[str, str]) -> str:
    raise NotImplementedError


def record_json(record: GestureRecord) -> dict[str, Any]:
    raise NotImplementedError


def record_from_json(data: Mapping[str, Any]) -> GestureRecord:
    raise NotImplementedError


def write_record(folder: str | Path, record: GestureRecord) -> Path:
    raise NotImplementedError


def read_record(folder: str | Path, key: str) -> GestureRecord:
    raise NotImplementedError


def cached_record(folder: str | Path, request: GestureRequest, inputs: Mapping[str, Any], code: Mapping[str, str],
                  make: Callable[[], Mapping[str, Any]]) -> GestureRecord:
    raise NotImplementedError
