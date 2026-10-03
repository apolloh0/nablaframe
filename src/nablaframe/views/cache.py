"""Caches for derived views (D11): keyed by their inputs and the code that made them.

Spec: docs/specs/phase-2.md, D11.

* ``cache_key(inputs, code_commit)``: SHA-256 of the canonical JSON of
  ``{"inputs": inputs, "code": code_commit}``, where ``inputs`` maps names to the
  SHA-256 of each input. Input order doesn't matter.
* ``ViewCache(root)``: ``get(key)`` returns the stored payload, or None.
  ``put(key, payload)`` writes ``<root>/<key>.json`` as canonical JSON, through a
  temporary file and a rename, and returns the path. Putting identical content again
  is a no-op. Putting different content under an existing key raises ValueError,
  because a key never changes meaning. Deleting the whole cache loses nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def cache_key(inputs: Mapping[str, str], code_commit: str) -> str:
    raise NotImplementedError


class ViewCache:
    def __init__(self, root: str | Path) -> None:
        raise NotImplementedError

    def get(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        raise NotImplementedError
