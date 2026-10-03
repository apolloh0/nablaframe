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

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nablaframe.trace.schema import canonical_json, sha256_hex

_KEY = re.compile(r"[0-9a-f]{64}")


def cache_key(inputs: Mapping[str, str], code_commit: str) -> str:
    """SHA-256 of the canonical JSON of the inputs' hashes and the code commit."""
    return sha256_hex(canonical_json({"inputs": dict(inputs), "code": code_commit}).encode("ascii"))


def _check_key(key: str) -> None:
    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        raise ValueError(f"cache key must be 64 lowercase hex characters, got {key!r}")


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


class ViewCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        _check_key(key)
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """The payload stored under ``key``, or None if there is none."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_bytes())

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        """Store ``payload`` under ``key`` as canonical JSON and return its path.

        Identical content again is a no-op; different content under an existing key raises ValueError.
        """
        path = self._path(key)
        data = canonical_json(payload).encode("ascii")
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"{path} holds different content; a cache key never changes meaning")
            return path
        self.root.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, data)
        return path
