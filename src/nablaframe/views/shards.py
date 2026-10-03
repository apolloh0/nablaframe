"""What the viewer hashes (D13).

* ``shard_tensor_bytes(path)``: the raw data slice of the shard's ``"tensor"`` entry,
  read straight from the safetensors layout: an 8-byte little-endian header length,
  that many bytes of JSON header, then the data, where the entry's ``data_offsets``
  locate its bytes. Standard library only, no safetensors import, so it mirrors
  exactly what the page does in the browser. Its SHA-256 equals the event's.
"""

from __future__ import annotations

from pathlib import Path


def shard_tensor_bytes(path: str | Path) -> bytes:
    raise NotImplementedError
