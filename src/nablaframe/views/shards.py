"""What the viewer hashes (D13).

* ``shard_tensor_bytes(path)``: the raw data slice of the shard's ``"tensor"`` entry,
  read straight from the safetensors layout: an 8-byte little-endian header length,
  that many bytes of JSON header, then the data, where the entry's ``data_offsets``
  locate its bytes. Standard library only, no safetensors import, so it mirrors
  exactly what the page does in the browser. Its SHA-256 equals the event's.
"""

from __future__ import annotations

import json
from pathlib import Path


def shard_tensor_bytes(path: str | Path) -> bytes:
    """The bytes of the shard's ``"tensor"`` entry, as ``viewer/app.js`` slices them.

    Raises ValueError if the file is too short for its header, the header isn't a JSON
    object with a ``"tensor"`` entry, or the entry's ``data_offsets`` fall outside the data.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: {len(raw)} bytes is too short for a safetensors header length")
    header_len = int.from_bytes(raw[:8], "little")
    data_start = 8 + header_len
    if data_start > len(raw):
        raise ValueError(f"{path}: header length {header_len} runs past the end of the file")
    header = json.loads(raw[8:data_start].decode("utf-8"))
    if not isinstance(header, dict) or not isinstance(header.get("tensor"), dict):
        raise ValueError(f"{path}: the header has no \"tensor\" entry")
    begin, end = header["tensor"]["data_offsets"]
    if not 0 <= begin <= end <= len(raw) - data_start:
        raise ValueError(f"{path}: data_offsets {[begin, end]} fall outside the {len(raw) - data_start} data bytes")
    return raw[data_start + begin:data_start + end]
