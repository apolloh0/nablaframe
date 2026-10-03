"""Exporting derived views for the viewer.

Spec: docs/specs/phase-2.md, "The scrubber": a view without a meter can't render.

* ``export_view(out_dir, name, payload)``: writes ``<out_dir>/<name>.json`` as
  canonical JSON (sorted keys, compact separators, ASCII), through a temporary file
  and a rename, and returns the path. Raises ValueError, writing nothing, unless
  ``payload`` has a ``"meter"`` entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nablaframe.trace.schema import canonical_json
from nablaframe.views.cache import _write_atomic


def export_view(out_dir: str | Path, name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` to ``<out_dir>/<name>.json`` as canonical JSON and return the path.

    Raises ValueError, writing nothing, if ``payload`` has no ``"meter"`` entry.
    """
    if "meter" not in payload:
        raise ValueError(f"derived view {name!r} has no meter; a view without one can't render")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"view name must be a plain file name, got {name!r}")
    data = canonical_json(payload).encode("ascii")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    _write_atomic(path, data)
    return path
