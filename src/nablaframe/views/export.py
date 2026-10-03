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


def export_view(out_dir: str | Path, name: str, payload: dict[str, Any]) -> Path:
    raise NotImplementedError
