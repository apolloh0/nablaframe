"""Pinned tolerance records.

Spec: docs/specs/phase-0.md, "Mode-equality oracle".

A pinned tolerance is a record, not a constant: its value, the report it was
measured in, the backend and the dtype. Changing one needs a D-entry in
docs/decisions.md plus a discrimination test, never a unilateral edit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# A source report is a committed gate report: a relative path under reports/.
SOURCE_REPORT_PATTERN = r"^reports/[^/].*\.json$"


@dataclass(frozen=True)
class ToleranceRecord:
    """A tolerance ``value`` for max abs logit difference, and where it came from.

    Construction raises ValueError unless all of these hold:

    * ``value`` and ``measured_max_abs_diff`` are finite and >= 0.
    * ``value >= measured_max_abs_diff``: a tolerance tighter than its own
      measurement contradicts the report it cites.
    * ``source_report`` is a str matching ``SOURCE_REPORT_PATTERN``.
    * ``backend`` is a non-empty str.
    * ``dtype`` is a numpy floating dtype name such as ``"float32"``.
    """

    value: float
    source_report: str
    backend: str
    dtype: str
    measured_max_abs_diff: float

    def __post_init__(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToleranceRecord:
        """Builds and validates; the same ValueErrors as construction."""
        raise NotImplementedError
