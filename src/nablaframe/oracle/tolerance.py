"""Pinned tolerance records.

Spec: docs/specs/phase-0.md, "Mode-equality oracle".

A pinned tolerance is a record, not a constant: its value, the report it was
measured in, the backend and the dtype. Changing one needs a D-entry in
docs/decisions.md plus a discrimination test, never a unilateral edit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any

import numpy as np

# A source report is a committed gate report: `reports/phase<N>/<name>.json`, nothing else (D3).
SOURCE_REPORT_PATTERN = r"^reports/phase[0-9]+/[A-Za-z0-9_-]+\.json$"


def _finite_nonnegative(name: str, x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, Real):
        raise ValueError(f"{name} must be a real number, got {x!r}")
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {x!r}")
    return x


def _is_float_dtype_name(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    try:
        dtype = np.dtype(name)
    except (TypeError, ValueError):
        return False
    return np.issubdtype(dtype, np.floating) and dtype.name == name


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
        value = _finite_nonnegative("value", self.value)
        measured = _finite_nonnegative("measured_max_abs_diff", self.measured_max_abs_diff)
        if value < measured:
            raise ValueError(
                f"value {value!r} is tighter than measured_max_abs_diff {measured!r} in its own source report"
            )
        if not isinstance(self.source_report, str) or re.fullmatch(SOURCE_REPORT_PATTERN, self.source_report) is None:
            raise ValueError(f"source_report must match {SOURCE_REPORT_PATTERN!r}, got {self.source_report!r}")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError(f"backend must be a non-empty str, got {self.backend!r}")
        if not _is_float_dtype_name(self.dtype):
            raise ValueError(f"dtype must be a numpy floating dtype name such as 'float32', got {self.dtype!r}")
        # Store plain floats so the JSON form round-trips exactly.
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "measured_max_abs_diff", measured)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToleranceRecord:
        """Builds and validates; the same ValueErrors as construction."""
        if not isinstance(data, Mapping):
            raise ValueError(f"expected a mapping, got {type(data).__name__}")
        names = {f.name for f in fields(cls)}
        missing = sorted(names - set(data))
        extra = sorted(map(str, set(data) - names))
        if missing or extra:
            raise ValueError(f"tolerance record fields: missing {missing}, unexpected {extra}")
        return cls(**{name: data[name] for name in names})
