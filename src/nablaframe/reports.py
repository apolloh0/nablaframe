"""Gate reports: the only place a gate result exists.

Spec: docs/specs/phase-0.md, "Gates".

Each gate script writes one JSON report under ``reports/phase0/``. The Phase 0
summary is generated from those files, never written by hand, and a report's
status must agree with its own measurements.

Report fields, all required, no others allowed:

==================  =============================================================
schema_version      1
gate_id             one of GATE_IDS
status              one of STATUSES; see STATUS RULES below
measurements        JSON object containing at least REQUIRED_MEASUREMENTS[gate_id];
                    every number finite
env_name            one of ENV_NAMES
env_lock_sha256     64 lowercase hex characters
xla_flags           str, may be empty
thread_count        int >= 1
chip                non-empty str, e.g. "Apple M3 Max"
timestamp_utc       ISO-8601 UTC ending in "Z", e.g. "2026-09-27T12:00:00Z"
==================  =============================================================

STATUS RULES (a report whose status contradicts its measurements is invalid):

* G0.1 pass     requires nonfinite_seen is False and loss_drop_fraction >= 0.30
* G0.2 pass     requires identical is True
  G0.2 conditional_pass requires identical is False and max_abs_param_diff <= 1e-6
* G0.3 pass     requires tokens_per_s_S >= 25000
* G0.4 pass     requires max_abs_logit_diff <= 1e-3 and top1_agreement == 1.0
* G0.5 pass     requires runs is True and max_rel_loss_diff <= 0.02
* G0.6          status is always "pass"; the measurements set the oracle policy
* conditional_pass is allowed only for G0.2
* fail is allowed for any gate except G0.6
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GATE_IDS: tuple[str, ...] = ("G0.1", "G0.2", "G0.3", "G0.4", "G0.5", "G0.6")
STATUSES: tuple[str, ...] = ("pass", "conditional_pass", "fail")
ENV_NAMES: tuple[str, ...] = ("nf-jax", "nf-metal", "nf-torch")

REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "gate_id",
    "status",
    "measurements",
    "env_name",
    "env_lock_sha256",
    "xla_flags",
    "thread_count",
    "chip",
    "timestamp_utc",
)

REQUIRED_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "G0.1": ("steps", "eval_loss_initial", "eval_loss_final", "loss_drop_fraction", "nonfinite_seen"),
    "G0.2": ("hash_run_a", "hash_run_b", "hash_resumed", "identical", "max_abs_param_diff", "single_threaded"),
    "G0.3": ("tokens_per_s_S", "tokens_per_s_M"),
    "G0.4": ("max_abs_logit_diff", "top1_agreement", "positions_compared", "positions_near_tie"),
    "G0.5": ("runs", "declined_ops", "max_rel_loss_diff", "tokens_per_s"),
    "G0.6": ("bitwise_equal", "max_abs_logit_diff", "seeds"),
}


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Every problem with ``report``, as human-readable strings. Empty means valid.

    Checks every rule in the module docstring. Never raises for a malformed
    report; a wrong type is reported as a problem. It does raise TypeError if
    ``report`` is not a Mapping at all.
    """
    if not isinstance(report, Mapping):
        raise TypeError(f"a report is a Mapping, got {type(report).__name__}")
    problems: list[str] = []

    missing = [name for name in REQUIRED_FIELDS if name not in report]
    problems += [f"missing field {name}" for name in missing]
    extra = sorted(repr(key) for key in report if key not in REQUIRED_FIELDS)
    problems += [f"unexpected field {key}" for key in extra]

    def check(name: str, ok: Any, expected: str) -> None:
        if name in report and not ok(report[name]):
            problems.append(f"{name} must be {expected}, got {report[name]!r}")

    check("schema_version", lambda v: _is_int(v) and v == SCHEMA_VERSION, str(SCHEMA_VERSION))
    check("gate_id", lambda v: isinstance(v, str) and v in GATE_IDS, f"one of {GATE_IDS}")
    check("status", lambda v: isinstance(v, str) and v in STATUSES, f"one of {STATUSES}")
    check("env_name", lambda v: isinstance(v, str) and v in ENV_NAMES, f"one of {ENV_NAMES}")
    check("env_lock_sha256", lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None,
          "64 lowercase hex characters")
    check("xla_flags", lambda v: isinstance(v, str), "a str")
    check("thread_count", lambda v: _is_int(v) and v >= 1, "an int >= 1")
    check("chip", lambda v: isinstance(v, str) and v != "", "a non-empty str")
    check("timestamp_utc", _is_utc_timestamp, 'an ISO-8601 UTC timestamp ending in "Z"')

    measurements = report.get("measurements")
    if "measurements" in report:
        if not isinstance(measurements, Mapping):
            problems.append(f"measurements must be a JSON object, got {measurements!r}")
            measurements = None
        else:
            problems += _json_problems(measurements, "measurements")

    gate_id, status = report.get("gate_id"), report.get("status")
    if isinstance(gate_id, str) and gate_id in GATE_IDS and measurements is not None:
        problems += [
            f"{gate_id} measurements missing {key}" for key in REQUIRED_MEASUREMENTS[gate_id] if key not in measurements
        ]
        if isinstance(status, str) and status in STATUSES:
            problems += _status_problems(gate_id, status, measurements)
    return problems


def write_report(report: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> None:
    """Validate ``report`` and write it as indented JSON with sorted keys.

    Raises ValueError listing every problem if the report is invalid, and
    FileExistsError if ``path`` exists and ``overwrite`` is False. Writes nothing
    in either case.
    """
    problems = validate_report(report)
    if problems:
        raise ValueError("invalid gate report:\n" + "\n".join(f"  - {p}" for p in problems))
    text = json.dumps(_plain(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Mode "x" refuses an existing file atomically, so nothing is ever half-overwritten by accident.
    with open(path, "w" if overwrite else "x", encoding="utf-8") as f:
        f.write(text)


# Numbers in a status rule: finite ints or floats, never bools.
_Number = (int, float)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, _Number) and not isinstance(v, bool) and _is_finite(v)


def _is_finite(v: int | float) -> bool:
    # Every int is finite; math.isfinite overflows on ints too large for a float.
    return isinstance(v, int) or math.isfinite(v)


def _is_utc_timestamp(v: Any) -> bool:
    if not isinstance(v, str) or not v.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(v)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)


def _json_problems(value: Any, where: str) -> list[str]:
    """Anything that is not a JSON value, or a number that is not finite. Null is allowed."""
    if value is None or isinstance(value, (bool, str)):
        return []
    if isinstance(value, _Number):
        return [] if _is_finite(value) else [f"{where} is not finite: {value!r}"]
    if isinstance(value, Mapping):
        problems = []
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{where} has a non-str key {key!r}")
            else:
                problems += _json_problems(item, f"{where}.{key}")
        return problems
    if isinstance(value, (list, tuple)):
        return [p for i, item in enumerate(value) for p in _json_problems(item, f"{where}[{i}]")]
    return [f"{where} is not a JSON value: {value!r} ({type(value).__name__})"]


def _status_problems(gate_id: str, status: str, m: Mapping[str, Any]) -> list[str]:
    """The STATUS RULES in the module docstring, for a known gate and status."""
    if gate_id == "G0.6":
        return [] if status == "pass" else [f'G0.6 status is always "pass", got {status!r}']
    if status == "fail":
        return []
    if status == "conditional_pass" and gate_id != "G0.2":
        return [f"conditional_pass is allowed only for G0.2, not {gate_id}"]

    def is_(key: str, expected: bool) -> tuple[bool, str]:
        return m.get(key) is expected, f"{key} is {expected}"

    def at_least(key: str, bound: float) -> tuple[bool, str]:
        v = m.get(key)
        return _is_number(v) and v >= bound, f"{key} >= {bound:g}"

    def at_most(key: str, bound: float) -> tuple[bool, str]:
        v = m.get(key)
        return _is_number(v) and v <= bound, f"{key} <= {bound:g}"

    def equals(key: str, target: float) -> tuple[bool, str]:
        v = m.get(key)
        return _is_number(v) and v == target, f"{key} == {target!r}"

    rules = {
        ("G0.1", "pass"): [is_("nonfinite_seen", False), at_least("loss_drop_fraction", 0.30)],
        ("G0.2", "pass"): [is_("identical", True)],
        ("G0.2", "conditional_pass"): [is_("identical", False), at_most("max_abs_param_diff", 1e-6)],
        ("G0.3", "pass"): [at_least("tokens_per_s_S", 25000)],
        ("G0.4", "pass"): [at_most("max_abs_logit_diff", 1e-3), equals("top1_agreement", 1.0)],
        ("G0.5", "pass"): [is_("runs", True), at_most("max_rel_loss_diff", 0.02)],
    }[(gate_id, status)]
    return [
        f"{gate_id} {status} requires {rule}, got {m.get(rule.split()[0])!r}" for ok, rule in rules if not ok
    ]


def _plain(value: Any) -> Any:
    """Mappings as dicts, recursively, so any Mapping serializes as a JSON object."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
