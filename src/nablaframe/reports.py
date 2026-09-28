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

from collections.abc import Mapping
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
    raise NotImplementedError


def write_report(report: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> None:
    """Validate ``report`` and write it as indented JSON with sorted keys.

    Raises ValueError listing every problem if the report is invalid, and
    FileExistsError if ``path`` exists and ``overwrite`` is False. Writes nothing
    in either case.
    """
    raise NotImplementedError
