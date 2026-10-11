"""Gate reports: the only place a gate result exists.

Spec: docs/specs/phase-0.md, "Gates"; docs/specs/phase-1.md, "Carried over from Phase 0" and "Gates"; docs/specs/phase-2.md, "Gates"; docs/specs/phase-3.md, "Gates"; docs/specs/phase-4.md, "Gates"; docs/deltas/phase-4-delta-3.md, "Contract changes"; docs/specs/phase-5.md, "Gates"; docs/deltas/phase-5-delta-4.md, "Contract changes"; docs/deltas/phase-5-delta-5.md, "Contract changes".

Each gate script writes one JSON report under ``reports/phase0/``, ``reports/phase1/``, ``reports/phase2/``, ``reports/phase3/``, ``reports/phase4/`` or ``reports/phase5/``. The Phase 0
summary is generated from those files, never written by hand, and a report's
status must agree with its own measurements.

Report fields, all required, no others allowed (version 2 adds two; see below):

==================  =============================================================
schema_version      1 or 2
gate_id             one of GATE_IDS and PHASE1_GATE_IDS to PHASE5_GATE_IDS
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

Schema version 2 adds two required fields: `git_commit`, 40 lowercase hex characters, and `git_dirty`, a bool. Version 1 reports have neither and stay valid. Phase 1 gates write version 2 only; a Phase 1 report on version 1 is invalid.

`GATE_IDS` stays the Phase 0 tuple, because Phase 0's tests and summary loop over it. Phase 1's gates are `PHASE1_GATE_IDS = ("G1.1", "G1.2", "G1.3", "G1.4", "G1.5", "G1.6")`, and the validator accepts both.

`PHASE2_GATE_IDS = ("G2.1", "G2.2", "G2.3", "G2.4", "G2.5", "G2.6")`. The validator accepts them, and a Phase 2 report must use schema version 2. `PHASE2_GATE_IDS` gains G2.7, G2.8 and G2.9. `PHASE2_GATE_IDS` gains G2.10.

`PHASE3_GATE_IDS = ("G3.1", "G3.2", "G3.3", "G3.4", "G3.5")`. The validator accepts them, and a Phase 3 report must use schema version 2.

`PHASE4_GATE_IDS = ("G4.1", "G4.2", "G4.3", "G4.4", "G4.5")`. The validator accepts them, and a Phase 4 report must use schema version 2. `PHASE4_GATE_IDS` gains G4.6.

`PHASE5_GATE_IDS = ("G5.1", "G5.2", "G5.3", "G5.4", "G5.5")`. The validator accepts them, and a Phase 5 report must use schema version 2. `PHASE5_GATE_IDS` gains G5.6.

STATUS RULES (a report whose status contradicts its measurements is invalid):

* G0.1 pass requires nonfinite_seen is False, loss_drop_fraction >= 0.30 and steps >= 1000.
* G0.2 pass     requires identical is True
  G0.2 conditional_pass requires identical is False and max_abs_param_diff <= 1e-6
* G0.2, under every status: identical must be True exactly when hash_run_a, hash_run_b and hash_resumed are all equal. A null hash was not measured and never counts as equal to another hash.
* G0.3 pass     requires tokens_per_s_S >= 25000
* G0.4 pass     requires max_abs_logit_diff <= 1e-3 and top1_agreement == 1.0
* G0.5 pass     requires runs is True and max_rel_loss_diff <= 0.02
* G0.6          status is always "pass"; the measurements set the oracle policy
* fail is allowed for any gate except G0.6, G2.5, G3.3, G3.4, G4.4 and G5.5

Status rules for Phase 1:

* G1.1 pass requires `builds_identical` is true and `vocab_selected` is 4,096 exactly when `unk_rate_4096 <= 0.01`, otherwise 8,192 (D8).
* G1.2 pass requires `n_params == n_params_expected`, `max_abs_logit_diff <= 1e-4` and `top1_agreement == 1.0`.
* G1.3 follows G0.2's rules, including `conditional_pass` and the hash-agreement rule under every status. `conditional_pass` is allowed only for G0.2 and G1.3.
* G1.4 pass requires `nonfinite_seen` is false, `checkpoints_written == checkpoints_scheduled` and `final_val_loss < bigram_val_loss`.
* G1.5 pass requires `bitwise_equal` and `defect_detected` both true.
* G1.6 pass requires `sample_matches == sample_size`, `retrace_new_events == 0` and `store_bytes` within 10% of `store_bytes_expected`.

Status rules for Phase 2:

* G2.1 pass requires `logits_events == logits_events_expected`, `other_events_written == 0`, `sample_matches == sample_size` and `logits_bytes` within 1% of `logits_bytes_expected`.
* G2.2 pass requires `sample_matches == sample_size` and `builds_identical` is true.
* G2.3 pass requires `planted_recovery >= 0.99`, `random_max_accuracy <= 2 * random_chance` and `last_layer_kl_max <= 1e-5`.
* G2.4 pass requires `planted_aligned_credit >= 0.99`, `planted_perpendicular_quiet >= 0.99` and `decomposition_error_max <= 1e-3`.
* G2.5's status is always `pass`, as G0.6's is.
* G2.6 pass requires `builds_identical` is true and `external_urls == 0`.
* G2.7 pass requires `export_matches_params` and `jacobian_recompute_identical` both true.
* G2.8 pass requires `tokens_identical` true, `export_logit_max_abs_diff <= 1e-4` and `jacobian_relative_difference <= 1e-4`.
* G2.9 pass requires `jacobians_written == jacobians_expected` and `recompute_identical` true.
* G2.10 pass requires `planted_recovery >= 0.99`, `identity_exact` true, `random_max_accuracy <= 2 * random_chance` and `last_reading_kl_max <= 1e-5`.

Status rules for Phase 3:

* G3.1 pass requires `identities_exact == identities_checked`.
* G3.2 pass requires `recompute_identical == sample_size` and `cache_keys_distinct` true.
* G3.3's and G3.4's status is always `pass`: they record, not judge.
* G3.5 pass requires `builds_identical` true and `external_urls == 0`.

Status rules for Phase 4:

* G4.1 pass requires:
  * `recompute_identical` true
  * `checkpoints_written == checkpoints_scheduled`
  * `events_recorded == events_expected`
  * `induction_mechanism_accuracy >= 0.9`
  * `induction_control_accuracy` and `induction_twin_accuracy` each `<= 2 * induction_chance`
  * `two_hop_accuracy >= 0.9`
  * `two_hop_broken_accuracy` and `two_hop_twin_accuracy` each `<= 2 * two_hop_chance`
* G4.2 pass requires:
  * `best_head_attention >= 0.5`
  * `twin_best_attention <= 2 * uniform_baseline`
  * `ablation_fraction_removed >= 0.5`
  * `swap_fraction_removed >= 0.5`
  * `patch_after_block0_min >= 0.9`
  * `patch_after_block1_max <= 0.1`
* G4.3 pass requires `control_unchanged_min >= 0.95`. Its swap and transplant profiles are recorded, not judged (D36).
* G4.4's status is always `pass`: it records, not judges (D34).
* G4.5 pass requires `builds_identical` true and `external_urls == 0`.
* G4.6 pass requires `builds_identical` true and `subjects_missing == 0`.

Status rules for Phase 5:

* G5.1 pass requires `files_identical` true, `cache_keys_distinct` true, `identities_checked >= 1` and `identities_exact == identities_checked`.
* G5.2 pass requires:
  * `fvu_max <= 0.1`
  * `recovered_min >= 0.9`
  * `faithful_ratio_min >= 0.9`
  * `break_excess_max <= 0.05`
  * `selective_f1_min >= 0.5`
  * `selective_margin_min >= 0`
* G5.3 pass requires:
  * `fvu_max <= 0.1`
  * `recovered_min >= 0.9`
  * `faithful_ratio_min >= 0.9`
  * `control_unchanged_min >= 0.95`
  * `selective_f1_min >= 0.5`
  * `selective_margin_min >= 0`
* G5.4 pass requires `edge_ratio_max <= 0.2`.
* G5.5's status is always `pass`: it records, not judges (D44).
* G5.6 pass requires `builds_identical` true, `external_urls == 0`, `dictionaries_missing == 0` and `subjects_missing == 0`.

Measurements:

Null is allowed for anything a gate could not measure; every number must be finite. Values must be plain JSON types: convert numpy scalars with int(), float() or bool().

Required measurements for Phase 1:

* G1.1: `vocab_selected`, `unk_rate_4096`, `unk_rate_8192`, `unk_rate_4096_cased`, `builds_identical`, `train_tokens`, `validation_tokens`, `stories_dropped`
* G1.2: `n_params`, `n_params_expected`, `max_abs_logit_diff`, `top1_agreement`, `positions_near_tie`
* G1.3: the same six as G0.2
* G1.4: `steps`, `tokens`, `nonfinite_seen`, `checkpoints_written`, `checkpoints_scheduled`, `final_val_loss`, `bigram_val_loss`
* G1.5: `bitwise_equal`, `steps_checked`, `defect_detected`
* G1.6: `events_recorded`, `sample_size`, `sample_matches`, `retrace_new_events`, `store_bytes`, `store_bytes_expected`

Required measurements for Phase 2:

* G2.1: `logits_events`, `logits_events_expected`, `other_events_written`, `sample_size`, `sample_matches`, `logits_bytes`, `logits_bytes_expected`
* G2.2: `sample_size`, `sample_matches`, `builds_identical`
* G2.3: `planted_recovery`, `random_max_accuracy`, `random_chance`, `last_layer_kl_max`
* G2.4: `planted_aligned_credit`, `planted_perpendicular_quiet`, `decomposition_error_max`
* G2.5: `exact_positions`, `estimator_errors`, `fit_sizes`, `fit_change_per_doubling`, `projected_seconds`, `crosscheck_feasible`
* G2.6: `builds_identical`, `external_urls`, `export_bytes`
* G2.7: `step`, `layer`, `windows`, `export_matches_params`, `jacobian_recompute_identical`
* G2.8: `windows`, `tokens_identical`, `export_logit_max_abs_diff`, `jacobian_relative_difference`
* G2.9: `jacobians_written`, `jacobians_expected`, `recompute_identical`, `seconds`
* G2.10: `planted_recovery`, `identity_exact`, `random_max_accuracy`, `random_chance`, `last_reading_kl_max`, `comparison_by_step`

Required measurements for Phase 3:

* G3.1: `identities_checked`, `identities_exact`, `identities_by_kind`
* G3.2: `sample_size`, `recompute_identical`, `cache_keys_distinct`
* G3.3: `layers`, `scale_factors`, `results_by_layer`
* G3.4: `meters_by_step`
* G3.5: `builds_identical`, `external_urls`, `payloads`

Required measurements for Phase 4:

* G4.1: `recompute_identical`, `checkpoints_written`, `checkpoints_scheduled`, `events_recorded`, `events_expected`, `induction_mechanism_accuracy`, `induction_control_accuracy`, `induction_twin_accuracy`, `induction_chance`, `two_hop_accuracy`, `two_hop_broken_accuracy`, `two_hop_twin_accuracy`, `two_hop_chance`
* G4.2: `mechanism_accuracy`, `chance`, `attention_by_head`, `best_head_attention`, `induction_heads`, `twin_best_attention`, `uniform_baseline`, `ablation_fraction_removed`, `fraction_removed_by_head`, `swap_step`, `swap_fraction_removed`, `patch_followed_by_tap`, `patch_after_block0_min`, `patch_after_block1_max`
* G4.3: `unpatched_accuracy`, `swap_by_site`, `transplant_by_site`, `control_unchanged_by_site`, `control_unchanged_min`, `best_swap`
* G4.4: `lens_ranks`, `jswap_by_layer`, `attribution`
* G4.5: `formation_loss_step`, `formation_attention_step`, `builds_identical`, `external_urls`
* G4.6: `badges`, `subjects_missing`, `builds_identical`, `payloads`

Required measurements for Phase 5:

* G5.1: `files_identical`, `identical_by_kind`, `kinds_checked`, `cache_keys_checked`, `cache_keys_distinct`, `identities_checked`, `identities_exact`, `identities_by_site`
* G5.2: `meters`, `fvu_max`, `recovered_min`, `faithful`, `faithful_ratio_min`, `break_excess_max`, `selectivity`, `selective_f1_min`, `selective_margin_min`, `recovery`, `robustness`, `train_seconds`
* G5.3: `meters`, `fvu_max`, `recovered_min`, `faithful`, `faithful_ratio_min`, `control_unchanged_min`, `selectivity`, `selective_f1_min`, `selective_margin_min`, `best_swap_block`, `recovery`, `robustness`, `train_seconds`
* G5.4: `edge_pairs`, `judged_mlps`, `edge_ratio_max`, `block_pairs`, `jsae`
* G5.5: `meters`, `edge_pairs`, `train_seconds`
* G5.6: `badges`, `subjects_missing`, `dictionaries`, `dictionaries_missing`, `features`, `builds_identical`, `external_urls`

G0.4's `top1_agreement` is the fraction of positions whose top-1 token agrees, counted only over positions where the Levanter logits' top-2 margin exceeds 1e-3. `positions_near_tie` counts the positions left out.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
GATE_IDS: tuple[str, ...] = ("G0.1", "G0.2", "G0.3", "G0.4", "G0.5", "G0.6")
PHASE1_GATE_IDS: tuple[str, ...] = ("G1.1", "G1.2", "G1.3", "G1.4", "G1.5", "G1.6")
PHASE2_GATE_IDS: tuple[str, ...] = ("G2.1", "G2.2", "G2.3", "G2.4", "G2.5", "G2.6", "G2.7", "G2.8", "G2.9", "G2.10")
PHASE3_GATE_IDS: tuple[str, ...] = ("G3.1", "G3.2", "G3.3", "G3.4", "G3.5")
PHASE4_GATE_IDS: tuple[str, ...] = ("G4.1", "G4.2", "G4.3", "G4.4", "G4.5", "G4.6")
PHASE5_GATE_IDS: tuple[str, ...] = ("G5.1", "G5.2", "G5.3", "G5.4", "G5.5")
_ALL_GATE_IDS: tuple[str, ...] = (
    GATE_IDS + PHASE1_GATE_IDS + PHASE2_GATE_IDS + PHASE3_GATE_IDS + PHASE4_GATE_IDS + PHASE5_GATE_IDS
)
# Gates written from Phase 1 on, which must use schema version 2.
_V2_ONLY_GATE_IDS: tuple[str, ...] = (
    PHASE1_GATE_IDS + PHASE2_GATE_IDS + PHASE3_GATE_IDS + PHASE4_GATE_IDS + PHASE5_GATE_IDS
)
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
# Required in addition to REQUIRED_FIELDS from schema version 2 on.
V2_FIELDS: tuple[str, ...] = ("git_commit", "git_dirty")

REQUIRED_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "G0.1": ("steps", "eval_loss_initial", "eval_loss_final", "loss_drop_fraction", "nonfinite_seen"),
    "G0.2": ("hash_run_a", "hash_run_b", "hash_resumed", "identical", "max_abs_param_diff", "single_threaded"),
    "G0.3": ("tokens_per_s_S", "tokens_per_s_M"),
    "G0.4": ("max_abs_logit_diff", "top1_agreement", "positions_compared", "positions_near_tie"),
    "G0.5": ("runs", "declined_ops", "max_rel_loss_diff", "tokens_per_s"),
    "G0.6": ("bitwise_equal", "max_abs_logit_diff", "seeds"),
    "G1.1": (
        "vocab_selected", "unk_rate_4096", "unk_rate_8192", "unk_rate_4096_cased", "builds_identical",
        "train_tokens", "validation_tokens", "stories_dropped",
    ),
    "G1.2": ("n_params", "n_params_expected", "max_abs_logit_diff", "top1_agreement", "positions_near_tie"),
    "G1.3": ("hash_run_a", "hash_run_b", "hash_resumed", "identical", "max_abs_param_diff", "single_threaded"),
    "G1.4": (
        "steps", "tokens", "nonfinite_seen", "checkpoints_written", "checkpoints_scheduled", "final_val_loss",
        "bigram_val_loss",
    ),
    "G1.5": ("bitwise_equal", "steps_checked", "defect_detected"),
    "G1.6": (
        "events_recorded", "sample_size", "sample_matches", "retrace_new_events", "store_bytes",
        "store_bytes_expected",
    ),
    "G2.1": (
        "logits_events", "logits_events_expected", "other_events_written", "sample_size", "sample_matches",
        "logits_bytes", "logits_bytes_expected",
    ),
    "G2.2": ("sample_size", "sample_matches", "builds_identical"),
    "G2.3": ("planted_recovery", "random_max_accuracy", "random_chance", "last_layer_kl_max"),
    "G2.4": ("planted_aligned_credit", "planted_perpendicular_quiet", "decomposition_error_max"),
    "G2.5": (
        "exact_positions", "estimator_errors", "fit_sizes", "fit_change_per_doubling", "projected_seconds",
        "crosscheck_feasible",
    ),
    "G2.6": ("builds_identical", "external_urls", "export_bytes"),
    "G2.7": ("step", "layer", "windows", "export_matches_params", "jacobian_recompute_identical"),
    "G2.8": ("windows", "tokens_identical", "export_logit_max_abs_diff", "jacobian_relative_difference"),
    "G2.9": ("jacobians_written", "jacobians_expected", "recompute_identical", "seconds"),
    "G2.10": (
        "planted_recovery", "identity_exact", "random_max_accuracy", "random_chance", "last_reading_kl_max",
        "comparison_by_step",
    ),
    "G3.1": ("identities_checked", "identities_exact", "identities_by_kind"),
    "G3.2": ("sample_size", "recompute_identical", "cache_keys_distinct"),
    "G3.3": ("layers", "scale_factors", "results_by_layer"),
    "G3.4": ("meters_by_step",),
    "G3.5": ("builds_identical", "external_urls", "payloads"),
    "G4.1": (
        "recompute_identical", "checkpoints_written", "checkpoints_scheduled", "events_recorded", "events_expected",
        "induction_mechanism_accuracy", "induction_control_accuracy", "induction_twin_accuracy", "induction_chance",
        "two_hop_accuracy", "two_hop_broken_accuracy", "two_hop_twin_accuracy", "two_hop_chance",
    ),
    "G4.2": (
        "mechanism_accuracy", "chance", "attention_by_head", "best_head_attention", "induction_heads",
        "twin_best_attention", "uniform_baseline", "ablation_fraction_removed", "fraction_removed_by_head",
        "swap_step", "swap_fraction_removed", "patch_followed_by_tap", "patch_after_block0_min",
        "patch_after_block1_max",
    ),
    "G4.3": (
        "unpatched_accuracy", "swap_by_site", "transplant_by_site", "control_unchanged_by_site",
        "control_unchanged_min", "best_swap",
    ),
    "G4.4": ("lens_ranks", "jswap_by_layer", "attribution"),
    "G4.5": ("formation_loss_step", "formation_attention_step", "builds_identical", "external_urls"),
    "G4.6": ("badges", "subjects_missing", "builds_identical", "payloads"),
    "G5.1": (
        "files_identical", "identical_by_kind", "kinds_checked", "cache_keys_checked", "cache_keys_distinct",
        "identities_checked", "identities_exact", "identities_by_site",
    ),
    "G5.2": (
        "meters", "fvu_max", "recovered_min", "faithful", "faithful_ratio_min", "break_excess_max", "selectivity",
        "selective_f1_min", "selective_margin_min", "recovery", "robustness", "train_seconds",
    ),
    "G5.3": (
        "meters", "fvu_max", "recovered_min", "faithful", "faithful_ratio_min", "control_unchanged_min",
        "selectivity", "selective_f1_min", "selective_margin_min", "best_swap_block", "recovery", "robustness",
        "train_seconds",
    ),
    "G5.4": ("edge_pairs", "judged_mlps", "edge_ratio_max", "block_pairs", "jsae"),
    "G5.5": ("meters", "edge_pairs", "train_seconds"),
}
# Gates whose status is always "pass": their measurements set a policy rather than test one.
_ALWAYS_PASS_GATES: tuple[str, ...] = ("G0.6", "G2.5", "G3.3", "G3.4", "G4.4", "G5.5")
# Gates whose three final-state hashes must agree with ``identical`` under every status.
_HASH_GATES: tuple[str, ...] = ("G0.2", "G1.3")


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Every problem with ``report``, as human-readable strings. Empty means valid.

    Checks every rule in the module docstring. Never raises for a malformed
    report; a wrong type is reported as a problem. It does raise TypeError if
    ``report`` is not a Mapping at all.
    """
    if not isinstance(report, Mapping):
        raise TypeError(f"a report is a Mapping, got {type(report).__name__}")
    problems: list[str] = []

    version = report.get("schema_version")
    fields = REQUIRED_FIELDS + V2_FIELDS if _is_int(version) and version >= 2 else REQUIRED_FIELDS
    missing = [name for name in fields if name not in report]
    problems += [f"missing field {name}" for name in missing]
    extra = sorted(repr(key) for key in report if key not in fields)
    problems += [f"unexpected field {key}" for key in extra]

    def check(name: str, ok: Any, expected: str) -> None:
        if name in report and not ok(report[name]):
            problems.append(f"{name} must be {expected}, got {report[name]!r}")

    check("schema_version", lambda v: _is_int(v) and v in SCHEMA_VERSIONS, f"one of {SCHEMA_VERSIONS}")
    check("gate_id", lambda v: isinstance(v, str) and v in _ALL_GATE_IDS, f"one of {_ALL_GATE_IDS}")
    check("status", lambda v: isinstance(v, str) and v in STATUSES, f"one of {STATUSES}")
    check("env_name", lambda v: isinstance(v, str) and v in ENV_NAMES, f"one of {ENV_NAMES}")
    check("env_lock_sha256", lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None,
          "64 lowercase hex characters")
    check("xla_flags", lambda v: isinstance(v, str), "a str")
    check("thread_count", lambda v: _is_int(v) and v >= 1, "an int >= 1")
    check("chip", lambda v: isinstance(v, str) and v != "", "a non-empty str")
    check("timestamp_utc", _is_utc_timestamp, 'an ISO-8601 UTC timestamp ending in "Z"')
    if "git_commit" in fields:
        check("git_commit", lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v) is not None,
              "40 lowercase hex characters")
        check("git_dirty", lambda v: isinstance(v, bool), "a bool")

    gate_id, status = report.get("gate_id"), report.get("status")
    if isinstance(gate_id, str) and gate_id in _V2_ONLY_GATE_IDS and _is_int(version) and version < 2:
        phase = gate_id[1]
        problems.append(f"{gate_id} is a Phase {phase} gate and must use schema version 2, got {version!r}")

    measurements = report.get("measurements")
    if "measurements" in report:
        if not isinstance(measurements, Mapping):
            problems.append(f"measurements must be a JSON object, got {measurements!r}")
            measurements = None
        else:
            problems += _json_problems(measurements, "measurements")

    if isinstance(gate_id, str) and gate_id in _ALL_GATE_IDS and measurements is not None:
        problems += [
            f"{gate_id} measurements missing {key}" for key in REQUIRED_MEASUREMENTS[gate_id] if key not in measurements
        ]
        if gate_id in _HASH_GATES:
            problems += _hash_consistency_problems(gate_id, measurements)
        if isinstance(status, str) and status in STATUSES:
            problems += _status_problems(gate_id, status, measurements)
    return problems


def write_report(report: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> None:
    """Validate ``report`` and write it as indented JSON with sorted keys.

    Raises ValueError listing every problem if the report is invalid, and
    FileExistsError if ``path`` exists and ``overwrite`` is False. Writes nothing
    in either case.

    Creates missing parent folders.
    """
    problems = validate_report(report)
    if problems:
        raise ValueError("invalid gate report:\n" + "\n".join(f"  - {p}" for p in problems))
    text = json.dumps(_plain(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    """Anything that is not a plain JSON value, or a number that is not finite. Null is allowed."""
    # numpy scalars first: np.float64 subclasses float and would otherwise pass.
    if isinstance(value, np.generic):
        return [f"{where} is a numpy scalar, not a plain JSON value: {value!r} ({type(value).__name__})"]
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


def _hash_consistency_problems(gate_id: str, m: Mapping[str, Any]) -> list[str]:
    """G0.2 and G1.3, under every status: ``identical`` is True exactly when the three
    hashes are all equal. A null hash was not measured, so it never counts as equal."""
    hashes = [m.get(key) for key in ("hash_run_a", "hash_run_b", "hash_resumed")]
    all_equal = all(isinstance(h, str) for h in hashes) and len(set(hashes)) == 1
    if (m.get("identical") is True) != all_equal:
        return [
            f"{gate_id} identical must be True exactly when hash_run_a, hash_run_b and hash_resumed are all equal;"
            f" identical is {m.get('identical')!r}, hashes are {'all equal' if all_equal else 'not all equal'}"
        ]
    return []


def _status_problems(gate_id: str, status: str, m: Mapping[str, Any]) -> list[str]:
    """The STATUS RULES in the module docstring, for a known gate and status."""
    if gate_id in _ALWAYS_PASS_GATES:
        return [] if status == "pass" else [f'{gate_id} status is always "pass", got {status!r}']
    if status == "fail":
        return []
    if status == "conditional_pass" and gate_id not in _HASH_GATES:
        return [f"conditional_pass is allowed only for G0.2 and G1.3, not {gate_id}"]

    # Each rule is (holds, what it requires, what the report has).
    def is_(key: str, expected: bool) -> tuple[bool, str, str]:
        return m.get(key) is expected, f"{key} is {expected}", f"{m.get(key)!r}"

    def at_least(key: str, bound: float) -> tuple[bool, str, str]:
        v = m.get(key)
        return _is_number(v) and v >= bound, f"{key} >= {bound:g}", f"{v!r}"

    def at_most(key: str, bound: float) -> tuple[bool, str, str]:
        v = m.get(key)
        return _is_number(v) and v <= bound, f"{key} <= {bound:g}", f"{v!r}"

    def equals(key: str, target: float) -> tuple[bool, str, str]:
        v = m.get(key)
        return _is_number(v) and v == target, f"{key} == {target!r}", f"{v!r}"

    def equals_key(key: str, other: str) -> tuple[bool, str, str]:
        v, w = m.get(key), m.get(other)
        return _is_number(v) and _is_number(w) and v == w, f"{key} == {other}", f"{v!r} and {w!r}"

    def below_key(key: str, other: str) -> tuple[bool, str, str]:
        v, w = m.get(key), m.get(other)
        return _is_number(v) and _is_number(w) and v < w, f"{key} < {other}", f"{v!r} and {w!r}"

    def within_fraction_of(key: str, other: str, fraction: float) -> tuple[bool, str, str]:
        v, w = m.get(key), m.get(other)
        ok = _is_number(v) and _is_number(w) and abs(v - w) <= fraction * abs(w)
        return ok, f"{key} within {fraction:.0%} of {other}", f"{v!r} and {w!r}"

    def at_most_times_key(key: str, factor: float, other: str) -> tuple[bool, str, str]:
        v, w = m.get(key), m.get(other)
        ok = _is_number(v) and _is_number(w) and v <= factor * w
        return ok, f"{key} <= {factor:g} * {other}", f"{v!r} and {w!r}"

    def d8_vocabulary() -> tuple[bool, str, str]:
        v, rate = m.get("vocab_selected"), m.get("unk_rate_4096")
        ok = _is_number(v) and _is_number(rate) and v == (4096 if rate <= 0.01 else 8192)
        rule = "vocab_selected == 4096 when unk_rate_4096 <= 0.01, otherwise 8192 (D8)"
        return ok, rule, f"vocab_selected {v!r} with unk_rate_4096 {rate!r}"

    rules = {
        ("G0.1", "pass"): [
            is_("nonfinite_seen", False), at_least("loss_drop_fraction", 0.30), at_least("steps", 1000)
        ],
        ("G0.2", "pass"): [is_("identical", True)],
        ("G0.2", "conditional_pass"): [is_("identical", False), at_most("max_abs_param_diff", 1e-6)],
        ("G0.3", "pass"): [at_least("tokens_per_s_S", 25000)],
        ("G0.4", "pass"): [at_most("max_abs_logit_diff", 1e-3), equals("top1_agreement", 1.0)],
        ("G0.5", "pass"): [is_("runs", True), at_most("max_rel_loss_diff", 0.02)],
        ("G1.1", "pass"): [is_("builds_identical", True), d8_vocabulary()],
        ("G1.2", "pass"): [
            equals_key("n_params", "n_params_expected"), at_most("max_abs_logit_diff", 1e-4),
            equals("top1_agreement", 1.0),
        ],
        ("G1.3", "pass"): [is_("identical", True)],
        ("G1.3", "conditional_pass"): [is_("identical", False), at_most("max_abs_param_diff", 1e-6)],
        ("G1.4", "pass"): [
            is_("nonfinite_seen", False), equals_key("checkpoints_written", "checkpoints_scheduled"),
            below_key("final_val_loss", "bigram_val_loss"),
        ],
        ("G1.5", "pass"): [is_("bitwise_equal", True), is_("defect_detected", True)],
        ("G1.6", "pass"): [
            equals_key("sample_matches", "sample_size"), equals("retrace_new_events", 0),
            within_fraction_of("store_bytes", "store_bytes_expected", 0.10),
        ],
        ("G2.1", "pass"): [
            equals_key("logits_events", "logits_events_expected"), equals("other_events_written", 0),
            equals_key("sample_matches", "sample_size"),
            within_fraction_of("logits_bytes", "logits_bytes_expected", 0.01),
        ],
        ("G2.2", "pass"): [equals_key("sample_matches", "sample_size"), is_("builds_identical", True)],
        ("G2.3", "pass"): [
            at_least("planted_recovery", 0.99), at_most_times_key("random_max_accuracy", 2, "random_chance"),
            at_most("last_layer_kl_max", 1e-5),
        ],
        ("G2.4", "pass"): [
            at_least("planted_aligned_credit", 0.99), at_least("planted_perpendicular_quiet", 0.99),
            at_most("decomposition_error_max", 1e-3),
        ],
        ("G2.6", "pass"): [is_("builds_identical", True), equals("external_urls", 0)],
        ("G2.7", "pass"): [is_("export_matches_params", True), is_("jacobian_recompute_identical", True)],
        ("G2.8", "pass"): [
            is_("tokens_identical", True), at_most("export_logit_max_abs_diff", 1e-4),
            at_most("jacobian_relative_difference", 1e-4),
        ],
        ("G2.9", "pass"): [equals_key("jacobians_written", "jacobians_expected"), is_("recompute_identical", True)],
        ("G2.10", "pass"): [
            at_least("planted_recovery", 0.99), is_("identity_exact", True),
            at_most_times_key("random_max_accuracy", 2, "random_chance"), at_most("last_reading_kl_max", 1e-5),
        ],
        ("G3.1", "pass"): [equals_key("identities_exact", "identities_checked")],
        ("G3.2", "pass"): [equals_key("recompute_identical", "sample_size"), is_("cache_keys_distinct", True)],
        ("G3.5", "pass"): [is_("builds_identical", True), equals("external_urls", 0)],
        ("G4.1", "pass"): [
            is_("recompute_identical", True), equals_key("checkpoints_written", "checkpoints_scheduled"),
            equals_key("events_recorded", "events_expected"), at_least("induction_mechanism_accuracy", 0.9),
            at_most_times_key("induction_control_accuracy", 2, "induction_chance"),
            at_most_times_key("induction_twin_accuracy", 2, "induction_chance"),
            at_least("two_hop_accuracy", 0.9),
            at_most_times_key("two_hop_broken_accuracy", 2, "two_hop_chance"),
            at_most_times_key("two_hop_twin_accuracy", 2, "two_hop_chance"),
        ],
        ("G4.2", "pass"): [
            at_least("best_head_attention", 0.5), at_most_times_key("twin_best_attention", 2, "uniform_baseline"),
            at_least("ablation_fraction_removed", 0.5), at_least("swap_fraction_removed", 0.5),
            at_least("patch_after_block0_min", 0.9), at_most("patch_after_block1_max", 0.1),
        ],
        ("G4.3", "pass"): [at_least("control_unchanged_min", 0.95)],
        ("G4.5", "pass"): [is_("builds_identical", True), equals("external_urls", 0)],
        ("G4.6", "pass"): [is_("builds_identical", True), equals("subjects_missing", 0)],
        ("G5.1", "pass"): [
            is_("files_identical", True), is_("cache_keys_distinct", True), at_least("identities_checked", 1),
            equals_key("identities_exact", "identities_checked"),
        ],
        ("G5.2", "pass"): [
            at_most("fvu_max", 0.1), at_least("recovered_min", 0.9), at_least("faithful_ratio_min", 0.9),
            at_most("break_excess_max", 0.05), at_least("selective_f1_min", 0.5),
            at_least("selective_margin_min", 0),
        ],
        ("G5.3", "pass"): [
            at_most("fvu_max", 0.1), at_least("recovered_min", 0.9), at_least("faithful_ratio_min", 0.9),
            at_least("control_unchanged_min", 0.95), at_least("selective_f1_min", 0.5),
            at_least("selective_margin_min", 0),
        ],
        ("G5.4", "pass"): [at_most("edge_ratio_max", 0.2)],
    }[(gate_id, status)]
    return [f"{gate_id} {status} requires {rule}, got {got}" for ok, rule, got in rules if not ok]


def _plain(value: Any) -> Any:
    """Mappings as dicts, recursively, so any Mapping serializes as a JSON object."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
