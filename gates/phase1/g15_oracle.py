"""Gate G1.5: Oracle on the real model.

Spec: docs/specs/phase-1.md, Gates table, G1.5 row. Report: reports/phase1/g15_oracle.json.

Loads G1.4's trace checkpoints at steps 0, 6,000 and 12,208 and runs
``check_mode_equality`` on the probe batch with every tap, under the bitwise
policy (D6). ``bitwise_equal`` is whether all three pass.

``defect_detected``: whether ``compare_logits`` under the bitwise policy fails
for the step-12,208 params when ``instrumented`` is given
``interventions={"blocks.0.hook_resid_post": lambda x: x + 1e-3}``.

Measurements: REQUIRED_MEASUREMENTS["G1.5"], plus each step's oracle result.


Every gate: runs on the Mac only, in nf-jax
(``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/<script>``);
refuses a dirty tree unless ``--allow-dirty``; refuses to overwrite its report unless
``--overwrite``; writes one schema 2 report with ``write_report``, measurements
passed through ``plain``; prints progress at least every 30 s during anything
longer than a minute; returns exit code 0 whenever it wrote a valid report,
whatever the gate's status, and non-zero only when it could not produce one.
"""

# How it measures:
#
# * The model is built from G1.4's manifest, and each checkpoint is loaded only
#   after its sidecar is checked to name G1.4's run (``_common.trace_checkpoint``).
#   The probe batch is the one G1.6 traces.
# * Each step's oracle result is recorded as the oracle returned it: passed,
#   max_abs_diff (null when not finite), n_mismatched and reason.
# * The defect check compares fast logits with instrumented logits under the
#   intervention, with every tap recorded, through ``compare_logits`` under the
#   bitwise policy. defect_detected is whether that comparison fails.
# * This gate enforces the policy D6 set and changes nothing in the oracle.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

GATE_ID = "G1.5"
REPORT_PATH = "reports/phase1/g15_oracle.json"

POLICY = "bitwise"  # D6
STEPS = (0, 6000, 12_208)
DEFECT_TAP = "blocks.0.hook_resid_post"
DEFECT_SHIFT = 1e-3
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    report = _common.start(args, REPORT_PATH)
    _common.require_data()
    manifest = _common.load_g14_manifest()
    measurements = measure(manifest, _common.load_split("validation"), steps=STEPS)
    measurements["versions"] = _common.phase0.package_versions(VERSIONS)
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G1.5 rule holds."""
    return "pass" if m["bitwise_equal"] is True and m["defect_detected"] is True else "fail"


def measure(manifest: Any, validation: np.ndarray, steps: tuple[int, ...] = STEPS) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.oracle.mode_equality import check_mode_equality, compare_logits
    from nablaframe.trace.registry import tap_names
    from nablaframe.train.checkpoint import load_params

    missing = [step for step in steps if step not in manifest.checkpoint_schedule]
    if missing:
        raise SystemExit(f"steps {missing} are not in G1.4's trace schedule. No report written.")
    config = _common.config_from_manifest(manifest)
    model = GPT2Model(config)
    taps = tap_names(config.n_layers)
    probe = _common.probe(validation, config.seq)
    like = init_params(config, 0)

    per_step, params = [], None
    for step in steps:
        params = load_params(_common.trace_checkpoint(manifest, step), like=like)
        result = check_mode_equality(model, params, probe, taps, POLICY)
        per_step.append({"step": step, **_result(result)})
        print(f"{GATE_ID}: step {step:,}: {'bitwise equal' if result.passed else result.reason}", flush=True)

    # The defect: the same instrumented pass with 1e-3 added at one tap. ``params`` is the last step's.
    fast = np.asarray(model.fast(params, probe))
    defective, _ = model.instrumented(params, probe, taps, interventions={DEFECT_TAP: _shift})
    defect = compare_logits(fast, np.asarray(defective), POLICY)
    print(f"{GATE_ID}: defect at {DEFECT_TAP}, step {steps[-1]:,}:"
          f" {'caught: ' + defect.reason if not defect.passed else 'NOT caught'}", flush=True)
    return {
        "bitwise_equal": all(entry["passed"] for entry in per_step),
        "steps_checked": list(steps),
        "defect_detected": not defect.passed,
        "per_step": per_step,
        "defect": {"step": steps[-1], "tap": DEFECT_TAP, "shift": DEFECT_SHIFT, **_result(defect)},
        "policy": POLICY,
        "taps": len(taps),
        "run_id": manifest.run_id,
        "probe_batch_hash": batch_hash(probe),
        "probe_shape": list(probe.shape),
    }


def _shift(x: Any) -> Any:
    return x + DEFECT_SHIFT


def _result(result: Any) -> dict[str, Any]:
    return {"passed": bool(result.passed), "max_abs_diff": _common.finite_or_none(result.max_abs_diff),
            "n_mismatched": int(result.n_mismatched), "reason": result.reason}


if __name__ == "__main__":
    sys.exit(main())
