"""Gate G0.5: Metal probe.

Spec: docs/specs/phase-0.md, Gates table, G0.5 row.
Environment: nf-metal. Run on apolloh0's Mac only; never in the cloud.

Measures: The G0.1 config under metaljax: declined ops, loss vs CPU over the first 200 steps, tokens/s.
Pass: Runs, with loss within 2% relative of CPU at every logged step.
Never decides the lane; failure means no accelerator.

Writes reports/phase0/g05_metal_probe.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.5"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures (gates/_levanter.py has the details):
#
# * First a probe process with JAX_PLATFORMS=metal asks JAX for its Metal devices
#   and runs one small computation there. If metaljax is not installed or does not
#   load, the report says fail with runs false, nulls for everything unmeasured, and
#   the macOS version and the load error in measurements.
# * Otherwise one child process runs the G0.1 configuration and overrides (1,000
#   planned steps, so the learning-rate schedule matches G0.1) with
#   JAX_PLATFORMS=metal, and stops after COMPARE_STEPS updates. Levanter logs the
#   training loss at every step, so every one of steps 0 to 199 is compared with
#   G0.1's CPU loss at the same step, read from G0.1's report ("train_loss").
# * runs is true when all COMPARE_STEPS steps completed on the Metal backend.
# * metaljax declines unsupported ops at compile time, naming them, and nothing
#   falls back to CPU. So declined_ops is [] for a run that completed, the op names
#   (stablehlo.*) found in the error of a run that did not, and null when the error
#   names none.
# * tokens_per_s is batch x sequence length per step over the mean duration of steps
#   WARMUP_STEPS to 199, as Levanter times them (step only, not hooks).
# * An interrupted run (Ctrl-C) or a missing G0.1 report means the gate could not
#   measure: no report, exit code 1. A training process that dies without a result
#   (a crash inside the backend) is reported as runs false.

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402
import _levanter  # noqa: E402

GATE_ID = "G0.5"
REPORT_PATH = "reports/phase0/g05_metal_probe.json"

ENV_NAME = "nf-metal"
WORK_DIR = _common.RUNS_DIR / "g05_metal_probe"
G01_REPORT_PATH = "reports/phase0/g01_cpu_train.json"
COMPARE_STEPS = 200
WARMUP_STEPS = 10
MAX_REL_LOSS_DIFF = 0.02
RUN_ID = "g05_metal_probe"
VERSIONS = ("metaljax", "jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "optax", "numpy")

PROBE_SOURCE = """
import json, jax, jax.numpy as jnp
devices = jax.devices("metal")
value = float(jax.device_put(jnp.arange(4.0), devices[0]).sum())
print(json.dumps({"devices": [str(d) for d in devices], "default_backend": jax.default_backend(), "check": value}))
"""


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)

    base = {
        "macos_version": _common.macos_version(),
        "versions": _common.package_versions(VERSIONS),
        "matmul_precision": os.environ.get("METALJAX_MATMUL_PRECISION"),
    }
    load_error, probe = probe_metal()
    if load_error is not None:
        measurements = {**unmeasured(), "load_error": load_error, "probe": probe, **base}
        report = _common.build_report(GATE_ID, "fail", measurements, ENV_NAME)
        return _common.emit_report(report, report_file, overwrite=args.overwrite)

    g01 = _common.load_report(_common.report_path(G01_REPORT_PATH))
    cpu_losses = g01["measurements"].get("train_loss") or []
    if len(cpu_losses) < COMPARE_STEPS or any(loss is None for loss in cpu_losses[:COMPARE_STEPS]):
        print(f"{GATE_ID} needs G0.1's finite CPU training losses for steps 0 to {COMPARE_STEPS - 1}; its report"
              f" has {len(cpu_losses)}. No report written.", file=sys.stderr)
        return 1

    work = _common.fresh_dir(WORK_DIR, overwrite=args.overwrite)
    runtime = _levanter.run_overrides(work, RUN_ID, _common.DATA_CACHE_DIR)
    run_args = _levanter.draccus_args({**_levanter.OVERRIDES, **runtime})
    config = _levanter.parse_config(run_args)
    tokens_per_step = config.trainer.train_batch_size * (config.train_seq_len or config.model.max_seq_len)

    code, result = _levanter.run_training(
        work, args=run_args, env={**os.environ, "JAX_PLATFORMS": "metal"}, stop_after_updates=COMPARE_STEPS
    )
    interrupted = code == -signal.SIGINT or (result is not None and result["error_type"] == "KeyboardInterrupt")
    if interrupted:
        print(f"{GATE_ID} was interrupted. No report written.", file=sys.stderr)
        return 1
    if result is None:
        # The process died without writing its result, as a crash inside a native backend does.
        # Not running is what this gate measures, so it is reported, not treated as a gate error.
        result = _levanter.new_result()
        result["error"] = f"training process exited with code {code} and no result:\n{_levanter.log_tail(work)}"

    measurements = {
        **measure(result, cpu_losses, tokens_per_step),
        "load_error": None,
        "probe": probe,
        "g01_report_timestamp_utc": g01["timestamp_utc"],
        "overrides": dict(_levanter.OVERRIDES),
        **base,
    }
    report = _common.build_report(GATE_ID, status(measurements), measurements, ENV_NAME)
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def probe_metal() -> tuple[str | None, dict[str, Any] | None]:
    """(None, probe output) when metaljax loads and computes; (the load error, None) otherwise."""
    if _common.package_versions(["metaljax"])["metaljax"] is None:
        return "metaljax is not installed in this environment", None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE_SOURCE],
            env={**os.environ, "JAX_PLATFORMS": "metal"},
            capture_output=True, text=True, timeout=600, check=False,
        )
    except subprocess.TimeoutExpired:
        return "the metal probe did not finish within 600 s", None
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-20:])
        return f"exit code {proc.returncode}: {tail}", None
    probe = json.loads(proc.stdout.strip().splitlines()[-1])
    if probe["check"] != 6.0:
        return f"the metal probe computed sum(0, 1, 2, 3) = {probe['check']}", None
    return None, probe


def measure(result: dict[str, Any], cpu_losses: list[float], tokens_per_step: int) -> dict[str, Any]:
    """Measurements from a Metal training child's result and G0.1's CPU losses. Pure."""
    metal = {step: loss for step, loss in zip(result["steps"], result["losses"])}
    completed = [step for step in range(COMPARE_STEPS) if step in metal]
    on_metal = result["backend"] == "metal"
    runs = on_metal and result["error"] is None and len(completed) == COMPARE_STEPS

    rel_diffs = []
    for step in completed:
        cpu, gpu = cpu_losses[step], _common.finite_or_none(metal[step])
        rel_diffs.append(None if gpu is None or cpu == 0 else abs(gpu - cpu) / abs(cpu))
    max_rel = max(rel_diffs) if rel_diffs and None not in rel_diffs else None

    timed = result["step_seconds"][WARMUP_STEPS:COMPARE_STEPS]
    tokens_per_s = tokens_per_step * len(timed) / sum(timed) if on_metal and timed and sum(timed) > 0 else None

    if runs:
        declined: list[str] | None = []
    else:
        names = sorted(set(re.findall(r"stablehlo\.[A-Za-z_0-9]+", result["error"] or "")))
        declined = names or None
    return {
        "runs": runs,
        "declined_ops": declined,
        "max_rel_loss_diff": max_rel,
        "tokens_per_s": tokens_per_s,
        "steps_completed": len(completed),
        "steps_compared": len(completed) if max_rel is not None else 0,
        "backend": result["backend"],
        "devices": result["devices"],
        "metal_loss": [_common.finite_or_none(metal[step]) for step in completed],
        "error": result["error"],
    }


def unmeasured() -> dict[str, Any]:
    return {"runs": False, "declined_ops": None, "max_rel_loss_diff": None, "tokens_per_s": None}


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G0.5 rule holds."""
    diff = m["max_rel_loss_diff"]
    return "pass" if m["runs"] is True and diff is not None and diff <= MAX_REL_LOSS_DIFF else "fail"


if __name__ == "__main__":
    sys.exit(main())
