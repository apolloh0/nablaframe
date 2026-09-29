"""Gate G0.1: CPU training.

Spec: docs/specs/phase-0.md, Gates table, G0.1 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: Levanter's own tiny GPT-2 config trains on CPU for 1,000 steps.

Config: gates/configs/gpt2_nano.yaml, vendored from the Marin monorepo (the
marin-levanter wheel ships no configs). Train through levanter.main.train_lm with
these overrides applied in code, never by editing the YAML, and record them in
the report's measurements as "overrides":
  trainer.num_train_steps = 1000
  trainer.mp = float32 (the YAML says bfloat16; the reference lane is float32)
  step-based checkpoints including step 500, for G0.2
The data source is a Hugging Face dataset, so the first run needs network access.
Pass: Completes with no NaN or Inf; eval loss at step 1,000 is at least 30% below step 0.
On failure the Phase 1 lane is PyTorch.

Writes reports/phase0/g01_cpu_train.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.1"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures (gates/_levanter.py has the details):
#
# * One child process runs train_lm.main on the vendored YAML with OVERRIDES, plus
#   runtime-only overrides (run id, output paths, a tracker that needs no account),
#   recorded separately as "runtime_overrides".
# * "Eval loss at step 0" is Levanter's own eval callback run on the initial
#   parameters before the first update; "at step 1,000" is the same callback at the
#   end of training, after 1,000 updates.
# * nonfinite_seen covers every training loss, both eval losses and the final
#   parameters. Levanter stops on a NaN or Inf loss by default; that ends the run
#   and the gate reports fail with the steps it completed.
# * Any other error means the gate could not measure: no report, exit code 1.
# * The per-step training losses go in the report as "train_loss", indexed by step;
#   G0.5 compares its Metal losses against the first 200.
# * The checkpoint directory is recorded for G0.4, and kept under runs/phase0/.

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402
import _levanter  # noqa: E402

GATE_ID = "G0.1"
REPORT_PATH = "reports/phase0/g01_cpu_train.json"

ENV_NAME = "nf-jax"
WORK_DIR = _common.RUNS_DIR / "g01_cpu_train"
RUN_ID = "g01_cpu_train"
MIN_LOSS_DROP = 0.30
VERSIONS = ("jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "optax", "numpy")


def checkpoint_dir() -> Path:
    """Where G0.1 leaves its checkpoints; G0.4 exports the final one."""
    return _levanter.checkpoint_dir(WORK_DIR, RUN_ID)


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)
    work = _common.fresh_dir(WORK_DIR, overwrite=args.overwrite)

    runtime = _levanter.run_overrides(work, RUN_ID, _common.DATA_CACHE_DIR)
    started = time.monotonic()
    code, result = _levanter.run_training(
        work, args=_levanter.draccus_args({**_levanter.OVERRIDES, **runtime}), eval_initial=True
    )
    wall_seconds = time.monotonic() - started

    if result is None or (result["error"] is not None and not _nonfinite_loss_error(result)):
        print(f"{GATE_ID} could not measure (child exit code {code}). No report written.", file=sys.stderr)
        print(result["error"] if result and result["error"] else _levanter.log_tail(work), file=sys.stderr)
        return 1

    measurements = measure(result)
    measurements["runtime_overrides"] = {key: _relative(value) for key, value in runtime.items()}
    measurements["wall_seconds"] = round(wall_seconds, 1)
    report = _common.build_report(GATE_ID, status(measurements), measurements, ENV_NAME)
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def measure(result: dict[str, Any]) -> dict[str, Any]:
    """Measurements from a training child's result. Pure, so it can be checked without training."""
    if result["steps"] != list(range(len(result["steps"]))):
        raise ValueError(f"expected steps 0, 1, 2, ... from a fresh run, got {result['steps'][:5]}...")
    losses = [_common.finite_or_none(loss) for loss in result["losses"]]
    steps = result["final_updates"] if result["final_updates"] is not None else len(result["steps"])

    evals = {entry["updates"]: entry["loss"] for entry in result["eval"]}
    initial, final = evals.get(0), evals.get(_levanter.NUM_TRAIN_STEPS)
    nonfinite_seen = (
        _nonfinite_loss_error(result)
        or any(loss is None for loss in losses)
        or any(loss is not None and not math.isfinite(loss) for loss in (initial, final))
        or bool(result["params_nonfinite"])
    )
    initial, final = _common.finite_or_none(initial), _common.finite_or_none(final)
    drop = (initial - final) / initial if initial is not None and final is not None and initial > 0 else None

    return {
        "steps": int(steps),
        "eval_loss_initial": initial,
        "eval_loss_final": final,
        "loss_drop_fraction": drop,
        "nonfinite_seen": nonfinite_seen,
        "overrides": dict(_levanter.OVERRIDES),
        "config": _common.rel(_levanter.CONFIG_PATH),
        "config_sha256": hashlib.sha256(_levanter.CONFIG_PATH.read_bytes()).hexdigest(),
        "train_loss": losses,
        "train_seconds": round(sum(result["step_seconds"]), 3),
        "backend": result["backend"],
        "checkpoint_dir": _common.rel(checkpoint_dir()),
        "error": _last_line(result["error"]),
        "versions": _common.package_versions(VERSIONS),
    }


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G0.1 rule holds."""
    drop = m["loss_drop_fraction"]
    ok = m["nonfinite_seen"] is False and m["steps"] >= _levanter.NUM_TRAIN_STEPS and drop is not None
    return "pass" if ok and drop >= MIN_LOSS_DROP else "fail"


def _nonfinite_loss_error(result: dict[str, Any]) -> bool:
    """Levanter's crash_on_nan and crash_on_inf raise RuntimeError("Loss is NaN") or ("Loss is Inf")."""
    error = result.get("error") or ""
    return result.get("error_type") == "RuntimeError" and ("Loss is NaN" in error or "Loss is Inf" in error)


def _last_line(text: str | None) -> str | None:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-1].strip() if lines else None


def _relative(value: Any) -> Any:
    return _common.rel(Path(value)) if isinstance(value, str) and value.startswith("/") else value


if __name__ == "__main__":
    sys.exit(main())
