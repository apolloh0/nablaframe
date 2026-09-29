"""Gate G0.2: Determinism.

Spec: docs/specs/phase-0.md, Gates table, G0.2 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: Same seed twice, plus a run killed at step 500 and resumed. SHA-256 over serialized parameter leaves at step 1,000.
Pass: All three hashes identical. If not, retry single-threaded (for example --xla_cpu_multi_thread_eigen=false). Still different: max abs parameter diff <= 1e-6 is conditional_pass with a pinned tolerance; larger is fail.
On failure the Phase 1 lane is PyTorch.

Writes reports/phase0/g02_determinism.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.2"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures (gates/_levanter.py has the details):
#
# * The G0.1 configuration and overrides, three times, each in its own process and
#   checkpoint directory: run A and run B train 1,000 steps from scratch; run C is
#   SIGKILLed once its step-500 checkpoint has committed, then a new process resumes
#   it from that checkpoint to step 1,000.
# * Each hash is SHA-256 over the final parameter leaves: pytree path, dtype, shape
#   and bytes of every leaf, in pytree order.
# * If the hashes differ, all three runs repeat with SINGLE_THREAD_FLAGS appended to
#   XLA_FLAGS, and the report describes that second attempt; the first is kept under
#   "attempts". The report's xla_flags field is the flags the reported runs used.
# * max_abs_param_diff is the largest difference in any parameter between any two of
#   the three runs, 0.0 when the hashes agree.
# * A run that fails for any reason means the gate could not measure: no report,
#   exit code 1.

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402
import _levanter  # noqa: E402

GATE_ID = "G0.2"
REPORT_PATH = "reports/phase0/g02_determinism.json"

ENV_NAME = "nf-jax"
WORK_DIR = _common.RUNS_DIR / "g02_determinism"
SINGLE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false"
MAX_CONDITIONAL_DIFF = 1e-6
VERSIONS = ("jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "optax", "numpy")


class RunFailed(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)
    work = _common.fresh_dir(WORK_DIR, overwrite=args.overwrite)

    base_flags = os.environ.get("XLA_FLAGS", "")
    try:
        attempts = [run_attempt(work / "default_threads", base_flags, single_threaded=False)]
        if not attempts[0]["identical"]:
            single_flags = f"{base_flags} {SINGLE_THREAD_FLAGS}".strip()
            attempts.append(run_attempt(work / "single_threaded", single_flags, single_threaded=True))
    except RunFailed as exc:
        print(f"{GATE_ID} could not measure: {exc}. No report written.", file=sys.stderr)
        return 1

    final = attempts[-1]
    measurements = {
        **{key: final[key] for key in ("hash_run_a", "hash_run_b", "hash_resumed", "identical")},
        "max_abs_param_diff": final["max_abs_param_diff"],
        "single_threaded": final["single_threaded"],
        "attempts": attempts,
        "overrides": dict(_levanter.OVERRIDES),
        "kill_step": _levanter.KILL_STEP,
        "versions": _common.package_versions(VERSIONS),
    }
    report = _common.build_report(GATE_ID, status(measurements), measurements, ENV_NAME, xla_flags=final["xla_flags"])
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def run_attempt(attempt_dir: Path, xla_flags: str, *, single_threaded: bool) -> dict[str, Any]:
    """Runs A, B and the killed-and-resumed C under ``xla_flags``; their hashes and largest parameter difference."""
    env = {**os.environ, "XLA_FLAGS": xla_flags}
    print(f"{GATE_ID}: three runs with XLA_FLAGS={xla_flags!r}", flush=True)

    def train(name: str, run_dir: Path, log_name: str = "train", **kwargs: Any) -> tuple[int, dict[str, Any] | None]:
        return _levanter.run_training(run_dir, args=_args(run_dir, name), env=env, name=log_name, **kwargs)

    hashes, params = {}, []
    for name in ("run_a", "run_b"):
        run_dir = attempt_dir / name
        code, result = train(name, run_dir, save_params=True)
        hashes[name] = _completed_hash(name, code, result, run_dir)
        params.append(_levanter.load_params(run_dir / "train.params.npz"))

    resumed_dir = attempt_dir / "resumed"
    code, _ = train("resumed", resumed_dir, kill_after_step=_levanter.KILL_STEP, log_name="killed")
    kill_checkpoint = _levanter.checkpoint_dir(resumed_dir, "resumed") / f"step-{_levanter.KILL_STEP}"
    if code != -signal.SIGKILL or not kill_checkpoint.exists():
        raise RunFailed(f"run C was to be killed at step {_levanter.KILL_STEP}, but exited with code {code}"
                        f" and {'a' if kill_checkpoint.exists() else 'no'} checkpoint at {kill_checkpoint}")
    code, resumed = train("resumed", resumed_dir, save_params=True, log_name="resumed")
    hashes["resumed"] = _completed_hash("resumed", code, resumed, resumed_dir, log="resumed")
    if resumed["start_updates"] != _levanter.KILL_STEP + 1:
        raise RunFailed(f"run C resumed from {resumed['start_updates']} updates, expected {_levanter.KILL_STEP + 1}")
    params.append(_levanter.load_params(resumed_dir / "resumed.params.npz"))

    identical = len(set(hashes.values())) == 1
    return {
        "xla_flags": xla_flags,
        "single_threaded": single_threaded,
        "hash_run_a": hashes["run_a"],
        "hash_run_b": hashes["run_b"],
        "hash_resumed": hashes["resumed"],
        "identical": identical,
        "max_abs_param_diff": 0.0 if identical else _levanter.max_abs_param_diff(params),
        "resumed_from_updates": resumed["start_updates"],
    }


def status(m: dict[str, Any]) -> str:
    """pass, conditional_pass or fail, exactly as reports.py's G0.2 rules allow."""
    if m["identical"] is True:
        return "pass"
    diff = m["max_abs_param_diff"]
    return "conditional_pass" if diff is not None and diff <= MAX_CONDITIONAL_DIFF else "fail"


def _args(run_dir: Path, run_id: str) -> list[str]:
    return _levanter.draccus_args(
        {**_levanter.OVERRIDES, **_levanter.run_overrides(run_dir, run_id, _common.DATA_CACHE_DIR)}
    )


def _completed_hash(name: str, code: int, result: dict[str, Any] | None, run_dir: Path, log: str = "train") -> str:
    if code != 0 or result is None or result["error"] is not None or result["param_sha256"] is None:
        detail = result["error"] if result and result["error"] else _levanter.log_tail(run_dir, log)
        raise RunFailed(f"{name} did not complete (exit code {code}):\n{detail}")
    if result["final_updates"] != _levanter.NUM_TRAIN_STEPS:
        raise RunFailed(f"{name} ended after {result['final_updates']} updates, not {_levanter.NUM_TRAIN_STEPS}")
    return result["param_sha256"]


if __name__ == "__main__":
    sys.exit(main())
