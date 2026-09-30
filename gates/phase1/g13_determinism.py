"""Gate G1.3: Trainer determinism.

Spec: docs/specs/phase-1.md, Gates table, G1.3 row. Report: reports/phase1/g13_determinism.json.

Size S at the selected vocabulary, 600 steps, with ``total_steps=600`` so the
schedule completes:

* Runs A and B from scratch.
* Run C in a child process, SIGKILLed once its step-300 resume checkpoint is on
  disk, then resumed from that checkpoint in a new process.
* Hashes each final state with ``state_hash``. If they differ, repeats all three
  with ``--xla_cpu_multi_thread_eigen=false`` and records both attempts.

Progress: a ``Heartbeat`` for each run; child processes write logs under
``runs/phase1/g13_determinism/`` and the gate prints its ``watch_hint`` first.
Measurements: REQUIRED_MEASUREMENTS["G1.3"], plus both attempts if there was a retry.


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
# * Every run is a child process, ``g13_determinism.py --child SPEC``, so that
#   XLA_FLAGS takes effect before JAX starts and run C can be killed for real. Its
#   output, including its heartbeat, goes to ``<run>/<name>.log``; the gate echoes
#   the newest log line to the terminal every 30 seconds.
# * Each run trains with ``run_until`` from ``init_state`` on ``train.bin``, saving a
#   resume checkpoint every KILL_STEP updates. Run C's child SIGKILLs itself in its
#   heartbeat hook right after the step-300 checkpoint is written and synced, the
#   same way G0.2 killed Levanter; a new child resumes from that checkpoint.
# * Each hash is ``state_hash`` of the final TrainState (params, optimizer state and
#   step) as loaded from the run's step-600 checkpoint, whose loader checks it
#   against the hash the child wrote beside it.
# * max_abs_param_diff is the largest difference in any parameter between any two
#   of the three runs, 0.0 when the hashes agree.
# * If the hashes differ, all three runs repeat with SINGLE_THREAD_FLAGS appended to
#   XLA_FLAGS and the report describes that attempt; both are kept under
#   "attempts", and the report's xla_flags field is the flags the reported runs used.
# * A run that fails for any reason means the gate could not measure: no report,
#   exit code 1.

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

GATE_ID = "G1.3"
REPORT_PATH = "reports/phase1/g13_determinism.json"

STEPS = 600
KILL_STEP = 300
SINGLE_THREAD_FLAGS = "--xla_cpu_multi_thread_eigen=false"
MAX_CONDITIONAL_DIFF = 1e-6
ECHO_EVERY_S = 30.0
VERSIONS = ("jax", "jaxlib", "equinox", "optax", "marin-haliax", "safetensors", "numpy")


class RunFailed(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = _common.arg_parser(__doc__)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child is not None:
        return child(args.child)

    report = _common.start(args, REPORT_PATH)
    _common.require_data()
    tokenizer = _common.load_tokenizer()
    model_config = _common.model_config(len(tokenizer.vocab))
    work = _common.fresh_dir(_common.RUN_DIR / "g13_determinism", overwrite=args.overwrite)
    _common.watch_hint(work)
    try:
        measurements = measure(model_config, train_config(), _common.split_path("train"), work, kill_step=KILL_STEP)
    except RunFailed as exc:
        print(f"{GATE_ID} could not measure: {exc}\nNo report written.", file=sys.stderr)
        return 1
    measurements.update(tokenizer_sha256=tokenizer.sha256, versions=_common.phase0.package_versions(VERSIONS))
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite,
                         xla_flags=measurements["xla_flags"])


def train_config() -> Any:
    """TrainConfig's defaults with ``total_steps = 600``, so the schedule completes."""
    from nablaframe.train.loop import TrainConfig

    return TrainConfig(total_steps=STEPS)


def status(m: dict[str, Any]) -> str:
    """pass, conditional_pass or fail, exactly as reports.py's G1.3 rules allow."""
    if m["identical"] is True:
        return "pass"
    diff = m["max_abs_param_diff"]
    return "conditional_pass" if diff is not None and diff <= MAX_CONDITIONAL_DIFF else "fail"


def measure(model_config: Any, train_config: Any, train_bin: Path, work: Path, *,
            kill_step: int = KILL_STEP) -> dict[str, Any]:
    base_flags = os.environ.get("XLA_FLAGS", "")
    attempts = [run_attempt(work / "default_threads", model_config, train_config, train_bin, kill_step, base_flags,
                            single_threaded=False)]
    if not attempts[0]["identical"]:
        flags = f"{base_flags} {SINGLE_THREAD_FLAGS}".strip()
        attempts.append(run_attempt(work / "single_threaded", model_config, train_config, train_bin, kill_step,
                                    flags, single_threaded=True))
    final = attempts[-1]
    return {
        **{key: final[key] for key in ("hash_run_a", "hash_run_b", "hash_resumed", "identical",
                                       "max_abs_param_diff", "single_threaded", "xla_flags")},
        "attempts": attempts,
        "steps": train_config.total_steps,
        "kill_step": kill_step,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }


def run_attempt(attempt_dir: Path, model_config: Any, train_config: Any, train_bin: Path, kill_step: int,
                xla_flags: str, *, single_threaded: bool) -> dict[str, Any]:
    """Runs A, B and the killed-and-resumed C under ``xla_flags``: their hashes and largest parameter difference."""
    from nablaframe.train.checkpoint import load_checkpoint, state_hash
    from nablaframe.train.loop import init_state

    print(f"{GATE_ID}: three runs with XLA_FLAGS={xla_flags!r}", flush=True)
    env = {**os.environ, "XLA_FLAGS": xla_flags}
    steps = train_config.total_steps

    def spec(run_dir: Path, name: str, **extra: Any) -> dict[str, Any]:
        return {"model_config": asdict(model_config), "train_config": asdict(train_config),
                "train_bin": str(train_bin), "checkpoint_dir": str(run_dir / "resume"), "save_every": kill_step,
                "until_step": steps, "kill_at": None, "resume": False,
                "result_path": str(run_dir / f"{name}.result.json"), **extra}

    for name in ("run_a", "run_b"):
        code, result = run_child(attempt_dir / name, "train", spec(attempt_dir / name, "train"), env)
        _check_completed(name, code, result, steps)

    resumed = attempt_dir / "resumed"
    code, _ = run_child(resumed, "killed", spec(resumed, "killed", kill_at=kill_step), env)
    kill_checkpoint = resumed / "resume" / f"step-{kill_step}.safetensors"
    if code != -signal.SIGKILL or not kill_checkpoint.exists():
        raise RunFailed(f"run C was to be killed after its step-{kill_step} checkpoint, but exited with code {code}"
                        f" and {'a' if kill_checkpoint.exists() else 'no'} checkpoint at {_common.rel(kill_checkpoint)}"
                        f"\n{_tail(resumed / 'killed.log')}")
    code, result = run_child(resumed, "resumed", spec(resumed, "resumed", resume=True), env)
    _check_completed("resumed", code, result, steps)
    if result["start_step"] != kill_step:
        raise RunFailed(f"run C resumed from step {result['start_step']}, expected {kill_step}")

    like = init_state(model_config, train_config)
    hashes, params = {}, []
    for name in ("run_a", "run_b", "resumed"):
        final = attempt_dir / name / "resume" / f"step-{steps}.safetensors"
        try:
            state = load_checkpoint(final, like=like)
        except (OSError, ValueError) as exc:
            raise RunFailed(f"{name}'s final checkpoint {_common.rel(final)} did not load: {exc}") from exc
        hashes[name] = state_hash(state)
        params.append(state.params)

    identical = len(set(hashes.values())) == 1
    return {
        "xla_flags": xla_flags,
        "single_threaded": single_threaded,
        "hash_run_a": hashes["run_a"],
        "hash_run_b": hashes["run_b"],
        "hash_resumed": hashes["resumed"],
        "identical": identical,
        "max_abs_param_diff": 0.0 if identical else max_abs_param_diff(params),
        "resumed_from_step": result["start_step"],
    }


def max_abs_param_diff(trees: list[Any]) -> float:
    """Largest absolute difference in any parameter between any two trees."""
    import jax
    import numpy as np

    flat = [{jax.tree_util.keystr(path): np.asarray(leaf, dtype=np.float64)
             for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]} for tree in trees]
    worst = 0.0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i].keys() != flat[j].keys():
                raise RunFailed("runs have different parameter trees")
            for key, a in flat[i].items():
                b = flat[j][key]
                if a.shape != b.shape:
                    raise RunFailed(f"parameter {key} differs in shape between runs")
                if a.size:
                    worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def run_child(run_dir: Path, name: str, spec: dict[str, Any], env: dict[str, str]) -> tuple[int, dict[str, Any] | None]:
    """Run one training child; its output goes to ``run_dir/<name>.log``. Returns its exit code and result."""
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / f"{name}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    result_path = Path(spec["result_path"])
    result_path.unlink(missing_ok=True)
    log_path = run_dir / f"{name}.log"
    print(f"  {run_dir.name}/{name}: training, log in {_common.rel(log_path)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", str(spec_path)],
                                cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        last_echo = time.monotonic()
        while proc.poll() is None:
            time.sleep(1.0)
            if time.monotonic() - last_echo >= ECHO_EVERY_S:
                line = _tail(log_path, 1)
                print(f"  {run_dir.name}/{name}: {line or 'starting'}", flush=True)
                last_echo = time.monotonic()
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    return proc.returncode, result


def _check_completed(name: str, code: int, result: dict[str, Any] | None, steps: int) -> None:
    if code != 0 or result is None or result["error"] is not None:
        detail = result["error"] if result and result["error"] else ""
        raise RunFailed(f"{name} did not complete (exit code {code}):\n{detail}")
    if result["final_step"] != steps:
        raise RunFailed(f"{name} ended at step {result['final_step']}, not {steps}")


def _tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


class _KillAfter:
    """Heartbeat hook: forwards every tick, and SIGKILLs this process once ``step`` is reached.

    ``run_until`` saves a resume checkpoint, synced, before it ticks, so the
    checkpoint for ``step`` is on disk when the process dies.
    """

    def __init__(self, heartbeat: Any, step: int | None) -> None:
        self.heartbeat = heartbeat
        self.step = step

    def tick(self, *, step: int, loss: float, tokens: int) -> None:
        self.heartbeat.tick(step=step, loss=loss, tokens=tokens)
        if self.step is not None and step == self.step:
            print(f"step {step:,}: checkpoint written; killing this process with SIGKILL", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)


def child(spec_path: Path) -> int:
    import traceback

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"start_step": None, "final_step": None, "error": None}
    code = 0
    try:
        import numpy as np

        from nablaframe.model.gpt2 import GPT2Config
        from nablaframe.train.checkpoint import load_checkpoint
        from nablaframe.train.heartbeat import Heartbeat
        from nablaframe.train.loop import TrainConfig, init_state, run_until

        model_config = GPT2Config(**spec["model_config"])
        train_config = TrainConfig(**{**spec["train_config"], "betas": tuple(spec["train_config"]["betas"])})
        stream = np.memmap(spec["train_bin"], dtype="<u2", mode="r")
        state = init_state(model_config, train_config)
        checkpoint_dir = Path(spec["checkpoint_dir"])
        if spec["resume"]:
            latest = _common.latest_checkpoint(checkpoint_dir)
            if latest is None:
                raise RuntimeError(f"no resume checkpoint in {checkpoint_dir}")
            state = load_checkpoint(latest, like=state)
        result["start_step"] = int(state.step)
        print(f"training from step {state.step} to {spec['until_step']}", flush=True)
        heartbeat = _KillAfter(Heartbeat(train_config.total_steps, write=lambda line: print(line, flush=True)),
                               spec["kill_at"])
        state = run_until(state, stream, model_config, train_config, until_step=spec["until_step"],
                          checkpoint_dir=checkpoint_dir, save_every=spec["save_every"], heartbeat=heartbeat)
        result["final_step"] = int(state.step)
        print(f"done at step {state.step}", flush=True)
    except BaseException:  # noqa: BLE001  every failure is recorded for the gate to judge
        result["error"] = traceback.format_exc()[-8000:]
        print(result["error"], flush=True)
        code = 1
    Path(spec["result_path"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
