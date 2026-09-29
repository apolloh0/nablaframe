"""Levanter training runs for gates G0.1, G0.2 and G0.5, and checkpoint helpers for G0.4. Not a gate.

Environment: nf-jax or nf-metal for everything that imports Levanter. Importing
this module imports nothing beyond the standard library and numpy, so the gates
that use it import cleanly anywhere.

Each training run happens in a child process, ``python gates/_levanter.py SPEC``,
so that ``XLA_FLAGS`` and ``JAX_PLATFORMS`` take effect before JAX starts, a
crash in a backend cannot take the gate down with it, and a run can be killed
for real. The child trains through ``levanter.main.train_lm.main`` on the
vendored config with the overrides given as draccus arguments, never by editing
the YAML. It patches three Levanter functions for the duration of the run, to
observe it and nothing else:

* ``Trainer.train_step``: records every step's training loss and duration. It
  can also SIGKILL the process once the checkpoint for a given step has
  committed (G0.2), or stop cleanly after a given number of updates (G0.5).
* ``Trainer.train``: evaluates the initial parameters with Levanter's own eval
  callback before the first update (G0.1's "eval loss at step 0"; Levanter's
  hooks never fire before a step), and hashes the final parameters.
* ``levanter.eval.cb_tagged_lm_evaluate``: keeps a reference to the eval
  callback ``train_lm`` registers, so the initial evaluation is the same code
  as the final one.

Step numbering. Levanter's ``StepInfo.step`` is the 0-based index of the step
just completed, so training for 1,000 steps runs steps 0 to 999 and leaves
parameters after 1,000 updates. Losses here are indexed the same way. Levanter
names checkpoints by that index: ``step-500`` holds the state after 501
updates, and ``step-999`` the final state after 1,000.

The child writes a result JSON (see ``new_result``) whatever happens, except
when it kills itself on purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "gates" / "configs" / "gpt2_nano.yaml"

# G0.1's overrides, per its docstring. G0.2 and G0.5 run the same configuration.
# These are the only ones that can change what the model computes.
NUM_TRAIN_STEPS = 1000
KILL_STEP = 500
OVERRIDES: dict[str, Any] = {
    "trainer.num_train_steps": NUM_TRAIN_STEPS,
    "trainer.mp": "float32",
    # Permanent step-based checkpoints at step 500 (G0.2 resumes from it) and at the end.
    # The YAML's every-50 policy would also include 500 but keeps twenty checkpoints.
    "trainer.checkpointer.keep": [{"every": KILL_STEP}],
}

# Where a run writes, and how it logs. None of these touches the numerics.
# wandb is Levanter's default tracker; it would need an account, so metrics go to the Python log instead.
TRACKER = {"type": "json_logger"}

EVAL_LOSS_KEY = "eval/loss"


def run_overrides(work_dir: Path, run_id: str, data_cache_dir: Path) -> dict[str, Any]:
    """Runtime-only overrides: run id, output paths and tracker. Values are absolute paths."""
    return {
        "trainer.id": run_id,
        "trainer.checkpointer.base_path": str(work_dir / "checkpoints"),
        "trainer.log_dir": str(work_dir / "logs"),
        "trainer.tracker": TRACKER,
        "data.cache_dir": str(data_cache_dir),
    }


def draccus_args(overrides: dict[str, Any]) -> list[str]:
    """``--key value`` pairs; non-string values as JSON, which draccus parses."""
    args: list[str] = []
    for key, value in overrides.items():
        args += [f"--{key}", value if isinstance(value, str) else json.dumps(value)]
    return args


def checkpoint_dir(work_dir: Path, run_id: str) -> Path:
    """Levanter appends the run id to the checkpointer's base path."""
    return work_dir / "checkpoints" / run_id


def run_training(
    work_dir: Path,
    *,
    args: list[str],
    config_path: Path = CONFIG_PATH,
    eval_initial: bool = False,
    kill_after_step: int | None = None,
    stop_after_updates: int | None = None,
    save_params: bool = False,
    env: dict[str, str] | None = None,
    python: str = sys.executable,
    name: str = "train",
) -> tuple[int, dict[str, Any] | None]:
    """Run one training child in ``work_dir``. Returns its exit code and its result, or None if it wrote none.

    Output goes to ``work_dir/<name>.log``. ``env`` replaces the child's environment when given.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "config_path": str(config_path),
        "args": args,
        "result_path": str(work_dir / f"{name}.result.json"),
        "params_path": str(work_dir / f"{name}.params.npz") if save_params else None,
        "eval_initial": eval_initial,
        "kill_after_step": kill_after_step,
        "stop_after_updates": stop_after_updates,
    }
    spec_path = work_dir / f"{name}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    result_path = Path(spec["result_path"])
    result_path.unlink(missing_ok=True)

    log_path = work_dir / f"{name}.log"
    print(f"  {name}: training, log in {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [python, str(Path(__file__).resolve()), str(spec_path)],
            cwd=work_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    return proc.returncode, result


def log_tail(work_dir: Path, name: str = "train", lines: int = 30) -> str:
    path = work_dir / f"{name}.log"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def load_params(path: Path) -> dict[str, np.ndarray]:
    """Parameters saved by a child with ``save_params``, keyed by their pytree path."""
    with np.load(path) as data:
        names = json.loads(str(data["__names__"]))
        return {name: data[f"a{i}"] for i, name in enumerate(names)}


def max_abs_param_diff(runs: list[dict[str, np.ndarray]]) -> float:
    """Largest absolute difference in any parameter between any two runs. Raises if the trees differ."""
    worst = 0.0
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            if a.keys() != b.keys():
                raise ValueError("runs have different parameter trees")
            for name in a:
                if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
                    raise ValueError(f"parameter {name} differs in shape or dtype between runs")
                if a[name].size:
                    diff = np.max(np.abs(a[name].astype(np.float64) - b[name].astype(np.float64)))
                    worst = max(worst, float(diff))
    return worst


def param_digest(model: Any) -> tuple[str, dict[str, np.ndarray]]:
    """SHA-256 over the model's array leaves, and the leaves themselves.

    Leaves are taken in pytree order. Each contributes its path, dtype (with byte
    order), shape and C-order bytes, so equal hashes mean equal parameters.
    """
    import equinox as eqx
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))
    digest = hashlib.sha256()
    arrays: dict[str, np.ndarray] = {}
    for path, leaf in leaves:
        name = jax.tree_util.keystr(path)
        array = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode() + b"\0")
        digest.update(array.tobytes())
        arrays[name] = array
    return digest.hexdigest(), arrays


def parse_config(args: list[str], config_path: Path = CONFIG_PATH) -> Any:
    """The vendored YAML plus draccus arguments, as Levanter's TrainLmConfig."""
    import draccus
    import levanter.config  # noqa: F401  registers Levanter's draccus codecs (jmp policies, timedeltas)
    from levanter.main.train_lm import TrainLmConfig

    return draccus.parse(TrainLmConfig, config_path=str(config_path), args=args)


def load_trained_model(config: Any, checkpoint_path: str) -> Any:
    """The model subtree of a Levanter checkpoint, built the way ``levanter.main.export_lm_to_hf`` builds it."""
    import equinox as eqx
    import jax
    from haliax import Axis
    from levanter.checkpoint import load_checkpoint
    from levanter.utils.jax_utils import is_inexact_arrayish

    vocab = Axis("vocab", len(config.data.the_tokenizer))
    shape = eqx.filter_eval_shape(config.model.build, vocab, key=jax.random.PRNGKey(0))
    trainable, non_trainable = eqx.partition(shape, is_inexact_arrayish)
    trainable = load_checkpoint(trainable, checkpoint_path, subpath="model")
    return eqx.combine(trainable, non_trainable)


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


class _StopTraining(Exception):
    """Raised from inside the training loop to end a run after a set number of updates."""


def new_result() -> dict[str, Any]:
    return {
        "backend": None,  # jax.default_backend() inside the run
        "devices": [],
        "start_updates": None,  # updates already in the state training began from (0, or a resumed checkpoint's)
        "steps": [],  # Levanter step index of each step this process ran
        "losses": [],  # training loss of each of those steps
        "step_seconds": [],  # Levanter's step duration, before hooks
        "eval": [],  # [{"updates": int, "loss": float}] from Levanter's eval_metrics.jsonl
        "final_updates": None,
        "param_sha256": None,
        "params_nonfinite": None,
        "stopped_early": False,
        "error": None,  # traceback text when the run raised
        "error_type": None,
    }


def _child(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result = new_result()
    result_path = Path(spec["result_path"])

    def dump() -> None:
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        tmp.replace(result_path)

    code = 0
    try:
        config = parse_config(spec["args"], Path(spec["config_path"]))
        _instrumented_train(config, spec, result, dump)
    except _StopTraining:
        result["stopped_early"] = True
    except BaseException as exc:  # noqa: BLE001  every failure is recorded for the gate to judge
        result["error"] = traceback.format_exc()[-8000:]
        result["error_type"] = type(exc).__name__
        code = 1
    _read_eval_metrics(spec, result)
    dump()
    return code


def _instrumented_train(config: Any, spec: dict[str, Any], result: dict[str, Any], dump) -> None:
    import jax
    import levanter.eval
    import levanter.trainer
    from levanter.callbacks import StepInfo
    from levanter.main import train_lm

    trainer_cls = levanter.trainer.Trainer
    original_step = trainer_cls.train_step
    original_train = trainer_cls.train
    original_eval_cb = levanter.eval.cb_tagged_lm_evaluate
    eval_callbacks: list[Any] = []
    kill_after = spec.get("kill_after_step")
    stop_after = spec.get("stop_after_updates")

    def cb_tagged_lm_evaluate(*args: Any, **kwargs: Any) -> Any:
        callback = original_eval_cb(*args, **kwargs)
        eval_callbacks.append(callback)
        return callback

    def train_step(self: Any, state: Any, *batch: Any, **batch_kwargs: Any) -> Any:
        info = original_step(self, state, *batch, **batch_kwargs)
        result["steps"].append(int(info.step))
        result["losses"].append(float(info.loss))
        result["step_seconds"].append(float(info.step_duration))
        if kill_after is not None and info.step == kill_after:
            # The checkpoint hook ran inside train_step; wait for the write to commit, then die
            # the way a killed process dies: no cleanup, no final hooks, no final checkpoint.
            self._checkpointer.wait_until_finished()
            dump()
            os.kill(os.getpid(), signal.SIGKILL)
        if stop_after is not None and info.next_step >= stop_after:
            raise _StopTraining
        return info

    def train(self: Any, state: Any, train_loader: Any) -> Any:
        result["backend"] = jax.default_backend()
        result["devices"] = [str(d) for d in jax.devices()]
        result["start_updates"] = int(state.step)
        if spec.get("eval_initial"):
            if not eval_callbacks:
                raise RuntimeError("train_lm registered no eval callback, so there is no eval loss to record")
            for callback in eval_callbacks:
                callback(StepInfo(state, 0.0, 0.0))
        info = original_train(self, state, train_loader)
        result["final_updates"] = int(info.state.step)
        digest, arrays = param_digest(info.state.model)
        result["param_sha256"] = digest
        result["params_nonfinite"] = not all(
            bool(np.all(np.isfinite(a))) for a in arrays.values() if np.issubdtype(a.dtype, np.inexact)
        )
        if spec.get("params_path"):
            names = list(arrays)
            np.savez(
                spec["params_path"],
                __names__=np.array(json.dumps(names)),
                **{f"a{i}": arrays[name] for i, name in enumerate(names)},
            )
        return info

    trainer_cls.train_step = train_step
    trainer_cls.train = train
    levanter.eval.cb_tagged_lm_evaluate = cb_tagged_lm_evaluate
    try:
        train_lm.main(config)
    finally:
        trainer_cls.train_step = original_step
        trainer_cls.train = original_train
        levanter.eval.cb_tagged_lm_evaluate = original_eval_cb


def _read_eval_metrics(spec: dict[str, Any], result: dict[str, Any]) -> None:
    """Eval losses from the file Levanter's eval callback writes next to the checkpoints.

    Its "step" is StepInfo.step, the index of the step just completed, so updates = step + 1:
    -1 is the initial evaluation, 999 the final one.
    """
    try:
        config = parse_config(spec["args"], Path(spec["config_path"]))
        path = Path(config.trainer.checkpointer.expanded_path(config.trainer.id)) / "eval_metrics.jsonl"
    except Exception:  # noqa: BLE001  a config that did not parse has already been recorded as the error
        return
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            loss = record.get(EVAL_LOSS_KEY)
            result["eval"].append({"updates": int(record["step"]) + 1, "loss": None if loss is None else float(loss)})


if __name__ == "__main__":
    sys.exit(_child(sys.argv[1]))
