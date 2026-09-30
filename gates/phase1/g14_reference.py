"""Gate G1.4: Reference run.

Spec: docs/specs/phase-1.md, Gates table, G1.4 row. Report: reports/phase1/g14_reference.json.

Trains the reference model: size S, the selected vocabulary, ``TrainConfig()``
defaults (12,208 steps, 100M tokens), data seed 0, parameter seed 0.

* Writes ``runs/phase1/g14_reference/manifest.json`` first: the canonical JSON of a
  ``RunManifest`` with the real corpus and tokenizer hashes, git state, lockfile
  hash, ``XLA_FLAGS`` and ``checkpoint_schedule = trace_steps(12_208)``. Its
  ``run_id`` goes into every checkpoint sidecar.
* Trace checkpoints (params only) at every step in the schedule, under
  ``runs/phase1/g14_reference/trace/``; resume checkpoints every 500 steps under
  ``runs/phase1/g14_reference/resume/``. The two never share a folder, since both
  use the name ``step-<n>.safetensors``.
* Validation loss on ``eval_windows(validation, EVAL_TOKENS, 256)`` every 500
  steps and at every trace checkpoint; the curve is recorded.
* ``bigram_val_loss``: the cross-entropy of an add-one-smoothed bigram model
  counted on ``train.bin``, evaluated on the same windows' next-token targets.
* ``--resume`` continues from the latest resume checkpoint after an interruption.

Progress: a ``Heartbeat`` throughout. Expect about 1.8 to 2.3 hours.
Measurements: REQUIRED_MEASUREMENTS["G1.4"], plus the validation-loss curve and
``run_id``.


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
# * Training runs in segments with ``run_until``, which writes a resume checkpoint
#   whenever the step reaches a multiple of SAVE_EVERY. At each segment's end the
#   gate writes the trace checkpoint if that step is scheduled, then evaluates if
#   the step is a multiple of EVAL_EVERY or scheduled. Step 0 gets both before
#   the first update.
# * Validation loss is the mean next-token cross-entropy over every position of
#   every evaluation window, computed with ``GPT2Model.fast``.
# * ``runs/phase1/g14_reference/progress.json`` keeps the validation curve, the
#   steps where a loss was not finite, and the steps the run resumed from, so a
#   resumed run reports the whole run. A resume keeps what was measured before
#   the resume checkpoint and redoes the rest; batch k depends only on the seed
#   and k, so the redone steps are the same updates.
# * ``--resume`` requires the manifest on disk to have the same run_id this checkout
#   and data would give: a change of commit, lockfile, data or XLA_FLAGS is a
#   different run, and must start over with --overwrite. It resumes from the
#   latest resume checkpoint whose sidecar names that run, or from step 0 if the
#   run was interrupted before its first one.
# * nonfinite_seen is true if any training or validation loss was NaN or infinite.
#   Training stops at the next checkpoint once one is, and the report says fail.
# * checkpoints_written counts the scheduled trace checkpoints on disk, complete
#   and naming this run. The bigram model counts every adjacent pair of train.bin,
#   with P(b | a) = (count(a, b) + 1) / (count(a) + vocab).

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

GATE_ID = "G1.4"
REPORT_PATH = "reports/phase1/g14_reference.json"

TOTAL_STEPS = 12_208
SAVE_EVERY = 500
EVAL_EVERY = 500
EVAL_BATCH = 32
CORPUS_ID = "tinystories-v2"
BIGRAM_CHUNK = 1 << 25
VERSIONS = ("jax", "jaxlib", "equinox", "optax", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    parser = _common.arg_parser(__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its latest resume checkpoint")
    args = parser.parse_args(argv)
    report = _common.start(args, REPORT_PATH)
    _common.require_data()

    from nablaframe.train.loop import TrainConfig

    tokenizer = _common.load_tokenizer()
    config = _common.model_config(len(tokenizer.vocab))
    train_config = TrainConfig()
    if train_config.total_steps != TOTAL_STEPS:
        raise SystemExit(f"TrainConfig() runs {train_config.total_steps:,} steps; the spec's reference run is"
                         f" {TOTAL_STEPS:,}. No report written.")
    manifest = build_manifest(config, train_config, tokenizer.sha256, _common.corpus_sha256())
    work = _common.g14_dir()
    if args.resume:
        manifest = check_resumable(work, manifest)
    else:
        if work.exists() and not args.overwrite:
            raise SystemExit(f"{_common.rel(work)} holds an earlier run. Pass --resume to continue it, or --overwrite"
                             " to start over.")
        _common.fresh_dir(work, overwrite=args.overwrite)
        write_manifest(work, manifest)
    print(f"{GATE_ID}: run {manifest.run_id}, manifest in {_common.rel(work / 'manifest.json')}", flush=True)

    measurements = run(manifest, config, train_config, _common.load_split("train"),
                       _common.load_split("validation"), work, resume=args.resume, save_every=SAVE_EVERY,
                       eval_every=EVAL_EVERY)
    measurements.update(tokenizer_sha256=tokenizer.sha256, versions=_common.phase0.package_versions(VERSIONS))
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G1.4 rule holds."""
    final, bigram = m["final_val_loss"], m["bigram_val_loss"]
    ok = (m["nonfinite_seen"] is False and m["checkpoints_written"] == m["checkpoints_scheduled"]
          and final is not None and bigram is not None and final < bigram)
    return "pass" if ok else "fail"


def build_manifest(config: Any, train_config: Any, tokenizer_sha256: str, corpus_sha256: str) -> Any:
    """The run's RunManifest; its run_id covers everything but ``created_at``."""
    import jax

    from nablaframe.trace.schema import RunManifest, Seeds

    commit, dirty = _common.git_state()
    optimizer = {
        "name": "adamw",
        "schedule": "linear warmup from lr / warmup_steps to lr, then cosine decay to lr * min_lr_ratio (D9)",
        **{key: list(value) if isinstance(value, tuple) else value for key, value in asdict(train_config).items()},
    }
    return RunManifest(
        model_config=asdict(config),
        optimizer=optimizer,
        seeds=Seeds(param_init=train_config.param_seed, data_order=train_config.data_seed, dropout=0),
        corpus_id=CORPUS_ID,
        corpus_sha256=corpus_sha256,
        tokenizer_id=f"words-{config.vocab}",
        tokenizer_sha256=tokenizer_sha256,
        git_commit=commit,
        git_dirty=dirty,
        env_lock_sha256=_common.phase0.env_lock_sha256(),
        backend=jax.default_backend(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        checkpoint_schedule=_common.trace_steps(train_config.total_steps),
        created_at=_common.phase0.utc_timestamp(),
    )


def write_manifest(work: Path, manifest: Any) -> None:
    _common.write_atomic(work / "manifest.json", manifest.to_json().encode("utf-8"))


def check_resumable(work: Path, manifest: Any) -> Any:
    """The manifest on disk, if it describes the run this checkout would start. Raises SystemExit otherwise."""
    from nablaframe.trace.schema import RunManifest

    path = work / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{_common.rel(path)} is missing, so there is no run to resume. Run without --resume.")
    existing = RunManifest.from_json(path.read_text(encoding="utf-8"))
    if existing.run_id != manifest.run_id:
        old, new = existing.to_dict(), manifest.to_dict()
        changed = sorted(key for key in new if key != "created_at" and old.get(key) != new[key])
        raise SystemExit(f"The run in {_common.rel(path)} is {existing.run_id}; this checkout and data would start"
                         f" {manifest.run_id} (differs in: {', '.join(changed)}). Resuming would mix two runs."
                         " Pass --overwrite without --resume to start over.")
    return existing


def run(manifest: Any, config: Any, train_config: Any, train: np.ndarray, validation: np.ndarray, work: Path, *,
        resume: bool = False, save_every: int = SAVE_EVERY, eval_every: int = EVAL_EVERY) -> dict[str, Any]:
    """Train, checkpoint and evaluate the run ``manifest`` describes; returns the measurements."""
    import jax
    import optax

    from nablaframe.data.corpus import eval_windows
    from nablaframe.model.gpt2 import GPT2Model
    from nablaframe.train.checkpoint import load_checkpoint, save_params
    from nablaframe.train.heartbeat import Heartbeat
    from nablaframe.train.loop import init_state, run_until

    run_id, total = manifest.run_id, train_config.total_steps
    schedule = tuple(manifest.checkpoint_schedule)
    trace_dir, resume_dir = work / "trace", work / "resume"
    eval_steps = set(range(0, total + 1, eval_every)) | set(schedule)
    boundaries = sorted(eval_steps | set(range(save_every, total + 1, save_every)) | {total})

    model = GPT2Model(config)
    windows = eval_windows(np.asarray(validation), _common.EVAL_TOKENS, config.seq)
    if len(windows) == 0:
        raise SystemExit("the validation stream is too short for one evaluation window. No report written.")
    loss_sum = jax.jit(lambda params, w: optax.softmax_cross_entropy_with_integer_labels(
        model.fast(params, w[:, :-1]), w[:, 1:]).sum())

    def evaluate(params: Any) -> float:
        total_loss = 0.0
        for i in range(0, len(windows), EVAL_BATCH):
            total_loss += float(loss_sum(params, windows[i:i + EVAL_BATCH]))
        return total_loss / (len(windows) * config.seq)

    bigram = bigram_val_loss(train, windows, config.vocab)
    print(f"{GATE_ID}: add-one bigram validation loss {bigram:.4f}", flush=True)

    progress_path = work / "progress.json"
    state = init_state(config, train_config)
    progress = {"run_id": run_id, "val_loss": [], "nonfinite_steps": [], "resumed_from": []}
    if resume:
        latest = _common.latest_checkpoint(resume_dir, run_id)
        if latest is not None:
            state = load_checkpoint(latest, like=state)
        if progress_path.exists():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("run_id") == run_id:
                progress = saved
        # Everything after the resume point is redone, so only what came before it is kept.
        progress["val_loss"] = [e for e in progress["val_loss"] if e["step"] < state.step]
        progress["nonfinite_steps"] = [s for s in progress["nonfinite_steps"] if s < state.step]
        progress["resumed_from"].append(int(state.step))
        print(f"{GATE_ID}: resuming at step {state.step:,}", flush=True)

    def save_progress() -> None:
        _common.write_atomic(progress_path, (json.dumps(progress, indent=2) + "\n").encode("utf-8"))

    def at_boundary(step: int) -> None:
        if step in schedule:
            save_params(state.params, trace_dir, step=step, run_id=run_id)
        if step in eval_steps:
            loss = evaluate(state.params)
            progress["val_loss"].append({"step": step, "loss": loss if math.isfinite(loss) else None})
            if not math.isfinite(loss):
                progress["nonfinite_steps"].append(step)
            print(f"{GATE_ID}: step {step:,} validation loss {loss:.4f}", flush=True)
        save_progress()

    watch = _NonfiniteWatch(Heartbeat(total), progress["nonfinite_steps"])
    at_boundary(state.step)
    for boundary in boundaries:
        if boundary <= state.step:
            continue
        if progress["nonfinite_steps"]:
            print(f"{GATE_ID}: a loss was not finite at step {progress['nonfinite_steps'][0]:,}; stopping",
                  flush=True)
            break
        state = run_until(state, train, config, train_config, until_step=boundary, checkpoint_dir=resume_dir,
                          save_every=save_every, heartbeat=watch, run_id=run_id)
        at_boundary(state.step)

    curve = sorted(progress["val_loss"], key=lambda e: e["step"])
    final = next((e["loss"] for e in curve if e["step"] == total), None)
    written = sum(1 for step in schedule if _complete(trace_dir / f"step-{step}.safetensors", run_id))
    resume_scheduled = list(range(save_every, total + 1, save_every))
    return {
        "steps": int(state.step),
        "tokens": int(state.step) * train_config.batch * train_config.seq,
        "nonfinite_seen": bool(progress["nonfinite_steps"]),
        "checkpoints_written": written,
        "checkpoints_scheduled": len(schedule),
        "final_val_loss": final,
        "bigram_val_loss": bigram,
        "run_id": run_id,
        "val_loss_curve": curve,
        "first_nonfinite_step": min(progress["nonfinite_steps"]) if progress["nonfinite_steps"] else None,
        "resumed_from": progress["resumed_from"],
        "resume_checkpoints_written": sum(
            1 for step in resume_scheduled if _complete(resume_dir / f"step-{step}.safetensors", run_id)),
        "resume_checkpoints_scheduled": len(resume_scheduled),
        "checkpoint_schedule": list(schedule),
        "eval_windows": int(len(windows)),
        "eval_tokens": _common.EVAL_TOKENS,
        "model_config": asdict(config),
        "train_config": asdict(train_config),
        "manifest": _common.rel(work / "manifest.json"),
    }


class _NonfiniteWatch:
    """Heartbeat hook: forwards every tick, and notes each step whose training loss is not finite."""

    def __init__(self, heartbeat: Any, nonfinite_steps: list[int]) -> None:
        self.heartbeat = heartbeat
        self.nonfinite_steps = nonfinite_steps

    def tick(self, *, step: int, loss: float, tokens: int) -> None:
        if not math.isfinite(loss):
            self.nonfinite_steps.append(step)
        self.heartbeat.tick(step=step, loss=loss, tokens=tokens)


def _complete(path: Path, run_id: str) -> bool:
    sidecar = path.with_suffix(".json")
    return path.exists() and sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get(
        "run_id") == run_id


def bigram_val_loss(train: np.ndarray, windows: np.ndarray, vocab: int) -> float:
    """Cross-entropy of an add-one-smoothed bigram model counted on ``train``, on the windows' next-token targets."""
    counts = np.zeros(vocab * vocab, dtype=np.int64)
    pairs = max(len(train) - 1, 0)
    progress = _common.Progress("counting bigrams", pairs)
    for start in range(0, pairs, BIGRAM_CHUNK):
        chunk = np.asarray(train[start:start + BIGRAM_CHUNK + 1], dtype=np.int64)
        counts += np.bincount(chunk[:-1] * vocab + chunk[1:], minlength=vocab * vocab)
        progress.advance(len(chunk) - 1)
    counts = counts.reshape(vocab, vocab)
    row = counts.sum(axis=1)
    previous, target = windows[:, :-1].ravel(), windows[:, 1:].ravel()
    probability = (counts[previous, target] + 1) / (row[previous] + vocab)
    return float(-np.mean(np.log(probability)))


if __name__ == "__main__":
    sys.exit(main())
