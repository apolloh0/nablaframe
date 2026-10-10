"""G4.1: both zoo models trained, traced and retrained to a seeded recompute, and held to their behaviour bars (D29, D30, D36).

Spec: docs/specs/phase-4.md, "The zoo" and "Gates"; helpers in ``_common.py``. Report:
``reports/phase4/g41_zoo_training.json``. Run in nf-jax, on the Mac.

For ``"induction"`` and then ``"two_hop"``, each with its task from ``zoo/tasks.py`` at its
defaults and its settings from ``configs(name)``:

* **Train:** from step 0 with ``run_until(..., batch_source=lambda k: task.batch_for_step(0, k,
  batch))``, saving a checkpoint at every ``checkpoint_steps(name)`` step under
  ``runs/phase4/<name>/checkpoints/``, beside a run manifest and the task's tokenizer file, so the
  trace engine and the views take the run as they take any (D29). The manifest's ``corpus_id`` is
  ``"zoo:<name>"`` and its ``corpus_sha256`` the SHA-256 of the task's canonical JSON, its class
  name and fields; its tokenizer fields name the tokenizer file and its SHA-256.
* **Trace:** every checkpoint, with every canonical tap and the logits, on what the model reads of
  ``probe(task)``, its first ``seq`` tokens, through ``TraceRun`` into
  ``runs/phase4/<name>/store/``, as G1.6 traces the reference run.
* **Recompute:** trains again from step 0 into a scratch folder under ``runs/phase4/scratch/`` and
  compares the final state hash with the first run's.
* **Behaviour,** from logits alone at the final checkpoint, with ``next_token_accuracy``:
  induction on ``eval_windows`` at its mechanism and its control positions, and its
  random-weights twin (``twin_params``) at the mechanism positions; two-hop on ``eval_windows``
  and on ``broken_windows`` at position 25, and its twin on ``eval_windows``.

Measurements: REQUIRED_MEASUREMENTS["G4.1"]. ``recompute_identical`` is true when both final
hashes match; ``checkpoints_scheduled`` is 62; ``events_expected`` is each model's canonical taps
plus the logits, times its checkpoints, 1,576. Also recorded: each model's final state hash,
mechanism accuracy at every checkpoint on the evaluation windows, and the seconds each stage
took. Pass needs every bar.

Run folders start fresh: an existing one is removed only with ``--overwrite``. ``--only
induction`` or ``--only two_hop`` runs one model and writes no report, so one model can be
exercised alone. If the trace engine's oracle refuses a checkpoint, or a recompute isn't
reproducible, the report fails and records the error, as G1.6's does.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g41_zoo_training.py``. Takes --overwrite, --allow-dirty and --only. With
#   --only it still runs on the Mac only and refuses a dirty tree, but never looks at or writes
#   the report.
# * Every folder a model needs, its run folder and its scratch folder, is checked before
#   anything trains, so the second model never stops on a folder that can't be started.
# * Training: ``init_state`` at the model's ``TrainConfig``, then ``run_until`` from one
#   checkpoint step to the next with the batch source, a ``Heartbeat`` printing progress. Each
#   scheduled step's params go to ``checkpoints/`` with ``save_params`` under the manifest's
#   ``run_id``; step 0 is saved before the first update. No resume checkpoints are written.
# * The manifest is G1.4's, for a zoo task: ``corpus_id`` ``"zoo:<name>"``, ``corpus_sha256``
#   the SHA-256 of ``canonical_json({"class": <the task's class name>, "fields": <its fields>})``,
#   ``tokenizer_id`` ``"tokenizer.json"`` (the file beside the manifest) and
#   ``tokenizer_sha256`` that file's SHA-256, with ``checkpoint_schedule``
#   ``checkpoint_steps(name)``. The tokenizer file holds exactly ``task.tokenizer_json()``.
# * Mechanism accuracy at every checkpoint: the checkpoint's params, as just saved, on
#   ``eval_windows`` at the task's mechanism positions. Every pass in this gate, those included,
#   is ``run_intervened`` with an empty spec, the unintervened pass.
# * The trace: one ``TraceRun`` on ``store/`` with the manifest, the checkpoints folder and the
#   probe's first ``seq`` tokens, recording every canonical tap and the logits at each step, as
#   G1.6 records. OracleFailure or NotReproducible stops that model's trace: its ``error`` and
#   ``error_stage`` (``"trace"``) are recorded and the report fails, since
#   ``events_recorded`` falls short. A provenance ValueError from the engine is a setup error:
#   exit 1, no report.
# * The recompute: the same training from step 0 in ``runs/phase4/scratch/<name>/``, saving
#   only its final state there with ``save_checkpoint``. Each run's final state hash is
#   ``state_hash`` of its final ``TrainState`` (params, optimizer state and step), as G1.3 hashes.
# * Behaviour: on the final checkpoint as loaded from disk, with ``next_token_accuracy`` on the
#   full windows. The twin is ``twin_params(config)``.
# * ``checkpoints_written`` counts scheduled checkpoints on disk, complete and naming their
#   run; ``events_recorded`` counts the events both traces wrote. ``run_ids`` names each run,
#   which G4.2 to G4.5 check their run folders against.

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.1"
REPORT_NAME = _common.G41_REPORT
VERSIONS = ("jax", "jaxlib", "equinox", "optax", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    parser = p1.arg_parser(__doc__)
    parser.add_argument("--only", choices=_common.NAMES,
                        help="train, trace and measure one zoo model, and write no report")
    args = parser.parse_args(argv)
    if args.only:
        p1.phase0.require_macos()
        p1.require_clean_tree(args.allow_dirty)
        report = None
    else:
        report = p1.start(args, _common.report_relative(REPORT_NAME))
    names = (args.only,) if args.only else _common.NAMES
    for name in names:
        for folder in (_common.run_dir(name), _common.scratch_dir() / name):
            if folder.exists() and not args.overwrite:
                raise SystemExit(f"{p1.rel(folder)} exists. Pass --overwrite to start it over.")
    results = {name: measure_model(name, overwrite=args.overwrite) for name in names}
    if report is None:
        print(f"{GATE_ID}: --only {args.only}: no report written. {_line(args.only, results[args.only])}",
              flush=True)
        return 0
    measurements = combine(results)
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G4.1 rule holds."""
    def number(key: str) -> bool:
        return isinstance(m[key], (int, float)) and not isinstance(m[key], bool)

    ok = (m["recompute_identical"] is True and m["checkpoints_written"] == m["checkpoints_scheduled"]
          and m["events_recorded"] == m["events_expected"]
          and number("induction_mechanism_accuracy") and m["induction_mechanism_accuracy"] >= _common.BARS["behaviour"]
          and number("induction_control_accuracy") and m["induction_control_accuracy"] <= 2 * m["induction_chance"]
          and number("induction_twin_accuracy") and m["induction_twin_accuracy"] <= 2 * m["induction_chance"]
          and number("two_hop_accuracy") and m["two_hop_accuracy"] >= _common.BARS["behaviour"]
          and number("two_hop_broken_accuracy") and m["two_hop_broken_accuracy"] <= 2 * m["two_hop_chance"]
          and number("two_hop_twin_accuracy") and m["two_hop_twin_accuracy"] <= 2 * m["two_hop_chance"])
    return "pass" if ok else "fail"


def combine(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The report's measurements from both models' results."""
    ind, two = results["induction"], results["two_hop"]
    return {
        "recompute_identical": all(r["recompute_identical"] is True for r in results.values()),
        "checkpoints_written": sum(r["checkpoints_written"] for r in results.values()),
        "checkpoints_scheduled": sum(r["checkpoints_scheduled"] for r in results.values()),
        "events_recorded": sum(r["events_recorded"] for r in results.values()),
        "events_expected": sum(r["events_expected"] for r in results.values()),
        "induction_mechanism_accuracy": ind["behaviour"]["mechanism_accuracy"],
        "induction_control_accuracy": ind["behaviour"]["control_accuracy"],
        "induction_twin_accuracy": ind["behaviour"]["twin_accuracy"],
        "induction_chance": ind["chance"],
        "two_hop_accuracy": two["behaviour"]["accuracy"],
        "two_hop_broken_accuracy": two["behaviour"]["broken_accuracy"],
        "two_hop_twin_accuracy": two["behaviour"]["twin_accuracy"],
        "two_hop_chance": two["chance"],
        "bars": _common.BARS,
        "run_ids": {name: r["run_id"] for name, r in results.items()},
        "models": results,
    }


def measure_model(name: str, *, overwrite: bool) -> dict[str, Any]:
    """Train, trace, retrain and measure one zoo model; returns its results."""
    from nablaframe.train.checkpoint import save_checkpoint, state_hash

    task = _common.task(name)
    config, train_config = _common.configs(name)
    steps = _common.checkpoint_steps(name)
    folder = _common.fresh_dir(_common.run_dir(name), overwrite=overwrite)
    tokenizer = folder / "tokenizer.json"
    p1.write_atomic(tokenizer, task.tokenizer_json().encode("utf-8"))
    manifest = build_manifest(name, task, config, train_config, p1.sha256_file(tokenizer))
    p1.write_atomic(folder / "manifest.json", manifest.to_json().encode("utf-8"))
    print(f"{GATE_ID}: {name}: run {manifest.run_id}, in {p1.rel(folder)}", flush=True)
    seconds: dict[str, float] = {}

    started = time.monotonic()
    curve: list[dict[str, Any]] = []
    windows = _common.eval_windows(task)

    def measure_checkpoint(step: int, params: Any) -> None:
        curve.append({"step": step, "accuracy": accuracy(config, params, task, windows, task.mechanism_positions)})

    state = train(task, config, train_config, steps, run_id=manifest.run_id, checkpoint_dir=folder / "checkpoints",
                  on_checkpoint=measure_checkpoint, label=name)
    seconds["train"] = time.monotonic() - started
    final_hash = state_hash(state)
    print(f"{GATE_ID}: {name}: trained to step {state.step:,}, final state {final_hash[:12]}, mechanism accuracy"
          f" {curve[-1]['accuracy']:.4f}", flush=True)

    started = time.monotonic()
    traced = trace(name, manifest, config, task, folder, steps)
    seconds["trace"] = time.monotonic() - started

    started = time.monotonic()
    scratch = _common.fresh_dir(_common.scratch_dir() / name, overwrite=overwrite)
    again = train(task, config, train_config, steps, run_id=manifest.run_id, label=f"{name}, recompute")
    save_checkpoint(again, scratch, run_id=manifest.run_id)
    recompute_hash = state_hash(again)
    seconds["recompute"] = time.monotonic() - started
    print(f"{GATE_ID}: {name}: the recompute's final state is {recompute_hash[:12]},"
          f" {'identical' if recompute_hash == final_hash else 'different'}", flush=True)

    started = time.monotonic()
    params = load_final(folder, config, manifest.run_id, steps[-1])
    behaviour = measure_behaviour(name, task, config, params)
    seconds["behaviour"] = time.monotonic() - started

    written = sum(1 for step in steps if _complete(folder / "checkpoints" / f"step-{step}.safetensors",
                                                    manifest.run_id))
    result = {
        "run_id": manifest.run_id,
        "final_state_sha256": final_hash,
        "recompute_state_sha256": recompute_hash,
        "recompute_identical": recompute_hash == final_hash,
        "checkpoints_written": written,
        "checkpoints_scheduled": len(steps),
        **traced,
        "chance": task.chance,
        "behaviour": behaviour,
        "mechanism_accuracy_by_step": curve,
        "seconds": seconds,
        "model_config": dataclasses.asdict(config),
        "train_config": dataclasses.asdict(train_config),
        "task": task_identity(task),
        "corpus_sha256": manifest.corpus_sha256,
        "tokenizer_sha256": manifest.tokenizer_sha256,
        "checkpoint_schedule": steps,
        "run_dir": p1.rel(folder),
        "recompute_dir": p1.rel(scratch),
    }
    print(f"{GATE_ID}: {name}: {_line(name, result)}", flush=True)
    return result


def task_identity(task: Any) -> dict[str, Any]:
    """The task's class name and fields, whose canonical JSON ``corpus_sha256`` hashes."""
    return {"class": type(task).__name__, "fields": dataclasses.asdict(task)}


def task_sha256(task: Any) -> str:
    from nablaframe.trace.schema import canonical_json, sha256_hex

    return sha256_hex(canonical_json(task_identity(task)).encode("utf-8"))


def build_manifest(name: str, task: Any, config: Any, train_config: Any, tokenizer_sha256: str) -> Any:
    """The zoo run's RunManifest, built as G1.4 builds the reference run's.

    Its run_id covers everything but ``created_at``.
    """
    import jax

    from nablaframe.trace.schema import RunManifest, Seeds

    commit, dirty = p1.git_state()
    optimizer = {
        "name": "adamw",
        "schedule": "linear warmup from lr / warmup_steps to lr, then cosine decay to lr * min_lr_ratio (D9)",
        **{key: list(value) if isinstance(value, tuple) else value
           for key, value in dataclasses.asdict(train_config).items()},
    }
    return RunManifest(
        model_config=dataclasses.asdict(config),
        optimizer=optimizer,
        seeds=Seeds(param_init=train_config.param_seed, data_order=train_config.data_seed, dropout=0),
        corpus_id=f"zoo:{name}",
        corpus_sha256=task_sha256(task),
        tokenizer_id="tokenizer.json",
        tokenizer_sha256=tokenizer_sha256,
        git_commit=commit,
        git_dirty=dirty,
        env_lock_sha256=p1.phase0.env_lock_sha256(),
        backend=jax.default_backend(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        checkpoint_schedule=tuple(_common.checkpoint_steps(name)),
        created_at=p1.phase0.utc_timestamp(),
    )


def train(task: Any, config: Any, train_config: Any, steps: list[int], *, run_id: str,
          checkpoint_dir: Path | None = None, on_checkpoint: Any = None, label: str) -> Any:
    """``train_config.total_steps`` updates from step 0 on the task's batches; at each step in ``steps``,
    the params are saved to ``checkpoint_dir`` (when given) and passed to ``on_checkpoint`` (when given)."""
    from nablaframe.train.checkpoint import save_params
    from nablaframe.train.heartbeat import Heartbeat
    from nablaframe.train.loop import init_state, run_until

    batch, seed = train_config.batch, train_config.data_seed

    def batch_source(k: int) -> np.ndarray:
        return task.batch_for_step(seed, k, batch)

    print(f"{GATE_ID}: training {label}: {train_config.total_steps:,} steps at batch {batch}", flush=True)
    heartbeat = Heartbeat(train_config.total_steps)
    state = init_state(config, train_config)
    for step in sorted(set(steps) | {train_config.total_steps}):
        if step > state.step:
            state = run_until(state, None, config, train_config, until_step=step, heartbeat=heartbeat,
                              run_id=run_id, batch_source=batch_source)
        if step in steps:
            if checkpoint_dir is not None:
                save_params(state.params, checkpoint_dir, step=step, run_id=run_id)
            if on_checkpoint is not None:
                on_checkpoint(step, state.params)
    return state


def accuracy(config: Any, params: Any, task: Any, windows: np.ndarray, positions: Any) -> float:
    """``next_token_accuracy`` of the unintervened pass on ``windows`` at ``positions``."""
    from nablaframe.zoo.ground_truth import next_token_accuracy

    logits, _ = _common.logits(_model(config), params, task, windows, n_layers=config.n_layers)
    return next_token_accuracy(logits, windows, positions)


_MODELS: dict[Any, Any] = {}


def _model(config: Any) -> Any:
    """One GPT2Model per config, so its compiled passes are reused."""
    from nablaframe.model.gpt2 import GPT2Model

    if config not in _MODELS:
        _MODELS[config] = GPT2Model(config)
    return _MODELS[config]


def trace(name: str, manifest: Any, config: Any, task: Any, folder: Path, steps: list[int]) -> dict[str, Any]:
    """Every canonical tap and the logits at every checkpoint, into ``folder/store``."""
    from nablaframe.data.corpus import batch_hash
    from nablaframe.model.gpt2 import init_params
    from nablaframe.trace.engine import NotReproducible, OracleFailure, TraceRun
    from nablaframe.trace.registry import LOGITS, tap_names

    taps = tap_names(config.n_layers) + (LOGITS,)
    probe = _common.tokens(task, _common.probe(task))
    events: list[Any] = []
    error = None
    with TraceRun.open(folder / "store", manifest, _model(config), checkpoint_dir=folder / "checkpoints", probe=probe,
                       like=init_params(config, 0)) as run:
        progress = p1.Progress(f"tracing {name}'s checkpoints", len(steps))
        try:
            for step in steps:
                events += _engine(run.record, step=step, taps=taps)
                progress.advance()
        except OracleFailure as exc:
            error = p1.scrub(f"OracleFailure: {exc}")
            print(f"{GATE_ID}: {name}: the engine's oracle refused a checkpoint: {exc}", flush=True)
        except NotReproducible as exc:
            error = p1.scrub(f"NotReproducible: {exc}")
            print(f"{GATE_ID}: {name}: the trace found a tap that does not reproduce: {exc}", flush=True)
    print(f"{GATE_ID}: {name}: {len(events):,} events recorded", flush=True)
    return {
        "events_recorded": len(events),
        "events_expected": len(taps) * len(steps),
        "taps_per_checkpoint": len(taps),
        "error": error,
        "error_stage": None if error is None else "trace",
        "probe_batch_hash": batch_hash(probe),
        "probe_shape": list(probe.shape),
        "store": p1.rel(folder / "store"),
    }


def _engine(call: Any, *args: Any, **kwargs: Any) -> Any:
    """``call(*args, **kwargs)``, with the engine's ValueError (a checkpoint without this run's
    provenance) turned into a setup error: SystemExit, so no report is written."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"{GATE_ID}: setup error, the engine refused a checkpoint: {exc}."
                         " No report written.") from exc


def load_final(folder: Path, config: Any, run_id: str, step: int) -> Any:
    """The final checkpoint's params, as on disk, after checking its sidecar names the run."""
    from nablaframe.model.gpt2 import init_params
    from nablaframe.train.checkpoint import load_params

    return load_params(_common.checkpoint(folder / "checkpoints", run_id, step), like=init_params(config, 0))


def measure_behaviour(name: str, task: Any, config: Any, params: Any) -> dict[str, Any]:
    """The behaviour bars' accuracies, from logits alone, at the final checkpoint."""
    from nablaframe.zoo.ground_truth import TWIN_SEED, twin_params

    twin = twin_params(config)
    windows = _common.eval_windows(task)
    mechanism = task.mechanism_positions
    if name == "induction":
        measured = {
            "mechanism_accuracy": accuracy(config, params, task, windows, mechanism),
            "control_accuracy": accuracy(config, params, task, windows, task.control_positions),
            "twin_accuracy": accuracy(config, twin, task, windows, mechanism),
            "mechanism_positions": [int(mechanism[0]), int(mechanism[-1])],
            "control_positions": [int(task.control_positions[0]), int(task.control_positions[-1])],
        }
    else:
        broken = _common.broken_windows(task)
        measured = {
            "accuracy": accuracy(config, params, task, windows, mechanism),
            "broken_accuracy": accuracy(config, params, task, broken, mechanism),
            "twin_accuracy": accuracy(config, twin, task, windows, mechanism),
            "mechanism_positions": [int(p) for p in mechanism],
            "broken_windows_sha256": _batch_hash(broken),
        }
    measured["twin_seed"] = TWIN_SEED
    measured["eval_windows"] = int(len(windows))
    measured["eval_windows_sha256"] = _batch_hash(windows)
    return measured


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


def _complete(path: Path, run_id: str) -> bool:
    sidecar = path.with_suffix(".json")
    return path.exists() and sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get(
        "run_id") == run_id


def _line(name: str, result: dict[str, Any]) -> str:
    """One line of what a model's results say, for the terminal."""
    b = result["behaviour"]
    if name == "induction":
        shown = (f"mechanism {b['mechanism_accuracy']:.4f}, control {b['control_accuracy']:.4f},"
                 f" twin {b['twin_accuracy']:.4f}")
    else:
        shown = f"two-hop {b['accuracy']:.4f}, broken {b['broken_accuracy']:.4f}, twin {b['twin_accuracy']:.4f}"
    return (f"{shown} (chance {result['chance']:.4f}); recompute identical: {result['recompute_identical']};"
            f" {result['checkpoints_written']}/{result['checkpoints_scheduled']} checkpoints,"
            f" {result['events_recorded']}/{result['events_expected']} events")


if __name__ == "__main__":
    raise SystemExit(main())
