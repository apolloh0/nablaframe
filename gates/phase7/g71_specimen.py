"""G7.1: the modular-addition specimen trained, traced and retrained, holding its behaviour and its clock (D54).

Spec: docs/specs/phase-7.md, D54 and "The specimen"; helpers in ``_common.py``. Report:
``reports/phase7/g71_specimen.json``. Run in nf-jax, on the Mac.

``ModularAdditionTask()`` at its defaults, with ``configs()``:

* **Train:** from step 0 with ``run_until(..., batch_source=lambda k: task.batch_for_step(seed, k,
  batch))``, ``seed`` and ``batch`` being the ``TrainConfig``'s ``data_seed`` and ``batch``, as
  G4.1 trains, saving a checkpoint at every ``checkpoint_steps()`` step under
  ``runs/phase7/modular_addition/checkpoints/``, beside a run manifest and the task's tokenizer
  file, as G4.1 saves a zoo run: ``corpus_id`` ``"zoo:modular_addition"``, ``corpus_sha256`` the
  SHA-256 of the task's canonical JSON, its class name and fields. G4.1's training, manifest and
  checkpoint code may be loaded by path and reused; nothing in Phase 4 changes.
* **Trace:** every checkpoint, with every canonical tap and the logits, on what the model reads
  of ``probe(task)``, through ``TraceRun`` into ``runs/phase7/modular_addition/store/``, as G4.1
  traces the zoo runs.
* **Recompute:** trains again from step 0 into ``runs/phase7/scratch/`` and compares the final
  state hash with the first run's.
* **Behaviour,** from logits alone at the final checkpoint: ``held_accuracy`` and
  ``train_accuracy`` at position 2 on the held-out and training pairs; ``control_accuracy`` at
  position 0 on the held-out pairs, where nothing predicts ``b`` from ``a``; ``twin_accuracy`` at
  position 2 on the held-out pairs, the twin being ``init_params(config, seed=1)``; ``chance``,
  ``1 / p``.
* **The clock,** on the final checkpoint's weights and logits, never a view: ``frequencies``
  and ``energy``, ``fewest_frequencies(params.wte[:p], p)``'s count and share;
  ``twin_frequencies``, the twin's count; and ``logit_r2``, ``logit_r2`` of the training pairs'
  and the held-out pairs' logits at position 2, at the frequencies
  ``crystallize.frequencies(params.wte[:p], p)["k"]``: the five or so the fourier form keeps.

Measurements: REQUIRED_MEASUREMENTS["G7.1"]. ``checkpoints_scheduled`` is 45 and
``events_expected`` the canonical taps plus the logits, times 45. Also recorded: the run id, the
final state hash, training and held-out accuracy at every checkpoint, the first step each
reaches 0.99, and the seconds each stage took. Pass needs every bar. The run folder starts
fresh: an existing one is removed only with ``--overwrite``. If the trace engine's oracle refuses
a checkpoint, or the recompute isn't reproducible, the report fails and records the error, as
G4.1's does.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g71_specimen.py``.
#   Takes --overwrite and --allow-dirty. It builds on no earlier report.
# * Its two folders, ``runs/phase7/modular_addition/`` and ``runs/phase7/scratch/modular_addition/``, are
#   checked before anything trains, so the gate never stops on a folder after training.
# * Training is G4.1's, written here for the specimen: ``init_state`` at ``configs()``'s ``TrainConfig``, then
#   ``run_until`` from one checkpoint step to the next with the task's batches, a ``Heartbeat`` printing
#   progress. Each scheduled step's params go to ``checkpoints/`` with ``save_params`` under the manifest's
#   ``run_id``; step 0 is saved before the first update. No resume checkpoints are written. The manifest is
#   G4.1's for a zoo task: ``corpus_id`` ``"zoo:modular_addition"``, ``corpus_sha256`` the SHA-256 of
#   ``canonical_json({"class": "ModularAdditionTask", "fields": <its fields>})``, ``tokenizer_id``
#   ``"tokenizer.json"``, the file beside it holding ``task.tokenizer_json()``, with ``checkpoint_schedule``
#   ``checkpoint_steps()``.
# * Accuracy at every checkpoint: the params as just saved, at position 2 on the training and the held-out
#   pairs. Every pass in this gate, those included, is ``run_intervened`` with an empty spec.
# * The trace: one ``TraceRun`` on ``store/`` with the manifest, the checkpoints folder and the probe's first
#   ``seq`` tokens, recording every canonical tap and the logits at each step. OracleFailure or
#   NotReproducible stops the trace: its ``error`` and ``error_stage`` (``"trace"``) are recorded and the report
#   fails, since ``events_recorded`` falls short. A provenance ValueError from the engine is a setup error:
#   exit 1, no report.
# * The recompute saves only its final state, with ``save_checkpoint``. Each run's final state hash is
#   ``state_hash`` of its final ``TrainState``, as G4.1 hashes.
# * Behaviour and the clock: on the final checkpoint as loaded from disk. ``first_at_099``: the first step at
#   which training and held-out accuracy each reach 0.99, None where neither does.

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

p1, p4 = _common.p1, _common.p4

GATE_ID = "G7.1"
REPORT_NAME = "g71_specimen.json"
NEEDS: tuple[str, ...] = ()
ANSWER, CONTROL = 2, 0
REACHED = 0.99


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, _ = _common.start(args, REPORT_NAME, NEEDS)
    for folder in (_common.run_dir(), scratch_dir()):
        if folder.exists() and not args.overwrite:
            raise SystemExit(f"{p1.rel(folder)} exists. Pass --overwrite to start it over.")
    measurements = measure(overwrite=args.overwrite)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G7.1 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (m["recompute_identical"] is True
          and n(m["checkpoints_written"]) and m["checkpoints_written"] == m["checkpoints_scheduled"]
          and n(m["events_recorded"]) and m["events_recorded"] == m["events_expected"]
          and n(m["held_accuracy"]) and m["held_accuracy"] >= bars["held_accuracy"]
          and n(m["chance"])
          and n(m["control_accuracy"]) and m["control_accuracy"] <= bars["control_ratio"] * m["chance"]
          and n(m["twin_accuracy"]) and m["twin_accuracy"] <= bars["control_ratio"] * m["chance"]
          and n(m["frequencies"]) and m["frequencies"] <= bars["frequencies"]
          and n(m["logit_r2"]) and m["logit_r2"] >= bars["logit_r2"])
    return "pass" if ok else "fail"


def scratch_dir() -> Path:
    """``runs/phase7/scratch/modular_addition/``: the recompute's final state."""
    return Path(_common.RUN_DIR) / "scratch" / _common.RUN_NAME


def measure(*, overwrite: bool) -> dict[str, Any]:
    from nablaframe.train.checkpoint import save_checkpoint, state_hash

    task = _common.task()
    config, train_config = _common.configs()
    steps = _common.checkpoint_steps()
    folder = _common.fresh_dir(_common.run_dir(), overwrite=overwrite)
    tokenizer = folder / "tokenizer.json"
    p1.write_atomic(tokenizer, task.tokenizer_json().encode("utf-8"))
    manifest = build_manifest(task, config, train_config, p1.sha256_file(tokenizer))
    p1.write_atomic(folder / "manifest.json", manifest.to_json().encode("utf-8"))
    _common.say(f"{GATE_ID}: run {manifest.run_id}, in {p1.rel(folder)}")
    seconds: dict[str, float] = {}
    training, held = task.training(), task.held_out()

    started = time.monotonic()
    curve: list[dict[str, Any]] = []

    def measure_checkpoint(step: int, params: Any) -> None:
        row = {"step": step, "train_accuracy": accuracy(config, params, training, ANSWER),
               "held_accuracy": accuracy(config, params, held, ANSWER)}
        curve.append(row)
        _common.say(f"{GATE_ID}: step {step:,}: accuracy at position 2, training {row['train_accuracy']:.4f},"
                    f" held out {row['held_accuracy']:.4f}")

    state = train(task, config, train_config, steps, run_id=manifest.run_id, checkpoint_dir=folder / "checkpoints",
                  on_checkpoint=measure_checkpoint, label="the specimen")
    seconds["train"] = time.monotonic() - started
    final_hash = state_hash(state)
    _common.say(f"{GATE_ID}: trained to step {state.step:,}, final state {final_hash[:12]}")

    started = time.monotonic()
    traced = trace(manifest, config, task, folder, steps)
    seconds["trace"] = time.monotonic() - started

    started = time.monotonic()
    scratch = _common.fresh_dir(scratch_dir(), overwrite=overwrite)
    again = train(task, config, train_config, steps, run_id=manifest.run_id, label="the recompute")
    save_checkpoint(again, scratch, run_id=manifest.run_id)
    recompute_hash = state_hash(again)
    seconds["recompute"] = time.monotonic() - started
    _common.say(f"{GATE_ID}: the recompute's final state is {recompute_hash[:12]},"
                f" {'identical' if recompute_hash == final_hash else 'different'}")

    started = time.monotonic()
    params = load_final(folder, config, manifest.run_id, steps[-1])
    behaviour = measure_behaviour(task, config, params)
    seconds["behaviour"] = time.monotonic() - started

    started = time.monotonic()
    clock = measure_clock(task, config, params, behaviour.pop("logits"))
    seconds["clock"] = time.monotonic() - started

    written = sum(1 for step in steps if _complete(folder / "checkpoints" / f"step-{step}.safetensors",
                                                    manifest.run_id))
    out = {
        "recompute_identical": recompute_hash == final_hash,
        "checkpoints_written": written,
        "checkpoints_scheduled": len(steps),
        **{key: traced[key] for key in ("events_recorded", "events_expected")},
        **{key: behaviour[key] for key in ("held_accuracy", "train_accuracy", "control_accuracy", "twin_accuracy")},
        "chance": task.chance,
        **{key: clock[key] for key in ("frequencies", "energy", "twin_frequencies", "logit_r2")},
        "bars": _common.BARS,
        "run_id": manifest.run_id,
        "final_state_sha256": final_hash,
        "recompute_state_sha256": recompute_hash,
        "trace": {key: value for key, value in traced.items() if key not in ("events_recorded", "events_expected")},
        "behaviour": behaviour,
        "clock": clock,
        "accuracy_by_step": curve,
        "first_at_099": {"train": _first(curve, "train_accuracy"), "held": _first(curve, "held_accuracy")},
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
    _common.say(f"{GATE_ID}: held out {_common.show(out['held_accuracy'])}, training"
                f" {_common.show(out['train_accuracy'])},"
                f" control {_common.show(out['control_accuracy'])}, twin {_common.show(out['twin_accuracy'])} (chance"
                f" {out['chance']:.4f}); {out['frequencies']} frequencies hold {_common.show(out['energy'])} (twin"
                f" {out['twin_frequencies']}), logits' R² {_common.show(out['logit_r2'])}; recompute identical"
                f" {out['recompute_identical']}; {written}/{len(steps)} checkpoints,"
                f" {out['events_recorded']}/{out['events_expected']} events")
    return out


def _first(curve: list[dict[str, Any]], key: str) -> int | None:
    return next((row["step"] for row in curve if row[key] >= REACHED), None)


def task_identity(task: Any) -> dict[str, Any]:
    """The task's class name and fields, whose canonical JSON ``corpus_sha256`` hashes."""
    return {"class": type(task).__name__, "fields": dataclasses.asdict(task)}


def task_sha256(task: Any) -> str:
    from nablaframe.trace.schema import canonical_json, sha256_hex

    return sha256_hex(canonical_json(task_identity(task)).encode("utf-8"))


def build_manifest(task: Any, config: Any, train_config: Any, tokenizer_sha256: str) -> Any:
    """The specimen's RunManifest, built as G4.1 builds a zoo run's. Its run_id covers everything but ``created_at``."""
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
        corpus_id=f"zoo:{_common.RUN_NAME}",
        corpus_sha256=task_sha256(task),
        tokenizer_id="tokenizer.json",
        tokenizer_sha256=tokenizer_sha256,
        git_commit=commit,
        git_dirty=dirty,
        env_lock_sha256=p1.phase0.env_lock_sha256(),
        backend=jax.default_backend(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        checkpoint_schedule=tuple(_common.checkpoint_steps()),
        created_at=p1.phase0.utc_timestamp(),
    )


def train(task: Any, config: Any, train_config: Any, steps: list[int], *, run_id: str,
          checkpoint_dir: Path | None = None, on_checkpoint: Any = None, label: str) -> Any:
    """``train_config.total_steps`` updates from step 0 on the task's batches; at each step in ``steps``, the params
    are saved to ``checkpoint_dir`` (when given) and passed to ``on_checkpoint`` (when given)."""
    from nablaframe.train.checkpoint import save_params
    from nablaframe.train.heartbeat import Heartbeat
    from nablaframe.train.loop import init_state, run_until

    batch, seed = train_config.batch, train_config.data_seed

    def batch_source(k: int) -> np.ndarray:
        return task.batch_for_step(seed, k, batch)

    _common.say(f"{GATE_ID}: training {label}: {train_config.total_steps:,} steps at batch {batch}")
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


_MODELS: dict[Any, Any] = {}


def _model(config: Any) -> Any:
    """One GPT2Model per config, so its compiled passes are reused."""
    from nablaframe.model.gpt2 import GPT2Model

    if config not in _MODELS:
        _MODELS[config] = GPT2Model(config)
    return _MODELS[config]


def logits(config: Any, params: Any, windows: np.ndarray) -> np.ndarray:
    """The unintervened pass's logits on what the model reads of ``windows``, through ``run_intervened``."""
    return p4.logits(_model(config), params, _common.task(), windows, n_layers=config.n_layers)[0]


def accuracy(config: Any, params: Any, windows: np.ndarray, position: int) -> float:
    """``next_token_accuracy`` of the unintervened pass on ``windows`` at ``position``."""
    from nablaframe.zoo.ground_truth import next_token_accuracy

    return next_token_accuracy(logits(config, params, windows), windows, [position])


def trace(manifest: Any, config: Any, task: Any, folder: Path, steps: list[int]) -> dict[str, Any]:
    """Every canonical tap and the logits at every checkpoint, on the probe, into ``folder/store``."""
    from nablaframe.data.corpus import batch_hash
    from nablaframe.model.gpt2 import init_params
    from nablaframe.trace.engine import NotReproducible, OracleFailure, TraceRun
    from nablaframe.trace.registry import LOGITS, tap_names

    taps = tap_names(config.n_layers) + (LOGITS,)
    probe = _common.tokens(_common.probe(task))
    events: list[Any] = []
    error = None
    with TraceRun.open(folder / "store", manifest, _model(config), checkpoint_dir=folder / "checkpoints", probe=probe,
                       like=init_params(config, 0)) as run:
        progress = p1.Progress(f"{GATE_ID}: tracing the specimen's checkpoints", len(steps))
        try:
            for step in steps:
                events += _engine(run.record, step=step, taps=taps)
                progress.advance()
        except OracleFailure as exc:
            error = p1.scrub(f"OracleFailure: {exc}")
            _common.say(f"{GATE_ID}: the engine's oracle refused a checkpoint: {exc}")
        except NotReproducible as exc:
            error = p1.scrub(f"NotReproducible: {exc}")
            _common.say(f"{GATE_ID}: the trace found a tap that does not reproduce: {exc}")
    _common.say(f"{GATE_ID}: {len(events):,} events recorded")
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
    """``call(*args, **kwargs)``, with the engine's ValueError (a checkpoint without this run's provenance) turned
    into a setup error: SystemExit, so no report is written."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"{GATE_ID}: setup error, the engine refused a checkpoint: {exc}. No report written.") from exc


def load_final(folder: Path, config: Any, run_id: str, step: int) -> Any:
    """The final checkpoint's params, as on disk, after checking its sidecar names the run."""
    from nablaframe.model.gpt2 import init_params
    from nablaframe.train.checkpoint import load_params

    return load_params(p4.checkpoint(folder / "checkpoints", run_id, step), like=init_params(config, 0))


def measure_behaviour(task: Any, config: Any, params: Any) -> dict[str, Any]:
    """The behaviour bars' accuracies, from logits alone, at the final checkpoint; ``logits`` holds the training
    and held-out pairs' logits at position 2, for the clock."""
    from nablaframe.zoo.ground_truth import TWIN_SEED, next_token_accuracy, twin_params

    training, held = task.training(), task.held_out()
    train_logits, held_logits = logits(config, params, training), logits(config, params, held)
    twin_logits = logits(config, twin_params(config), held)
    return {
        "held_accuracy": next_token_accuracy(held_logits, held, [ANSWER]),
        "train_accuracy": next_token_accuracy(train_logits, training, [ANSWER]),
        "control_accuracy": next_token_accuracy(held_logits, held, [CONTROL]),
        "twin_accuracy": next_token_accuracy(twin_logits, held, [ANSWER]),
        "answer_position": ANSWER,
        "control_position": CONTROL,
        "twin_seed": TWIN_SEED,
        "training_pairs": int(len(training)),
        "held_out_pairs": int(len(held)),
        "training_sha256": _batch_hash(training),
        "held_out_sha256": _batch_hash(held),
        "logits": {"train": train_logits[:, ANSWER], "held": held_logits[:, ANSWER]},
    }


def measure_clock(task: Any, config: Any, params: Any, at_answer: dict[str, np.ndarray]) -> dict[str, Any]:
    """The clock, on the final checkpoint's weights and logits: the embedding's frequencies, the twin's count, and
    the logits against the fourier form's frequencies' cosines."""
    from nablaframe.gestures import crystallize
    from nablaframe.zoo.ground_truth import twin_params

    p = task.p
    embedding = np.asarray(params.wte, dtype=np.float64)[:p]
    fewest = _common.fewest_frequencies(embedding, p)
    twin = _common.fewest_frequencies(np.asarray(twin_params(config).wte, dtype=np.float64)[:p], p)
    kept = crystallize.frequencies(embedding, p)
    r2 = _common.logit_r2(at_answer["train"], task.training(), at_answer["held"], task.held_out(), p, kept["k"])
    return {"frequencies": fewest["count"], "energy": fewest["held"], "twin_frequencies": twin["count"],
            "twin_energy": twin["held"], "logit_r2": r2, "fourier_k": list(kept["k"]), "fourier_held": kept["held"]}


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


def _complete(path: Path, run_id: str) -> bool:
    sidecar = path.with_suffix(".json")
    return path.exists() and sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")).get(
        "run_id") == run_id


if __name__ == "__main__":
    raise SystemExit(main())
