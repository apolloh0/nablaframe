"""G2.1: logits in the trace (D12).

Opens a ``TraceRun`` on the reference store with G1.4's manifest and checkpoints and
G1.6's probe, and records every canonical tap plus ``LOGITS`` at each of the 28 trace
steps. Identical events are skipped, so only logits are added, and a rerun adds nothing.
A ``NotReproducible`` is a finding: a fail report with ``error``.

Measurements: ``logits_events`` (current logits events after the run),
``logits_events_expected`` (the number of trace steps), ``other_events_written`` (events
this run wrote for any other tap), ``sample_size`` (20) and ``sample_matches`` (a seeded
sample of logits events that verify bit for bit), ``logits_bytes`` (the total size of the
logits events' shard files) and ``logits_bytes_expected``
(``logits_bytes_expected(28, 8, 256, 4096)``). Counting totals rather than new events
makes a rerun after success pass again.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g21_logits.py``. Takes --overwrite and --allow-dirty, as Phase 1's
#   gates do. --overwrite only allows replacing the report: the store is G1.6's, and
#   is append-only, so this gate never clears it.
# * The store is G1.6's, at runs/phase1/g16_trace/store/; it must already hold G1.6's
#   events, or there is nothing to add logits to (exit 1, no report).
# * ``logits_events`` counts the store's current events with tap ``logits`` for this
#   run and probe, after the run. ``logits_bytes`` sums the shard files of those same
#   events. ``logits_bytes_expected`` uses the probe's shape and the model's vocabulary,
#   which are 8, 256 and 4,096 for the reference run.
# * The sample is ``default_rng(0).choice(n, 20, replace=False)`` over those events,
#   sorted by step, verified in the order drawn with the engine's ``verify``, which
#   recomputes the logits from the checkpoint and compares hashes. ``sample_size`` is
#   the number actually sampled: fewer than 20 only if fewer logits events exist.
# * ``OracleFailure`` and ``NotReproducible`` stop the recording: the report says fail,
#   with ``error`` ("<class>: <message>") and ``error_stage`` "record". Everything else
#   is measured on what the store holds then.
# * A ValueError from the engine (a checkpoint without this run's provenance) is a setup
#   error, as in G1.6: exit 1, no report.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.1"
REPORT_NAME = "g21_logits.json"
SAMPLE_SIZE = 20
SAMPLE_SEED = 0
MAX_BYTES_DEVIATION = 0.01
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    probe = p1.probe(p1.load_split("validation"), ref.config.seq)
    store_root = _common.g16_store()
    if not (store_root / "events.jsonl").exists():
        raise SystemExit(f"{p1.rel(store_root)} holds no trace store. Run G1.6 first. No report written.")
    measurements = measure(ref, probe, store_root)
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.1 rule holds."""
    stored, expected = m["logits_bytes"], m["logits_bytes_expected"]
    ok = (m["logits_events"] == m["logits_events_expected"] and m["other_events_written"] == 0
          and m["sample_matches"] is not None and m["sample_matches"] == m["sample_size"]
          and stored is not None and abs(stored - expected) <= MAX_BYTES_DEVIATION * abs(expected))
    return "pass" if ok else "fail"


def measure(ref: Any, probe: np.ndarray, store_root: Path, *, sample_size: int = SAMPLE_SIZE) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.trace.engine import NotReproducible, OracleFailure, TraceRun
    from nablaframe.trace.registry import LOGITS, tap_names

    config, steps = ref.config, ref.steps
    taps = tap_names(config.n_layers) + (LOGITS,)
    written: list[Any] = []
    error, error_stage, sample = None, None, []
    with TraceRun.open(store_root, ref.manifest, ref.model, checkpoint_dir=ref.checkpoint_dir, probe=probe,
                       like=ref.like) as run:
        progress = p1.Progress("recording checkpoints", len(steps))
        try:
            for step in steps:
                written += _engine(run.record, step=step, taps=taps)
                progress.advance()
        except (OracleFailure, NotReproducible) as exc:
            error, error_stage = p1.scrub(f"{type(exc).__name__}: {exc}"), "record"
            print(f"{GATE_ID}: recording stopped: {error}", flush=True)
        new_logits = sum(1 for event in written if event.tap == LOGITS)
        print(f"{GATE_ID}: {new_logits} logits events and {len(written) - new_logits} other events written",
              flush=True)

        logits = logits_events(store_root, ref.manifest.run_id, batch_hash(probe))
        n = min(sample_size, len(logits))
        drawn = np.random.default_rng(SAMPLE_SEED).choice(len(logits), size=n, replace=False) if n else []
        progress = p1.Progress("verifying logits events", n)
        for index in drawn:
            event = logits[int(index)]
            sample.append({"index": int(index), "event_id": event.event_id, "step": event.step,
                           "matches": bool(_engine(run.verify, event))})
            progress.advance()
        print(f"{GATE_ID}: {sum(e['matches'] for e in sample)} of {n} sampled logits events recompute to their"
              " hash", flush=True)

    batch, seq = probe.shape
    return {
        "logits_events": len(logits),
        "logits_events_expected": len(steps),
        "other_events_written": sum(1 for event in written if event.tap != LOGITS),
        "logits_events_written": new_logits,
        "sample_size": len(sample),
        "sample_size_planned": sample_size,
        "sample_matches": sum(1 for entry in sample if entry["matches"]),
        "logits_bytes": _common.file_bytes(store_root / event.tensor_ref for event in logits),
        "logits_bytes_expected": _common.logits_bytes_expected(len(steps), batch, seq, config.vocab),
        "logits_steps_missing": sorted(set(steps) - {event.step for event in logits}),
        "sample_seed": SAMPLE_SEED,
        "sample": sample,
        "error": error,
        "error_stage": error_stage,
        "run_id": ref.manifest.run_id,
        "probe_batch_hash": batch_hash(probe),
        "probe_shape": list(probe.shape),
        "store": p1.rel(store_root),
    }


def logits_events(store_root: Path, run_id: str, probe_hash: str) -> list[Any]:
    """The store's current logits events for this run and probe, sorted by step."""
    from nablaframe.trace.registry import LOGITS
    from nablaframe.trace.store import TraceStore

    with TraceStore(store_root, mode="r") as store:
        events = [event for event in store.current()
                  if event.tap == LOGITS and event.run_id == run_id and event.batch_hash == probe_hash]
    return sorted(events, key=lambda event: event.step)


def _engine(call: Any, *args: Any, **kwargs: Any) -> Any:
    """``call(*args, **kwargs)``, with the engine's ValueError (a checkpoint without this run's
    provenance) turned into a setup error: SystemExit, so no report is written."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"{GATE_ID}: setup error, the engine refused a checkpoint: {exc}."
                         " No report written.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
