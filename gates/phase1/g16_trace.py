"""Gate G1.6: Trace and recompute.

Spec: docs/specs/phase-1.md, Gates table, G1.6 row. Report: reports/phase1/g16_trace.json.

Opens a ``TraceRun`` at ``runs/phase1/g16_trace/store/`` with G1.4's manifest and
trace checkpoints and the probe batch. Records all 35 taps at every trace
checkpoint (``events_recorded``, expected 980). Then:

* Verifies a sample of 50 events drawn with ``numpy.random.default_rng(0)``
  (``sample_matches``).
* Records every checkpoint again (``retrace_new_events``, expected 0).
* ``store_bytes``: the total size of the store's shard files.
  ``store_bytes_expected``: ``trace_bytes_per_checkpoint(config, 8)`` times the
  number of trace checkpoints.

Progress: a ``Progress`` line per checkpoint recorded and per event verified.
Measurements: REQUIRED_MEASUREMENTS["G1.6"].

* If `record` raises `NotReproducible` during either trace, G1.6 writes a fail
  report: `error_stage` names the trace (`first` or `retrace`), `error` holds the
  message, and any measurement it could not take is null. A provenance `ValueError`
  from the engine is a setup error: the gate exits non-zero without a report.


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
# * The store starts empty: its folder is created fresh, and an existing one is
#   removed only with --overwrite, so events_recorded counts this run's events.
# * The model is built from G1.4's manifest, and every trace checkpoint's sidecar
#   is checked to name G1.4's run before anything is traced.
# * The sample is ``default_rng(0).choice(n, 50, replace=False)`` over the events
#   in the order they were recorded, verified in the order drawn. sample_size is the
#   number of events actually sampled, which is fewer than 50 only when the first
#   trace stopped early; sample_size_planned is always 50.
# * If the engine's oracle refuses a checkpoint (``OracleFailure``), tracing stops
#   there: the report says fail and records the error, and the retrace is not run,
#   so retrace_new_events is null. error_stage is "first", the only trace that
#   catches it.
# * NotReproducible in the first trace is handled the same way: the events recorded
#   before it are still sampled and verified, and the retrace is not run.
# * In the retrace, OracleFailure and NotReproducible both stop the retrace and make
#   the report fail with error_stage "retrace"; retrace_new_events is null, since the
#   retrace never finished, and every other measurement is taken as usual.
# * error holds "<exception class>: <message>", with absolute paths shortened.
# * A ValueError from record or verify (the engine's provenance check on a
#   checkpoint's sidecar) is raised again as SystemExit, so the gate exits 1 with no
#   report. The gate checks every sidecar before tracing, so this only fires if a
#   sidecar changes during the run.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

GATE_ID = "G1.6"
REPORT_PATH = "reports/phase1/g16_trace.json"

SAMPLE_SIZE = 50
SAMPLE_SEED = 0
MAX_STORE_DEVIATION = 0.10
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    report = _common.start(args, REPORT_PATH)
    _common.require_data()
    manifest = _common.load_g14_manifest()
    work = _common.fresh_dir(_common.RUN_DIR / "g16_trace", overwrite=args.overwrite)
    measurements = measure(manifest, _common.load_split("validation"), work / "store")
    measurements["versions"] = _common.phase0.package_versions(VERSIONS)
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G1.6 rule holds."""
    stored, expected = m["store_bytes"], m["store_bytes_expected"]
    ok = (m["sample_matches"] is not None and m["sample_matches"] == m["sample_size"]
          and m["retrace_new_events"] == 0
          and stored is not None and abs(stored - expected) <= MAX_STORE_DEVIATION * abs(expected))
    return "pass" if ok else "fail"


def measure(manifest: Any, validation: np.ndarray, store_root: Path) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.trace.engine import NotReproducible, OracleFailure, TraceRun
    from nablaframe.trace.registry import tap_names

    config = _common.config_from_manifest(manifest)
    steps = tuple(manifest.checkpoint_schedule)
    trace_dir = _common.g14_trace_dir()
    for step in steps:
        _common.trace_checkpoint(manifest, step)
    model = GPT2Model(config)
    taps = tap_names(config.n_layers)
    probe = _common.probe(validation, config.seq)
    per_checkpoint = _common.trace_bytes_per_checkpoint(config, len(probe))

    events: list[Any] = []
    error, error_stage, sample, retrace_new = None, None, [], None
    with TraceRun.open(store_root, manifest, model, checkpoint_dir=trace_dir, probe=probe,
                       like=init_params(config, 0)) as run:
        progress = _common.Progress("tracing checkpoints", len(steps))
        try:
            for step in steps:
                events += _engine(run.record, step=step, taps=taps)
                progress.advance()
        except OracleFailure as exc:
            error, error_stage = _common.scrub(f"OracleFailure: {exc}"), "first"
            print(f"{GATE_ID}: the engine's oracle refused a checkpoint: {exc}", flush=True)
        except NotReproducible as exc:
            error, error_stage = _common.scrub(f"NotReproducible: {exc}"), "first"
            print(f"{GATE_ID}: the first trace found a tap that does not reproduce: {exc}", flush=True)
        print(f"{GATE_ID}: {len(events):,} events recorded", flush=True)

        n = min(SAMPLE_SIZE, len(events))
        drawn = np.random.default_rng(SAMPLE_SEED).choice(len(events), size=n, replace=False) if n else []
        progress = _common.Progress("verifying events", n)
        for index in drawn:
            event = events[int(index)]
            sample.append({"index": int(index), "event_id": event.event_id, "step": event.step, "tap": event.tap,
                           "matches": bool(_engine(run.verify, event))})
            progress.advance()
        print(f"{GATE_ID}: {sum(e['matches'] for e in sample)} of {n} sampled events recompute to their hash",
              flush=True)

        if error is None:
            retrace_new = 0
            progress = _common.Progress("tracing again", len(steps))
            try:
                for step in steps:
                    retrace_new += len(_engine(run.record, step=step, taps=taps))
                    progress.advance()
                print(f"{GATE_ID}: the second trace wrote {retrace_new} events", flush=True)
            except OracleFailure as exc:
                error, error_stage, retrace_new = _common.scrub(f"OracleFailure: {exc}"), "retrace", None
                print(f"{GATE_ID}: the engine's oracle refused a checkpoint during the retrace: {exc}", flush=True)
            except NotReproducible as exc:
                error, error_stage, retrace_new = _common.scrub(f"NotReproducible: {exc}"), "retrace", None
                print(f"{GATE_ID}: the retrace found a tap that does not reproduce: {exc}", flush=True)

    store_bytes = sum(path.stat().st_size for path in (store_root / "shards").glob("*.safetensors"))
    return {
        "events_recorded": len(events),
        "sample_size": len(sample),
        "sample_size_planned": SAMPLE_SIZE,
        "sample_matches": sum(1 for entry in sample if entry["matches"]),
        "retrace_new_events": retrace_new,
        "store_bytes": store_bytes,
        "store_bytes_expected": per_checkpoint * len(steps),
        "events_expected": len(taps) * len(steps),
        "trace_checkpoints": len(steps),
        "taps": len(taps),
        "bytes_per_checkpoint_expected": per_checkpoint,
        "sample_seed": SAMPLE_SEED,
        "sample": sample,
        "error": error,
        "error_stage": error_stage,
        "run_id": manifest.run_id,
        "probe_batch_hash": batch_hash(probe),
        "probe_shape": list(probe.shape),
        "store": _common.rel(store_root),
    }


def _engine(call: Any, *args: Any, **kwargs: Any) -> Any:
    """``call(*args, **kwargs)``, with the engine's ValueError (a checkpoint without this run's
    provenance) turned into a setup error: SystemExit, so no report is written."""
    try:
        return call(*args, **kwargs)
    except ValueError as exc:
        raise SystemExit(f"{GATE_ID}: setup error, the engine refused a checkpoint: {exc}."
                         " No report written.") from exc


if __name__ == "__main__":
    sys.exit(main())
