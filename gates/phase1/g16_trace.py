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


Every gate: runs on the Mac only, in nf-jax
(``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/<script>``);
refuses a dirty tree unless ``--allow-dirty``; refuses to overwrite its report unless
``--overwrite``; writes one schema 2 report with ``write_report``, measurements
passed through ``plain``; prints progress at least every 30 s during anything
longer than a minute; returns exit code 0 whenever it wrote a valid report,
whatever the gate's status, and non-zero only when it could not produce one.
"""

from __future__ import annotations

import sys

GATE_ID = "G1.6"
REPORT_PATH = "reports/phase1/g16_trace.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
