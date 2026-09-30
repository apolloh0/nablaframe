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

from __future__ import annotations

import sys

GATE_ID = "G1.3"
REPORT_PATH = "reports/phase1/g13_determinism.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
