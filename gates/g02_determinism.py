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

from __future__ import annotations

import sys

GATE_ID = "G0.2"
REPORT_PATH = "reports/phase0/g02_determinism.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
