"""Gate G0.3: Throughput.

Spec: docs/specs/phase-0.md, Gates table, G0.3 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: Training tokens/s on CPU at size S (4 layers, d=256, batch 32x256) and size M (4 layers, d=512, batch 64x256).
Pass: S >= 25,000 tokens/s. M is recorded only.
On failure, revisit scale targets.

Writes reports/phase0/g03_throughput.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.3"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

from __future__ import annotations

import sys

GATE_ID = "G0.3"
REPORT_PATH = "reports/phase0/g03_throughput.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
