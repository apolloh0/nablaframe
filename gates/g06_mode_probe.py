"""Gate G0.6: Mode probe.

Spec: docs/specs/phase-0.md, Gates table, G0.6 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: nablaframe.lanes.jax_ref.toy.ToyGPT under jit: logits only vs logits plus all taps, on 3 seeds.
Pass: Always passes: records bitwise equality or the measured max abs diff.
The result sets the oracle policy through a D-entry.

Writes reports/phase0/g06_mode_probe.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.6"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

from __future__ import annotations

import sys

GATE_ID = "G0.6"
REPORT_PATH = "reports/phase0/g06_mode_probe.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
