"""Gate G0.5: Metal probe.

Spec: docs/specs/phase-0.md, Gates table, G0.5 row.
Environment: nf-metal. Run on apolloh0's Mac only; never in the cloud.

Measures: The G0.1 config under metaljax: declined ops, loss vs CPU over the first 200 steps, tokens/s.
Pass: Runs, with loss within 2% relative of CPU at every logged step.
Never decides the lane; failure means no accelerator.

Writes reports/phase0/g05_metal_probe.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.5"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

from __future__ import annotations

import sys

GATE_ID = "G0.5"
REPORT_PATH = "reports/phase0/g05_metal_probe.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
