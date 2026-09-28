"""Generate reports/phase0/summary.md from the six gate reports.

Spec: docs/specs/phase-0.md, Deliverables: "Regenerates identically from the same reports."

The summary is a pure function of the report files: no timestamps of its own, no
reading of anything else, rows sorted by gate id. It lists each gate's status and
key measurements, then states which D0 outcome the evidence supports, using the
Purpose-and-decision table in the spec. It never states a result that is not in
a report, and it refuses to run if any report fails validate_report.
"""

from __future__ import annotations

import sys
from pathlib import Path


def summarize(report_paths: list[Path]) -> str:
    """Markdown summary. Same inputs, same output, byte for byte."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
