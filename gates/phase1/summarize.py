"""Generate reports/phase1/summary.md from the six Phase 1 gate reports: the H1 evidence.

Spec: docs/specs/phase-1.md, "Gates". A pure function of the report files, byte
for byte: no timestamps of its own, rows sorted by gate id. It refuses to run if
any report is missing or fails ``validate_report``. It lists each gate's status
and key measurements, the vocabulary D8 selected, and a verdict: Phase 2 can build
on the engine only if all six pass; otherwise it names what failed. It never
states a result that is not in a report. apolloh0 rules H1.
"""

from __future__ import annotations

import sys
from pathlib import Path


def summarize(report_paths: list[Path]) -> str:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
