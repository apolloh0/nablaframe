"""The Phase 2 summary: reports/phase2/summary.md, from the six Phase 2 reports.

Same shape and rules as gates/phase1/summarize.py: refuses a missing report or one whose
status contradicts its measurements, lists each gate that did not pass with the spec's
next step, and gives byte-identical output on two runs. It ends with the evidence for
human gate H2, including G2.5's numbers for D15's ruling.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
