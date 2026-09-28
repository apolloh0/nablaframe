"""Gate G0.4: Cross-check.

Spec: docs/specs/phase-0.md, Gates table, G0.4 row.
Environment: nf-torch. Run on apolloh0's Mac only; never in the cloud.

Measures: The G0.1 checkpoint, exported to Hugging Face format with
levanter.main.export_lm_to_hf, loaded through TransformerLens TransformerBridge (without compatibility mode, so numerics stay raw Hugging Face), logits compared on a fixed 8x128 batch.
Pass: Max abs logit diff <= 1e-3, and identical top-1 tokens wherever the top-2 margin exceeds 1e-3.
On failure, diagnose the export or load path before D0.

Writes reports/phase0/g04_crosscheck.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.4"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

from __future__ import annotations

import sys

GATE_ID = "G0.4"
REPORT_PATH = "reports/phase0/g04_crosscheck.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
