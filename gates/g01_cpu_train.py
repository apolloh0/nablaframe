"""Gate G0.1: CPU training.

Spec: docs/specs/phase-0.md, Gates table, G0.1 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: Levanter's own tiny GPT-2 config trains on CPU for 1,000 steps.

Config: gates/configs/gpt2_nano.yaml, vendored from the Marin monorepo (the
marin-levanter wheel ships no configs). Train through levanter.main.train_lm with
these overrides applied in code, never by editing the YAML, and record them in
the report's measurements as "overrides":
  trainer.num_train_steps = 1000
  trainer.mp = float32 (the YAML says bfloat16; the reference lane is float32)
  step-based checkpoints including step 500, for G0.2
The data source is a Hugging Face dataset, so the first run needs network access.
Pass: Completes with no NaN or Inf; eval loss at step 1,000 is at least 30% below step 0.
On failure the Phase 1 lane is PyTorch.

Writes reports/phase0/g01_cpu_train.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.1"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

from __future__ import annotations

import sys

GATE_ID = "G0.1"
REPORT_PATH = "reports/phase0/g01_cpu_train.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
