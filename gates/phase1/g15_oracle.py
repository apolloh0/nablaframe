"""Gate G1.5: Oracle on the real model.

Spec: docs/specs/phase-1.md, Gates table, G1.5 row. Report: reports/phase1/g15_oracle.json.

Loads G1.4's trace checkpoints at steps 0, 6,000 and 12,208 and runs
``check_mode_equality`` on the probe batch with every tap, under the bitwise
policy (D6). ``bitwise_equal`` is whether all three pass.

``defect_detected``: whether ``compare_logits`` under the bitwise policy fails
for the step-12,208 params when ``instrumented`` is given
``interventions={"blocks.0.hook_resid_post": lambda x: x + 1e-3}``.

Measurements: REQUIRED_MEASUREMENTS["G1.5"], plus each step's oracle result.


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

GATE_ID = "G1.5"
REPORT_PATH = "reports/phase1/g15_oracle.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
