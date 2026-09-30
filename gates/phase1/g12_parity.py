"""Gate G1.2: Parity with Levanter.

Spec: docs/specs/phase-1.md, Gates table, G1.2 row. Report: reports/phase1/g12_parity.json.

Builds a Levanter GPT-2 at size S with the vocabulary in
``data/tinystories/tokenizer.json``, seeded, and exports it to Hugging Face
format with Levanter's own converter, the path G0.4 used. Loads the export with
``load_hf_gpt2`` and compares logits with Levanter's on the probe batch:
``probe_batch(validation, EVAL_TOKENS, PROBE_SEED, PROBE_SEQUENCES, 256)``.

* ``n_params``: the loaded params' element count. ``n_params_expected``:
  ``GPT2Config.from_size("S", vocab).n_params()``.
* ``max_abs_logit_diff`` over every position and vocabulary entry.
* ``top1_agreement`` over positions where Levanter's top-2 margin exceeds 1e-3;
  ``positions_near_tie`` counts the rest.

Measurements: REQUIRED_MEASUREMENTS["G1.2"].


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

GATE_ID = "G1.2"
REPORT_PATH = "reports/phase1/g12_parity.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
