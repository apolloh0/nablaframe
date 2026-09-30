"""Gate G1.4: Reference run.

Spec: docs/specs/phase-1.md, Gates table, G1.4 row. Report: reports/phase1/g14_reference.json.

Trains the reference model: size S, the selected vocabulary, ``TrainConfig()``
defaults (12,208 steps, 100M tokens), data seed 0, parameter seed 0.

* Writes ``runs/phase1/g14_reference/manifest.json`` first: the canonical JSON of a
  ``RunManifest`` with the real corpus and tokenizer hashes, git state, lockfile
  hash, ``XLA_FLAGS`` and ``checkpoint_schedule = trace_steps(12_208)``. Its
  ``run_id`` goes into every checkpoint sidecar.
* Trace checkpoints (params only) at every step in the schedule, under
  ``runs/phase1/g14_reference/trace/``; resume checkpoints every 500 steps under
  ``runs/phase1/g14_reference/resume/``. The two never share a folder, since both
  use the name ``step-<n>.safetensors``.
* Validation loss on ``eval_windows(validation, EVAL_TOKENS, 256)`` every 500
  steps and at every trace checkpoint; the curve is recorded.
* ``bigram_val_loss``: the cross-entropy of an add-one-smoothed bigram model
  counted on ``train.bin``, evaluated on the same windows' next-token targets.
* ``--resume`` continues from the latest resume checkpoint after an interruption.

Progress: a ``Heartbeat`` throughout. Expect about 1.8 to 2.3 hours.
Measurements: REQUIRED_MEASUREMENTS["G1.4"], plus the validation-loss curve and
``run_id``.


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

GATE_ID = "G1.4"
REPORT_PATH = "reports/phase1/g14_reference.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
