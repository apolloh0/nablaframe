"""G2.9: the summed J for every layer and trace checkpoint, in the jax lane (D17). About 4 hours.

Refuses to start unless ``reports/phase2/g28_jlens_crosscheck.json`` exists and passes, so the
long computation runs only on a definition the reference has confirmed. For each of the 28
trace steps and each of layers 0 to 2 (layer 3's J is the identity, by definition), computes
``summed_jacobian`` on all 128 windows of D17's fit sample and saves it with
``save_jacobian`` under ``runs/phase2/jlens/fit/``, keyed by ``jacobian_key`` with the
checkpoint sidecar's state hash, the fit sample's token hash and ``code_identity()``. A J
already saved under its key is skipped, so an interrupted run resumes. Prints a progress
line at least every 30 seconds. At the end, recomputes one seeded (step, layer) and compares.

Measurements: ``jacobians_written`` (saved, now or before), ``jacobians_expected`` (84),
``recompute_identical``, ``seconds``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
