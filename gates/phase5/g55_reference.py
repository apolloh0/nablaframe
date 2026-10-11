"""G5.5: every judged dictionary on the reference run, its meters and the meter's curves (recorded; D44).

Spec: docs/specs/phase-5.md, D44 and "The reference run"; helpers in ``_common.py``. Report:
``reports/phase5/g55_reference.json``. Run in nf-jax, on the Mac, after G5.1 passes, with G1.4's
run and the reference corpus in place.

On G1.4's run at its final checkpoint, read through Phase 1's helpers, the model reading each
window's first ``seq`` tokens. Every dictionary comes through ``cached``, under ``dictionary_key``
with that checkpoint's state hash, trained with ``settings("reference", ...)`` on
``reference_windows(train, validation, seq, "train")``: ``topk`` at every block, the
``staircase``, a ``transcoder`` at every block, and TopK pairs (the ``jsae`` at ``lam`` 0.0) across
blocks 1 and 2's MLPs. Everything is measured on ``reference_windows(..., "held_out")``.

* ``meters``: as G5.2's, without a twin, the readout being every position. TinyStories has no
  ground truth, so nothing is judged.
* ``edge_pairs``: for the two TopK pairs, as G5.4's ``"trained"`` entries, the meter's tokens
  being the first ``METER_WINDOWS["reference"]`` held-out windows, each curve in full.
* ``train_seconds``: per dictionary, None when it came from the cache.

Measurements: REQUIRED_MEASUREMENTS["G5.5"]. Its status is always ``pass``: it records, not judges.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
