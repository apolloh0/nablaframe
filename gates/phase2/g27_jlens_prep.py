"""G2.7: the J-lens cross-check's inputs, in the jax lane (D17). Runs before G2.8.

At the final checkpoint (step 12,208), for layer 0 (the longest path, through blocks 1 to 3):

* Exports the checkpoint with ``export_hf`` to ``runs/phase2/jlens/hf-step-12208/``, with the
  run's tokenizer, ``<eos>`` and ``<unk>``, and reloads it to compare every tensor with its
  parameter: ``export_matches_params``.
* Computes ``summed_jacobian`` for ``block_map(..., layer=0)`` on the first 8 windows of
  D17's fit sample (``batch_for_step(train, 1, 0, 128, 256)``), saves it with
  ``save_jacobian`` under ``runs/phase2/jlens/crosscheck/``, writes those 8 windows' token
  ids beside it for G2.8, and recomputes it once: ``jacobian_recompute_identical``.

Measurements: ``step``, ``layer``, ``windows`` (8), ``export_matches_params``,
``jacobian_recompute_identical``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
