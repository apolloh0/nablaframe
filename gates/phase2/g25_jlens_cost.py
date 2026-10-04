"""G2.5: the J-lens cost probe (D15). Always passes; its numbers set D15's method.

At the final checkpoint, for the residual stream entering block 2, with ``f`` the map to
the last block's output through the model's intervention hook:

* The exact reference on causal prefixes. The model is causal, so the same-position
  Jacobian at position t depends only on positions up to t. The exact computation on each
  sequence's first ``EXACT_PREFIX`` positions is therefore exact for those positions, at
  ``exact_jacobian_bytes(32, 256)`` per sequence rather than 17 GB. ``exact_positions`` is
  the number of positions it covers (64 sequences: 2,048), ``exact_seconds`` its time.
* The estimator on those same prefix positions, like for like: ``estimator_errors`` maps
  draws (1, 8, 32) to the relative Frobenius error against the exact reference, and
  ``estimator_seconds`` to the times.
* Convergence on full-length sequences from the training stream with seed 1:
  ``fit_sizes`` is ``FIT_SIZES``, ``fit_change_per_doubling`` the relative change in J
  between consecutive sizes, ``fit_seconds`` the times, and ``projected_seconds`` the
  projected time for 4 layers and 28 checkpoints at the largest size.
* ``reference_method``: how ``anthropics/jacobian-lens`` computes J, in a sentence, and
  ``crosscheck_feasible``: whether it can run on our Hugging Face export, both read from
  its source by the implementing thread and written into this module.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
