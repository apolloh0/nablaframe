"""G2.10: the J-lens view and its discrimination tests (D14, D17, D20).

Refuses to start unless G2.9's report exists and passes. Then:

* Planted, at decoder level, at size S: source directions that a J carries exactly to each
  target token's unembedding direction, as in ``tests/test_jlens_view.py``, with a
  standard-normal unembedding and an identity layer norm. ``planted_recovery`` is the
  fraction of positions whose top-1 J-lens token is the target.
* ``identity_exact``: with J the identity, ``jlens_lens`` equals ``logit_lens`` bit for bit
  on the reference run's final-step residual streams.
* Random weights, at model level: a freshly initialized size-S model; each layer's J fitted
  with ``summed_jacobian`` on the first 16 windows of D17's fit sample; the J-lens over the
  262,144-token evaluation slice. ``random_accuracy_by_layer`` is ``next_token_accuracy`` per
  reading, ``random_max_accuracy`` its maximum, ``random_chance`` is ``1 / 4096``.
* The meter and the comparison on the reference run's probe at every trace step, with G2.9's
  J's: ``meters`` maps each step to its J-lens ``lens_meter``; ``last_reading_kl_max`` is the
  largest last-reading ``kl_mean``; ``comparison_by_step`` maps each step to
  ``lens_divergence`` against the logit lens: recorded, not judged.

Prints a progress line at least every 30 seconds.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
