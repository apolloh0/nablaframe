"""G2.3: the logit lens and its discrimination tests (D14).

* Planted, at decoder level: residual streams at size S carrying a chosen token's
  direction plus noise, through ``logit_lens`` with the reference run's final layer norm
  and unembedding. ``planted_recovery`` is the fraction of positions whose top-1 lens
  token is the planted one.
* Random weights, at model level: a freshly initialized size-S model over the
  262,144-token evaluation slice, in windows of 256. ``random_accuracy_by_layer`` is
  ``next_token_accuracy`` per reading, ``random_max_accuracy`` its maximum,
  ``random_chance`` is ``1 / 4096`` and ``random_positions`` the positions scored.
* The meter on the reference run's probe at every trace step: ``meters`` maps each step
  to its ``lens_meter``; ``last_layer_kl_max`` is the largest last-reading ``kl_mean``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
