"""G3.3: the J-lens experiment, the first answer to D21's open question. Recorded, not judged.

At the final checkpoint, for each block before the last and each of ``SCALE_FACTORS``: at
``EXPERIMENT_POSITIONS`` seeded positions in ``fit_range``, each with a seeded token (never
``<unk>`` or ``<eos>``), steer block l's output at that position along the J-lens direction
and along the logit-lens direction, scaled by that factor times ``steering_scales``' mean
residual norm. Record the change in the token's logit at the steered position and its mean
change over ``later_positions``. Measurements: ``layers``, ``scale_factors``, and
``results_by_layer`` with, per layer and scale, both directions' mean same-position and
later-position changes and their ratio.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
