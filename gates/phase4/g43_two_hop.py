"""G4.3: the two-hop aligned patch, the signature demo (D36). Judged on its distractor control; the swap and transplant profiles are recorded.

Spec: docs/specs/phase-4.md, "Patch, on the two-hop model"; helpers in ``_common.py``. Report:
``reports/phase4/g43_two_hop.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's two-hop run at its final checkpoint, on ``eval_windows``:
``aligned_sources(task, windows, numpy.random.default_rng([SOURCE_SEED, 0]))`` gives each window
its aligned source, its control and its counterpart. ``unpatched_accuracy``: how often the
unpatched model's answer, the argmax at position 25, is ``c``. Then for each site in
``TWO_HOP_SITES`` and each block's ``hook_resid_post``, two patches through ``run_intervened`` at
the site's positions, one from the sources and one from the controls, each read at position 25:

* ``swap_by_site`` and ``transplant_by_site``: ``patch_outcomes`` of the source patch, as
  ``{site: [rate per block]}``
* ``control_unchanged_by_site``: how often the control patch's answer equals the unpatched
  model's own answer, in the same shape; ``control_unchanged_min``, its minimum over every site
  and block
* ``best_swap``: ``{"site", "block", "rate"}`` for the largest swap rate, ties going to the
  earlier site in ``TWO_HOP_SITES`` order, then the earlier block

Measurements: REQUIRED_MEASUREMENTS["G4.3"]. Also recorded: how often each source patch leaves
the answer ``c``. Pass needs ``control_unchanged_min >= BARS["control_unchanged"]``.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
