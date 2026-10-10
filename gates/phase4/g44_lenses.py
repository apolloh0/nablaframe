"""G4.4: the lenses, the J-lens coordinate swap and direct logit attribution at model level. Recorded, not judged (D34).

Spec: docs/specs/phase-4.md, "Lenses", "J-lens coordinate swap" and "Direct logit attribution";
helpers in ``_common.py``. Report: ``reports/phase4/g44_lenses.json``. Run in nf-jax, on the
Mac, after G4.1 passes.

At each zoo model's final checkpoint, from ``instrumented`` passes on ``eval_windows``:

* **J:** for the two-hop model and its twin, each block's summed J but the last's, which is the
  identity: ``summed_jacobian(block_map(...), xs)`` on that block's ``hook_resid_post`` for
  ``fit_windows(task)``, over D17's valid positions for a sequence of 26, 16 to 24. Saved with
  ``save_jacobian`` under ``runs/phase4/two_hop/jlens/``, keyed by ``jacobian_key``, and reused
  when present.
* **Lens ranks:** at each of ``QUERY_POSITIONS`` and each block's output, ``logit_lens`` and
  ``jlens_lens`` readings, and the median over windows of ``token_ranks`` of the intermediate
  ``b`` and the answer ``c``, for the two-hop model and its twin: ``lens_ranks``, as
  ``{"trained"|"twin": {"logit_lens"|"jlens": {"<position>": {"<block>": {"intermediate",
  "answer"}}}}}``.
* **The coordinate swap:** at each block's output, ``jswap`` with that block's J of the
  intermediate ``b`` and the counterpart ``k``, the ``aligned_sources`` counterpart drawn with
  ``SOURCE_SEED``, at ``QUERY_POSITIONS``; at the last block J is the identity, so it's the logit
  lens's swap. ``jswap_by_layer``: per block, ``{"swap", "unchanged"}``, how often the answer
  becomes the counterpart's object ``d`` and how often it stays ``c``.
* **Direct logit attribution:** for both zoo models and their twins, ``attribution`` of the
  correct next token's logit, from each part ``parts_from_trace`` names, taken from the
  ``instrumented`` pass, averaged over windows and mechanism positions: ``attribution``, as
  ``{"induction"|"two_hop": {"trained"|"twin": {part: mean}}}``.

Measurements: REQUIRED_MEASUREMENTS["G4.4"]. Its status is always ``pass``. Prints a progress
line per J and per block.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
