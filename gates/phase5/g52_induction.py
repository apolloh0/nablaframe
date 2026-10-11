"""G5.2: dictionaries on the induction model, against its twin's (judged; D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase5/g52_induction.json``. Run in nf-jax, on the Mac, after G4.1 and G5.1 pass.

On G4.1's induction run at its final checkpoint and on its twin (``twin_params``), the model
reading each window's first 63 tokens. Every dictionary comes through ``cached``, under
``dictionary_key`` with the checkpoint's state hash or ``twin_state``, trained with ``settings`` on
``zoo_windows(task, "induction", "train")``:

* on both models: ``topk`` at blocks 0 and 1, the ``staircase``, and a ``transcoder`` at blocks 0
  and 1
* on the trained model only: ``topk`` at blocks 0 and 1 at each of ``ROBUST_SEEDS``, each on its
  own training windows

Each site is named ``"<kind> <output tap>"``. A dictionary's ``tokens``, for its key and for its training
alike, are what the model reads of its training windows, ``p4.tokens(task, windows)``; G5.6 finds each
shipped dictionary by recomputing that key. Everything below is measured on
``zoo_windows(task, "induction", "held_out")``.

* **Meters (judged):** per site, ``fvu`` and ``dead_share`` over every position, on both models.
  On the trained model, ``readout_loss`` at ``READOUT["induction"]`` of the unintervened pass, of
  ``spliced_logits`` and of ``ablated_logits`` at the site's output tap, and ``loss_recovered``.
  ``meters``: ``{site: {"fvu", "dead_share", "twin_fvu", "twin_dead_share", "clean_loss",
  "spliced_loss", "ablated_loss", "cost", "recovered"}}``; ``fvu_max`` over the trained model's
  sites; ``recovered_min`` over every site where ``recovered`` isn't None, or None when none is.
* **Faithful code (judged):** on G4.1's run, ``induction_patch_windows`` (G4.2's windows), for
  each ``j`` in ``PATCH_TARGETS``, ``patch_predictions`` at position ``j``, read at ``j +
  PATCH_LAG``, at block 0's ``hook_resid_post`` for the ``topk`` and the ``staircase`` (its site
  0), and at block 0's ``hook_mlp_out`` for the transcoder:

  - **following:** from ``induction_sources(task, windows, j, "follow")``, how often the
    prediction is the source's token at ``j``
  - **breaking:** from ``induction_sources(task, windows, j, "break")``, how often the prediction
    is the unpatched pass's own

  For each, the ``"residual"``, ``"code"`` and ``"latents"`` passes' rates, one per target. The
  ``"latents"`` pass moves the latents ``assigned`` at the site to ``cur`` for following and to
  ``prev`` for breaking, from ``f1_table``s of every variable at the mechanism positions.
  ``faithful``: ``{kind: {"follow": {pass: [rate per target]}, "break": {...}, "assigned":
  {"cur": count, "prev": count}}}``, ``"assigned"`` counting the latents the ``"latents"`` pass
  moves. ``faithful_ratio_min``: the least ``faithful(residual,
  code)["ratio"]`` over every kind and target of following where it applies, or None when it
  applies nowhere. ``break_excess_max``: the greatest ``code - residual`` over every kind and
  target of breaking.
* **Selective latents (judged):** at the mechanism positions, ``mean_best_f1`` of
  ``f1_table`` for ``cur`` at block 0 and ``next`` at block 1: in the ``topk`` at that block, the
  ``staircase`` at that site, and the ``transcoder`` at that block, whose code reads its MLP's
  input. ``selectivity``: ``{"<site> <variable>": {"trained", "twin", "margin"}}``, the margin
  being ``selective_margin``; ``selective_f1_min``, the least trained value; and
  ``selective_margin_min``, the least margin.
* **Recorded:** ``recovery``: at every site, for each of ``cur``, ``prev`` and ``next``, the
  trained and twin ``mean_best_f1`` and ``probe_accuracy`` from the site's input activation and
  from its code. ``robustness``: for each of ``ROBUST_SEEDS``, each ``topk``'s ``fvu`` and its
  ``mean_best_f1`` for the variable judged at its block. ``train_seconds``: per dictionary, None
  when it came from the cache.

Measurements: REQUIRED_MEASUREMENTS["G5.2"]. Pass needs ``fvu_max <= BARS["fvu"]``,
``recovered_min >= BARS["recovered"]``, ``faithful_ratio_min >= BARS["faithful_ratio"]``,
``break_excess_max <= BARS["break_excess"]``, ``selective_f1_min >= BARS["selective_f1"]`` and
``selective_margin_min >= 0``. The residual patch's following rates at block 0 are G4.2's
``patch_followed_by_tap`` for ``blocks.0.hook_resid_post``: the same windows, sources and Mac.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
