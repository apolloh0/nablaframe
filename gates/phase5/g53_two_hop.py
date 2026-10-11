"""G5.3: dictionaries on the two-hop model, against its twin's (judged; D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase5/g53_two_hop.json``. Run in nf-jax, on the Mac, after G4.1, G4.3 and G5.1
pass.

On G4.1's two-hop run at its final checkpoint and on its twin (``twin_params``), the model reading
each window's first 26 tokens. Every dictionary comes through ``cached``, as in G5.2, trained on
``zoo_windows(task, "two_hop", "train")``:

* on both models: ``topk`` at every block, the ``staircase``, and a ``transcoder`` at every block
* on the trained model only: ``topk`` at every block at each of ``ROBUST_SEEDS``

Sites are named, and dictionaries keyed, as in G5.2. Everything below is measured on ``zoo_windows(task, "two_hop",
"held_out")`` unless it says otherwise. The best swap block ``B`` is ``best_swap["block"]`` from
G4.3's report, which must exist, pass ``validate_report`` and say ``pass``; it is recorded as
``best_swap_block``.

* **Meters (judged):** as G5.2's, at ``READOUT["two_hop"]``: ``meters``, ``fvu_max`` and
  ``recovered_min``. The MLPs whose ablation costs less than ``BARS["ablation_cost"]`` have
  ``recovered`` None, and their change is recorded.
* **Faithful code (judged):** on G4.3's windows, ``eval_windows`` from Phase 4, with
  ``aligned_sources(task, windows, numpy.random.default_rng([SOURCE_SEED, 0]))``, at every block's
  ``hook_resid_post`` for the ``topk`` there and the ``staircase``'s site: ``patch_predictions`` at
  the query, positions 24 and 25, read at 25, once from the sources and once from the controls.
  ``patch_outcomes`` gives each pass's swap and transplant rates from the sources; its
  ``control_unchanged`` is how often the control patch's prediction is the unpatched pass's own.
  The ``"latents"`` pass moves, at each position, the latents ``assigned`` to ``b`` there, from
  ``f1_table``s of ``a``, ``b`` and ``c`` at that position. ``faithful``: ``{kind: {block:
  {pass: {"swap", "transplant", "control_unchanged"}, "assigned": {position: count}}}}``,
  ``"assigned"`` counting the latents the ``"latents"`` pass moves at each position.
  ``faithful_ratio_min``: the least ``faithful(residual, code)["ratio"]`` over every kind, block
  and rate (swap and transplant) where it applies, or None when it applies nowhere.
  ``control_unchanged_min``: the least ``"code"`` pass's ``control_unchanged`` over every kind and
  block. The transcoders aren't patched: two-hop's MLP outputs carry no planted variable at the
  query.
* **Selective latents (judged):** at position 25, ``b`` after block ``B`` and ``c`` after the last
  block, in the ``topk`` and the ``staircase`` there, and ``b`` in the code of block ``B``'s
  ``transcoder``. ``selectivity``, ``selective_f1_min`` and ``selective_margin_min`` as G5.2's.
* **Recorded:** ``recovery``: at every site and both positions 24 and 25, for each of ``a``, ``b``
  and ``c``, as G5.2's, which holds the other variables the spec lists. ``robustness`` and
  ``train_seconds`` as G5.2's, the variables being those judged at each block, or ``b`` and ``c``
  where neither is.

Measurements: REQUIRED_MEASUREMENTS["G5.3"]. Pass needs ``fvu_max <= BARS["fvu"]``,
``recovered_min >= BARS["recovered"]``, ``faithful_ratio_min >= BARS["faithful_ratio"]``,
``control_unchanged_min >= BARS["control_unchanged"]``, ``selective_f1_min >=
BARS["selective_f1"]`` and ``selective_margin_min >= 0``. The residual patch's rates are G4.3's
``swap_by_site``, ``transplant_by_site`` and ``control_unchanged_by_site`` at ``"query"``.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
