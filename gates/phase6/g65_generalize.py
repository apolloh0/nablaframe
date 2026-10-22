"""G6.5: generalize-this for the judged names, across held-out slices, dictionary seeds and every checkpoint (judged; D52).

Spec: docs/specs/phase-6.md, D44, D52 and "Discrimination tests on the zoo"; helpers in
``_common.py``. Report: ``reports/phase6/g65_generalize.json``. Run in nf-jax, on the Mac, after
G4.1, G5.2, G5.3 and G6.1 pass.

For each name in ``GENERALIZE`` (two-hop's ``b`` after block 2 at ``query``, induction's
``token(p+1)`` after block 1 at ``mechanism``): ``cli.record(RECORD_DIR / run, request,
context)``, the request being ``{"dictionary", "site": 0, "class", "description"}`` with the
block's TopK SAE, ``INSTANCES["name"]`` instances and ``SEEDS["generalize"]``, at the final
checkpoint. The run's context reads every checkpoint of G4.1's run through the final
dictionaries (D44), and holds the TopK SAEs of ``ROBUST_SEEDS`` at that block as its seeds. The
record holds the slices, the seeds, every checkpoint's row and the envelope.

Beside each checkpoint, the task's residual patch, which the record doesn't hold, since it is task
code (D29), each pass through ``run_intervened``:

* two-hop: G4.3's patch at the query, position 25, of ``blocks.2.hook_resid_post``, on G4.3's
  windows and aligned sources (``p4.eval_windows``; ``aligned_sources`` with
  ``numpy.random.default_rng([SOURCE_SEED, 0])``), its ``swap_rate`` from ``patch_outcomes``
* induction: G4.2's patch at ``blocks.0.hook_resid_post``, on ``induction_patch_windows``: for each
  ``j`` in ``PATCH_TARGETS``, from ``new_token_sources(windows, j, numpy.random.default_rng(j),
  vocab)``, how often the argmax at ``j + PATCH_LAG`` is the source's token at ``j``; their mean

``names``: ``{"<run> <description> after block <l> at <class>": {"record", "results", "patch",
"checks"}}``: the record's key; its results; ``[{"step", "rate"}]`` for every checkpoint; and
``envelope_checks`` of the results' envelope over the run's checkpoint steps. Over both names:

* ``slice_f1_min`` and ``slice_follows_min``: the least of each name's ``slices["f1"]`` and
  ``slices["follows"]`` lower ends, a range of None counting as None
* ``seed_f1_min`` and ``seed_follows_min``: the least of every seed row's ``f1`` and ``follows``
* ``step0_outside`` and ``one_final_run``: whether each holds for both names
* ``start_offset_max``: the greatest ``start_offset``, None where either name's is

the least and greatest through ``least`` and ``greatest``.

Measurements: REQUIRED_MEASUREMENTS["G6.5"]. Pass needs ``slice_f1_min >= BARS["slice_f1"]``,
``slice_follows_min >= BARS["slice_follows"]``, ``seed_f1_min >= BARS["seed_f1"]``,
``seed_follows_min >= BARS["seed_follows"]``, ``step0_outside``, ``one_final_run`` and
``start_offset_max <= BARS["start_offset"]``. Round 3 read, for two-hop and induction: slices' F1
0.780 to 0.825 and 0.787 to 0.794, follows 0.902 to 0.945 and 0.953 to 1.000; seeds' F1 0.756 to
0.832 and 0.747 to 0.772, follows 0.920 to 0.996; each envelope one run to the end, from step 6,000
(the probe at 6,000) and from step 700 (the probe at 750); step 0 outside both.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
