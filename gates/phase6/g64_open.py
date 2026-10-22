"""G6.4: open-this-up on the judged targets, on the trained zoo models and their twins (judged; D51).

Spec: docs/specs/phase-6.md, D51 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase6/g64_open.json``. Run in nf-jax, on the Mac, after G4.1, G5.2, G5.3 and
G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins, with the Phase 5 TopK SAEs from the
cache. Each target in ``OPEN`` is opened along a relation the run declares (two-hop's ``carry``
and ``hop``, induction's ``copy``). With ``R`` that relation, ``up`` the reading of the TopK SAE
at ``R.up``'s block and ``down`` the reading at ``R.down``'s, on each model:
``open_targets(up, down, relation_candidates(up, down, R.name), instances=INSTANCES["open"],
seed=SEEDS["open"])``. So each instance opens the downstream place's set for its own value, at its
downstream position, into the block below, and its planted set is the upstream place's set for its
own value at the upstream position:

* the carry: ``b``'s set at ``query`` after block 2, into block 1; planted, ``b``'s set at
  ``query`` after block 1
* the hop: ``token(p+1)``'s set at ``query`` after block 3, into block 2; planted, ``b``'s set at
  ``query`` after block 2
* the copy: ``token(p+1)``'s set at ``s + P - 1`` after block 1, into block 0; planted,
  ``token(p)``'s set at the sole target ``s`` after block 0

``targets``: ``{"<run> <relation>": {"target", "planted", "bar", "twin_judged", "trained",
"twin"}}``: the target's and the planted set's places, each ``{"description", "block",
"class"}``; the bar from ``OPEN``; whether ``TWIN_OPEN`` holds it; and each model's
``open_targets`` result. Over the targets:

* ``reading_ratio_max``: the greatest ``ratio`` of a trained absolute-order ``reading_median``
  to its bar
* ``within_min``: the least trained absolute-order ``within``
* ``planted_top_min``: the least trained ``planted["top"]``
* ``random_within_max``: the greatest trained random-order ``within``
* ``twin_planted_top_max``: the greatest twin ``planted["top"]`` over ``TWIN_OPEN``'s targets

each through ``greatest`` or ``least``. Recorded beside them: signed order, and the two-hop twin,
whose residual stream carries the same position's set one block earlier, so its planted set ranks
high as well.

Measurements: REQUIRED_MEASUREMENTS["G6.4"]. Pass needs ``reading_ratio_max <=
BARS["reading_ratio"]``, ``within_min >= BARS["within"]``, ``planted_top_min >=
BARS["planted_top"]``, ``random_within_max <= BARS["random_within"]`` and ``twin_planted_top_max
<= BARS["twin_planted_top"]``. Round 3 read, for the carry, the hop and the copy: medians of 8, 32
and 24 parts; within 128 parts for 1.000, 0.930 and 0.992; the planted set's first part in the top
8 for 1.000, 0.781 and 0.797; in random order within 128 for at most 0.016; on the induction twin,
the planted set in the top 8 for 0.000.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
