"""G6.3: every relation, on the trained zoo models and their twins (judged where a task requires it; D50).

Spec: docs/specs/phase-6.md, D50, "What each run declares" and "Discrimination tests on the zoo";
helpers in ``_common.py``. Report: ``reports/phase6/g63_edges.json``. Run in nf-jax, on the Mac,
after G4.1, G5.2, G5.3 and G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins, with the Phase 5 TopK SAEs from the
cache. For every relation the run declares (two-hop's ``hop``, ``carry`` and ``two back``;
induction's ``copy``), its upstream and downstream places read with the TopK SAEs at their blocks:

* **trained:** ``cli.record(RECORD_DIR / run, request, context)``, the request being
  ``{"relation", "up", "up_site": 0, "down", "down_site": 0}`` with the two TopK SAEs' content
  hashes, ``INSTANCES["edge"]`` instances and ``SEEDS["edge"]``, at the final checkpoint: the
  run's edge record, which the record cache answers when it already holds it
* **twin:** ``edge.edge(up, down, relation, instances=INSTANCES["edge"], seed=SEEDS["edge"])`` on
  the twin's readings with the twin's TopK SAEs. The twin isn't a checkpoint of the run, so it gets
  no record.

``edges``: ``{"<run> <relation>": {"judged", "record", "trained", "twin"}}``: the declaration's
flag, the record's key, and each model's results. Over the judged relations, the hop and the copy:

* ``follows_min`` and ``null_follows_max``: the least trained ``follows`` and the greatest
  trained ``null_follows``
* ``output_follows_min``: the least trained ``output["follows"]``, a judged edge without an
  output counting as None
* ``kept_max`` and ``null_kept_min``: the greatest trained ``kept`` and the least ``null_kept``
* ``residual_alone_max``: the greatest trained ``residual_alone["follows"]`` over the judged
  edges whose ``residual_alone`` isn't None, the edges at one position; None where none has one
* ``labels_computed``: every judged trained edge is labelled ``computed``
* ``twin_excess_max``: the greatest ``twin_excess`` of each judged twin edge's ``follows`` and
  ``null_follows``

each through ``least`` or ``greatest``. The carry and two back are recorded beside them.

Measurements: REQUIRED_MEASUREMENTS["G6.3"]. Pass needs ``follows_min >= BARS["edge_follows"]``,
``null_follows_max <= BARS["edge_null_follows"]``, ``output_follows_min >=
BARS["output_follows"]``, ``kept_max <= BARS["kept"]``, ``null_kept_min >= BARS["null_kept"]``,
``residual_alone_max <= BARS["residual_alone"]``, ``labels_computed`` and ``twin_excess_max <=
BARS["twin_excess"]``. Round 3 read the hop and the copy: follows 0.939 and 1.000 (nulls 0.029 and
0.008), the output 0.934 and 1.000, kept 0.211 and 0.000 (nulls 1.013 and 0.962), the hop 0.025
through the residual alone; the twins' excess -0.032 and 0.000; the carry 0.988, and 0.918 through
the residual alone, so carried.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
