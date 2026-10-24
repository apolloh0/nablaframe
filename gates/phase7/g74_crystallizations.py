"""G7.4: every judged and recorded crystallization, trained and twin, with its counterfactual and its envelope (D56 to D58).

Spec: docs/specs/phase-7.md, D56 to D58, "What each run declares" and "Discrimination tests on the
zoo"; helpers in ``_common.py``. Report: ``reports/phase7/g74_crystallizations.json``. Run in
nf-jax, on the Mac, after G4.1, G5.2, G5.3, G6.3, G6.7 and G7.1 to G7.3 pass.

For each of ``JUDGED`` and then ``RECORDED``, in order, a crystallize-this record through
``cli.record`` into ``runs/phase7/gestures/<run>/``. The request: the run's id, its final
checkpoint, the crystallization's block, part, class, function and form, ``instances`` the
held-out windows' count, ``SEED``, and ``cites`` as below. The run's ``cli.Context``:

* **induction and two-hop:** G4.1's checkpoints, every one; Phase 5's training and held-out
  windows (``zoo_windows``) as the fitting and held-out windows; the twin,
  ``init_params(config, seed=1)``; ``functions_for(run)``; and ``cited`` reading Phase 6's
  records, ``runs/phase6/gestures/<run>/``.
* **modular addition:** G7.1's checkpoints, every one; its training and held-out pairs; its twin
  and functions; and ``cited`` reading ``runs/phase7/gestures/modular_addition/``.

``cites``: ``crystallize.cite`` of every record in the folder ``cited`` reads, by its
``index.json``, with the crystallization's class, ``describes`` of its run's held-out windows,
the dictionary at its part's output site, and its function's relation. The dictionary is the run's Phase 5 TopK SAE at the block's
``hook_resid_post`` for induction and two-hop, and G7.2's for modular addition, each at site 0.

Each record's results are summarized in ``crystallizations``, ``{"<run> <part> <function>
<form>": {"key", "judged", "meter", "wrong", "counterfactual", "twin", "envelope", "cites"}}``,
those being the record's, ``twin`` holding the twin's meter and wrong functions. The judged ones'
``crystal_checks`` go through ``judged``: its seven values are the report's measurements of the
same names.

Measurements: REQUIRED_MEASUREMENTS["G7.4"]. Pass needs every bar. Records already in a folder
under their keys are read, not built again, as ``cli.record`` does.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
