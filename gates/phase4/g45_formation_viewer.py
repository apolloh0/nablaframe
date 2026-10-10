"""G4.5: the induction formation demo, and both zoo models' views.

Spec: docs/specs/phase-4.md, "The induction formation demo" and "Gates"; helpers in
``_common.py``. Report: ``reports/phase4/g45_formation_viewer.json``. Run in nf-jax, on the Mac,
after G4.1 passes.

* **Formation,** from G4.1's induction trace, what the scrubber shows: at every checkpoint, the
  mechanism positions' mean loss, from the traced logits against the full ``probe(task)``
  windows, and the best block-1 head's ``induction_scores`` of the traced
  ``blocks.1.attn.hook_pattern``. ``formation_loss_step`` is ``steepest_drop`` of the losses and
  ``formation_attention_step`` ``steepest_rise`` of the scores. Both curves are recorded.
* **Views:** each zoo model's views built with ``build_views``, with the task's vocabulary, into
  ``runs/phase4/<name>/views/``, then again into a scratch folder, compared file by file:
  ``builds_identical``. ``external_urls`` counts external references in the viewer as G2.6 does.

Measurements: REQUIRED_MEASUREMENTS["G4.5"]. Pass needs ``builds_identical`` true and
``external_urls == 0``; the formation steps are recorded.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
