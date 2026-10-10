"""G4.2: the induction model's attention view, ablation, component swap and patch (judged; D36).

Spec: docs/specs/phase-4.md, "Discrimination tests at model level"; helpers in ``_common.py``.
Report: ``reports/phase4/g42_induction.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's induction run at its final checkpoint, on ``eval_windows``, the model reading each
window's first 63 tokens, at the task's mechanism positions:

* **Attention:** ``induction_scores`` of ``blocks.1.attn.hook_pattern``, the tap the attention
  view reads, from ``instrumented``: ``attention_by_head`` and ``best_head_attention``.
  ``induction_heads`` lists the block-1 heads at or above ``BARS["induction_attention"]``. The
  twin's (``twin_params``) best head, ``twin_best_attention``, beside ``uniform_baseline`` on the
  same windows and positions.
* **Ablation:** every head in ``induction_heads`` zeroed together through
  ``blocks.1.attn.hook_z``, one ``zero`` edit with ``head`` per head in a single spec:
  ``ablation_fraction_removed``, the ``fraction_removed`` of mechanism accuracy, null when no
  head qualifies. Each of the eight heads zeroed alone with ``head_zero``:
  ``fraction_removed_by_head``, keyed ``"<layer>.<head>"``.
* **Component swap:** mechanism accuracy at every G4.1 checkpoint; ``swap_step`` is
  ``last_at_chance`` of them, and ``swap_fraction_removed`` the ``fraction_removed`` by a ``swap``
  of ``blocks.1.attn`` from that step, ``params_at`` loading G4.1's checkpoints.
* **Patch:** on ``induction_patch_windows``, for each ``j`` in ``PATCH_TARGETS``, sources from
  ``new_token_sources(windows, j, numpy.random.default_rng(j), 64)``, as the feasibility run drew
  them, patched at position ``j`` at ``blocks.0.hook_resid_pre``, ``blocks.0.hook_resid_post`` and
  ``blocks.1.hook_resid_post``. Each records how often the argmax at ``j + PATCH_LAG`` is the
  source's token at ``j``: ``patch_followed_by_tap``, a list per tap with one rate per target;
  ``patch_after_block0_min`` and ``patch_after_block1_max``.

Measurements: REQUIRED_MEASUREMENTS["G4.2"], ``mechanism_accuracy`` and ``chance`` among them.
Pass needs every bar. G4.1's induction run retraces the feasibility run for seed 0 at batch 32, so
on the same Mac these numbers should reproduce that run's (docs/deltas/phase-4-delta-3.md,
"Feasibility").
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
