"""Model-level badges: each view's and each operation's result on Phase 4's zoo, from the gate reports (D34, D38).

Spec: docs/specs/phase-4.md, D34 and D38; contract: docs/deltas/phase-4-delta-4.md. Nothing here
is written by hand: every number on a badge is a value in a Phase 4 report, cited by its gate and
its path within that report's measurements.

* ``SUBJECTS``: the shipped views, then the operations: ``("per_token_loss", "logit_lens",
  "jlens", "attribution", "attention_pattern", "residual_norm", "layer_structure", "zero",
  "mean", "resample", "patch", "add", "swap", "jswap")``.
* ``UNITS = ("fraction", "fraction removed", "rank", "step", "tokens", "logit")``: how the viewer
  shows a value. A fraction as a percentage; a fraction removed as a number, which can pass 1; a
  rank 0-based as the reports hold it; a step; a count of tokens; a logit contribution, signed.
* The notes, fixed text:

  - ``JLENS_FIT_NOTE = "J is fitted over D17's valid positions, 16 to 24, which leave out ⇒, the
    position the answer is read from."``
  - ``ATTRIBUTION_NOTE = "Attribution measures each part's direct contribution to the logit, not
    whether the model needs that part."``
  - ``NOT_TESTED_NOTE = "No Phase 4 test exercises this operation on a model with known
    structure."``
  - ``READOUT_NOTE = "An exact readout of the trace, checked by its hash; there is no model-level
    structure for it to recover."``

* ``model_level_badges(reports)``: ``reports`` maps gate ids to report dicts, and must hold
  G4.1 to G4.4, each passing ``validate_report`` and carrying its own gate id; any other key is
  ignored. ValueError otherwise, naming the problem. Returns one badge per subject, in
  ``SUBJECTS`` order: ``{"status", "sources", "facts", "notes"}``, plain JSON.

  - ``facts``: a list of ``{"label", "value", "unit", "gate", "path"}``. ``path`` is the list
    of keys and list indices leading to ``value`` inside that gate's ``measurements``, so
    ``value`` is exactly what the report holds there. ``label`` is a non-empty phrase.
  - ``sources``: the sorted, distinct gate ids its facts cite.
  - ``status``: ``"readout"`` for ``per_token_loss``, ``residual_norm`` and
    ``layer_structure``, with ``READOUT_NOTE`` and no facts. ``"not tested"`` for ``mean``,
    ``resample`` and ``add``, with ``NOT_TESTED_NOTE`` and no facts. ``"recorded"`` for
    ``logit_lens``, ``jlens``, ``attribution`` and ``jswap``. ``"judged: pass"``, or
    ``"judged: fail"`` when any judged source's status isn't ``pass``, for
    ``attention_pattern``, ``zero`` and ``swap`` (judged by G4.2) and ``patch`` (G4.2 and
    G4.3).

  With ``s`` and ``b`` G4.3's ``best_swap`` site and block, and ``last`` the two-hop model's
  last block (G4.1's ``models.two_hop.model_config.n_layers`` less one), the facts are:

  - ``logit_lens`` and ``jlens``, each from its own lens in G4.4's ``lens_ranks``: the
    intermediate's median rank at positions 24 and 25 after block ``b``, for ``trained`` and
    ``twin``; the answer's at position 25 after block ``last``, trained; G4.3's
    ``swap_by_site[s][b]``; and G4.1's ``models.two_hop.model_config.vocab``. ``jlens`` adds
    G4.4's ``jswap_by_layer[str(b)]`` ``swap`` and ``unchanged``, and the note
    ``JLENS_FIT_NOTE``; ``logit_lens`` has no notes.
  - ``attribution``: for each of ``induction`` and ``two_hop``, the two parts of G4.4's
    ``attribution[model]["trained"]`` with the largest absolute mean, ties to the part name
    earlier in sorted order; and G4.2's ``ablation_fraction_removed``. Note:
    ``ATTRIBUTION_NOTE``.
  - ``attention_pattern``: G4.2's ``best_head_attention``, ``twin_best_attention`` and
    ``uniform_baseline``.
  - ``zero``: G4.2's ``ablation_fraction_removed``.
  - ``swap``: G4.2's ``swap_step`` and ``swap_fraction_removed``.
  - ``patch``: G4.2's ``patch_after_block0_min`` and ``patch_after_block1_max``, and G4.3's
    ``control_unchanged_min`` and ``best_swap.rate``.
  - ``jswap``: G4.4's ``jswap_by_layer[str(b)].swap`` and G4.3's ``best_swap.rate``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUBJECTS = ("per_token_loss", "logit_lens", "jlens", "attribution", "attention_pattern", "residual_norm",
            "layer_structure", "zero", "mean", "resample", "patch", "add", "swap", "jswap")
UNITS = ("fraction", "fraction removed", "rank", "step", "tokens", "logit")
JLENS_FIT_NOTE = ("J is fitted over D17's valid positions, 16 to 24, which leave out ⇒, the position the answer "
                  "is read from.")
ATTRIBUTION_NOTE = ("Attribution measures each part's direct contribution to the logit, not whether the model "
                    "needs that part.")
NOT_TESTED_NOTE = "No Phase 4 test exercises this operation on a model with known structure."
READOUT_NOTE = "An exact readout of the trace, checked by its hash; there is no model-level structure for it to recover."


def model_level_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    raise NotImplementedError
