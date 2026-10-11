"""Dictionary badges: each shipped dictionary kind's results on the zoo, and dpatch's, from the Phase 5 reports (D38, D41, D44).

Spec: docs/specs/phase-5.md, D41, D44 and D45; contract: docs/deltas/phase-5-delta-5.md. Nothing
here is written by hand: every number on a badge is a value in a report, cited by its gate and its
path within that report's measurements, as in ``views/badges.py``, whose form these badges share.
A badge is about a kind, wherever its dictionaries are trained: the reference run's TopK SAEs carry
the TopK badge, beside their own recorded meters.

* ``SUBJECTS = ("topk", "staircase", "transcoder", "dpatch")``: the shipped kinds, then the
  operation.
* ``UNITS = ("fraction", "score", "ratio", "edges", "count")``: how the viewer shows a value. A
  fraction as a percentage; a score (a mean best F1) and a ratio to three decimals; edges and a
  count as integers.
* The notes, fixed text:

  - ``BOUNDS_NOTE = "A figure for any kind is its gate's bound over every kind that gate judges, not
    this kind's alone."``
  - ``TWIN_NOTE = "Each twin figure is a dictionary trained the same way on the model's
    random-weights twin, which has nothing to find."``
  - ``EDGE_NOTE = "The edge meter is judged across the MLPs whose mean ablation costs at least 0.1
    nats. Across a block it is recorded, not judged: masking a latent also moves attention (D42)."``
  - ``TRANSCODER_NOTE = "Two-hop's MLP outputs carry no planted variable at the query, so a
    transcoder's code is patched on the induction model only."``
  - ``DPATCH_NOTE = "The identities show dpatch changes nothing where it should change nothing. The
    rates judge a patch through a dictionary's code against the residual patch, so they judge the
    operation and the dictionaries together."``

* ``dictionary_badges(reports)``: ``reports`` maps gate ids to report dicts, and must hold G4.1 and
  G5.1 to G5.4, each passing ``validate_report`` and carrying its own gate id; any other key is
  ignored. ValueError otherwise, naming the problem. Returns one badge per subject, in
  ``SUBJECTS`` order: ``{"status", "sources", "facts", "notes"}``, plain JSON.

  - ``facts``: a list of ``{"label", "value", "unit", "gate", "path"}``, as ``views/badges.py``
    builds them: ``value`` is a copy of what the report holds at ``path`` in that gate's
    ``measurements``, and ``label`` a non-empty phrase, distinct within the badge. A path the
    report doesn't have is a ValueError.
  - ``sources``: the sorted, distinct gate ids its facts cite.
  - ``status``: ``"judged: pass"``, or ``"judged: fail"`` when any of its judging gates' status
    isn't ``pass``. ``topk`` is judged by G5.2, G5.3 and G5.4; ``staircase`` and ``transcoder`` by
    G5.2 and G5.3; ``dpatch`` by G5.1, G5.2 and G5.3.
  - ``notes``: ``topk`` has ``[BOUNDS_NOTE, TWIN_NOTE, EDGE_NOTE]``; ``staircase``
    ``[BOUNDS_NOTE, TWIN_NOTE]``; ``transcoder`` ``[BOUNDS_NOTE, TWIN_NOTE, TRANSCODER_NOTE]``;
    ``dpatch`` ``[BOUNDS_NOTE, DPATCH_NOTE]``.

  With ``B`` G5.3's ``best_swap_block``, ``last`` the two-hop model's last block (G4.1's
  ``models.two_hop.model_config.n_layers`` less one), and each site named ``"<kind> <output
  tap>"`` as the gates name them, the facts are, in order:

  - ``topk`` and ``staircase``, ``K`` being the kind:

    1. selective latents, each as ``"trained"`` then ``"twin"``, a score: G5.2's
       ``selectivity["K blocks.0.hook_resid_post cur"]`` and ``["K blocks.1.hook_resid_post
       next"]``; G5.3's ``selectivity["K blocks.<B>.hook_resid_post b"]`` and ``["K
       blocks.<last>.hook_resid_post c"]``
    2. the faithful code at the query after block ``B``, fractions: G5.3's ``faithful[K][str(B)]``
       ``["code"]["swap"]``, ``["residual"]["swap"]`` and ``["latents"]["swap"]``, the last being
       the single-latent patch, recorded
    3. the gates' bounds: G5.2's ``faithful_ratio_min`` (a ratio) and ``break_excess_max`` (a
       fraction); G5.3's ``faithful_ratio_min`` (a ratio) and ``control_unchanged_min`` (a
       fraction)
    4. ``topk`` only, for each key of G5.4's ``judged_mlps`` in its order:
       ``edge_pairs[key]["trained"]["meter"]`` and ``edge_pairs[key]["twin"]["meter"]``, edges

  - ``transcoder``: selective latents as above, a score, ``"trained"`` then ``"twin"``: G5.2's
    ``selectivity["transcoder blocks.0.hook_mlp_out cur"]`` and ``["transcoder
    blocks.1.hook_mlp_out next"]``; G5.3's ``selectivity["transcoder blocks.<B>.hook_mlp_out b"]``.
    Then G5.2's ``faithful_ratio_min`` (a ratio) and ``break_excess_max`` (a fraction).
  - ``dpatch``: G5.1's ``identities_exact`` and ``identities_checked``, counts; G5.2's
    ``faithful_ratio_min`` (a ratio) and ``break_excess_max`` (a fraction); G5.3's
    ``faithful_ratio_min`` (a ratio) and ``control_unchanged_min`` (a fraction).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUBJECTS = ("topk", "staircase", "transcoder", "dpatch")
UNITS = ("fraction", "score", "ratio", "edges", "count")
BOUNDS_NOTE = "A figure for any kind is its gate's bound over every kind that gate judges, not this kind's alone."
TWIN_NOTE = ("Each twin figure is a dictionary trained the same way on the model's random-weights twin, which has "
             "nothing to find.")
EDGE_NOTE = ("The edge meter is judged across the MLPs whose mean ablation costs at least 0.1 nats. Across a block "
             "it is recorded, not judged: masking a latent also moves attention (D42).")
TRANSCODER_NOTE = ("Two-hop's MLP outputs carry no planted variable at the query, so a transcoder's code is patched "
                   "on the induction model only.")
DPATCH_NOTE = ("The identities show dpatch changes nothing where it should change nothing. The rates judge a patch "
               "through a dictionary's code against the residual patch, so they judge the operation and the "
               "dictionaries together.")


def dictionary_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    raise NotImplementedError
