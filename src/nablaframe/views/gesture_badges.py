"""Gesture badges: each gesture's results on the zoo, and dsteer's, from the Phase 6 reports (D38, D47 to D52).

Spec: docs/specs/phase-6.md, "Records, the viewer and badges"; contract:
docs/deltas/phase-6-delta-5.md. Nothing here is written by hand: every number on a badge is a value
in a report, cited by its gate and its path within that report's measurements, as in
``views/badges.py`` and ``views/dictionary_badges.py``, whose form these badges share. A badge is
about a gesture wherever it runs: the reference run's records carry the same badges, whose notes say
that there the gestures are recorded, not judged.

* ``SUBJECTS = ("name", "edge", "open", "generalize", "dsteer")``: the four gestures, named as
  ``records.KINDS`` names them, then the operation.
* ``UNITS = ("fraction", "ratio", "score", "count", "parts", "step", "span", "flag", "label")``: how
  the viewer shows a value. A fraction as a percentage; a ratio, and a score (an F1), to three
  decimals; a count and a step as integers; parts, a number of upstream parts, as an integer, or to
  one decimal where a median falls between two; a span, ``[first, last]`` steps, as "first to
  last"; a flag as yes or no; a label as the report holds it. A null, in any unit, reads as none.
* The notes, fixed text:

  - ``REFERENCE_NOTE = "On the reference run every gesture is recorded, not judged: it has no
    planted structure to recover (D44). G6.6's report holds its readings."``
  - ``STEERING_NOTE = "Steering swaps a value's set for another value's, drawn from every value of
    the description that the window doesn't hold. Where few of those values have a set that names
    them, as on the reference run, steering rarely follows."``
  - ``LABEL_NOTE = "An edge at one position is labelled carried where the swap still follows at
    least half as often with every block between the two sites frozen: the residual stream passes
    on whatever is put into it. A label says little where follows is no higher than its null."``
  - ``TWIN_OPEN_NOTE = "On the random-weights twin, a planted set at its target's own position ranks
    in the top 8 as well, because the residual stream carries it. So only the induction target, read
    across positions, is judged on the twin."``
  - ``CHECKPOINTS_NOTE = "Every checkpoint is read through the final dictionaries (D44), so a name
    formed another way earlier in training can't be seen. The gap shows the checkpoints where the
    probe reads the variable but the name doesn't hold."``
  - ``PATCH_NOTE = "Beside each checkpoint, G6.5 patches two-hop's residual stream at position 25
    only, where the name is read. G4.3's query patch moves positions 24 and 25."``
  - ``DSTEER_NOTE = "The identities show dsteer changes nothing where it should change nothing:
    removing and adding nothing, removing only latents whose code is zero, and adding only zero
    amounts."``

* ``gesture_badges(reports)``: ``reports`` maps gate ids to report dicts, and must hold G6.1 to
  G6.5, each passing ``validate_report`` and carrying its own gate id; any other key is ignored.
  ValueError otherwise, naming the problem. Returns one badge per subject, in ``SUBJECTS`` order:
  ``{"status", "sources", "facts", "notes"}``, plain JSON.

  - ``facts``: a list of ``{"label", "value", "unit", "gate", "path"}``, as ``views/badges.py``
    builds them: ``value`` is a copy of what the report holds at ``path`` in that gate's
    ``measurements``, and ``label`` a non-empty phrase, naming its place where it has one, distinct
    within the badge. A path the report doesn't have is a ValueError.
  - ``sources``: the sorted, distinct gate ids its facts cite.
  - ``status``: ``"judged: pass"``, or ``"judged: fail"`` when any of its judging gates' status
    isn't ``pass``: ``name`` is judged by G6.1 and G6.2, ``edge`` by G6.1 and G6.3, ``open`` by
    G6.1 and G6.4, ``generalize`` by G6.1 and G6.5, and ``dsteer`` by G6.1.
  - ``notes``: ``name`` has ``[STEERING_NOTE, REFERENCE_NOTE]``; ``edge`` ``[LABEL_NOTE,
    REFERENCE_NOTE]``; ``open`` ``[TWIN_OPEN_NOTE, REFERENCE_NOTE]``; ``generalize``
    ``[CHECKPOINTS_NOTE, PATCH_NOTE, REFERENCE_NOTE]``; ``dsteer`` ``[DSTEER_NOTE]``.

  A place is a key of a report's ``names``, ``edges`` or ``targets`` mapping (``"<run> <description>
  after block <l> at <class>"`` or ``"<run> <relation>"``), and each gate's places are taken in
  sorted order. The facts are, in order:

  - every gesture first, from G6.1, flags: ``records_by_kind[<kind>][0]``, whether a record built
    twice from one request was byte-identical, and ``records_by_kind[<kind>][1]``, whether the
    command line's was too
  - ``name``: G6.2's ``named_as_planted_min``, ``follows_min``, ``null_follows_max``,
    ``twin_named_max`` and ``twin_follows_max``, fractions. Then for each of G6.2's places whose
    ``judged`` is true: ``names[place]["trained"]``'s ``["named_as_planted"]``,
    ``["steering"]["follows"]``, ``["effect"]["first"]`` and ``["effect"]["last"]``, fractions:
    the shares of values named as planted, followed, and ranked first and last by their set's
    effect
  - ``edge``: G6.3's ``follows_min``, ``null_follows_max``, ``output_follows_min``, ``kept_max``,
    ``null_kept_min`` and ``residual_alone_max``, fractions; ``labels_computed``, a flag; and
    ``twin_excess_max``, a fraction. Then for each of G6.3's places, judged or recorded:
    ``edges[place]["trained"]``'s ``["follows"]`` and ``["null_follows"]``, fractions, and
    ``["label"]``, a label, each label saying "(recorded)" where the place's ``judged`` is false
  - ``open``: G6.4's ``reading_ratio_max``, a ratio; ``within_min``, ``planted_top_min``,
    ``random_within_max`` and ``twin_planted_top_max``, fractions. Then for each of G6.4's places:
    ``targets[place]["trained"]["orders"]["absolute"]["reading_median"]`` and
    ``targets[place]["bar"]``, parts; ``targets[place]["trained"]["orders"]["absolute"]["within"]``
    and ``targets[place]["trained"]["planted"]["top"]``, fractions; and, where the place's
    ``twin_judged`` is true, ``targets[place]["twin"]["planted"]["top"]``, a fraction
  - ``generalize``: G6.5's ``slice_f1_min``, a score; ``slice_follows_min``, a fraction;
    ``seed_f1_min``, a score; ``seed_follows_min``, a fraction; ``step0_outside`` and
    ``one_final_run``, flags; and ``start_offset_max``, a count. Then for each of G6.5's places:
    ``names[place]["results"]["envelope"]["final"]``, a span, and
    ``names[place]["results"]["envelope"]["probe_reads_from"]``, a step
  - ``dsteer``: G6.1's ``identities_exact`` and ``identities_checked``, counts
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUBJECTS = ("name", "edge", "open", "generalize", "dsteer")
UNITS = ("fraction", "ratio", "score", "count", "parts", "step", "span", "flag", "label")
REFERENCE_NOTE = ("On the reference run every gesture is recorded, not judged: it has no planted structure to "
                  "recover (D44). G6.6's report holds its readings.")
STEERING_NOTE = ("Steering swaps a value's set for another value's, drawn from every value of the description that "
                 "the window doesn't hold. Where few of those values have a set that names them, as on the reference "
                 "run, steering rarely follows.")
LABEL_NOTE = ("An edge at one position is labelled carried where the swap still follows at least half as often with "
              "every block between the two sites frozen: the residual stream passes on whatever is put into it. A "
              "label says little where follows is no higher than its null.")
TWIN_OPEN_NOTE = ("On the random-weights twin, a planted set at its target's own position ranks in the top 8 as well, "
                  "because the residual stream carries it. So only the induction target, read across positions, is "
                  "judged on the twin.")
CHECKPOINTS_NOTE = ("Every checkpoint is read through the final dictionaries (D44), so a name formed another way "
                    "earlier in training can't be seen. The gap shows the checkpoints where the probe reads the "
                    "variable but the name doesn't hold.")
PATCH_NOTE = ("Beside each checkpoint, G6.5 patches two-hop's residual stream at position 25 only, where the name is "
              "read. G4.3's query patch moves positions 24 and 25.")
DSTEER_NOTE = ("The identities show dsteer changes nothing where it should change nothing: removing and adding "
               "nothing, removing only latents whose code is zero, and adding only zero amounts.")


def gesture_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    raise NotImplementedError
