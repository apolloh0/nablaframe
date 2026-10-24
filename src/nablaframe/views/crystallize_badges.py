"""Crystallize-this's badge and set's, from the Phase 7 reports (D38, D55 to D57, C11).

Spec: docs/specs/phase-7.md, "Records, the viewer and badges"; contract:
docs/deltas/phase-7-delta-4.md. Nothing here is written by hand: every number on a badge is a value
in a report, cited by its gate and its path within that report's measurements, in the form of
``views/gesture_badges.py``'s badges. A badge is about crystallize-this wherever it runs, judged on
the zoo and recorded elsewhere, as its notes say.

* ``SUBJECTS = ("crystallize", "set")``: the gesture, named as ``records.KINDS`` names it, then the
  operation it puts a component to its function with (D55).
* ``UNITS = ("fraction", "ratio", "nats", "count", "span", "flag", "label")``: how the viewer shows
  a value. A fraction as a percentage; a ratio to three decimals; a KL in nats to three decimals,
  followed by "nats"; a count as an integer; a span, ``[first, last]`` steps, as "first to last"; a
  flag as yes or no; a label as the report holds it. A null, in any unit, reads as none.
* The notes, fixed text:

  - ``METER_NOTE = "A crystallization is judged by the KL from the model's own prediction, beside a
    mean ablation's, never by accuracy or loss: a table keyed by the answer can predict better than
    a model that hasn't learned yet. The loss delta is recorded, never judged."``
  - ``ENVELOPE_NOTE = "A crystallization is demonstrated over an envelope, never proven. At every
    checkpoint the form is fitted again, since a table is read off the model it replaces; the
    fourier form keeps the frequencies of the request's checkpoint. The gap shows the checkpoints
    where the model is accurate but the crystallization doesn't hold."``
  - ``NAMES_NOTE = "Modular addition's names, recorded by G7.2, don't hold its sum: the sum is
    written in a few frequencies, which no set of four latents carries. So its crystallizations'
    own tests stand alone, and each record says why (C11)."``
  - ``RECORDED_NOTE = "Recorded crystallizations are measured and shown, not judged: modular
    addition's MLP and attention, and two-hop's block 3 attention alone (D57)."``
  - ``SET_NOTE = "The identities show set changes nothing where it should change nothing: putting a
    tap to its own values reproduces the pass bit for bit, at every residual-shaped tap but hook_z,
    on every zoo model (D55)."``

* ``crystallize_badges(reports)``: ``reports`` maps gate ids to report dicts, and must hold G7.2,
  G7.3 and G7.4, each passing ``validate_report`` and carrying its own gate id; any other key is
  ignored. ValueError otherwise, naming the problem. Returns one badge per subject, in ``SUBJECTS``
  order: ``{"status", "sources", "facts", "notes"}``, plain JSON, sharing no object with the
  reports.

  - ``facts``: a list of ``{"label", "value", "unit", "gate", "path"}``, as
    ``views/gesture_badges.py`` builds them: ``value`` is a copy of what the report holds at
    ``path`` in that gate's ``measurements``, and ``label`` a non-empty phrase, naming its place
    where it has one, distinct within the badge. A path the report doesn't have is a ValueError.
  - ``sources``: the sorted, distinct gate ids its facts cite.
  - ``status``: ``"judged: pass"``, or ``"judged: fail"`` when any of its judging gates' status
    isn't ``pass``: ``crystallize`` is judged by G7.3 and G7.4, and ``set`` by G7.3. G7.2 records,
    never judges.
  - ``notes``: ``crystallize`` has ``[METER_NOTE, ENVELOPE_NOTE, NAMES_NOTE, RECORDED_NOTE]``;
    ``set`` ``[SET_NOTE]``.

  A place is a key of G7.4's ``crystallizations`` mapping (``"<run> <part> <function> <form>"``)
  or of G7.2's ``names`` (a description), and each gate's places, and G7.3's forms, are taken in
  sorted order. The facts are, in order:

  - ``crystallize``:

    - from G7.3, for each form of ``records_by_form``, flags: ``records_by_form[<form>][0]``,
      whether its record built twice from one request was byte-identical, and
      ``records_by_form[<form>][1]``, whether the command line's was too
    - from G7.4: ``kl_max``, ``ablation_min`` and ``wrong_min``, in nats;
      ``reads_agreement_min`` and ``answer_agreement_max``, fractions; ``twin_within_max``, in
      nats; and ``envelope_ok``, a flag
    - then for each of G7.4's places, judged or recorded: ``crystallizations[place]``'s
      ``["meter"]["label"]``, a label; ``["meter"]["kl"]`` and ``["meter"]["ablation"]``, in
      nats; ``["envelope"]["final"]``, a span; and ``["twin"]["meter"]["label"]``, a label; each
      label saying "(recorded)" where the place's ``judged`` is false
    - from G7.2, recorded: ``fvu`` and ``twin_fvu``, ratios; then for each of its places,
      ``names[place]["trained"]``'s ``["named_as_planted"]``, ``["steering"]["follows"]`` and
      ``["steering"]["null_follows"]``, fractions, each label naming modular addition and saying
      "(recorded)"

  - ``set``: G7.3's ``identities_exact`` and ``identities_checked``, counts
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUBJECTS = ("crystallize", "set")
UNITS = ("fraction", "ratio", "nats", "count", "span", "flag", "label")
METER_NOTE = ("A crystallization is judged by the KL from the model's own prediction, beside a mean ablation's, "
              "never by accuracy or loss: a table keyed by the answer can predict better than a model that hasn't "
              "learned yet. The loss delta is recorded, never judged.")
ENVELOPE_NOTE = ("A crystallization is demonstrated over an envelope, never proven. At every checkpoint the form is "
                 "fitted again, since a table is read off the model it replaces; the fourier form keeps the "
                 "frequencies of the request's checkpoint. The gap shows the checkpoints where the model is accurate "
                 "but the crystallization doesn't hold.")
NAMES_NOTE = ("Modular addition's names, recorded by G7.2, don't hold its sum: the sum is written in a few frequencies, "
              "which no set of four latents carries. So its crystallizations' own tests stand alone, and each record "
              "says why (C11).")
RECORDED_NOTE = ("Recorded crystallizations are measured and shown, not judged: modular addition's MLP and attention, "
                 "and two-hop's block 3 attention alone (D57).")
SET_NOTE = ("The identities show set changes nothing where it should change nothing: putting a tap to its own values "
            "reproduces the pass bit for bit, at every residual-shaped tap but hook_z, on every zoo model (D55).")


def crystallize_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    raise NotImplementedError
