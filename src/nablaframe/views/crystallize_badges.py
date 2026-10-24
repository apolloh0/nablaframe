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

import copy
from collections.abc import Mapping
from typing import Any

from nablaframe.reports import validate_report

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


_GATES = ("G7.2", "G7.3", "G7.4")
_JUDGED_BY = {"crystallize": ("G7.3", "G7.4"), "set": ("G7.3",)}
_NOTES = {"crystallize": [METER_NOTE, ENVELOPE_NOTE, NAMES_NOTE, RECORDED_NOTE], "set": [SET_NOTE]}


def _checked(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for gate_id in _GATES:
        if gate_id not in reports:
            raise ValueError(f"reports must hold {', '.join(_GATES)}; {gate_id} is missing")
        report = reports[gate_id]
        problems = validate_report(report)
        if problems:
            raise ValueError(f"the {gate_id} report is invalid: {'; '.join(problems)}")
        if report["gate_id"] != gate_id:
            raise ValueError(f"the report filed under {gate_id} carries gate id {report['gate_id']!r}")
        out[gate_id] = report
    return out


class _Facts:
    """Collects facts, reading each value from the report at its path."""

    def __init__(self, reports: Mapping[str, Mapping[str, Any]]) -> None:
        self.reports = reports
        self.facts: list[dict[str, Any]] = []

    def value(self, gate: str, path: list) -> Any:
        value: Any = self.reports[gate]["measurements"]
        for segment in path:
            try:
                value = value[segment]
            except (KeyError, IndexError, TypeError):
                raise ValueError(f"the {gate} report has no measurement at {path!r}") from None
        return value

    def add(self, label: str, unit: str, gate: str, path: list) -> None:
        self.facts.append({"label": label, "value": copy.deepcopy(self.value(gate, path)), "unit": unit,
                           "gate": gate, "path": list(path)})

    def flag(self, gate: str, path: list, what: str) -> bool:
        value = self.value(gate, path)
        if not isinstance(value, bool):
            raise ValueError(f"the {gate} report's {what} must be a flag")
        return value

    def places(self, gate: str, key: str) -> list[str]:
        mapping = self.value(gate, [key])
        if not isinstance(mapping, Mapping):
            raise ValueError(f"the {gate} report's {key} must be a mapping of places")
        return sorted(mapping)


def _crystallize(f: _Facts) -> None:
    for form in f.places("G7.3", "records_by_form"):
        f.add(f"{form} records: one request built twice gives identical bytes", "flag", "G7.3",
              ["records_by_form", form, 0])
        f.add(f"{form} records: the command line's build gives identical bytes too", "flag", "G7.3",
              ["records_by_form", form, 1])
    for key, unit, label in (
            ("kl_max", "nats", "KL from the model's own prediction, most of any judged crystallization"),
            ("ablation_min", "nats", "KL of the mean ablation, least of any judged crystallization"),
            ("wrong_min", "nats", "KL of a wrong function, least of any wrong function of any judged "
                                  "crystallization"),
            ("reads_agreement_min", "fraction", "agreement keyed by what the component reads, least of any "
                                                "judged crystallization with a counterfactual"),
            ("answer_agreement_max", "fraction", "agreement keyed by the answer, most of any judged "
                                                 "crystallization with a counterfactual"),
            ("twin_within_max", "nats", "twin: a wrong function's KL from the twin's own KL, most of any judged "
                                        "crystallization"),
            ("envelope_ok", "flag", "every judged crystallization holds at the final checkpoint, with step 0 "
                                    "outside")):
        f.add(label, unit, "G7.4", [key])
    for place in f.places("G7.4", "crystallizations"):
        at = ["crystallizations", place]
        tag = "" if f.flag("G7.4", at + ["judged"], f"judged flag at {place}") else " (recorded)"
        f.add(f"{place}: the crystallization's verdict{tag}", "label", "G7.4", at + ["meter", "label"])
        f.add(f"{place}: KL from the model's own prediction{tag}", "nats", "G7.4", at + ["meter", "kl"])
        f.add(f"{place}: KL of the mean ablation{tag}", "nats", "G7.4", at + ["meter", "ablation"])
        f.add(f"{place}: the run of checkpoints where it holds, ending at the final one{tag}", "span", "G7.4",
              at + ["envelope", "final"])
        f.add(f"{place}: twin: the crystallization's verdict{tag}", "label", "G7.4", at + ["twin", "meter", "label"])
    f.add("modular addition's dictionary: fraction of variance unexplained (recorded)", "ratio", "G7.2", ["fvu"])
    f.add("modular addition's twin dictionary: fraction of variance unexplained (recorded)", "ratio", "G7.2",
          ["twin_fvu"])
    for place in f.places("G7.2", "names"):
        trained = ["names", place, "trained"]
        f.add(f"modular addition {place}: values named as planted (recorded)", "fraction", "G7.2",
              trained + ["named_as_planted"])
        f.add(f"modular addition {place}: steering follows (recorded)", "fraction", "G7.2",
              trained + ["steering", "follows"])
        f.add(f"modular addition {place}: steering follows under the null (recorded)", "fraction", "G7.2",
              trained + ["steering", "null_follows"])


def _set(f: _Facts) -> None:
    f.add("identities reproduced bit for bit", "count", "G7.3", ["identities_exact"])
    f.add("identities checked", "count", "G7.3", ["identities_checked"])


_BUILDERS = {"crystallize": _crystallize, "set": _set}


def crystallize_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    checked = _checked(reports)
    f = _Facts(checked)
    badges: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        f.facts = []
        _BUILDERS[subject](f)
        labels = [fact["label"] for fact in f.facts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"the {subject} badge would repeat a label")
        passed = all(checked[g]["status"] == "pass" for g in _JUDGED_BY[subject])
        badges[subject] = {"status": "judged: pass" if passed else "judged: fail",
                           "sources": sorted({fact["gate"] for fact in f.facts}), "facts": f.facts,
                           "notes": list(_NOTES[subject])}
    return badges
