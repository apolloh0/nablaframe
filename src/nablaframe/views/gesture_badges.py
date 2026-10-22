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

import copy
from collections.abc import Mapping
from typing import Any

from nablaframe.reports import validate_report

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


_GATES = ("G6.1", "G6.2", "G6.3", "G6.4", "G6.5")
_JUDGED_BY = {"name": ("G6.1", "G6.2"), "edge": ("G6.1", "G6.3"), "open": ("G6.1", "G6.4"),
              "generalize": ("G6.1", "G6.5"), "dsteer": ("G6.1",)}
_NOTES = {"name": [STEERING_NOTE, REFERENCE_NOTE], "edge": [LABEL_NOTE, REFERENCE_NOTE],
          "open": [TWIN_OPEN_NOTE, REFERENCE_NOTE], "generalize": [CHECKPOINTS_NOTE, PATCH_NOTE, REFERENCE_NOTE],
          "dsteer": [DSTEER_NOTE]}


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


def _built(f: _Facts, kind: str) -> None:
    f.add(f"{kind} records: one request built twice gives identical bytes", "flag", "G6.1",
          ["records_by_kind", kind, 0])
    f.add(f"{kind} records: the command line's build gives identical bytes too", "flag", "G6.1",
          ["records_by_kind", kind, 1])


def _name(f: _Facts) -> None:
    _built(f, "name")
    for key, label in (("named_as_planted_min", "values named as planted, least of any judged name"),
                       ("follows_min", "steering follows, least of any judged name"),
                       ("null_follows_max", "steering follows under the null, most of any judged name"),
                       ("twin_named_max", "twin: values named at all, most of any judged name"),
                       ("twin_follows_max", "twin: steering follows, most of any judged name")):
        f.add(label, "fraction", "G6.2", [key])
    for place in f.places("G6.2", "names"):
        if not f.flag("G6.2", ["names", place, "judged"], f"judged flag at {place}"):
            continue
        trained = ["names", place, "trained"]
        f.add(f"{place}: values named as planted", "fraction", "G6.2", trained + ["named_as_planted"])
        f.add(f"{place}: steering follows", "fraction", "G6.2", trained + ["steering", "follows"])
        f.add(f"{place}: values whose set's effect ranks them first", "fraction", "G6.2",
              trained + ["effect", "first"])
        f.add(f"{place}: values whose set's effect ranks them last", "fraction", "G6.2",
              trained + ["effect", "last"])


def _edge(f: _Facts) -> None:
    _built(f, "edge")
    for key, label in (("follows_min", "the swap follows, least of any judged edge"),
                       ("null_follows_max", "the swap follows under the null, most of any judged edge"),
                       ("output_follows_min", "the swap follows at the output, least of any judged edge"),
                       ("kept_max", "with the upstream set removed, the downstream set's mass kept (median), "
                                    "most of any judged edge"),
                       ("null_kept_min", "with the null's latents removed instead, the mass kept, least of any "
                                         "judged edge"),
                       ("residual_alone_max", "with every block between the sites frozen, the swap still follows, "
                                              "most of any judged edge")):
        f.add(label, "fraction", "G6.3", [key])
    f.add("every judged edge is labelled computed", "flag", "G6.3", ["labels_computed"])
    f.add("twin: follows less its null's follows, most of any judged edge", "fraction", "G6.3",
          ["twin_excess_max"])
    for place in f.places("G6.3", "edges"):
        judged = f.flag("G6.3", ["edges", place, "judged"], f"judged flag at {place}")
        tag = "" if judged else " (recorded)"
        trained = ["edges", place, "trained"]
        f.add(f"{place}: the swap follows{tag}", "fraction", "G6.3", trained + ["follows"])
        f.add(f"{place}: the swap follows under the null{tag}", "fraction", "G6.3", trained + ["null_follows"])
        f.add(f"{place}: the edge is labelled{tag}", "label", "G6.3", trained + ["label"])


def _open(f: _Facts) -> None:
    _built(f, "open")
    f.add("median reading over its bar, most of any target", "ratio", "G6.4",
          ["reading_ratio_max"])
    for key, label in (("within_min", "instances read within 128 parts, least of any target"),
                       ("planted_top_min", "the planted set's first part in the top 8 parts, least of any target"),
                       ("random_within_max", "in random order of parts, instances read within 128 parts, most of any "
                                          "target"),
                       ("twin_planted_top_max", "twin: the planted set's first part in the top 8 parts, most of any "
                                              "target")):
        f.add(label, "fraction", "G6.4", [key])
    for place in f.places("G6.4", "targets"):
        target = ["targets", place]
        absolute = target + ["trained", "orders", "absolute"]
        f.add(f"{place}: median number of upstream parts to read the set", "parts", "G6.4",
              absolute + ["reading_median"])
        f.add(f"{place}: the bar, in upstream parts", "parts", "G6.4", target + ["bar"])
        f.add(f"{place}: instances read within 128 parts", "fraction", "G6.4", absolute + ["within"])
        f.add(f"{place}: the planted set's first part in the top 8 parts", "fraction", "G6.4",
              target + ["trained", "planted", "top"])
        if f.flag("G6.4", target + ["twin_judged"], f"twin_judged flag at {place}"):
            f.add(f"{place}: twin: the planted set's first part in the top 8 parts", "fraction", "G6.4",
                  target + ["twin", "planted", "top"])


def _generalize(f: _Facts) -> None:
    _built(f, "generalize")
    for key, unit, label in (
            ("slice_f1_min", "score", "held-out slices: F1, least"),
            ("slice_follows_min", "fraction", "held-out slices: steering follows, least"),
            ("seed_f1_min", "score", "dictionary seeds 1 and 2, the name derived again: F1, least"),
            ("seed_follows_min", "fraction", "dictionary seeds 1 and 2, the name derived again: steering follows, "
             "least"),
            ("step0_outside", "flag", "step 0 lies outside the envelope"),
            ("one_final_run", "flag", "each envelope is one run, ending at the final checkpoint"),
            ("start_offset_max", "count", "checkpoints between the envelope's start and the probe's first reading, "
             "most of any judged name")):
        f.add(label, unit, "G6.5", [key])
    for place in f.places("G6.5", "names"):
        envelope = ["names", place, "results", "envelope"]
        f.add(f"{place}: the envelope's run to the final checkpoint", "span", "G6.5", envelope + ["final"])
        f.add(f"{place}: step the probe reads the variable from", "step", "G6.5",
              envelope + ["probe_reads_from"])


def _dsteer(f: _Facts) -> None:
    f.add("identities reproduced bit for bit", "count", "G6.1", ["identities_exact"])
    f.add("identities checked", "count", "G6.1", ["identities_checked"])


_BUILDERS = {"name": _name, "edge": _edge, "open": _open, "generalize": _generalize, "dsteer": _dsteer}


def gesture_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
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
