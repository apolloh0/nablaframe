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

import copy
from collections.abc import Mapping
from typing import Any

from nablaframe.reports import validate_report

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


_GATES = ("G4.1", "G5.1", "G5.2", "G5.3", "G5.4")
_JUDGED_BY = {"topk": ("G5.2", "G5.3", "G5.4"), "staircase": ("G5.2", "G5.3"), "transcoder": ("G5.2", "G5.3"),
              "dpatch": ("G5.1", "G5.2", "G5.3")}
_NOTES = {"topk": [BOUNDS_NOTE, TWIN_NOTE, EDGE_NOTE], "staircase": [BOUNDS_NOTE, TWIN_NOTE],
          "transcoder": [BOUNDS_NOTE, TWIN_NOTE, TRANSCODER_NOTE], "dpatch": [BOUNDS_NOTE, DPATCH_NOTE]}
_TAPS = {"topk": "hook_resid_post", "staircase": "hook_resid_post", "transcoder": "hook_mlp_out"}


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

    def selective(self, label: str, gate: str, kind: str, block: int, variable: str) -> None:
        key = f"{kind} blocks.{block}.{_TAPS[kind]} {variable}"
        for side in ("trained", "twin"):
            self.add(label if side == "trained" else f"{label}, twin", "score", gate, ["selectivity", key, side])

    def induction_bounds(self) -> None:
        self.add("induction: the code patch's rate over the residual patch's, least of any kind", "ratio", "G5.2",
                 ["faithful_ratio_min"])
        self.add("induction, breaking: how much more often the code patch keeps the prediction than the residual "
                 "patch does, most of any kind", "fraction", "G5.2", ["break_excess_max"])

    def bounds(self) -> None:
        self.induction_bounds()
        self.add("two-hop: the code patch's rate over the residual patch's, least of any kind", "ratio", "G5.3",
                 ["faithful_ratio_min"])
        self.add("two-hop: the control patch through the code leaves the answer unchanged, least of any kind",
                 "fraction", "G5.3", ["control_unchanged_min"])


def _int_value(f: _Facts, gate: str, path: list, what: str) -> int:
    value = f.value(gate, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"the {gate} report's {what} must be an integer")
    return value


def _facts(f: _Facts, subject: str, best: int, last: int) -> list[dict[str, Any]]:
    f.facts = []
    if subject in ("topk", "staircase"):
        f.selective("induction: mean best F1 for the current token after block 0", "G5.2", subject, 0, "cur")
        f.selective("induction: mean best F1 for the next token after block 1", "G5.2", subject, 1, "next")
        f.selective(f"two-hop: mean best F1 for the intermediate at ⇒ after block {best}", "G5.3", subject, best, "b")
        f.selective(f"two-hop: mean best F1 for the answer at ⇒ after block {last}", "G5.3", subject, last, "c")
        swap = ["faithful", subject, str(best)]
        f.add(f"two-hop: the code patch at the query after block {best} swaps the answer", "fraction", "G5.3",
              swap + ["code", "swap"])
        f.add("two-hop: the residual patch there swaps it", "fraction", "G5.3", swap + ["residual", "swap"])
        f.add("two-hop: moving only the intermediate's latents swaps it (recorded)", "fraction", "G5.3",
              swap + ["latents", "swap"])
        f.bounds()
        if subject == "topk":
            judged = f.value("G5.4", ["judged_mlps"])
            if not isinstance(judged, list) or not all(isinstance(key, str) for key in judged):
                raise ValueError("the G5.4 report's judged_mlps must be a list of names")
            for key in judged:
                for side in ("trained", "twin"):
                    label = f"{key}: edges the meter needs"
                    f.add(label if side == "trained" else f"{label}, twin", "edges", "G5.4",
                          ["edge_pairs", key, side, "meter"])
    elif subject == "transcoder":
        f.selective("induction: mean best F1 for the current token, from block 0's MLP input", "G5.2", subject, 0, "cur")
        f.selective("induction: mean best F1 for the next token, from block 1's MLP input", "G5.2", subject, 1, "next")
        f.selective(f"two-hop: mean best F1 for the intermediate at ⇒, from block {best}'s MLP input", "G5.3",
                    subject, best, "b")
        f.induction_bounds()
    else:
        f.add("identities reproduced bit for bit", "count", "G5.1", ["identities_exact"])
        f.add("identities checked", "count", "G5.1", ["identities_checked"])
        f.bounds()
    return f.facts


def dictionary_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    checked = _checked(reports)
    f = _Facts(checked)
    best = _int_value(f, "G5.3", ["best_swap_block"], "best_swap_block")
    n_layers = _int_value(f, "G4.1", ["models", "two_hop", "model_config", "n_layers"], "two-hop n_layers")
    last = n_layers - 1
    badges: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        facts = _facts(f, subject, best, last)
        passed = all(checked[g]["status"] == "pass" for g in _JUDGED_BY[subject])
        badges[subject] = {"status": "judged: pass" if passed else "judged: fail",
                           "sources": sorted({fact["gate"] for fact in facts}), "facts": facts,
                           "notes": list(_NOTES[subject])}
    return badges
