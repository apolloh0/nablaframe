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

import copy
from collections.abc import Mapping
from typing import Any

from nablaframe.reports import validate_report

SUBJECTS = ("per_token_loss", "logit_lens", "jlens", "attribution", "attention_pattern", "residual_norm",
            "layer_structure", "zero", "mean", "resample", "patch", "add", "swap", "jswap")
UNITS = ("fraction", "fraction removed", "rank", "step", "tokens", "logit")
JLENS_FIT_NOTE = ("J is fitted over D17's valid positions, 16 to 24, which leave out ⇒, the position the answer "
                  "is read from.")
ATTRIBUTION_NOTE = ("Attribution measures each part's direct contribution to the logit, not whether the model "
                    "needs that part.")
NOT_TESTED_NOTE = "No Phase 4 test exercises this operation on a model with known structure."
READOUT_NOTE = "An exact readout of the trace, checked by its hash; there is no model-level structure for it to recover."

_GATES = ("G4.1", "G4.2", "G4.3", "G4.4")
_READOUTS = ("per_token_loss", "residual_norm", "layer_structure")
_NOT_TESTED = ("mean", "resample", "add")
_JUDGED_BY = {"attention_pattern": ("G4.2",), "zero": ("G4.2",), "swap": ("G4.2",), "patch": ("G4.2", "G4.3")}
_ATTRIBUTION_PARTS = 2


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


def _lens_facts(f: _Facts, lens: str, site: str, best: int, last: str) -> None:
    where = f"after block {best}"
    for position in ("24", "25"):
        for model in ("trained", "twin"):
            f.add(f"{lens}: median rank of the intermediate at position {position} {where}, {model}", "rank",
                  "G4.4", ["lens_ranks", model, lens, position, str(best), "intermediate"])
    f.add(f"{lens}: median rank of the answer at position 25 after block {last}, trained", "rank", "G4.4",
          ["lens_ranks", "trained", lens, "25", last, "answer"])
    f.add(f"patch: share of answers swapped at {site}, {where}", "fraction", "G4.3", ["swap_by_site", site, best])
    f.add("vocabulary of the two-hop model", "tokens", "G4.1", ["models", "two_hop", "model_config", "vocab"])


def _top_parts(f: _Facts, model: str) -> list[str]:
    means = f.value("G4.4", ["attribution", model, "trained"])
    if not isinstance(means, Mapping) or not all(isinstance(v, (int, float)) for v in means.values()):
        raise ValueError(f"the G4.4 report's attribution for {model} must map each part to a number")
    return sorted(means, key=lambda part: (-abs(means[part]), part))[:_ATTRIBUTION_PARTS]


def model_level_badges(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    checked = _checked(reports)
    f = _Facts(checked)
    best = f.value("G4.3", ["best_swap"])
    if not isinstance(best, Mapping) or not isinstance(best.get("block"), int) or not isinstance(best.get("site"), str):
        raise ValueError("the G4.3 report's best_swap must name a site and an integer block")
    site, block = best["site"], best["block"]
    n_layers = f.value("G4.1", ["models", "two_hop", "model_config", "n_layers"])
    last = str(n_layers - 1)
    jswap = ["jswap_by_layer", str(block)]

    facts: dict[str, list[dict[str, Any]]] = {}

    def collect(subject: str, build) -> None:
        f.facts = []
        build()
        facts[subject] = f.facts

    collect("logit_lens", lambda: _lens_facts(f, "logit_lens", site, block, last))

    def jlens() -> None:
        _lens_facts(f, "jlens", site, block, last)
        f.add(f"J-lens coordinate swap: share of answers swapped, block {block}", "fraction", "G4.4", jswap + ["swap"])
        f.add(f"J-lens coordinate swap: share of answers unchanged, block {block}", "fraction", "G4.4",
              jswap + ["unchanged"])

    collect("jlens", jlens)

    def attribution() -> None:
        for model in ("induction", "two_hop"):
            for part in _top_parts(f, model):
                f.add(f"{model}: mean contribution of {part}, trained", "logit", "G4.4",
                      ["attribution", model, "trained", part])
        f.add("induction: accuracy above chance removed by zeroing the induction heads together", "fraction removed",
              "G4.2", ["ablation_fraction_removed"])

    collect("attribution", attribution)

    def attention() -> None:
        f.add("best head's attention to induction targets", "fraction", "G4.2", ["best_head_attention"])
        f.add("the random-weights twin's best head", "fraction", "G4.2", ["twin_best_attention"])
        f.add("uniform baseline", "fraction", "G4.2", ["uniform_baseline"])

    collect("attention_pattern", attention)
    collect("zero", lambda: f.add("accuracy above chance removed by zeroing the induction heads together",
                                  "fraction removed", "G4.2", ["ablation_fraction_removed"]))

    def swap() -> None:
        f.add("checkpoint step the block-1 attention is swapped from", "step", "G4.2", ["swap_step"])
        f.add("accuracy above chance removed by the swap", "fraction removed", "G4.2", ["swap_fraction_removed"])

    collect("swap", swap)

    def patch() -> None:
        f.add("copy target followed after a patch at block 0, least", "fraction", "G4.2", ["patch_after_block0_min"])
        f.add("copy target followed after a patch at block 1, most", "fraction", "G4.2", ["patch_after_block1_max"])
        f.add("distractor-only control: answer unchanged, least", "fraction", "G4.3", ["control_unchanged_min"])
        f.add("aligned patch: share of answers swapped at the best site and block", "fraction", "G4.3",
              ["best_swap", "rate"])

    collect("patch", patch)

    def jswap_facts() -> None:
        f.add(f"J-lens coordinate swap: share of answers swapped, block {block}", "fraction", "G4.4", jswap + ["swap"])
        f.add("aligned patch: share of answers swapped at the best site and block", "fraction", "G4.3",
              ["best_swap", "rate"])

    collect("jswap", jswap_facts)

    notes = {"jlens": [JLENS_FIT_NOTE], "attribution": [ATTRIBUTION_NOTE]}
    badges: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        if subject in _READOUTS:
            status, subject_notes = "readout", [READOUT_NOTE]
        elif subject in _NOT_TESTED:
            status, subject_notes = "not tested", [NOT_TESTED_NOTE]
        else:
            subject_notes = list(notes.get(subject, []))
            if subject in _JUDGED_BY:
                passed = all(checked[g]["status"] == "pass" for g in _JUDGED_BY[subject])
                status = "judged: pass" if passed else "judged: fail"
            else:
                status = "recorded"
        subject_facts = facts.get(subject, [])
        badges[subject] = {"status": status, "sources": sorted({fact["gate"] for fact in subject_facts}),
                           "facts": subject_facts, "notes": subject_notes}
    return badges
