"""Phase 4 delta 4: model-level badges, generated from the Phase 4 reports (D34, D38). Needs no JAX.

Contract: docs/deltas/phase-4-delta-4.md, "src/nablaframe/views/badges.py".
"""

from __future__ import annotations

import copy
import json

import pytest

from nablaframe.reports import validate_report
from nablaframe.views import badges as b  # looked up per test, so a missing name fails its test alone
from test_phase1_store_reports import _v2
from test_reports import passing_report

BLOCKS = ("0", "1", "2", "3")


def _ranks(base: float) -> dict:
    return {str(p): {k: {"intermediate": base + 10 * (p == 25) + int(k), "answer": base + int(k)} for k in BLOCKS}
            for p in (24, 25)}


MEASUREMENTS = {
    "G4.1": {"recompute_identical": True, "checkpoints_written": 62, "checkpoints_scheduled": 62, "events_recorded": 1576,
             "events_expected": 1576, "induction_mechanism_accuracy": 1.0, "induction_control_accuracy": 0.003,
             "induction_twin_accuracy": 0.017, "induction_chance": 1 / 64, "two_hop_accuracy": 0.996,
             "two_hop_broken_accuracy": 0.012, "two_hop_twin_accuracy": 0.008, "two_hop_chance": 1 / 48,
             "models": {"induction": {"model_config": {"vocab": 64, "n_layers": 2}},
                        "two_hop": {"model_config": {"vocab": 51, "n_layers": 4}}}},
    "G4.2": {"mechanism_accuracy": 1.0, "chance": 1 / 64, "attention_by_head": [0.98, 0.97, 0.96, 0.99],
             "best_head_attention": 0.99, "induction_heads": [0, 1, 2, 3], "twin_best_attention": 0.048,
             "uniform_baseline": 0.046, "ablation_fraction_removed": 1.0005, "fraction_removed_by_head": {"1.0": 0.0},
             "swap_step": 50, "swap_fraction_removed": 0.929, "patch_followed_by_tap": {"blocks.0.hook_resid_post": [1.0]},
             "patch_after_block0_min": 1.0, "patch_after_block1_max": 0.0},
    "G4.3": {"unpatched_accuracy": 0.996,
             "swap_by_site": {"query": [0.09, 0.836, 0.852, 0.0], "fact positions": [0.0] * 4, "every position": [0.0] * 4},
             "transplant_by_site": {"query": [0.0, 0.148, 0.141, 1.0]},
             "control_unchanged_by_site": {"query": [1.0, 1.0, 0.996, 1.0]}, "control_unchanged_min": 0.996,
             "best_swap": {"site": "query", "block": 2, "rate": 0.852}},
    "G4.4": {"lens_ranks": {"trained": {"logit_lens": _ranks(20), "jlens": _ranks(21)},
                            "twin": {"logit_lens": _ranks(26), "jlens": _ranks(27)}},
             "jswap_by_layer": {k: {"swap": 0.02 + 0.001 * int(k), "unchanged": 0.9 - 0.1 * int(k)} for k in BLOCKS},
             "attribution": {
                 "induction": {"trained": {"hook_embed": 0.002, "blocks.1.hook_attn_out": 3.35, "blocks.1.hook_mlp_out": 9.95,
                                           "blocks.0.hook_mlp_out": -4.0},
                               "twin": {"hook_embed": 0.0, "blocks.1.hook_attn_out": 0.005}},
                 "two_hop": {"trained": {"blocks.3.hook_attn_out": 11.13, "blocks.3.hook_mlp_out": 2.61,
                                         "blocks.1.hook_attn_out": -1.25},
                             "twin": {"blocks.3.hook_attn_out": 0.01}}}},
}


def _reports(**statuses) -> dict:
    out = {}
    for gate_id, measurements in MEASUREMENTS.items():
        report = _v2(passing_report("G0.3"))
        report.update(gate_id=gate_id, status=statuses.get(gate_id, "pass"), env_name="nf-jax",
                      measurements=copy.deepcopy(measurements))
        assert validate_report(report) == [], gate_id
        out[gate_id] = report
    return out


def _at(report: dict, path: list):
    value = report["measurements"]
    for segment in path:
        value = value[segment]
    return value


def _facts(badge: dict) -> set:
    return {(fact["gate"], tuple(fact["path"])) for fact in badge["facts"]}


def test_every_view_and_operation_has_a_badge():
    assert b.SUBJECTS == ("per_token_loss", "logit_lens", "jlens", "attribution", "attention_pattern", "residual_norm",
                          "layer_structure", "zero", "mean", "resample", "patch", "add", "swap", "jswap")
    assert b.UNITS == ("fraction", "fraction removed", "rank", "step", "tokens", "logit")
    reports = _reports()
    badges = b.model_level_badges(reports)
    assert list(badges) == list(b.SUBJECTS)
    expected = {"per_token_loss": "readout", "residual_norm": "readout", "layer_structure": "readout",
                "logit_lens": "recorded", "jlens": "recorded", "attribution": "recorded", "jswap": "recorded",
                "attention_pattern": "judged: pass", "zero": "judged: pass", "swap": "judged: pass", "patch": "judged: pass",
                "mean": "not tested", "resample": "not tested", "add": "not tested"}
    assert {subject: badge["status"] for subject, badge in badges.items()} == expected
    for subject, badge in badges.items():
        assert set(badge) == {"status", "sources", "facts", "notes"}, subject
        assert badge["sources"] == sorted({fact["gate"] for fact in badge["facts"]}), subject
        for fact in badge["facts"]:
            assert set(fact) == {"label", "value", "unit", "gate", "path"} and fact["label"], subject
            assert fact["unit"] in b.UNITS, (subject, fact)
            assert fact["value"] == _at(reports[fact["gate"]], fact["path"]), (subject, fact)
    for subject in ("mean", "resample", "add"):
        assert badges[subject]["facts"] == [] and badges[subject]["notes"] == [b.NOT_TESTED_NOTE]
    for subject in ("per_token_loss", "residual_norm", "layer_structure"):
        assert badges[subject]["facts"] == [] and badges[subject]["notes"] == [b.READOUT_NOTE]
    assert json.loads(json.dumps(badges)) == badges, "plain JSON"
    assert b.model_level_badges(_reports()) == badges, "a pure function of the reports"


def test_the_lens_badges_read_where_the_patch_finds_the_intermediate():
    badges = b.model_level_badges(_reports())
    for lens in ("logit_lens", "jlens"):
        facts = _facts(badges[lens])
        for position in ("24", "25"):
            for model in ("trained", "twin"):
                assert ("G4.4", ("lens_ranks", model, lens, position, "2", "intermediate")) in facts, (lens, position, model)
        assert ("G4.4", ("lens_ranks", "trained", lens, "25", "3", "answer")) in facts, "the answer after the last block"
        assert ("G4.3", ("swap_by_site", "query", 2)) in facts, "the patch's swap at that site and block"
        assert ("G4.1", ("models", "two_hop", "model_config", "vocab")) in facts, "ranks are out of the vocabulary"
    assert {("G4.4", ("jswap_by_layer", "2", "swap")), ("G4.4", ("jswap_by_layer", "2", "unchanged"))} <= _facts(badges["jlens"])
    assert badges["jlens"]["notes"] == [b.JLENS_FIT_NOTE] and badges["logit_lens"]["notes"] == []
    assert b.JLENS_FIT_NOTE == "J is fitted over D17's valid positions, 16 to 24, which leave out ⇒, the position the answer is read from."

    moved = _reports()
    moved["G4.3"]["measurements"]["best_swap"] = {"site": "fact positions", "block": 1, "rate": 0.0}
    facts = _facts(b.model_level_badges(moved)["logit_lens"])
    assert ("G4.4", ("lens_ranks", "trained", "logit_lens", "25", "1", "intermediate")) in facts, "follows G4.3's best block"
    assert ("G4.3", ("swap_by_site", "fact positions", 1)) in facts


def test_attribution_and_the_operations_cite_their_gates():
    badges = b.model_level_badges(_reports())
    attribution = _facts(badges["attribution"])
    for model, parts in (("induction", ("blocks.1.hook_mlp_out", "blocks.0.hook_mlp_out")),
                         ("two_hop", ("blocks.3.hook_attn_out", "blocks.3.hook_mlp_out"))):
        for part in parts:
            assert ("G4.4", ("attribution", model, "trained", part)) in attribution, "the two largest by magnitude"
    assert ("G4.4", ("attribution", "induction", "trained", "blocks.1.hook_attn_out")) not in attribution
    assert ("G4.2", ("ablation_fraction_removed",)) in attribution
    assert badges["attribution"]["notes"] == [b.ATTRIBUTION_NOTE]
    assert b.ATTRIBUTION_NOTE == "Attribution measures each part's direct contribution to the logit, not whether the model needs that part."
    assert _facts(badges["attention_pattern"]) == {("G4.2", ("best_head_attention",)), ("G4.2", ("twin_best_attention",)),
                                                   ("G4.2", ("uniform_baseline",))}
    assert _facts(badges["zero"]) == {("G4.2", ("ablation_fraction_removed",))}
    assert _facts(badges["swap"]) == {("G4.2", ("swap_step",)), ("G4.2", ("swap_fraction_removed",))}
    assert _facts(badges["patch"]) == {("G4.2", ("patch_after_block0_min",)), ("G4.2", ("patch_after_block1_max",)),
                                       ("G4.3", ("control_unchanged_min",)), ("G4.3", ("best_swap", "rate"))}
    assert _facts(badges["jswap"]) == {("G4.4", ("jswap_by_layer", "2", "swap")), ("G4.3", ("best_swap", "rate"))}
    assert b.NOT_TESTED_NOTE == "No Phase 4 test exercises this operation on a model with known structure."
    assert b.READOUT_NOTE == "An exact readout of the trace, checked by its hash; there is no model-level structure for it to recover."


def test_status_follows_the_reports_and_bad_reports_are_refused():
    failed = _reports()
    failed["G4.3"]["status"] = "fail"
    failed["G4.3"]["measurements"]["control_unchanged_min"] = 0.5
    badges = b.model_level_badges(failed)
    assert badges["patch"]["status"] == "judged: fail", "one judged source failing fails the badge"
    assert badges["zero"]["status"] == "judged: pass" and badges["jswap"]["status"] == "recorded"
    missing = _reports()
    del missing["G4.4"]
    with pytest.raises(ValueError):
        b.model_level_badges(missing)
    invalid = _reports()
    del invalid["G4.2"]["measurements"]["swap_step"]
    with pytest.raises(ValueError):
        b.model_level_badges(invalid)
    swapped = _reports()
    swapped["G4.3"] = swapped["G4.2"]
    with pytest.raises(ValueError):
        b.model_level_badges(swapped), "a report filed under another gate's id"
