"""Phase 7 delta 4: crystallize-this's badge and set's, from the Phase 7 reports, and D53's line (D38, D53, D55 to D57).

Contract: src/nablaframe/views/crystallize_badges.py, and the open-this-up bullet of
src/nablaframe/views/gesture_badges.py. Every place in these reports carries its own numbers, and the places are listed
out of order, so a wrong path, a wrong place or an unsorted walk reads a wrong value. Needs no JAX.
"""

from __future__ import annotations

import copy
import json

import pytest

from nablaframe.reports import validate_report
from nablaframe.views import crystallize_badges as cb  # looked up per test, so a missing name fails its test alone
from nablaframe.views import gesture_badges as gb
from test_phase1_store_reports import _v2
from test_reports import passing_report
from test_views_gesture_badges import _reports as phase6_reports

PLACES = {  # G7.4's crystallizations, out of order, each judged or recorded
    "two_hop write hop table": True,
    "modular_addition mlp sum and operands additive": False,
    "induction attention copied table": True,
    "modular_addition write sum fourier": True,
    "two_hop attention hop table": False,
    "modular_addition write sum table": True,
    "modular_addition attention operands additive": False,
}
DESCRIPTIONS = ("token(p-1)", "token(p+1)", "token(p-2)")  # G7.2's names, out of order
FORMS = {"table": [True, False], "fourier": [False, True], "additive": [True, True]}  # G7.3's, out of order


def _meter(i: int, label: str) -> dict:
    return {"label": label, "kl": round(0.01 * (i + 1), 4), "ablation": round(4.0 + 0.5 * i, 4),
            "removed": round(0.99 - 0.01 * i, 4), "agreement": round(0.95 + 0.001 * i, 4),
            "loss_delta": round(-0.01 * i, 4)}


def _crystallization(i: int, judged: bool) -> dict:
    final = None if i == 4 else [100 * (i + 1), 10000 + i]  # one recorded crystallization doesn't hold at the end
    return {"key": f"{i + 1:x}" * 64, "judged": judged,
            "meter": _meter(i, "holds" if final else "doesn't hold"), "wrong": [{"key": "a", "kl": 5.0 + i}],
            "counterfactual": None, "cites": None,
            "twin": {"meter": _meter(i + 20, "inert" if i % 2 else "doesn't hold"), "wrong": []},
            "envelope": {"steps": [], "runs": [final] if final else [], "final": final, "gap": [i],
                         "accurate_from": 1700, "before": []}}


def _name(i: int) -> dict:
    side = lambda base: {"values": 113, "named": base, "named_as_planted": round(base - 0.05, 4),  # noqa: E731
                         "steering": {"follows": round(0.001 * (i + 1), 4), "null_follows": round(0.0001 * i, 5)}}
    return {"trained": side(round(0.2 + 0.03 * i, 4)), "twin": side(round(0.01 * (i + 1), 4))}


MEASUREMENTS = {
    "G7.2": {"fvu": 0.0175, "twin_fvu": 0.045, "names": {d: _name(i) for i, d in enumerate(DESCRIPTIONS)},
             "records": 130, "seconds": {"dictionary": 31.6}},
    "G7.3": {"identities_checked": 51, "identities_exact": 51, "identities_by_run": {"induction": {"t": True}},
             "request_hashes_checked": 64, "request_hashes_distinct": True, "request_hashes_stable": True,
             "records_identical": True, "records_by_form": FORMS},
    "G7.4": {"crystallizations": {p: _crystallization(i, j) for i, (p, j) in enumerate(PLACES.items())},
             "kl_max": 0.0243, "ablation_min": 4.1414, "wrong_min": 4.4285, "reads_agreement_min": 0.9565,
             "answer_agreement_max": 0.0269, "twin_within_max": 0.0022, "envelope_ok": True},
}


def _reports(measurements=None, **statuses) -> dict:
    out = {}
    for gate_id, m in (measurements or MEASUREMENTS).items():
        report = _v2(passing_report("G0.3"))
        report.update(gate_id=gate_id, status=statuses.get(gate_id.replace(".", "_"), "pass"), env_name="nf-jax",
                      measurements=copy.deepcopy(m))
        out[gate_id] = report
    return out


def _at(report: dict, path: list):
    value = report["measurements"]
    for step in path:
        value = value[step]
    return value


def _expected(m: dict) -> dict:
    """Each badge's facts as (gate, path, unit), in the contract's order."""
    crystallize = []
    for form in sorted(m["G7.3"]["records_by_form"]):
        crystallize += [("G7.3", ["records_by_form", form, 0], "flag"), ("G7.3", ["records_by_form", form, 1], "flag")]
    crystallize += [("G7.4", [key], unit) for key, unit in (
        ("kl_max", "nats"), ("ablation_min", "nats"), ("wrong_min", "nats"), ("reads_agreement_min", "fraction"),
        ("answer_agreement_max", "fraction"), ("twin_within_max", "nats"), ("envelope_ok", "flag"))]
    for place in sorted(m["G7.4"]["crystallizations"]):
        at = ["crystallizations", place]
        crystallize += [("G7.4", at + ["meter", "label"], "label"), ("G7.4", at + ["meter", "kl"], "nats"),
                        ("G7.4", at + ["meter", "ablation"], "nats"), ("G7.4", at + ["envelope", "final"], "span"),
                        ("G7.4", at + ["twin", "meter", "label"], "label")]
    crystallize += [("G7.2", ["fvu"], "ratio"), ("G7.2", ["twin_fvu"], "ratio")]
    for place in sorted(m["G7.2"]["names"]):
        at = ["names", place, "trained"]
        crystallize += [("G7.2", at + ["named_as_planted"], "fraction"),
                        ("G7.2", at + ["steering", "follows"], "fraction"),
                        ("G7.2", at + ["steering", "null_follows"], "fraction")]
    return {"crystallize": crystallize,
            "set": [("G7.3", ["identities_exact"], "count"), ("G7.3", ["identities_checked"], "count")]}


def _check(badges: dict, reports: dict, m: dict) -> None:
    assert list(badges) == list(cb.SUBJECTS)
    expected = _expected(m)
    for subject, badge in badges.items():
        assert set(badge) == {"status", "sources", "facts", "notes"}, subject
        assert [(x["gate"], x["path"], x["unit"]) for x in badge["facts"]] == expected[subject], subject
        for fact in badge["facts"]:
            assert set(fact) == {"label", "value", "unit", "gate", "path"}
            assert fact["value"] == _at(reports[fact["gate"]], fact["path"]), (subject, fact["path"])
            assert fact["unit"] in cb.UNITS
            assert isinstance(fact["label"], str) and fact["label"].strip(), (subject, fact["path"])
            place = next((p for p in fact["path"][1:2] if p in (*m["G7.4"]["crystallizations"], *m["G7.2"]["names"])),
                         None)
            if place is not None:
                assert place in fact["label"], f"{subject}: the label names its place, {place}"
            if fact["path"][:1] == ["crystallizations"]:
                recorded = not m["G7.4"]["crystallizations"][fact["path"][1]]["judged"]
                assert ("(recorded)" in fact["label"]) == recorded, fact["label"]
            if fact["gate"] == "G7.2" and fact["path"][:1] == ["names"]:
                assert "(recorded)" in fact["label"] and "modular addition" in fact["label"], fact["label"]
        labels = [fact["label"] for fact in badge["facts"]]
        assert len(set(labels)) == len(labels), f"{subject}: each label names one fact"
        assert badge["sources"] == sorted({fact["gate"] for fact in badge["facts"]})
    assert json.loads(json.dumps(badges)) == badges, "plain JSON"


def test_crystallize_badges_cite_the_reports():
    assert cb.SUBJECTS == ("crystallize", "set")
    assert cb.UNITS == ("fraction", "ratio", "nats", "count", "span", "flag", "label")
    reports = _reports()
    for gate_id, report in reports.items():
        assert validate_report(report) == [], gate_id
    kept = copy.deepcopy(reports)
    badges = cb.crystallize_badges({**reports, "G6.1": {"anything": "ignored"}})
    _check(badges, reports, MEASUREMENTS)
    assert reports == kept, "the reports are read, never changed"
    assert badges["crystallize"]["status"] == "judged: pass" and badges["set"]["status"] == "judged: pass"
    assert badges["crystallize"]["notes"] == [cb.METER_NOTE, cb.ENVELOPE_NOTE, cb.NAMES_NOTE, cb.RECORDED_NOTE]
    assert badges["set"]["notes"] == [cb.SET_NOTE]
    assert badges["crystallize"]["sources"] == ["G7.2", "G7.3", "G7.4"] and badges["set"]["sources"] == ["G7.3"]
    for note in (cb.METER_NOTE, cb.ENVELOPE_NOTE, cb.NAMES_NOTE, cb.RECORDED_NOTE, cb.SET_NOTE):
        assert isinstance(note, str) and note.endswith(".")
    assert "C11" in cb.NAMES_NOTE and "D57" in cb.RECORDED_NOTE and "D55" in cb.SET_NOTE

    final = next(f for f in badges["crystallize"]["facts"]
                 if f["path"] == ["crystallizations", "two_hop attention hop table", "envelope", "final"])
    assert final["value"] is None and final["unit"] == "span", "a crystallization that doesn't hold at the end"

    for fact in badges["crystallize"]["facts"]:
        if isinstance(fact["value"], list):
            fact["value"].append(-1)
    assert reports == kept, "a badge shares no object with the reports"

    for statuses, crystallize, set_ in (({"G7_4": "fail"}, "judged: fail", "judged: pass"),
                                        ({"G7_3": "fail"}, "judged: fail", "judged: fail")):
        failing = _reports(**statuses)
        for gate_id, report in failing.items():
            assert validate_report(report) == [], gate_id
        badges = cb.crystallize_badges(failing)
        _check(badges, failing, MEASUREMENTS)
        assert (badges["crystallize"]["status"], badges["set"]["status"]) == (crystallize, set_), statuses

    other = copy.deepcopy(MEASUREMENTS)
    other["G7.3"]["records_by_form"] = {"fourier": [True, True], "table": [True, True]}
    del other["G7.4"]["crystallizations"]["induction attention copied table"]
    del other["G7.2"]["names"]["token(p-2)"]
    _check(cb.crystallize_badges(_reports(other)), _reports(other), other)


def test_crystallize_badges_refuse_bad_reports():
    reports = _reports()
    for gate_id in ("G7.2", "G7.3", "G7.4"):
        missing = dict(reports)
        del missing[gate_id]
        with pytest.raises(ValueError, match=gate_id.replace(".", r"\.")):
            cb.crystallize_badges(missing)
    invalid = copy.deepcopy(reports)
    invalid["G7.4"]["measurements"]["kl_max"] = 0.3  # a pass its measurements contradict, every path still there
    assert validate_report(invalid["G7.4"])
    with pytest.raises(ValueError, match=r"G7\.4"):
        cb.crystallize_badges(invalid)
    both = copy.deepcopy(reports["G7.4"])
    both["measurements"].update(copy.deepcopy(MEASUREMENTS["G7.3"]))  # a valid G7.4 report that G7.3's paths read
    assert validate_report(both) == []
    with pytest.raises(ValueError, match=r"G7\.3"):
        cb.crystallize_badges(dict(reports, **{"G7.3": both}))
    for gate_id, cut in (("G7.4", ["crystallizations", "modular_addition write sum table", "envelope"]),
                         ("G7.4", ["crystallizations", "two_hop write hop table", "twin"]),
                         ("G7.2", ["names", "token(p+1)", "trained", "steering"]),
                         ("G7.2", ["names", "token(p-1)", "trained", "named_as_planted"])):
        m = copy.deepcopy(MEASUREMENTS)
        node = m[gate_id]
        for step in cut[:-1]:
            node = node[step]
        del node[cut[-1]]
        broken = _reports(m)
        assert validate_report(broken[gate_id]) == [], "still a valid report"
        with pytest.raises(ValueError):
            cb.crystallize_badges(broken)


def test_open_badge_twin_line():
    badges = gb.gesture_badges(phase6_reports())
    facts = badges["open"]["facts"]
    twin = [f for f in facts if f["path"] == ["twin_planted_top_max"]]
    assert len(twin) == 1 and twin[0]["gate"] == "G6.4" and twin[0]["unit"] == "fraction"
    assert twin[0]["label"].endswith("most of any judged target"), \
        f"G6.4 measures it over the judged target only (D53): {twin[0]['label']!r}"
    old = "twin: the planted set's first part in the top 8 parts, most of any target"
    assert all(fact["label"] != old for fact in facts), "the line D53 corrects is gone"
