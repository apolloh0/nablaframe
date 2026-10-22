"""Phase 6 delta 5: gesture badges, generated from the Phase 6 reports (D38, D47 to D52). Needs no JAX.

Contract: src/nablaframe/views/gesture_badges.py. Every place in these reports carries its own numbers, and the places
are listed out of order, so a wrong path, a wrong place or an unsorted walk reads a wrong value.
"""

from __future__ import annotations

import copy
import json

import pytest

from nablaframe.reports import validate_report
from nablaframe.views import gesture_badges as gb  # looked up per test, so a missing name fails its test alone
from test_phase1_store_reports import _v2
from test_reports import passing_report

NAMES = {  # G6.2's places, out of order, each judged or recorded
    "two_hop token(p+1) after block 3 at query": True,
    "induction token(p) after block 0 at every": True,
    "two_hop b after block 1 at query": False,
    "two_hop b after block 2 at query": True,
    "induction token(p+1) after block 1 at mechanism": True,
}
EDGES = {"two_hop two back": False, "induction copy": True, "two_hop hop": True, "two_hop carry": False}
TARGETS = {"two_hop hop": (64, False), "induction copy": (64, True), "two_hop carry": (16, False)}
GENERAL = ("two_hop b after block 2 at query", "induction token(p+1) after block 1 at mechanism")


def _name(i: int) -> dict:
    side = lambda base: {"named": base, "named_as_planted": round(base - 0.01, 4),  # noqa: E731
                         "steering": {"follows": round(base - 0.02, 4), "null_follows": 0.001 * i, "values": 48},
                         "effect": {"first": round(0.1 * i, 4), "last": round(1 - 0.1 * i, 4), "vocabulary": 51}}
    return {"trained": side(round(0.9 + 0.01 * i, 4)), "twin": side(round(0.001 * i, 4))}


def _edge(i: int, judged: bool) -> dict:
    side = lambda base, label: {"follows": base, "null_follows": round(base / 20, 4), "kept": 0.1,  # noqa: E731
                                "label": label}
    return {"judged": judged, "record": f"{i:x}" * 64,
            "trained": side(round(0.91 + 0.01 * i, 4), "carried" if i % 2 else "computed"),
            "twin": side(round(0.1 + 0.01 * i, 4), "carried")}


def _target(i: int, bar: int, twin_judged: bool) -> dict:
    side = lambda reading, within, top: {"orders": {"absolute": {"reading_median": reading,  # noqa: E731
                                                                 "within": within, "readings": []}},
                                         "planted": {"top": top, "ranks": [], "rank_median": 2.0}}
    return {"bar": bar, "twin_judged": twin_judged, "target": {}, "planted": {},
            "trained": side(8.0 * (i + 1), round(0.95 - 0.01 * i, 4), round(0.8 + 0.01 * i, 4)),
            "twin": side(None, round(0.5 - 0.01 * i, 4), round(0.01 * i, 4))}


def _general(i: int) -> dict:
    final = [6000 if i else 700, 20000 if i else 4000]
    return {"record": f"{i + 7:x}" * 64, "checks": {},
            "results": {"envelope": {"final": final, "runs": [final], "steps": [], "gap": [],
                                     "probe_reads_from": final[0] + 50 * i}}}


MEASUREMENTS = {
    "G6.1": {"identities_checked": 18, "identities_exact": 18, "identities_by_site": {},
             "request_hashes_checked": 64, "request_hashes_distinct": True, "request_hashes_stable": True,
             "records_identical": True,
             "records_by_kind": {"name": [True, False], "edge": [False, True], "open": [True, True],
                                 "generalize": [False, False]}},
    "G6.2": {"names": {place: {"judged": judged, **_name(i)} for i, (place, judged) in enumerate(NAMES.items())},
             "named_as_planted_min": 0.812, "follows_min": 0.926, "null_follows_max": 0.0103, "twin_named_max": 0.0,
             "twin_follows_max": 0.021},
    "G6.3": {"edges": {place: _edge(i, judged) for i, (place, judged) in enumerate(EDGES.items())},
             "follows_min": 0.916, "null_follows_max": 0.0332, "output_follows_min": 0.9219, "kept_max": 0.198,
             "null_kept_min": 0.972, "residual_alone_max": 0.0312, "labels_computed": True, "twin_excess_max": 0.0},
    "G6.4": {"targets": {place: _target(i, *spec) for i, (place, spec) in enumerate(TARGETS.items())},
             "reading_ratio_max": 0.5, "within_min": 0.914, "planted_top_min": 0.766, "random_within_max": 0.039,
             "twin_planted_top_max": 0.0},
    "G6.5": {"names": {place: _general(i) for i, place in enumerate(GENERAL)}, "slice_f1_min": 0.785,
             "slice_follows_min": 0.898, "seed_f1_min": 0.747, "seed_follows_min": 0.898, "step0_outside": True,
             "one_final_run": True, "start_offset_max": 1},
}


def _reports(measurements=None, **statuses) -> dict:
    out = {}
    for gate_id, m in (measurements or MEASUREMENTS).items():
        report = _v2(passing_report("G0.3"))
        report.update(gate_id=gate_id, status=statuses.get(gate_id.replace(".", ""), "pass"), env_name="nf-jax",
                      measurements=copy.deepcopy(m))
        assert validate_report(report) == [], (gate_id, validate_report(report))
        out[gate_id] = report
    return out


def _at(report: dict, path: list):
    value = report["measurements"]
    for segment in path:
        value = value[segment]
    return value


def _expected(m: dict) -> dict:
    """Each subject's facts as (gate, path, unit), in the contract's order, for measurements ``m``."""
    def built(kind):
        return [("G6.1", ["records_by_kind", kind, 0], "flag"), ("G6.1", ["records_by_kind", kind, 1], "flag")]

    name = built("name") + [("G6.2", [k], "fraction") for k in ("named_as_planted_min", "follows_min",
                                                                "null_follows_max", "twin_named_max",
                                                                "twin_follows_max")]
    for place in sorted(m["G6.2"]["names"]):
        if m["G6.2"]["names"][place]["judged"]:
            name += [("G6.2", ["names", place, "trained", *path], "fraction")
                     for path in (["named_as_planted"], ["steering", "follows"], ["effect", "first"],
                                  ["effect", "last"])]
    edge = built("edge") + [("G6.3", [k], "fraction") for k in ("follows_min", "null_follows_max",
                                                                "output_follows_min", "kept_max", "null_kept_min",
                                                                "residual_alone_max")]
    edge += [("G6.3", ["labels_computed"], "flag"), ("G6.3", ["twin_excess_max"], "fraction")]
    for place in sorted(m["G6.3"]["edges"]):
        edge += [("G6.3", ["edges", place, "trained", "follows"], "fraction"),
                 ("G6.3", ["edges", place, "trained", "null_follows"], "fraction"),
                 ("G6.3", ["edges", place, "trained", "label"], "label")]
    opened = built("open") + [("G6.4", ["reading_ratio_max"], "ratio")]
    opened += [("G6.4", [k], "fraction") for k in ("within_min", "planted_top_min", "random_within_max",
                                                   "twin_planted_top_max")]
    for place in sorted(m["G6.4"]["targets"]):
        absolute = ["targets", place, "trained", "orders", "absolute"]
        opened += [("G6.4", absolute + ["reading_median"], "parts"), ("G6.4", ["targets", place, "bar"], "parts"),
                   ("G6.4", absolute + ["within"], "fraction"),
                   ("G6.4", ["targets", place, "trained", "planted", "top"], "fraction")]
        if m["G6.4"]["targets"][place]["twin_judged"]:
            opened += [("G6.4", ["targets", place, "twin", "planted", "top"], "fraction")]
    general = built("generalize") + [
        ("G6.5", ["slice_f1_min"], "score"), ("G6.5", ["slice_follows_min"], "fraction"),
        ("G6.5", ["seed_f1_min"], "score"), ("G6.5", ["seed_follows_min"], "fraction"),
        ("G6.5", ["step0_outside"], "flag"), ("G6.5", ["one_final_run"], "flag"), ("G6.5", ["start_offset_max"], "count")]
    for place in sorted(m["G6.5"]["names"]):
        envelope = ["names", place, "results", "envelope"]
        general += [("G6.5", envelope + ["final"], "span"), ("G6.5", envelope + ["probe_reads_from"], "step")]
    dsteer = [("G6.1", ["identities_exact"], "count"), ("G6.1", ["identities_checked"], "count")]
    return {"name": name, "edge": edge, "open": opened, "generalize": general, "dsteer": dsteer}


def _check(badges: dict, reports: dict, m: dict) -> None:
    assert list(badges) == list(gb.SUBJECTS)
    expected = _expected(m)
    for subject, badge in badges.items():
        assert set(badge) == {"status", "sources", "facts", "notes"}, subject
        assert [(x["gate"], x["path"], x["unit"]) for x in badge["facts"]] == expected[subject], subject
        for fact in badge["facts"]:
            assert set(fact) == {"label", "value", "unit", "gate", "path"}
            assert fact["value"] == _at(reports[fact["gate"]], fact["path"]), (subject, fact["path"])
            assert fact["unit"] in gb.UNITS
            assert isinstance(fact["label"], str) and fact["label"].strip(), (subject, fact["path"])
            place = next((p for p in fact["path"][1:2] if p in (*m["G6.2"]["names"], *m["G6.3"]["edges"],
                                                                 *m["G6.4"]["targets"], *m["G6.5"]["names"])), None)
            if place is not None:
                assert place in fact["label"], f"{subject}: the label names its place, {place}"
        labels = [fact["label"] for fact in badge["facts"]]
        assert len(set(labels)) == len(labels), f"{subject}: each label names one fact"
        assert badge["sources"] == sorted({fact["gate"] for fact in badge["facts"]})
    for fact in badges["edge"]["facts"]:
        if fact["path"][:1] == ["edges"]:
            recorded = not m["G6.3"]["edges"][fact["path"][1]]["judged"]
            assert ("(recorded)" in fact["label"]) == recorded, fact["label"]
    assert json.loads(json.dumps(badges)) == badges, "plain JSON"


def test_gesture_badges_cite_the_reports():
    assert gb.SUBJECTS == ("name", "edge", "open", "generalize", "dsteer")
    assert gb.UNITS == ("fraction", "ratio", "score", "count", "parts", "step", "span", "flag", "label")
    assert gb.REFERENCE_NOTE == ("On the reference run every gesture is recorded, not judged: it has no planted "
                                 "structure to recover (D44). G6.6's report holds its readings.")
    assert gb.STEERING_NOTE == ("Steering swaps a value's set for another value's, drawn from every value of the "
                                "description that the window doesn't hold. Where few of those values have a set that "
                                "names them, as on the reference run, steering rarely follows.")
    assert gb.LABEL_NOTE == ("An edge at one position is labelled carried where the swap still follows at least half as "
                             "often with every block between the two sites frozen: the residual stream passes on "
                             "whatever is put into it. A label says little where follows is no higher than its null.")
    assert gb.TWIN_OPEN_NOTE == ("On the random-weights twin, a planted set at its target's own position ranks in the "
                                 "top 8 as well, because the residual stream carries it. So only the induction target, "
                                 "read across positions, is judged on the twin.")
    assert gb.CHECKPOINTS_NOTE == ("Every checkpoint is read through the final dictionaries (D44), so a name formed "
                                   "another way earlier in training can't be seen. The gap shows the checkpoints where "
                                   "the probe reads the variable but the name doesn't hold.")
    assert gb.PATCH_NOTE == ("Beside each checkpoint, G6.5 patches two-hop's residual stream at position 25 only, where "
                             "the name is read. G4.3's query patch moves positions 24 and 25.")
    assert gb.DSTEER_NOTE == ("The identities show dsteer changes nothing where it should change nothing: removing and "
                              "adding nothing, removing only latents whose code is zero, and adding only zero amounts.")

    reports = _reports()
    badges = gb.gesture_badges(reports)
    _check(badges, reports, MEASUREMENTS)
    assert {s: b["notes"] for s, b in badges.items()} == {
        "name": [gb.STEERING_NOTE, gb.REFERENCE_NOTE], "edge": [gb.LABEL_NOTE, gb.REFERENCE_NOTE],
        "open": [gb.TWIN_OPEN_NOTE, gb.REFERENCE_NOTE],
        "generalize": [gb.CHECKPOINTS_NOTE, gb.PATCH_NOTE, gb.REFERENCE_NOTE], "dsteer": [gb.DSTEER_NOTE]}
    assert all(b["status"] == "judged: pass" for b in badges.values())
    assert badges["name"]["sources"] == ["G6.1", "G6.2"] and badges["dsteer"]["sources"] == ["G6.1"]
    assert reports == _reports(), "the reports are read, never changed"
    with_extra = {**reports, "G6.6": copy.deepcopy(reports["G6.5"]), "G5.3": {"anything": 1}}
    assert gb.gesture_badges(with_extra) == badges, "other keys are ignored"

    m = copy.deepcopy(MEASUREMENTS)  # the facts follow the reports' places and flags
    m["G6.2"]["names"]["two_hop b after block 1 at query"]["judged"] = True
    m["G6.2"]["names"]["induction token(p) after block 0 at every"]["judged"] = False
    m["G6.3"]["edges"]["two_hop carry"]["judged"] = True
    m["G6.4"]["targets"]["two_hop carry"]["twin_judged"] = True
    m["G6.4"]["targets"]["induction copy"]["twin_judged"] = False
    del m["G6.4"]["targets"]["two_hop hop"]
    m["G6.5"]["names"]["a_first two"] = _general(5)
    m["G6.5"]["names"]["two_hop b after block 2 at query"]["results"]["envelope"].update(final=None,
                                                                                         probe_reads_from=None)
    reports = _reports(m)
    _check(gb.gesture_badges(reports), reports, m)

    def status(**statuses):
        return {s: b["status"] for s, b in gb.gesture_badges(_reports(**statuses)).items()}

    passing, failing = "judged: pass", "judged: fail"
    assert status(G61="fail") == {s: failing for s in gb.SUBJECTS}
    assert status(G62="fail") == {"name": failing, "edge": passing, "open": passing, "generalize": passing,
                                  "dsteer": passing}
    assert status(G63="fail") == {"name": passing, "edge": failing, "open": passing, "generalize": passing,
                                  "dsteer": passing}
    assert status(G64="fail") == {"name": passing, "edge": passing, "open": failing, "generalize": passing,
                                  "dsteer": passing}
    assert status(G65="fail") == {"name": passing, "edge": passing, "open": passing, "generalize": failing,
                                  "dsteer": passing}


def test_bad_reports_are_refused():
    def without(gate_id):
        return {g: r for g, r in _reports().items() if g != gate_id}

    def changed(gate_id, edit):
        reports = _reports()
        edit(reports[gate_id]["measurements"])
        return reports

    def misfiled():
        reports = _reports()
        impostor = copy.deepcopy(reports["G6.5"])  # a valid G6.5 report holding every path G6.2's badge facts read
        g62 = reports["G6.2"]["measurements"]
        impostor["measurements"].update({k: v for k, v in g62.items() if k != "names"})
        impostor["measurements"]["names"].update(copy.deepcopy(g62["names"]))
        assert validate_report(impostor) == []
        reports["G6.2"] = impostor
        return reports

    def invalid():
        reports = _reports()
        reports["G6.3"]["measurements"]["follows_min"] = 0.5  # its status says pass
        return reports

    def not_a_report():
        reports = _reports()
        reports["G6.4"] = {"gate_id": "G6.4"}
        return reports

    cases = {f"{g} missing": without(g) for g in ("G6.1", "G6.2", "G6.3", "G6.4", "G6.5")}
    cases |= {
        "a report filed under another gate's id": misfiled(),
        "a report its own rules refuse": invalid(),
        "not a report at all": not_a_report(),
        "a judged name without its effect": changed(
            "G6.2", lambda m: m["names"]["two_hop b after block 2 at query"]["trained"].pop("effect")),
        "an edge without its label": changed("G6.3", lambda m: m["edges"]["two_hop carry"]["trained"].pop("label")),
        "a target without its bar": changed("G6.4", lambda m: m["targets"]["two_hop hop"].pop("bar")),
        "a twin-judged target without its twin": changed("G6.4", lambda m: m["targets"]["induction copy"].pop("twin")),
        "a name without its envelope": changed(
            "G6.5", lambda m: m["names"]["two_hop b after block 2 at query"]["results"].pop("envelope")),
        "a place whose judged flag is missing": changed(
            "G6.2", lambda m: m["names"]["two_hop b after block 1 at query"].pop("judged")),
        "an edge whose judged flag is missing": changed("G6.3", lambda m: m["edges"]["two_hop two back"].pop("judged")),
        "a target whose twin flag is missing": changed("G6.4", lambda m: m["targets"]["two_hop hop"].pop("twin_judged")),
        "a kind's records not checked": changed("G6.1", lambda m: m["records_by_kind"].pop("open")),
        "a kind's records checked once": changed("G6.1", lambda m: m["records_by_kind"].update(edge=[True])),
    }
    for label, reports in cases.items():
        with pytest.raises(ValueError):
            gb.gesture_badges(reports)
        assert label  # each case names what it breaks
    recorded = changed("G6.2", lambda m: m["names"]["two_hop b after block 1 at query"]["trained"].pop("effect"))
    assert gb.gesture_badges(recorded)["name"]["facts"], "a recorded name's facts aren't read, so it can lack them"
