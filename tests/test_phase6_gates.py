"""Phase 6 gate helpers: settings, bars, seeded instances and judgements, fixed before any gate runs (D49 to D52).

Contract: gates/phase6/_common.py. The settings are feasibility round 3's, so the gates retrace it. The tests that
read a model use phase6_fixtures.py's toy models and need the jax lane.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT


def _common():
    gates = REPO_ROOT / "gates" / "phase6"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase6_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # registered, as an import would, so its dataclasses resolve their annotations
    spec.loader.exec_module(module)
    return module


def test_settings_and_bars_are_fixed():
    c = _common()
    assert c.INSTANCES == {"name": 512, "edge": 512, "open": 128}
    assert c.SEEDS == {"name": 31, "edge": 32, "open": 35, "generalize": 40, "reference": 70}
    assert c.REFERENCE == {"latents": 256, "values": 32, "per_value": 8, "open": 32}
    assert (c.IDENTITY_WINDOWS, c.RECORD_INSTANCES) == (64, 32)
    assert c.BARS == {"named_as_planted": 0.7, "follows": 0.8, "null_follows": 0.05, "twin_named": 0.1,
                      "twin_follows": 0.05, "edge_follows": 0.8, "edge_null_follows": 0.1, "output_follows": 0.8,
                      "kept": 0.4, "null_kept": 0.8, "residual_alone": 0.1, "twin_excess": 0.1, "reading_ratio": 1.0,
                      "within": 0.8, "planted_top": 0.6, "random_within": 0.1, "twin_planted_top": 0.1,
                      "slice_f1": 0.5, "slice_follows": 0.8, "seed_f1": 0.5, "seed_follows": 0.8, "start_offset": 2}
    assert c.OPEN == {"two_hop": {"carry": 16, "hop": 64}, "induction": {"copy": 64}}
    assert c.TWIN_OPEN == (("induction", "copy"),)
    assert c.GENERALIZE == {"two_hop": ("b", 2, "query"), "induction": ("token(p+1)", 1, "mechanism")}
    assert c.IDENTITIES == ("empty", "zero codes", "zero amounts")


def test_judgements():
    c = _common()
    assert c.least([0.9, 0.81, 0.95]) == pytest.approx(0.81) and c.greatest([0.1, 0.3]) == pytest.approx(0.3)
    for empty in ([], [0.9, None]):
        assert c.least(empty) is None and c.greatest(empty) is None, "a judged reading that wasn't made fails its bar"
    assert c.ratio(8.0, 16) == 0.5 and c.ratio(64, 64) == 1.0 and c.ratio(None, 16) is None
    assert c.twin_excess(0.515, 0.208) == pytest.approx(0.307) and c.twin_excess(0.5, None) is None
    assert c.twin_excess(0.0, 0.032) == pytest.approx(-0.032)

    steer = lambda n, f, u, a, nf, nu: {"causal": True, "instances": n, "follows": f, "unchanged": u,  # noqa: E731
                                        "ablated_changed": a, "null_follows": nf, "null_unchanged": nu}
    pooled = c.pool([steer(10, 1.0, 0.0, 0.5, 0.0, 1.0), None, steer(30, 0.6, 0.2, 0.1, 0.1, 0.9),
                     {"causal": False, "instances": 50, "kl": 0.3, "null_kl": 0.0}])
    assert pooled["values"] == 2 and pooled["instances"] == 40, "None and non-causal steerings are left out"
    assert pooled["follows"] == pytest.approx((10 + 18) / 40) and pooled["unchanged"] == pytest.approx(6 / 40)
    assert pooled["ablated_changed"] == pytest.approx(8 / 40) and pooled["null_follows"] == pytest.approx(3 / 40)
    assert pooled["null_unchanged"] == pytest.approx(37 / 40), "weighted by instances: the share over all of them"
    assert c.pool([None]) == {"values": 0, "instances": 0, "follows": None, "unchanged": None, "ablated_changed": None,
                              "null_follows": None, "null_unchanged": None}

    width = 10
    order = np.array([3 * width + 7, 5 * width + 2, 2 * width + 1, 5 * width + 4, 3 * width + 2])
    assert c.planted_rank(order, 5, [4, 9], width) == 3 and c.planted_rank(order, 5, [2, 4], width) == 1
    assert c.planted_rank(order, 3, [2], width) == 4, "by position and latent, not latent alone"
    with pytest.raises(ValueError):
        c.planted_rank(order, 4, [7], width)

    steps = [0, 1000, 2000, 3000, 4000, 5000, 6000]
    env = {"steps": [4000, 5000, 6000], "runs": [[4000, 6000]], "final": [4000, 6000], "gap": [],
           "probe_reads_from": 3000}
    assert c.envelope_checks(env, steps) == {"step0_outside": True, "one_final_run": True, "start_offset": 1}
    broken = dict(env, steps=[0, 5000, 6000], runs=[[0, 0], [5000, 6000]], final=[5000, 6000], probe_reads_from=6000)
    assert c.envelope_checks(broken, steps) == {"step0_outside": False, "one_final_run": False, "start_offset": 1}
    lost = dict(env, runs=[[4000, 5000]], final=None, steps=[4000, 5000], probe_reads_from=None)
    assert c.envelope_checks(lost, steps) == {"step0_outside": True, "one_final_run": False, "start_offset": None}
    assert c.envelope_checks(env, steps[1:])["step0_outside"] is False, "no step 0, nothing outside it"
    for value in (c.least([np.float32(0.5)]), c.ratio(np.int64(8), 16), c.twin_excess(np.float32(0.5), 0.25),
                  c.pool([steer(3, np.float64(1.0), 0.0, 0.0, 0.0, 1.0)])["follows"]):
        assert type(value) is float, "plain floats, ready for a report"
    assert type(c.planted_rank(order, 5, [2], width)) is int
    assert all(type(v) in (bool, int) for v in c.envelope_checks(env, steps).values())


def test_ticking():
    c = _common()
    import time

    lines = []
    with c.ticking("naming", interval_s=0.05, write=lines.append):
        time.sleep(0.3)
    assert len(lines) >= 3 and all(line.startswith("naming: still working") for line in lines)
    count = len(lines)
    time.sleep(0.2)
    assert len(lines) == count, "it stops on exit"
    with pytest.raises(RuntimeError):
        with c.ticking("failing", interval_s=0.05, write=lines.append):
            raise RuntimeError("boom")
    count = len(lines)
    time.sleep(0.2)
    assert len(lines) == count, "and on an exception"


def _table():
    """A class of one position over 30 windows: d = 1 on rows 0 to 9, 2 on rows 10 to 19, 3 on rows 20 to 29.

    Latent 0 fires exactly where d = 1; latent 1 there and on rows 10 to 14, with large codes; latent 2 everywhere
    (shared); latent 3 exactly where d = 2; latent 4 on rows 10 to 12 with the largest codes; latent 5 on rows 10 to 12
    too, with the same codes as latent 4; latent 6 exactly where d = 3, with codes as large as latent 4's; latent 7 on
    rows 20 to 24 with larger codes still, and faintly on rows 0, 1 and 10 to 12, so its F1 for d = 3 is exactly 0.5.
    """
    from nablaframe.gestures.sets import ClassTable

    codes = np.zeros((30, 1, 8), np.float32)
    codes[0:10, 0, 0] = 1.0
    codes[0:15, 0, 1] = 3.0
    codes[:, 0, 2] = 5.0
    codes[10:20, 0, 3] = 1.0
    codes[10:13, 0, 4] = 9.0
    codes[10:13, 0, 5] = 9.0
    codes[20:30, 0, 6] = 9.0
    codes[20:25, 0, 7] = 20.0
    codes[[0, 1, 10, 11, 12], 0, 7] = 1.0
    d = np.repeat([1, 2, 3], 10).reshape(30, 1).astype(np.int64)
    return ClassTable(codes, {"d": d}, [0])


def test_recovery():
    c = _common()
    table = _table()
    one = c.recovery(table, "d", 1)
    assert one["best_f1"] == {"latent": 0, "name": {"description": "d", "value": 1, "f1": 1.0}, "named": True,
                              "as_planted": True, "in_set": True}
    big = one["largest_code"]
    assert big["latent"] == 1, "the largest mean code, the shared latent left out"
    assert big["name"] == {"description": "d", "value": 1, "f1": pytest.approx(0.8)} and big["as_planted"]
    two = c.recovery(table, "d", 2)
    assert two["best_f1"]["latent"] == 3 and two["best_f1"]["as_planted"]
    weak = two["largest_code"]
    assert weak["latent"] == 4, "ties to the lower latent"
    assert weak["name"]["f1"] == pytest.approx(6 / 13) and weak["named"] is False and weak["as_planted"] is False
    assert weak["in_set"] is (4 in table.set_for("d", 2))
    three = c.recovery(table, "d", 3)
    assert three["best_f1"]["latent"] == 6 and three["largest_code"]["latent"] == 7
    assert three["largest_code"]["name"] == {"description": "d", "value": 3, "f1": 0.5}
    assert three["largest_code"]["named"] and three["largest_code"]["as_planted"], "an F1 of exactly 0.5 is named"
    with pytest.raises(ValueError):
        c.recovery(table, "d", 4)
    assert json.loads(json.dumps(one)) == one, "plain JSON"
    rng = np.random.default_rng(0)
    assert c.reference_values(table, "d", 32, rng) == [1, 2, 3]
    from nablaframe.gestures.sets import ClassTable

    codes = np.zeros((20, 1, 2), np.float32)
    codes[5:10, 0, 0] = 1.0
    codes[10:15, 0, 0] = 3.0
    codes[10:20, 0, 1] = 1.0
    half = ClassTable(codes, {"d": np.repeat([1, 2], 10).reshape(20, 1).astype(np.int64)}, [0])
    assert c.reference_values(half, "d", 32, rng) == [1, 2], "a value whose best F1 is exactly 0.5 is named"
    assert c.recovery(half, "d", 2)["largest_code"] == {
        "latent": 0, "name": {"description": "d", "value": 1, "f1": 0.5}, "named": True, "as_planted": False,
        "in_set": True}, "named, but for the other value: ties go to the lower value"


def test_requests_and_their_hashes():
    c = _common()
    from nablaframe.gestures.records import request_hash

    a, b = "a" * 64, "b" * 64
    got = c.requests("e" * 64, [0, 1000, 20000], (a, b))
    assert len(got) == 64 and got == c.requests("e" * 64, [0, 1000, 20000], (a, b))
    assert [(r.step, r.instances, r.seed) for r in got[:8]] == [
        (0, 32, 0), (0, 32, 1), (0, 512, 0), (0, 512, 1), (20000, 32, 0), (20000, 32, 1), (20000, 512, 0), (20000, 512, 1)]
    assert [got[8 * k].kind for k in range(8)] == ["name", "name", "edge", "open", "open", "generalize", "generalize",
                                                   "generalize"]
    assert dict(got[0].args) == {"dictionary": a, "site": 0, "class": "query", "latent": 0}
    assert dict(got[8].args)["dictionary"] == b
    assert dict(got[16].args) == {"relation": "hop", "up": a, "up_site": 0, "down": b, "down_site": 0}
    assert dict(got[24].args) == {"dictionary": b, "site": 0, "class": "query", "upstream": a, "upstream_site": 0,
                                  "latent": 0}
    assert dict(got[32].args)["description"] == "b" and dict(got[32].args)["value"] == 0
    assert dict(got[40].args) == {"dictionary": a, "site": 0, "class": "query", "description": "b"}
    assert dict(got[48].args)["dictionary"] == b and dict(got[56].args) == dict(got[16].args)
    assert c.request_hashes(got) == {"checked": 64, "distinct": True, "stable": True}
    assert c.request_hashes(got + got[:1])["distinct"] is False
    assert len({request_hash(r) for r in got}) == 64


def _world(**kwargs):
    import phase6_fixtures as F

    w = F.world("identity", **kwargs)
    w.up, w.down = F.read(w, 0), F.read(w, 1)
    return w


def test_planted_names():
    pytest.importorskip("jax")
    c = _common()
    import phase6_fixtures as F
    from nablaframe.gestures import name

    w = _world()
    got = c.planted_name(w.up, "every", "token(p)", instances=6, seed=4)
    assert (got["description"], got["class"], got["block"], got["tap"]) == \
        ("token(p)", "every", 0, "blocks.0.hook_resid_post")
    assert got["values"] == 11 and [e["value"] for e in got["by_value"]] == list(range(1, 12))
    assert got["named"] == 1.0 and got["named_as_planted"] == 1.0 and got["largest_code"]["in_set"] == 1.0
    assert got["steering"]["values"] == 11 and got["steering"]["instances"] == 66
    assert got["steering"]["follows"] == 1.0 and got["steering"]["null_follows"] == 0.0, "the token's set steers"
    assert got["effect"] == {"last": 0.0, "first": 1.0, "mean_rank_from_top": 0.0, "vocabulary": F.VOCAB}
    assert got["coverage"]["token(p)"] == pytest.approx(1.0) and got["coverage"]["token(p-1)"] == pytest.approx(0.0)
    table = w.up.table("every")
    for entry in got["by_value"]:
        v = entry["value"]
        assert entry["best_f1"] == c.recovery(table, "token(p)", v)["best_f1"]
        effect = name.latent_effects(w.up.dictionary, 0, w.up.params, table.set_for("token(p)", v))
        ends = name.ends(table.mean_codes("token(p)", v)[list(table.set_for("token(p)", v))].astype(np.float64) @ effect, v)
        assert entry["effect"] == {"rank_from_top": ends["rank_from_top"], "rank_from_bottom": ends["rank_from_bottom"]}
    assert json.loads(json.dumps(got)) == got, "plain JSON"

    negative = _world(decoder="negative")
    flipped = c.planted_name(negative.up, "every", "token(p)", instances=6, seed=4)
    assert flipped["effect"]["last"] == 1.0 and flipped["effect"]["first"] == 0.0, "a set that writes minus its token"
    assert flipped["steering"]["follows"] == 0.0
    silent = c.planted_name(F.read(F.world("silent"), 0), "every", "token(p)", instances=6, seed=4)
    assert silent["steering"]["values"] == 0 and silent["steering"]["follows"] is None, "no prediction, nothing pooled"
    for cls, d in (("late", "g"), ("nowhere", "token(p)")):
        with pytest.raises(ValueError):
            c.planted_name(w.up, cls, d, instances=6, seed=4)


def test_instances_are_seeded():
    pytest.importorskip("jax")
    c = _common()
    import phase6_fixtures as F
    from nablaframe.gestures import name

    w = _world()
    quiet = F.read(F.world("silent"), 0)
    got = c.planted_name(quiet, "every", "token(p)", instances=6, seed=4)
    for entry in got["by_value"]:
        want = name.steer(quiet, "every", "token(p)", entry["value"], instances=6, rng=np.random.default_rng(4))
        assert entry["steering"] == want and want["examples"], \
            "each value steers with a fresh generator from the seed: with no prediction, its examples show the draws"

    table = w.up.table("every")
    rng = np.random.default_rng(9)
    active = [int(i) for i in np.flatnonzero(table.active > 0)]
    assert c.reference_sample(table, 100, rng) == active, "all of them, when there are few enough"
    assert rng.bit_generator.state == np.random.default_rng(9).bit_generator.state, "and the generator untouched"
    want = sorted(int(x) for x in np.random.default_rng(9).choice(np.asarray(active, np.int64), size=5, replace=False))
    assert c.reference_sample(table, 5, np.random.default_rng(9)) == want
    named = [v for v in range(1, 12)]
    assert c.reference_values(table, "token(p)", 32, np.random.default_rng(9)) == named
    want = sorted(int(x) for x in np.random.default_rng(9).choice(np.asarray(named, np.int64), size=3, replace=False))
    assert c.reference_values(table, "token(p)", 3, np.random.default_rng(9)) == want
    assert c.reference_values(table, "token(p+1)", 32, np.random.default_rng(9)) == [], "none named"
    with pytest.raises(ValueError):
        c.reference_values(w.up.table("late"), "g", 3, np.random.default_rng(9))

    candidates = c.relation_candidates(w.up, w.down, "carry")
    first = c.open_targets(w.up, w.down, candidates, instances=7, seed=11)
    cut = sorted(np.random.default_rng(11).choice(len(candidates), size=7, replace=False))
    assert first["candidates"] == len(candidates) and first["instances"] == 7
    assert first["windows"] == [[candidates[k].window, candidates[k].position] for k in cut], "cut with the seed"
    assert c.open_targets(w.up, w.down, candidates, instances=7, seed=11) == first
    assert c.open_targets(w.up, w.down, candidates, instances=7, seed=12)["windows"] != first["windows"]
    assert F.PAIRED == (4, 5)


def test_relation_candidates_and_planted_ranks():
    pytest.importorskip("jax")
    c = _common()
    import phase6_fixtures as F

    w = _world()
    got = c.relation_candidates(w.up, w.down, "carry")
    want = [c.OpenCandidate(i, p, tuple(F.latent(int(w.windows[i, p]), j) for j in range(4)), p,
                            tuple(F.latent(int(w.windows[i, p]), j) for j in range(4)))
            for i in range(len(w.windows)) for p in F.PAIRED]
    assert got == want, "every pair, window by window, the target's set and the planted set"
    nxt = c.relation_candidates(w.up, w.down, "next")
    Tu, Td = w.up.table("every"), w.down.table("every")
    assert nxt and len(nxt) < len(want), "only where the downstream set fires: no latent here reads token(p-1)"
    for x in nxt:
        u = int(w.windows[x.window, x.position - 1])
        assert x.planted_position == x.position - 1 and x.planted_latents == Tu.set_for("token(p)", u)
        assert x.latents == Td.set_for("token(p-1)", u) and w.down.codes[x.window, x.position, list(x.latents)].sum() > 0
    silent = _world(silent={1: (3,)})
    fired = c.relation_candidates(silent.up, silent.down, "carry")
    assert len(fired) < len(want) and all(w.windows[x.window, x.position] != 3 for x in fired), "a set that never fires"
    for bad in ((w.down, w.down, "carry"), (w.up, w.down, "nowhere"), (w.up, F.read(F.world("identity"), 1), "carry")):
        with pytest.raises(ValueError):
            c.relation_candidates(*bad)

    measured = c.open_targets(w.up, w.down, got, instances=10, seed=0)
    assert measured["measured"] == 10 and measured["planted"]["ranks"] == [0] * 10
    assert measured["planted"]["top"] == 1.0 and measured["planted"]["rank_median"] == 0.0, \
        "the token's own set, at its own position, ranks first"
    elsewhere = [c.OpenCandidate(x.window, x.position, x.latents, x.position - 1, x.latents) for x in got]
    missed = c.open_targets(w.up, w.down, elsewhere, instances=10, seed=0)
    assert missed["planted"]["top"] == 0.0 and min(missed["planted"]["ranks"]) >= 8, "the wrong position ranks low"
    assert missed["orders"] == measured["orders"], "the planted set doesn't change the measurement"
    for latent, rank, top in ((3, 7, 1.0), (4, 8, 0.0)):
        # Nothing moves between positions, so every part before the target's position ranks after its four, by index.
        at_zero = [c.OpenCandidate(x.window, x.position, x.latents, 0, (latent,)) for x in got]
        edge = c.open_targets(w.up, w.down, at_zero, instances=10, seed=0)["planted"]
        assert edge["ranks"] == [rank] * 10 and edge["top"] == top, "the top 8 are ranks 0 to 7"
    bare = [c.OpenCandidate(x.window, x.position, x.latents, None, None) for x in got]
    assert c.open_targets(w.up, w.down, bare, instances=10, seed=0)["planted"] is None
    with pytest.raises(ValueError):
        c.open_targets(w.up, w.down, bare[:3] + got[3:], instances=10, seed=0)
    for bad in ({"instances": 0, "seed": 0}, {"instances": 3, "seed": -1}):
        with pytest.raises(ValueError):
            c.open_targets(w.up, w.down, got, **bad)
    dead = [c.OpenCandidate(x.window, x.position, (F.DEAD_LATENT,), x.planted_position, x.planted_latents)
            for x in got[:5]]
    spanless = c.open_targets(w.up, w.down, dead, instances=10, seed=0)
    assert (spanless["instances"], spanless["no_span"], spanless["measured"]) == (5, 5, 0), \
        "a target no upstream code moves is left out, and counted"
    assert spanless["planted"] == {"ranks": [], "rank_median": None, "top": None}
    negative = _world(decoder="negative")
    flipped = c.open_targets(negative.up, negative.down, c.relation_candidates(negative.up, negative.down, "carry"),
                             instances=10, seed=0)
    assert flipped["planted"]["top"] == 1.0, "ranked by absolute attribution: a set that writes minus its token"
    assert flipped["orders"]["signed"]["within"] < flipped["orders"]["absolute"]["within"] or \
        flipped["orders"]["signed"]["readings"] != flipped["orders"]["absolute"]["readings"]
    empty = c.open_targets(w.up, w.down, [], instances=10, seed=0)
    assert empty["instances"] == 0 and empty["planted"] is None and empty["orders"]["absolute"]["readings"] == []


def test_open_targets_match_open_up():
    pytest.importorskip("jax")
    c = _common()
    from nablaframe.gestures import open as opening

    w = _world()
    table = w.down.table("late")
    latents = table.set_for("token(p)", 5)
    positions = list(w.declaration.classes["late"])
    candidates = [c.OpenCandidate(i, p, latents, None, None) for i in range(len(w.windows)) for p in positions
                  if w.windows[i, p] == 5 and w.down.codes[i, p, list(latents)].sum() > 0]
    got = c.open_targets(w.up, w.down, candidates, instances=6, seed=3)
    want = opening.open_up(w.up, w.down, "late", instances=6, seed=3, description="token(p)", value=5)
    for key in ("candidates", "instances", "no_span", "measured", "windows"):
        assert got[key] == want[key], key
    for kind in opening.ORDERS:
        assert got["orders"][kind]["readings"] == want["orders"][kind]["readings"], kind
        assert got["orders"][kind]["within"] == want["orders"][kind]["within"], kind
        assert got["orders"][kind]["reading_median"] == want["orders"][kind]["reading_median"], kind
        np.testing.assert_allclose(got["orders"][kind]["error_median"], want["orders"][kind]["error_median"],
                                   rtol=1e-6, atol=1e-9)
    assert got["planted"] is None


def test_dsteer_identities():
    pytest.importorskip("jax")
    c = _common()
    import phase6_fixtures as F

    codes = np.zeros((2, 3, 4), np.float32)
    codes[0, 0] = [0.5, 0.0, 0.7, 0.0]
    codes[0, 1] = [1.0, 2.0, 3.0, 4.0]
    codes[0, 2] = [0.0, 0.2, 0.2, 0.0]
    codes[1, 0] = [0.3, 0.3, 0.1, 0.0]
    edits = c.identity_edits(codes)
    assert list(edits) == list(c.IDENTITIES)
    remove, add, amounts = edits["empty"]
    assert remove.shape == (0, 3) and add.shape == (0, 3) and amounts.shape == (0,)
    assert remove.dtype == add.dtype == np.int32 and amounts.dtype == np.float32
    remove, add, amounts = edits["zero codes"]
    assert remove.tolist() == [[0, 0, 1], [0, 2, 0], [1, 0, 3], [1, 1, 0], [1, 2, 0]] and add.shape == (0, 3), \
        "the lowest latent whose code is zero, none where every code is above zero"
    remove, add, amounts = edits["zero amounts"]
    assert add.tolist() == [[0, 0, 2], [0, 1, 3], [0, 2, 1], [1, 0, 0], [1, 1, 0], [1, 2, 0]] and remove.shape == (0, 3)
    assert amounts.tolist() == [0.0] * 6 and amounts.dtype == np.float32, "the largest code, ties to the lower latent"

    w = F.world("identity")
    tokens = w.windows[:8, :F.SEQ]
    for block in (0, 1):
        assert c.dsteer_identities(w.model, w.params, n_layers=w.n_layers, dictionary=w.dictionaries[block],
                                   tokens=tokens) == [True, True, True]
