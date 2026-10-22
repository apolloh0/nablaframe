"""G6.4: open-this-up on the judged targets, on the trained zoo models and their twins (judged; D51).

Spec: docs/specs/phase-6.md, D51 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase6/g64_open.json``. Run in nf-jax, on the Mac, after G4.1, G5.2, G5.3 and
G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins, with the Phase 5 TopK SAEs from the
cache. Each target in ``OPEN`` is opened along a relation the run declares (two-hop's ``carry``
and ``hop``, induction's ``copy``). With ``R`` that relation, ``up`` the reading of the TopK SAE
at ``R.up``'s block and ``down`` the reading at ``R.down``'s, on each model:
``open_targets(up, down, relation_candidates(up, down, R.name), instances=INSTANCES["open"],
seed=SEEDS["open"])``. So each instance opens the downstream place's set for its own value, at its
downstream position, into the block below, and its planted set is the upstream place's set for its
own value at the upstream position:

* the carry: ``b``'s set at ``query`` after block 2, into block 1; planted, ``b``'s set at
  ``query`` after block 1
* the hop: ``token(p+1)``'s set at ``query`` after block 3, into block 2; planted, ``b``'s set at
  ``query`` after block 2
* the copy: ``token(p+1)``'s set at ``s + P - 1`` after block 1, into block 0; planted,
  ``token(p)``'s set at the sole target ``s`` after block 0

``targets``: ``{"<run> <relation>": {"target", "planted", "bar", "twin_judged", "trained",
"twin"}}``: the target's and the planted set's places, each ``{"description", "block",
"class"}``; the bar from ``OPEN``; whether ``TWIN_OPEN`` holds it; and each model's
``open_targets`` result. Over the targets:

* ``reading_ratio_max``: the greatest ``ratio`` of a trained absolute-order ``reading_median``
  to its bar
* ``within_min``: the least trained absolute-order ``within``
* ``planted_top_min``: the least trained ``planted["top"]``
* ``random_within_max``: the greatest trained random-order ``within``
* ``twin_planted_top_max``: the greatest twin ``planted["top"]`` over ``TWIN_OPEN``'s targets

each through ``greatest`` or ``least``. Recorded beside them: signed order, and the two-hop twin,
whose residual stream carries the same position's set one block earlier, so its planted set ranks
high as well.

Measurements: REQUIRED_MEASUREMENTS["G6.4"]. Pass needs ``reading_ratio_max <=
BARS["reading_ratio"]``, ``within_min >= BARS["within"]``, ``planted_top_min >=
BARS["planted_top"]``, ``random_within_max <= BARS["random_within"]`` and ``twin_planted_top_max
<= BARS["twin_planted_top"]``. Round 3 read, for the carry, the hop and the copy: medians of 8, 32
and 24 parts; within 128 parts for 1.000, 0.930 and 0.992; the planted set's first part in the top
8 for 1.000, 0.781 and 0.797; in random order within 128 for at most 0.016; on the induction twin,
the planted set in the top 8 for 0.000.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g64_open.py``.
#   Takes --overwrite and --allow-dirty. It reads G4.1's runs after ``require`` of G4.1, G5.2, G5.3 and G6.1;
#   every dictionary comes from the Phase 5 cache (``_common.topk``). It writes nothing but its report.
# * Each model's sites are read once per block over the held-out windows, with that model's params and the
#   run's declaration, so both readings of a relation share them as ``relation_candidates`` requires.
#   Open-this-up's passes are ``open_targets``'s, through ``gestures/open.py``.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G6.4"
REPORT_NAME = "g64_open.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G6.1")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, g41 = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(g41)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.4 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["reading_ratio_max"]) and m["reading_ratio_max"] <= bars["reading_ratio"]
          and n(m["within_min"]) and m["within_min"] >= bars["within"]
          and n(m["planted_top_min"]) and m["planted_top_min"] >= bars["planted_top"]
          and n(m["random_within_max"]) and m["random_within_max"] <= bars["random_within"]
          and n(m["twin_planted_top_max"]) and m["twin_planted_top_max"] <= bars["twin_planted_top"])
    return "pass" if ok else "fail"


def _place(place: Any) -> dict[str, Any]:
    return {"description": place.description, "block": int(place.block), "class": place.cls}


def measure(g41: dict[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    run_ids, steps = {}, {}
    for run, relations in _common.OPEN.items():
        m = _common.zoo_model(run, g41)
        run_ids[run], steps[run] = m.run_id, m.step
        _common.say(f"{GATE_ID}: {run} run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows")
        places = {name: m.declaration.relation(name) for name in relations}
        blocks = sorted({place.block for R in places.values() for place in (R.up, R.down)})
        for who, params in (("trained", m.params), ("twin", m.twin)):
            readings = {l: _common.read(m, params, _common.topk(m, l, twin=who == "twin"),
                                        f"{GATE_ID}: {run} {who}, reading block {l}") for l in blocks}
            for name, R in places.items():
                key = f"{run} {name}"
                entry = targets.setdefault(key, {"target": _place(R.down), "planted": _place(R.up),
                                                 "bar": relations[name], "twin_judged": (run, name) in _common.TWIN_OPEN})
                up, down = readings[R.up.block], readings[R.down.block]
                with _common.ticking(f"{GATE_ID}: {key}, {who}: opening the targets", write=_common.say):
                    candidates = _common.relation_candidates(up, down, R.name)
                    entry[who] = _common.open_targets(up, down, candidates, instances=_common.INSTANCES["open"],
                                                      seed=_common.SEEDS["open"])
                got = entry[who]
                planted = got["planted"] or {}
                _common.say(f"{GATE_ID}: {key}, {who}: {got['measured']} of {got['instances']} instances measured"
                            f" ({got['candidates']} candidates, {got['no_span']} without a span); reading median"
                            f" {_common.show(got['orders']['absolute']['reading_median'])}, within 128"
                            f" {_common.show(got['orders']['absolute']['within'])}, random within"
                            f" {_common.show(got['orders']['random']['within'])}; planted in the top"
                            f" {_common.opening.TOP_PARTS} {_common.show(planted.get('top'))}")
            del readings

    trained = [(entry, entry["trained"]) for entry in targets.values()]
    out = {
        "targets": targets,
        "reading_ratio_max": _common.greatest([_common.ratio(t["orders"]["absolute"]["reading_median"], entry["bar"])
                                               for entry, t in trained]),
        "within_min": _common.least([t["orders"]["absolute"]["within"] for _, t in trained]),
        "planted_top_min": _common.least([None if t["planted"] is None else t["planted"]["top"] for _, t in trained]),
        "random_within_max": _common.greatest([t["orders"]["random"]["within"] for _, t in trained]),
        "twin_planted_top_max": _common.greatest([None if entry["twin"]["planted"] is None
                                                  else entry["twin"]["planted"]["top"]
                                                  for entry in targets.values() if entry["twin_judged"]]),
        "bars": _common.BARS,
        "instances": _common.INSTANCES["open"],
        "seed": _common.SEEDS["open"],
        "parts": list(_common.opening.PARTS),
        "run_ids": run_ids,
        "steps": steps,
    }
    _common.say(f"{GATE_ID}: reading over its bar at most {_common.show(out['reading_ratio_max'])}; within 128 at least"
                f" {_common.show(out['within_min'])}; planted in the top {_common.opening.TOP_PARTS} at least"
                f" {_common.show(out['planted_top_min'])}; random order within 128 at most"
                f" {_common.show(out['random_within_max'])}; the induction twin's planted set in the top"
                f" {_common.opening.TOP_PARTS} at most {_common.show(out['twin_planted_top_max'])}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
