"""G6.2: every planted name, on the trained zoo models and their twins (judged; D49).

Spec: docs/specs/phase-6.md, D49, "What each run declares" and "Discrimination tests on the zoo";
helpers in ``_common.py``. Report: ``reports/phase6/g62_names.json``. Run in nf-jax, on the Mac,
after G4.1, G5.2, G5.3 and G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins (``twin_params``), each with its own
Phase 5 TopK SAEs from the cache, read over the held-out windows. For every planted name the run
declares (``declaration.planted``: two-hop's ``b`` after blocks 2 and 1 and ``token(p+1)`` after
block 3, at ``query``; induction's ``token(p)`` after block 0 at ``every`` and ``token(p+1)`` after
block 1 at ``mechanism``), on each model: ``planted_name(reading, class, description,
instances=INSTANCES["name"], seed=SEEDS["name"])``, the reading being the block's TopK SAE's.

``names``: ``{"<run> <description> after block <l> at <class>": {"judged", "trained",
"twin"}}``, ``judged`` the declaration's flag and the other two ``planted_name``'s results. Over the
judged names, each through ``least`` or ``greatest``, so a reading that wasn't made fails:

* ``named_as_planted_min``: the least trained ``named_as_planted``
* ``follows_min`` and ``null_follows_max``: the least trained ``steering["follows"]`` and the
  greatest trained ``steering["null_follows"]``
* ``twin_named_max`` and ``twin_follows_max``: the greatest twin ``named`` and twin
  ``steering["follows"]``

Measurements: REQUIRED_MEASUREMENTS["G6.2"]. Pass needs ``named_as_planted_min >=
BARS["named_as_planted"]``, ``follows_min >= BARS["follows"]``, ``null_follows_max <=
BARS["null_follows"]``, ``twin_named_max <= BARS["twin_named"]`` and ``twin_follows_max <=
BARS["twin_follows"]``. Round 3 read, for two-hop's ``b`` and answer and induction's current and
next token: named as planted 0.812, 1.000, 0.969 and 1.000; follows 0.939, 1.000, 1.000 and 0.951,
the null at most 0.008; on the twins, no value named and follows at most 0.018.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g62_names.py``.
#   Takes --overwrite and --allow-dirty. It reads G4.1's runs after ``require`` of G4.1, G5.2, G5.3 and G6.1;
#   every dictionary comes from the Phase 5 cache (``_common.topk``). It writes nothing but its report.
# * Each planted name is read on one model at a time, its block's site read once over the held-out windows,
#   so at most one reading is held at once.
# * ``planted_name``'s steering pools each value's ``name.steer``, a fresh generator from ``SEEDS["name"]``
#   for each value (delta 4).

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G6.2"
REPORT_NAME = "g62_names.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G6.1")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, g41 = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(g41)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.2 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["named_as_planted_min"]) and m["named_as_planted_min"] >= bars["named_as_planted"]
          and n(m["follows_min"]) and m["follows_min"] >= bars["follows"]
          and n(m["null_follows_max"]) and m["null_follows_max"] <= bars["null_follows"]
          and n(m["twin_named_max"]) and m["twin_named_max"] <= bars["twin_named"]
          and n(m["twin_follows_max"]) and m["twin_follows_max"] <= bars["twin_follows"])
    return "pass" if ok else "fail"


def measure(g41: dict[str, Any]) -> dict[str, Any]:
    names: dict[str, Any] = {}
    run_ids, steps = {}, {}
    for run in _common.ZOO:
        m = _common.zoo_model(run, g41)
        run_ids[run], steps[run] = m.run_id, m.step
        _common.say(f"{GATE_ID}: {run} run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows")
        for planted in m.declaration.planted:
            place = planted.place
            key = _common.place_name(run, place.description, place.block, place.cls)
            names[key] = {"judged": bool(planted.judged),
                          "trained": planted_name(m, m.params, _common.topk(m, place.block), place, "trained"),
                          "twin": planted_name(m, m.twin, _common.topk(m, place.block, twin=True), place, "twin")}
            for who in ("trained", "twin"):
                entry = names[key][who]
                _common.say(f"{GATE_ID}: {key}, {who}: {entry['values']} values, named {entry['named']:.3f}, as planted"
                            f" {entry['named_as_planted']:.3f}; follows {_common.show(entry['steering']['follows'])},"
                            f" null {_common.show(entry['steering']['null_follows'])}")

    judged = [entry for entry in names.values() if entry["judged"]]
    trained = [entry["trained"] for entry in judged]
    twin = [entry["twin"] for entry in judged]
    out = {
        "names": names,
        "named_as_planted_min": _common.least([e["named_as_planted"] for e in trained]),
        "follows_min": _common.least([e["steering"]["follows"] for e in trained]),
        "null_follows_max": _common.greatest([e["steering"]["null_follows"] for e in trained]),
        "twin_named_max": _common.greatest([e["named"] for e in twin]),
        "twin_follows_max": _common.greatest([e["steering"]["follows"] for e in twin]),
        "bars": _common.BARS,
        "instances": _common.INSTANCES["name"],
        "seed": _common.SEEDS["name"],
        "run_ids": run_ids,
        "steps": steps,
    }
    _common.say(f"{GATE_ID}: named as planted at least {_common.show(out['named_as_planted_min'])}; follows at least"
                f" {_common.show(out['follows_min'])}, null at most {_common.show(out['null_follows_max'])}; on the twins,"
                f" named at most {_common.show(out['twin_named_max'])}, follows at most"
                f" {_common.show(out['twin_follows_max'])}")
    return out


def planted_name(m: Any, params: Any, dictionary: Any, place: Any, who: str) -> dict[str, Any]:
    """``planted_name`` of one place on one model: its block's site read, then every value pointed and steered."""
    label = f"{GATE_ID}: {_common.place_name(m.name, place.description, place.block, place.cls)}, {who}"
    reading = _common.read(m, params, dictionary, f"{label}: reading the site")
    with _common.ticking(f"{label}: naming and steering every value", write=_common.say):
        return _common.planted_name(reading, place.cls, place.description, instances=_common.INSTANCES["name"],
                                    seed=_common.SEEDS["name"])


if __name__ == "__main__":
    raise SystemExit(main())
