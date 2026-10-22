"""G6.3: every relation, on the trained zoo models and their twins (judged where a task requires it; D50).

Spec: docs/specs/phase-6.md, D50, "What each run declares" and "Discrimination tests on the zoo";
helpers in ``_common.py``. Report: ``reports/phase6/g63_edges.json``. Run in nf-jax, on the Mac,
after G4.1, G5.2, G5.3 and G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins, with the Phase 5 TopK SAEs from the
cache. For every relation the run declares (two-hop's ``hop``, ``carry`` and ``two back``;
induction's ``copy``), its upstream and downstream places read with the TopK SAEs at their blocks:

* **trained:** ``cli.record(RECORD_DIR / run, request, context)``, the request being
  ``{"relation", "up", "up_site": 0, "down", "down_site": 0}`` with the two TopK SAEs' content
  hashes, ``INSTANCES["edge"]`` instances and ``SEEDS["edge"]``, at the final checkpoint: the
  run's edge record, which the record cache answers when it already holds it
* **twin:** ``edge.edge(up, down, relation, instances=INSTANCES["edge"], seed=SEEDS["edge"])`` on
  the twin's readings with the twin's TopK SAEs. The twin isn't a checkpoint of the run, so it gets
  no record.

``edges``: ``{"<run> <relation>": {"judged", "record", "trained", "twin"}}``: the declaration's
flag, the record's key, and each model's results. Over the judged relations, the hop and the copy:

* ``follows_min`` and ``null_follows_max``: the least trained ``follows`` and the greatest
  trained ``null_follows``
* ``output_follows_min``: the least trained ``output["follows"]``, a judged edge without an
  output counting as None
* ``kept_max`` and ``null_kept_min``: the greatest trained ``kept`` and the least ``null_kept``
* ``residual_alone_max``: the greatest trained ``residual_alone["follows"]`` over the judged
  edges whose ``residual_alone`` isn't None, the edges at one position; None where none has one
* ``labels_computed``: every judged trained edge is labelled ``computed``
* ``twin_excess_max``: the greatest ``twin_excess`` of each judged twin edge's ``follows`` and
  ``null_follows``

each through ``least`` or ``greatest``. The carry and two back are recorded beside them.

Measurements: REQUIRED_MEASUREMENTS["G6.3"]. Pass needs ``follows_min >= BARS["edge_follows"]``,
``null_follows_max <= BARS["edge_null_follows"]``, ``output_follows_min >=
BARS["output_follows"]``, ``kept_max <= BARS["kept"]``, ``null_kept_min >= BARS["null_kept"]``,
``residual_alone_max <= BARS["residual_alone"]``, ``labels_computed`` and ``twin_excess_max <=
BARS["twin_excess"]``. Round 3 read the hop and the copy: follows 0.939 and 1.000 (nulls 0.029 and
0.008), the output 0.934 and 1.000, kept 0.211 and 0.000 (nulls 1.013 and 0.962), the hop 0.025
through the residual alone; the twins' excess -0.032 and 0.000; the carry 0.988, and 0.918 through
the residual alone, so carried.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g63_edges.py``.
#   Takes --overwrite and --allow-dirty. It reads G4.1's runs after ``require`` of G4.1, G5.2, G5.3 and G6.1;
#   every dictionary comes from the Phase 5 cache (``_common.topk``). Beside its report it writes the
#   zoo runs' edge records, through ``cli.record``, under ``runs/phase6/gestures/<run>/``.
# * The trained edge is the record's: ``cli.record`` reads both sites itself, at the final checkpoint, and
#   the record cache answers when it already holds the key. ``"trained"`` is the record's results.
# * The twin's sites are read once per block and kept while the run's relations are measured.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G6.3"
REPORT_NAME = "g63_edges.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G6.1")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, g41 = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(g41)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.3 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["follows_min"]) and m["follows_min"] >= bars["edge_follows"]
          and n(m["null_follows_max"]) and m["null_follows_max"] <= bars["edge_null_follows"]
          and n(m["output_follows_min"]) and m["output_follows_min"] >= bars["output_follows"]
          and n(m["kept_max"]) and m["kept_max"] <= bars["kept"]
          and n(m["null_kept_min"]) and m["null_kept_min"] >= bars["null_kept"]
          and n(m["residual_alone_max"]) and m["residual_alone_max"] <= bars["residual_alone"]
          and m["labels_computed"] is True
          and n(m["twin_excess_max"]) and m["twin_excess_max"] <= bars["twin_excess"])
    return "pass" if ok else "fail"


def measure(g41: dict[str, Any]) -> dict[str, Any]:
    from nablaframe.gestures import cli, edge
    from nablaframe.gestures.records import GestureRequest

    edges: dict[str, Any] = {}
    run_ids, steps = {}, {}
    for run in _common.ZOO:
        m = _common.zoo_model(run, g41)
        run_ids[run], steps[run] = m.run_id, m.step
        _common.say(f"{GATE_ID}: {run} run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows")
        blocks = sorted({place.block for R in m.declaration.relations for place in (R.up, R.down)})
        trained = {l: _common.topk(m, l) for l in blocks}
        twins = {l: _common.topk(m, l, twin=True) for l in blocks}
        ctx = _common.context(m, list(trained.values()))
        twin_readings: dict[int, Any] = {}

        def twin_reading(l: int) -> Any:
            if l not in twin_readings:
                twin_readings[l] = _common.read(m, m.twin, twins[l], f"{GATE_ID}: {run} twin, reading block {l}")
            return twin_readings[l]

        for R in m.declaration.relations:
            key = f"{run} {R.name}"
            content = _common.core.content_hash
            request = GestureRequest("edge", m.run_id, m.step,
                                     {"relation": R.name, "up": content(trained[R.up.block]), "up_site": 0,
                                      "down": content(trained[R.down.block]), "down_site": 0},
                                     _common.INSTANCES["edge"], _common.SEEDS["edge"])
            with _common.ticking(f"{GATE_ID}: {key}, the trained model's edge record", write=_common.say):
                record = cli.record(Path(_common.RECORD_DIR) / run, request, ctx)
            up, down = twin_reading(R.up.block), twin_reading(R.down.block)
            with _common.ticking(f"{GATE_ID}: {key}, the twin's edge", write=_common.say):
                twin = edge.edge(up, down, R.name, instances=_common.INSTANCES["edge"], seed=_common.SEEDS["edge"])
            edges[key] = {"judged": bool(R.judged), "record": record.key, "trained": _common.results_of(record),
                          "twin": twin}
            for who in ("trained", "twin"):
                e = edges[key][who]
                alone = e["residual_alone"]
                _common.say(f"{GATE_ID}: {key}, {who}: {e['instances']} instances, follows {_common.show(e['follows'])}"
                            f" (null {_common.show(e['null_follows'])}), kept {_common.show(e['kept'])} (null"
                            f" {_common.show(e['null_kept'])}), through the residual alone"
                            f" {_common.show(None if alone is None else alone['follows'])}, {e['label']}")

    judged = [entry for entry in edges.values() if entry["judged"]]
    trained = [entry["trained"] for entry in judged]
    alone = [e["residual_alone"]["follows"] for e in trained if e["residual_alone"] is not None]
    out = {
        "edges": edges,
        "follows_min": _common.least([e["follows"] for e in trained]),
        "null_follows_max": _common.greatest([e["null_follows"] for e in trained]),
        "output_follows_min": _common.least([None if e["output"] is None else e["output"]["follows"] for e in trained]),
        "kept_max": _common.greatest([e["kept"] for e in trained]),
        "null_kept_min": _common.least([e["null_kept"] for e in trained]),
        "residual_alone_max": _common.greatest(alone),
        "labels_computed": bool(trained) and all(e["label"] == edge.COMPUTED for e in trained),
        "twin_excess_max": _common.greatest([_common.twin_excess(entry["twin"]["follows"], entry["twin"]["null_follows"])
                                             for entry in judged]),
        "bars": _common.BARS,
        "instances": _common.INSTANCES["edge"],
        "seed": _common.SEEDS["edge"],
        "run_ids": run_ids,
        "steps": steps,
    }
    _common.say(f"{GATE_ID}: follows at least {_common.show(out['follows_min'])}, null at most"
                f" {_common.show(out['null_follows_max'])}; output at least {_common.show(out['output_follows_min'])};"
                f" kept at most {_common.show(out['kept_max'])}, null at least {_common.show(out['null_kept_min'])};"
                f" through the residual alone at most {_common.show(out['residual_alone_max'])}; computed"
                f" {out['labels_computed']}; the twins' excess at most {_common.show(out['twin_excess_max'])}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
