"""G6.5: generalize-this for the judged names, across held-out slices, dictionary seeds and every checkpoint (judged; D52).

Spec: docs/specs/phase-6.md, D44, D52 and "Discrimination tests on the zoo"; helpers in
``_common.py``. Report: ``reports/phase6/g65_generalize.json``. Run in nf-jax, on the Mac, after
G4.1, G5.2, G5.3 and G6.1 pass.

For each name in ``GENERALIZE`` (two-hop's ``b`` after block 2 at ``query``, induction's
``token(p+1)`` after block 1 at ``mechanism``): ``cli.record(RECORD_DIR / run, request,
context)``, the request being ``{"dictionary", "site": 0, "class", "description"}`` with the
block's TopK SAE, ``INSTANCES["name"]`` instances and ``SEEDS["generalize"]``, at the final
checkpoint. The run's context reads every checkpoint of G4.1's run through the final
dictionaries (D44), and holds the TopK SAEs of ``ROBUST_SEEDS`` at that block as its seeds. The
record holds the slices, the seeds, every checkpoint's row and the envelope.

Beside each checkpoint, the task's residual patch, which the record doesn't hold, since it is task
code (D29), each pass through ``run_intervened``:

* two-hop: G4.3's patch at the query, position 25, of ``blocks.2.hook_resid_post``, on G4.3's
  windows and aligned sources (``p4.eval_windows``; ``aligned_sources`` with
  ``numpy.random.default_rng([SOURCE_SEED, 0])``), its ``swap_rate`` from ``patch_outcomes``
* induction: G4.2's patch at ``blocks.0.hook_resid_post``, on ``induction_patch_windows``: for each
  ``j`` in ``PATCH_TARGETS``, from ``new_token_sources(windows, j, numpy.random.default_rng(j),
  vocab)``, how often the argmax at ``j + PATCH_LAG`` is the source's token at ``j``; their mean

``names``: ``{"<run> <description> after block <l> at <class>": {"record", "results", "patch",
"checks"}}``: the record's key; its results; ``[{"step", "rate"}]`` for every checkpoint; and
``envelope_checks`` of the results' envelope over the run's checkpoint steps. Over both names:

* ``slice_f1_min`` and ``slice_follows_min``: the least of each name's ``slices["f1"]`` and
  ``slices["follows"]`` lower ends, a range of None counting as None
* ``seed_f1_min`` and ``seed_follows_min``: the least of every seed row's ``f1`` and ``follows``
* ``step0_outside`` and ``one_final_run``: whether each holds for both names
* ``start_offset_max``: the greatest ``start_offset``, None where either name's is

the least and greatest through ``least`` and ``greatest``.

Measurements: REQUIRED_MEASUREMENTS["G6.5"]. Pass needs ``slice_f1_min >= BARS["slice_f1"]``,
``slice_follows_min >= BARS["slice_follows"]``, ``seed_f1_min >= BARS["seed_f1"]``,
``seed_follows_min >= BARS["seed_follows"]``, ``step0_outside``, ``one_final_run`` and
``start_offset_max <= BARS["start_offset"]``. Round 3 read, for two-hop and induction: slices' F1
0.780 to 0.825 and 0.787 to 0.794, follows 0.902 to 0.945 and 0.953 to 1.000; seeds' F1 0.756 to
0.832 and 0.747 to 0.772, follows 0.920 to 0.996; each envelope one run to the end, from step 6,000
(the probe at 6,000) and from step 700 (the probe at 750); step 0 outside both.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g65_generalize.py``.
#   Takes --overwrite and --allow-dirty. It reads G4.1's runs after ``require`` of G4.1, G5.2, G5.3 and G6.1;
#   every dictionary comes from the Phase 5 cache (``_common.topk``). Beside its report it writes the zoo
#   runs' generalize records, through ``cli.record``, under ``runs/phase6/gestures/<run>/``.
# * The run's context holds the block's TopK SAE and, as its seeds, ``{seed: {its content hash: the seed's
#   TopK SAE's}}`` for each of ``ROBUST_SEEDS``. The record reads every checkpoint itself.
# * The two-hop patch is at the run's ``query`` class, position 25, and its answer the argmax there, ties
#   to the lower token, as G4.3's; the induction patch at each of ``PATCH_TARGETS``, read ``PATCH_LAG``
#   positions on, as G4.2's. Each checkpoint's patch is one ``run_intervened`` pass per source batch.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p3, p4 = _common.p1, _common.p3, _common.p4

GATE_ID = "G6.5"
REPORT_NAME = "g65_generalize.json"
NEEDS = ("G4.1", "G5.2", "G5.3", "G6.1")
PATCH_TAP = {"two_hop": "blocks.2.hook_resid_post", "induction": "blocks.0.hook_resid_post"}


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, g41 = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(g41)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.5 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["slice_f1_min"]) and m["slice_f1_min"] >= bars["slice_f1"]
          and n(m["slice_follows_min"]) and m["slice_follows_min"] >= bars["slice_follows"]
          and n(m["seed_f1_min"]) and m["seed_f1_min"] >= bars["seed_f1"]
          and n(m["seed_follows_min"]) and m["seed_follows_min"] >= bars["seed_follows"]
          and m["step0_outside"] is True and m["one_final_run"] is True
          and n(m["start_offset_max"]) and m["start_offset_max"] <= bars["start_offset"])
    return "pass" if ok else "fail"


def measure(g41: dict[str, Any]) -> dict[str, Any]:
    from nablaframe.gestures import cli
    from nablaframe.gestures.records import GestureRequest

    names: dict[str, Any] = {}
    run_ids = {}
    for run, (description, block, cls) in _common.GENERALIZE.items():
        m = _common.zoo_model(run, g41)
        run_ids[run] = m.run_id
        steps = [int(s) for s in m.run.steps]
        _common.say(f"{GATE_ID}: {run} run {m.run_id}, {len(steps)} checkpoints to step {m.step:,};"
                    f" {len(m.held_out):,} held-out windows")
        dictionary = _common.topk(m, block)
        seeds = {s: _common.topk(m, block, seed=s) for s in _common.p5.ROBUST_SEEDS}
        content = _common.core.content_hash
        ctx = _common.context(m, [dictionary, *seeds.values()],
                              {s: {content(dictionary): content(d)} for s, d in seeds.items()})
        request = GestureRequest("generalize", m.run_id, m.step,
                                 {"dictionary": content(dictionary), "site": 0, "class": cls,
                                  "description": description},
                                 _common.INSTANCES["name"], _common.SEEDS["generalize"])
        key = _common.place_name(run, description, block, cls)
        with _common.ticking(f"{GATE_ID}: {key}, the generalize record over {len(steps)} checkpoints",
                             write=_common.say):
            record = cli.record(Path(_common.RECORD_DIR) / run, request, ctx)
        results = _common.results_of(record)
        patch = patches(m, steps)
        checks = _common.envelope_checks(results["envelope"], steps)
        names[key] = {"record": record.key, "results": results, "patch": patch, "checks": checks}
        for row, rate in zip(results["checkpoints"], patch):
            _common.say(f"{GATE_ID}: {key}, step {row['step']:,}: F1 {_common.show(row['f1'])}, follows"
                        f" {_common.show(row['follows'])} (null {_common.show(row['null_follows'])}), probe"
                        f" {_common.show(row['probe'])}, patch {_common.show(rate['rate'])}, accuracy"
                        f" {_common.show(row['accuracy'])}, holds {row['holds']}")
        _common.say(f"{GATE_ID}: {key}: slices F1 {results['slices']['f1']}, follows {results['slices']['follows']};"
                    f" envelope {results['envelope']['runs']}, the probe reads from"
                    f" {results['envelope']['probe_reads_from']}; {checks}")

    entries = list(names.values())
    lower = [_low(e["results"]["slices"]["f1"]) for e in entries]
    out = {
        "names": names,
        "slice_f1_min": _common.least(lower),
        "slice_follows_min": _common.least([_low(e["results"]["slices"]["follows"]) for e in entries]),
        "seed_f1_min": _common.least([row["f1"] for e in entries for row in e["results"]["seeds"]]),
        "seed_follows_min": _common.least([row["follows"] for e in entries for row in e["results"]["seeds"]]),
        "step0_outside": bool(entries) and all(e["checks"]["step0_outside"] for e in entries),
        "one_final_run": bool(entries) and all(e["checks"]["one_final_run"] for e in entries),
        "start_offset_max": _common.greatest([e["checks"]["start_offset"] for e in entries]),
        "bars": _common.BARS,
        "instances": _common.INSTANCES["name"],
        "seed": _common.SEEDS["generalize"],
        "run_ids": run_ids,
    }
    _common.say(f"{GATE_ID}: slices' F1 at least {_common.show(out['slice_f1_min'])}, follows at least"
                f" {_common.show(out['slice_follows_min'])}; seeds' F1 at least {_common.show(out['seed_f1_min'])},"
                f" follows at least {_common.show(out['seed_follows_min'])}; step 0 outside {out['step0_outside']};"
                f" one run to the end {out['one_final_run']}; start offset at most {_common.show(out['start_offset_max'])}")
    return out


def _low(span: Any) -> Any:
    """A ``[least, greatest]`` range's lower end, None for a range of None."""
    return None if span is None else span[0]


def patches(m: Any, steps: list[int]) -> list[dict[str, Any]]:
    """The task's residual patch at every checkpoint, ``[{"step", "rate"}]``, each pass through ``run_intervened``."""
    from nablaframe.interventions.spec import InterventionSpec, edit

    task, tap = m.task, PATCH_TAP[m.name]
    objects = p3.Objects()
    if m.name == "two_hop":
        from nablaframe.zoo.two_hop import aligned_sources, patch_outcomes

        windows = p4.eval_windows(task)
        sources, _, counterparts = aligned_sources(task, windows, np.random.default_rng([p4.SOURCE_SEED, 0]))
        query = [int(p) for p in m.declaration.classes["query"]]
        source = objects.batch(p4.tokens(task, sources))
        specs = [(InterventionSpec((edit(tap, query, "patch", source=source),)), None)]

        def rate(logits: list[np.ndarray]) -> float:
            answers = np.argmax(logits[0][:, query[-1]], axis=-1)
            return float(patch_outcomes(answers, windows, sources, counterparts)["swap_rate"])
    else:
        from nablaframe.zoo.induction import new_token_sources

        windows = p4.induction_patch_windows(task)
        drawn = {j: new_token_sources(windows, j, np.random.default_rng(j), task.vocab_size) for j in p4.PATCH_TARGETS}
        specs = [(InterventionSpec((edit(tap, [j], "patch", source=objects.batch(p4.tokens(task, drawn[j]))),)), j)
                 for j in p4.PATCH_TARGETS]

        def rate(logits: list[np.ndarray]) -> float:
            return float(np.mean([np.mean(np.argmax(out[:, j + p4.PATCH_LAG], axis=-1) == drawn[j][:, j])
                                  for out, (_, j) in zip(logits, specs)]))
    tokens = p4.tokens(task, windows)
    rows = []
    progress = p1.Progress(f"{GATE_ID}: {m.name} patches", len(steps) * len(specs))
    with _common.ticking(f"{GATE_ID}: {m.name}: the residual patch at every checkpoint", write=_common.say):
        for step in steps:
            params = m.run.params_for(step)
            logits = []
            for spec, _ in specs:
                logits.append(p3.run(m.model, params, tokens, spec, n_layers=m.n_layers, resolve=objects.resolve)[0])
                progress.advance()
            rows.append({"step": int(step), "rate": rate(logits)})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
