"""G5.3: dictionaries on the two-hop model, against its twin's (judged; D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase5/g53_two_hop.json``. Run in nf-jax, on the Mac, after G4.1, G4.3 and G5.1
pass.

On G4.1's two-hop run at its final checkpoint and on its twin (``twin_params``), the model reading
each window's first 26 tokens. Every dictionary comes through ``cached``, as in G5.2, trained on
``zoo_windows(task, "two_hop", "train")``:

* on both models: ``topk`` at every block, the ``staircase``, and a ``transcoder`` at every block
* on the trained model only: ``topk`` at every block at each of ``ROBUST_SEEDS``

Sites are named, and dictionaries keyed, as in G5.2. Everything below is measured on ``zoo_windows(task, "two_hop",
"held_out")`` unless it says otherwise. The best swap block ``B`` is ``best_swap["block"]`` from
G4.3's report, which must exist, pass ``validate_report`` and say ``pass``; it is recorded as
``best_swap_block``.

* **Meters (judged):** as G5.2's, at ``READOUT["two_hop"]``: ``meters``, ``fvu_max`` and
  ``recovered_min``. The MLPs whose ablation costs less than ``BARS["ablation_cost"]`` have
  ``recovered`` None, and their change is recorded.
* **Faithful code (judged):** on G4.3's windows, ``eval_windows`` from Phase 4, with
  ``aligned_sources(task, windows, numpy.random.default_rng([SOURCE_SEED, 0]))``, at every block's
  ``hook_resid_post`` for the ``topk`` there and the ``staircase``'s site: ``patch_predictions`` at
  the query, positions 24 and 25, read at 25, once from the sources and once from the controls.
  ``patch_outcomes`` gives each pass's swap and transplant rates from the sources; its
  ``control_unchanged`` is how often the control patch's prediction is the unpatched pass's own.
  The ``"latents"`` pass moves, at each position, the latents ``assigned`` to ``b`` there, from
  ``f1_table``s of ``a``, ``b`` and ``c`` at that position. ``faithful``: ``{kind: {block:
  {pass: {"swap", "transplant", "control_unchanged"}, "assigned": {position: count}}}}``,
  ``"assigned"`` counting the latents the ``"latents"`` pass moves at each position.
  ``faithful_ratio_min``: the least ``faithful(residual, code)["ratio"]`` over every kind, block
  and rate (swap and transplant) where it applies, or None when it applies nowhere.
  ``control_unchanged_min``: the least ``"code"`` pass's ``control_unchanged`` over every kind and
  block. The transcoders aren't patched: two-hop's MLP outputs carry no planted variable at the
  query.
* **Selective latents (judged):** at position 25, ``b`` after block ``B`` and ``c`` after the last
  block, in the ``topk`` and the ``staircase`` there, and ``b`` in the code of block ``B``'s
  ``transcoder``. ``selectivity``, ``selective_f1_min`` and ``selective_margin_min`` as G5.2's.
* **Recorded:** ``recovery``: at every site and both positions 24 and 25, for each of ``a``, ``b``
  and ``c``, as G5.2's, which holds the other variables the spec lists. ``robustness`` and
  ``train_seconds`` as G5.2's, the variables being those judged at each block, or ``b`` and ``c``
  where neither is.

Measurements: REQUIRED_MEASUREMENTS["G5.3"]. Pass needs ``fvu_max <= BARS["fvu"]``,
``recovered_min >= BARS["recovered"]``, ``faithful_ratio_min >= BARS["faithful_ratio"]``,
``control_unchanged_min >= BARS["control_unchanged"]``, ``selective_f1_min >=
BARS["selective_f1"]`` and ``selective_margin_min >= 0``. The residual patch's rates are G4.3's
``swap_by_site``, ``transplant_by_site`` and ``control_unchanged_by_site`` at ``"query"``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g53_two_hop.py``. Takes --overwrite and --allow-dirty. It reads G4.1's two-hop run,
#   after ``require_g41``, ``require_g43``, ``require_g51`` and ``zoo_run``'s checks; it writes
#   dictionaries to the cache and nothing else outside its report.
# * Dictionaries, activations and meters as G5.2's. The query's positions are those ``variables``
#   gives, 24 and 25, and the answer is read at ``READOUT["two_hop"]``, 25.
# * The faithful code: per kind and block, two ``patch_predictions`` (sources, controls), the
#   sources' and controls' first 26 tokens patched at both query positions. ``patch_outcomes`` reads
#   the full windows. ``control_unchanged`` compares with the unpatched pass's argmax at 25, from
#   ``run_intervened`` with an empty spec. Blocks are keyed as strings, ``faithful[kind]["2"]``,
#   as JSON keeps them.
# * ``recovery``: ``{site: {"<position>": {variable: {"trained"|"twin": {"mean_best_f1",
#   "probe_activation", "probe_code"}}}}}``. ``robustness`` and ``train_seconds`` as G5.2's, at
#   position 25.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p4 = _common.p1, _common.p4

GATE_ID = "G5.3"
REPORT_NAME = "g53_two_hop.json"
NAME = "two_hop"
PASSES = ("residual", "code", "latents")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    g43 = _common.require_g43()
    _common.require_g51()
    m = _common.zoo_model(NAME, g41)
    best = (g43["measurements"].get("best_swap") or {}).get("block")
    if not isinstance(best, int) or isinstance(best, bool) or not 0 <= best < m.n_layers:
        raise SystemExit(f"G4.3's best swap block is {best!r}, not a block of the two-hop model. No report written.")
    measurements = measure(m, best)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G5.3 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["fvu_max"]) and m["fvu_max"] <= bars["fvu"]
          and n(m["recovered_min"]) and m["recovered_min"] >= bars["recovered"]
          and n(m["faithful_ratio_min"]) and m["faithful_ratio_min"] >= bars["faithful_ratio"]
          and n(m["control_unchanged_min"]) and m["control_unchanged_min"] >= bars["control_unchanged"]
          and n(m["selective_f1_min"]) and m["selective_f1_min"] >= bars["selective_f1"]
          and n(m["selective_margin_min"]) and m["selective_margin_min"] >= 0)
    return "pass" if ok else "fail"


def judged(block: int, best: int, last: int) -> list[str]:
    """The variables judged after ``block``: ``b`` at the best swap block, ``c`` at the last, both elsewhere."""
    out = (["b"] if block == best else []) + (["c"] if block == last else [])
    return out or ["b", "c"]


def measure(m: Any, best: int) -> dict[str, Any]:
    from nablaframe.dictionaries import measure as dm

    task, layers, last = m.task, range(m.n_layers), m.n_layers - 1
    print(f"{GATE_ID}: two-hop run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows; G4.3's best"
          f" swap block {best}", flush=True)
    shipped = [("topk", l) for l in layers] + [("staircase", None)] + [("transcoder", l) for l in layers]
    trained = {(kind, layer): _common.dictionary(m, kind, layer) for kind, layer in shipped}
    twins = {(kind, layer): _common.dictionary(m, kind, layer, twin=True) for kind, layer in shipped}
    robust = {(seed, l): _common.dictionary(m, "topk", l, seed=seed) for seed in _common.ROBUST_SEEDS for l in layers}
    sites = [(kind, layer, i) for kind, layer in shipped for i in range(len(trained[kind, layer].sites))]
    taps = [tap for d in trained.values() for site in d.sites for tap in site]
    acts = _common.activations(m, m.params, m.held_tokens, taps, f"{GATE_ID}: held-out activations")
    twin_acts = _common.activations(m, m.twin, m.held_tokens, taps, f"{GATE_ID}: the twin's held-out activations")

    def name(kind: str, layer: int | None, i: int) -> str:
        return _common.site_name(kind, trained[kind, layer].sites[i][1])

    def block(kind: str, layer: int | None, i: int) -> int:
        return i if kind == "staircase" else layer

    with _common.ticking(f"{GATE_ID}: meters", write=print):
        meter = _common.Meters(m, acts)
        meters = {}
        for kind, layer, i in sites:
            entry = meters[name(kind, layer, i)] = meter.site(trained[kind, layer], i, (twins[kind, layer], twin_acts))
            print(f"{GATE_ID}: {name(kind, layer, i)}: FVU {entry['fvu']:.4f} (twin {entry['twin_fvu']:.4f}),"
                  f" dead {entry['dead_share']:.3f}; loss {entry['clean_loss']:.4f}, spliced {entry['spliced_loss']:.4f},"
                  f" ablated {entry['ablated_loss']:.4f}, cost {entry['cost']:.4f}, recovered"
                  f" {_show(entry['recovered'])}", flush=True)

    by_position, n_values = _common.variables(task, NAME, m.held_out)
    query = sorted(by_position)
    read = _common.READOUT[NAME][0]
    with _common.ticking(f"{GATE_ID}: selectivity and readability", write=print):
        tables: dict[str, dict[int, Any]] = {}
        twin_tables: dict[str, dict[int, Any]] = {}
        recovery: dict[str, Any] = {}
        for kind, layer, i in sites:
            key = name(kind, layer, i)
            tables[key], twin_tables[key], recovery[key] = {}, {}, {}
            for p in query:
                values = by_position[p]
                tables[key][p] = _common.f1_tables(trained[kind, layer], i, acts, [p], values, n_values)
                twin_tables[key][p] = _common.f1_tables(twins[kind, layer], i, twin_acts, [p], values, n_values)
                mine = _common.readability(trained[kind, layer], i, acts, [p], values, n_values, tables[key][p])
                theirs = _common.readability(twins[kind, layer], i, twin_acts, [p], values, n_values,
                                             twin_tables[key][p])
                recovery[key][str(p)] = {v: {"trained": mine[v], "twin": theirs[v]} for v in values}

    selected = [("topk", best, "b"), ("staircase", best, "b"), ("topk", last, "c"), ("staircase", last, "c"),
                ("transcoder", best, "b")]
    selectivity = {}
    for kind, at, variable in selected:
        (layer, i), = [(layer, i) for k, layer, i in sites if k == kind and block(k, layer, i) == at]
        key = name(kind, layer, i)
        selectivity[f"{key} {variable}"] = _common.selectivity_entry(tables[key][read][variable],
                                                                     twin_tables[key][read][variable])
    for key, entry in selectivity.items():
        print(f"{GATE_ID}: selective latents, {key}: {entry['trained']:.3f} (twin {entry['twin']:.3f})", flush=True)

    faithful = faithful_code(m, trained, tables, query, read)

    robustness = {}
    for seed in _common.ROBUST_SEEDS:
        robustness[str(seed)] = {}
        for l in layers:
            d = robust[seed, l]
            reads, writes = d.sites[0]
            variables = judged(l, best, last)
            table = _common.f1_tables(d, 0, acts, [read], {v: by_position[read][v] for v in variables}, n_values)
            robustness[str(seed)][_common.site_name("topk", writes)] = {
                "fvu": dm.fvu(d, 0, _common.rows(acts[reads]), _common.rows(acts[writes])),
                "mean_best_f1": {v: dm.mean_best_f1(table[v]) for v in variables},
            }

    ratios = [_common.faithful(entry["residual"][rate], entry["code"][rate])["ratio"]
              for by_block in faithful.values() for entry in by_block.values() for rate in ("swap", "transplant")]
    ratios = [r for r in ratios if r is not None]
    out = {
        "meters": meters,
        **_common.meter_summary(meters),
        "faithful": faithful,
        "faithful_ratio_min": min(ratios) if ratios else None,
        "control_unchanged_min": min(entry["code"]["control_unchanged"] for by_block in faithful.values()
                                     for entry in by_block.values()),
        "selectivity": selectivity,
        "selective_f1_min": min(entry["trained"] for entry in selectivity.values()),
        "selective_margin_min": min(entry["margin"] for entry in selectivity.values()),
        "best_swap_block": best,
        "recovery": recovery,
        "robustness": robustness,
        "train_seconds": m.train_seconds,
        "run_id": m.run_id,
        "step": m.step,
        "twin_seed": m.twin_seed,
        "bars": _common.BARS,
        "readout_positions": m.readout,
        "query_positions": query,
        "held_out_windows": int(len(m.held_out)),
        "held_out_sha256": _batch_hash(m.held_out),
        "source_seed": p4.SOURCE_SEED,
    }
    print(f"{GATE_ID}: FVU at most {out['fvu_max']:.4f}; recovered at least {_show(out['recovered_min'])};"
          f" faithful ratio at least {_show(out['faithful_ratio_min'])}; control unchanged at least"
          f" {out['control_unchanged_min']:.4f}; selective F1 at least {out['selective_f1_min']:.3f}, margin at least"
          f" {out['selective_margin_min']:.3f}", flush=True)
    return out


def faithful_code(m: Any, trained: dict[Any, Any], tables: dict[str, dict[int, Any]], query: list[int],
                  read: int) -> dict[str, Any]:
    """G4.3's aligned patch at the query after every block, through the residual, the code and ``b``'s latents."""
    from nablaframe.dictionaries import measure
    from nablaframe.zoo.two_hop import aligned_sources, patch_outcomes

    task = m.task
    windows = p4.eval_windows(task)
    sources, controls, counterparts = aligned_sources(task, windows, np.random.default_rng([p4.SOURCE_SEED, 0]))
    tokens = m.tokens(windows)
    clean, _ = _common.p3.unintervened(m.model, m.params, tokens, n_layers=m.n_layers)
    answers = clean[:, read].argmax(axis=-1)
    faithful: dict[str, Any] = {}
    progress = p1.Progress("faithful-code patches", 2 * 2 * m.n_layers)
    with _common.ticking(f"{GATE_ID}: faithful code", write=print):
        for kind in ("topk", "staircase"):
            faithful[kind] = {}
            for l in range(m.n_layers):
                tap = f"blocks.{l}.hook_resid_post"
                d = trained[kind, l if kind == "topk" else None]
                site_tables = tables[_common.site_name(kind, tap)]
                moved = {p: measure.assigned(site_tables[p], "b") for p in query}
                predicted = {}
                for label, batch in (("source", sources), ("control", controls)):
                    predicted[label] = measure.patch_predictions(m.model, m.params, tokens, m.tokens(batch), tap=tap,
                                                                 positions=query, read=read, n_layers=m.n_layers,
                                                                 dictionary=d, latents=moved)
                    progress.advance()
                entry: dict[str, Any] = {}
                for name in PASSES:
                    outcomes = patch_outcomes(predicted["source"][name], windows, sources, counterparts)
                    entry[name] = {"swap": outcomes["swap_rate"], "transplant": outcomes["transplant_rate"],
                                   "control_unchanged": float(np.mean(predicted["control"][name] == answers))}
                entry["assigned"] = {str(p): len(moved[p]) for p in query}
                faithful[kind][str(l)] = entry
                print(f"{GATE_ID}: faithful code, {kind} after block {l}: "
                      + "; ".join(f"{name} swap {entry[name]['swap']:.3f}, transplant {entry[name]['transplant']:.3f},"
                                  f" control unchanged {entry[name]['control_unchanged']:.3f}" for name in PASSES)
                      + f"; {', '.join(f'{n} at {p}' for p, n in entry['assigned'].items())} latents moved", flush=True)
    return faithful


def _show(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
