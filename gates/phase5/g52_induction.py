"""G5.2: dictionaries on the induction model, against its twin's (judged; D41).

Spec: docs/specs/phase-5.md, D41 and "Discrimination tests on the zoo"; helpers in ``_common.py``.
Report: ``reports/phase5/g52_induction.json``. Run in nf-jax, on the Mac, after G4.1 and G5.1 pass.

On G4.1's induction run at its final checkpoint and on its twin (``twin_params``), the model
reading each window's first 63 tokens. Every dictionary comes through ``cached``, under
``dictionary_key`` with the checkpoint's state hash or ``twin_state``, trained with ``settings`` on
``zoo_windows(task, "induction", "train")``:

* on both models: ``topk`` at blocks 0 and 1, the ``staircase``, and a ``transcoder`` at blocks 0
  and 1
* on the trained model only: ``topk`` at blocks 0 and 1 at each of ``ROBUST_SEEDS``, each on its
  own training windows

Each site is named ``"<kind> <output tap>"``. A dictionary's ``tokens``, for its key and for its training
alike, are what the model reads of its training windows, ``p4.tokens(task, windows)``; G5.6 finds each
shipped dictionary by recomputing that key. Everything below is measured on
``zoo_windows(task, "induction", "held_out")``.

* **Meters (judged):** per site, ``fvu`` and ``dead_share`` over every position, on both models.
  On the trained model, ``readout_loss`` at ``READOUT["induction"]`` of the unintervened pass, of
  ``spliced_logits`` and of ``ablated_logits`` at the site's output tap, and ``loss_recovered``.
  ``meters``: ``{site: {"fvu", "dead_share", "twin_fvu", "twin_dead_share", "clean_loss",
  "spliced_loss", "ablated_loss", "cost", "recovered"}}``; ``fvu_max`` over the trained model's
  sites; ``recovered_min`` over every site where ``recovered`` isn't None, or None when none is.
* **Faithful code (judged):** on G4.1's run, ``induction_patch_windows`` (G4.2's windows), for
  each ``j`` in ``PATCH_TARGETS``, ``patch_predictions`` at position ``j``, read at ``j +
  PATCH_LAG``, at block 0's ``hook_resid_post`` for the ``topk`` and the ``staircase`` (its site
  0), and at block 0's ``hook_mlp_out`` for the transcoder:

  - **following:** from ``induction_sources(task, windows, j, "follow")``, how often the
    prediction is the source's token at ``j``
  - **breaking:** from ``induction_sources(task, windows, j, "break")``, how often the prediction
    is the unpatched pass's own

  For each, the ``"residual"``, ``"code"`` and ``"latents"`` passes' rates, one per target. The
  ``"latents"`` pass moves the latents ``assigned`` at the site to ``cur`` for following and to
  ``prev`` for breaking, from ``f1_table``s of every variable at the mechanism positions.
  ``faithful``: ``{kind: {"follow": {pass: [rate per target]}, "break": {...}, "assigned":
  {"cur": count, "prev": count}}}``, ``"assigned"`` counting the latents the ``"latents"`` pass
  moves. ``faithful_ratio_min``: the least ``faithful(residual,
  code)["ratio"]`` over every kind and target of following where it applies, or None when it
  applies nowhere. ``break_excess_max``: the greatest ``code - residual`` over every kind and
  target of breaking.
* **Selective latents (judged):** at the mechanism positions, ``mean_best_f1`` of
  ``f1_table`` for ``cur`` at block 0 and ``next`` at block 1: in the ``topk`` at that block, the
  ``staircase`` at that site, and the ``transcoder`` at that block, whose code reads its MLP's
  input. ``selectivity``: ``{"<site> <variable>": {"trained", "twin", "margin"}}``, the margin
  being ``selective_margin``; ``selective_f1_min``, the least trained value; and
  ``selective_margin_min``, the least margin.
* **Recorded:** ``recovery``: at every site, for each of ``cur``, ``prev`` and ``next``, the
  trained and twin ``mean_best_f1`` and ``probe_accuracy`` from the site's input activation and
  from its code. ``robustness``: for each of ``ROBUST_SEEDS``, each ``topk``'s ``fvu`` and its
  ``mean_best_f1`` for the variable judged at its block. ``train_seconds``: per dictionary, None
  when it came from the cache.

Measurements: REQUIRED_MEASUREMENTS["G5.2"]. Pass needs ``fvu_max <= BARS["fvu"]``,
``recovered_min >= BARS["recovered"]``, ``faithful_ratio_min >= BARS["faithful_ratio"]``,
``break_excess_max <= BARS["break_excess"]``, ``selective_f1_min >= BARS["selective_f1"]`` and
``selective_margin_min >= 0``. The residual patch's following rates at block 0 are G4.2's
``patch_followed_by_tap`` for ``blocks.0.hook_resid_post``: the same windows, sources and Mac.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g52_induction.py``. Takes --overwrite and --allow-dirty. It reads G4.1's induction
#   run, after ``require_g41``, ``require_g51`` and ``zoo_run``'s checks; it writes dictionaries to
#   the cache and nothing else outside its report.
# * Every dictionary comes from ``_common.dictionary``: ``cached`` under the key of the model's state
#   hash (``twin_state`` for the twin), its sites, ``settings`` and the tokens the model reads of its
#   training windows.
# * The held-out activations are collected once per model, every tap of every site, through
#   ``measure.activations``. The meters' passes: the unintervened one through ``run_intervened``,
#   ``spliced_logits`` and ``ablated_logits`` from ``measure``, each ablation once per tap.
# * Codes and variables at the mechanism positions are rows in window-major order, as
#   ``variables`` gives the values. ``f1_table``s cover ``cur``, ``prev`` and ``next`` in that order,
#   so ``assigned`` gives a tie to the first.
# * ``recovery``: ``{site: {variable: {"trained"|"twin": {"mean_best_f1", "probe_activation",
#   "probe_code"}}}}``. ``robustness``: ``{"<seed>": {site: {"fvu", "mean_best_f1": {variable:
#   value}}}}``. ``train_seconds``: ``{"<trained|twin|seed n> <dictionary>": seconds or null}``, a
#   dictionary named ``"<kind> <output tap>"``, or ``"staircase"``.
# * The faithful code: one ``patch_predictions`` per kind, target and test, its sources' first 63
#   tokens patched at ``j``. Breaking compares with the unpatched pass's argmax at ``j +
#   PATCH_LAG``, from ``run_intervened`` with an empty spec.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p4 = _common.p1, _common.p4

GATE_ID = "G5.2"
REPORT_NAME = "g52_induction.json"
NAME = "induction"
SHIPPED = (("topk", 0), ("topk", 1), ("staircase", None), ("transcoder", 0), ("transcoder", 1))
JUDGED = {0: "cur", 1: "next"}  # the variable judged after each block
PATCHED = {"topk": ("topk", 0, "blocks.0.hook_resid_post"), "staircase": ("staircase", None, "blocks.0.hook_resid_post"),
           "transcoder": ("transcoder", 0, "blocks.0.hook_mlp_out")}
MOVED = {"follow": "cur", "break": "prev"}  # the variable whose latents the "latents" pass moves


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    _common.require_g51()
    m = _common.zoo_model(NAME, g41)
    measurements = measure(m)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G5.2 rule holds."""
    bars, n = _common.BARS, _common.number
    ok = (n(m["fvu_max"]) and m["fvu_max"] <= bars["fvu"]
          and n(m["recovered_min"]) and m["recovered_min"] >= bars["recovered"]
          and n(m["faithful_ratio_min"]) and m["faithful_ratio_min"] >= bars["faithful_ratio"]
          and n(m["break_excess_max"]) and m["break_excess_max"] <= bars["break_excess"]
          and n(m["selective_f1_min"]) and m["selective_f1_min"] >= bars["selective_f1"]
          and n(m["selective_margin_min"]) and m["selective_margin_min"] >= 0)
    return "pass" if ok else "fail"


def measure(m: Any) -> dict[str, Any]:
    from nablaframe.dictionaries import measure as dm

    task = m.task
    print(f"{GATE_ID}: induction run {m.run_id}, final step {m.step:,}; {len(m.held_out):,} held-out windows", flush=True)
    trained = {(kind, layer): _common.dictionary(m, kind, layer) for kind, layer in SHIPPED}
    twins = {(kind, layer): _common.dictionary(m, kind, layer, twin=True) for kind, layer in SHIPPED}
    robust = {(seed, layer): _common.dictionary(m, "topk", layer, seed=seed)
              for seed in _common.ROBUST_SEEDS for layer in JUDGED}
    sites = [(kind, layer, i) for kind, layer in SHIPPED for i in range(len(trained[kind, layer].sites))]
    taps = [tap for d in trained.values() for site in d.sites for tap in site]
    acts = _common.activations(m, m.params, m.held_tokens, taps, f"{GATE_ID}: held-out activations")
    twin_acts = _common.activations(m, m.twin, m.held_tokens, taps, f"{GATE_ID}: the twin's held-out activations")

    def name(kind: str, layer: int | None, i: int) -> str:
        return _common.site_name(kind, trained[kind, layer].sites[i][1])

    with _common.ticking(f"{GATE_ID}: meters", write=print):
        meter = _common.Meters(m, acts)
        meters = {}
        for kind, layer, i in sites:
            meters[name(kind, layer, i)] = meter.site(trained[kind, layer], i, (twins[kind, layer], twin_acts))
            entry = meters[name(kind, layer, i)]
            print(f"{GATE_ID}: {name(kind, layer, i)}: FVU {entry['fvu']:.4f} (twin {entry['twin_fvu']:.4f}),"
                  f" dead {entry['dead_share']:.3f}; loss {entry['clean_loss']:.4f}, spliced {entry['spliced_loss']:.4f},"
                  f" ablated {entry['ablated_loss']:.4f}, recovered {_show(entry['recovered'])}", flush=True)

    positions = list(task.mechanism_positions)
    by_position, n_values = _common.variables(task, NAME, m.held_out)
    values = by_position["mechanism"]
    with _common.ticking(f"{GATE_ID}: selectivity and readability", write=print):
        tables, twin_tables, recovery = {}, {}, {}
        for kind, layer, i in sites:
            key = name(kind, layer, i)
            tables[key] = _common.f1_tables(trained[kind, layer], i, acts, positions, values, n_values)
            twin_tables[key] = _common.f1_tables(twins[kind, layer], i, twin_acts, positions, values, n_values)
            mine = _common.readability(trained[kind, layer], i, acts, positions, values, n_values, tables[key])
            theirs = _common.readability(twins[kind, layer], i, twin_acts, positions, values, n_values, twin_tables[key])
            recovery[key] = {v: {"trained": mine[v], "twin": theirs[v]} for v in values}
    selectivity = {}
    for block, variable in JUDGED.items():
        for kind, layer, i in sites:
            if (layer if kind != "staircase" else i) == block:
                key = name(kind, layer, i)
                selectivity[f"{key} {variable}"] = _common.selectivity_entry(tables[key][variable],
                                                                             twin_tables[key][variable])
    for key, entry in selectivity.items():
        print(f"{GATE_ID}: selective latents, {key}: {entry['trained']:.3f} (twin {entry['twin']:.3f})", flush=True)

    faithful = faithful_code(m, trained, tables)
    robustness = {}
    for seed in _common.ROBUST_SEEDS:
        robustness[str(seed)] = {}
        for layer, variable in JUDGED.items():
            d = robust[seed, layer]
            reads, writes = d.sites[0]
            table = _common.f1_tables(d, 0, acts, positions, {variable: values[variable]}, n_values)
            robustness[str(seed)][_common.site_name("topk", writes)] = {
                "fvu": dm.fvu(d, 0, _common.rows(acts[reads]), _common.rows(acts[writes])),
                "mean_best_f1": {variable: dm.mean_best_f1(table[variable])},
            }

    ratios = [_common.faithful(res, code)["ratio"] for entry in faithful.values()
              for res, code in zip(entry["follow"]["residual"], entry["follow"]["code"])]
    ratios = [r for r in ratios if r is not None]
    excess = [code - res for entry in faithful.values()
              for res, code in zip(entry["break"]["residual"], entry["break"]["code"])]
    out = {
        "meters": meters,
        **_common.meter_summary(meters),
        "faithful": faithful,
        "faithful_ratio_min": min(ratios) if ratios else None,
        "break_excess_max": max(excess),
        "selectivity": selectivity,
        "selective_f1_min": min(entry["trained"] for entry in selectivity.values()),
        "selective_margin_min": min(entry["margin"] for entry in selectivity.values()),
        "recovery": recovery,
        "robustness": robustness,
        "train_seconds": m.train_seconds,
        "run_id": m.run_id,
        "step": m.step,
        "twin_seed": m.twin_seed,
        "bars": _common.BARS,
        "readout_positions": m.readout,
        "mechanism_positions": [positions[0], positions[-1]],
        "held_out_windows": int(len(m.held_out)),
        "held_out_sha256": _batch_hash(m.held_out),
        "patch_targets": list(p4.PATCH_TARGETS),
        "patch_lag": p4.PATCH_LAG,
    }
    print(f"{GATE_ID}: FVU at most {out['fvu_max']:.4f}; recovered at least {_show(out['recovered_min'])};"
          f" faithful ratio at least {_show(out['faithful_ratio_min'])}; breaking at most"
          f" {out['break_excess_max']:+.4f}; selective F1 at least {out['selective_f1_min']:.3f}, margin at least"
          f" {out['selective_margin_min']:.3f}", flush=True)
    return out


def faithful_code(m: Any, trained: dict[Any, Any], tables: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """G4.2's patch after block 0 (and at block 0's MLP output), through the residual, the code and the assigned latents."""
    from nablaframe.dictionaries import measure

    task = m.task
    windows = p4.induction_patch_windows(task)
    tokens = m.tokens(windows)
    clean, _ = _common.p3.unintervened(m.model, m.params, tokens, n_layers=m.n_layers)
    faithful: dict[str, Any] = {}
    progress = p1.Progress("faithful-code patches", len(PATCHED) * len(p4.PATCH_TARGETS) * len(MOVED))
    with _common.ticking(f"{GATE_ID}: faithful code", write=print):
        for kind, (dict_kind, layer, tap) in PATCHED.items():
            d = trained[dict_kind, layer]
            site_tables = tables[_common.site_name(dict_kind, tap)]
            moved = {test: measure.assigned(site_tables, variable) for test, variable in MOVED.items()}
            entry: dict[str, Any] = {test: {"residual": [], "code": [], "latents": []} for test in MOVED}
            for j in p4.PATCH_TARGETS:
                read = j + p4.PATCH_LAG
                for test in MOVED:
                    sources = _common.induction_sources(task, windows, j, test)
                    predicted = measure.patch_predictions(m.model, m.params, tokens, m.tokens(sources), tap=tap,
                                                          positions=[j], read=read, n_layers=m.n_layers, dictionary=d,
                                                          latents={j: moved[test]})
                    expected = sources[:, j] if test == "follow" else clean[:, read].argmax(axis=-1)
                    for name, values in predicted.items():
                        entry[test][name].append(float(np.mean(values == expected)))
                    progress.advance()
            entry["assigned"] = {variable: len(moved[test]) for test, variable in MOVED.items()}
            faithful[kind] = entry
            print(f"{GATE_ID}: faithful code, {kind} at {tap}: following residual {_rates(entry['follow']['residual'])},"
                  f" code {_rates(entry['follow']['code'])}, latents {_rates(entry['follow']['latents'])}; breaking"
                  f" residual {_rates(entry['break']['residual'])}, code {_rates(entry['break']['code'])}", flush=True)
    return faithful


def _show(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _rates(rates: list[float]) -> str:
    return "[" + ", ".join(f"{rate:.3f}" for rate in rates) + "]"


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
