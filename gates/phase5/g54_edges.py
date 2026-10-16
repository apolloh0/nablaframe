"""G5.4: the edge-sparsity meter on the zoo, against the twins (judged across the MLPs that matter; D42, D43).

Spec: docs/specs/phase-5.md, D42, D43 and "The edge-sparsity meter"; helpers in ``_common.py``.
Report: ``reports/phase5/g54_edges.json``. Run in nf-jax, on the Mac, after G4.1 and G5.1 pass.

On G4.1's runs at their final checkpoints and on their twins, every dictionary through
``cached`` as in G5.2 and G5.3, the meter's ``tokens`` being the first ``METER_WINDOWS`` of each
model's held-out windows.

* **Across every MLP:** a TopK pair, the ``jsae`` at ``lam`` 0.0 at that block, its sites 0 and 1
  the two sides, on both models, with ``train.mlp_weights`` of that block. The MLP's ``cost``:
  ``readout_loss`` at the model's ``READOUT`` of ``ablated_logits`` at the block's
  ``hook_mlp_out``, minus the unintervened pass's, on all its held-out windows. It is judged where
  ``cost >= BARS["ablation_cost"]``. For each model: ``mlp_edge_scores``, ``mlp_curve`` and
  ``meter``. ``edge_pairs``: ``{"<run> mlp <l>": {"cost", "judged", "trained": {"meter",
  "share", "scored_edges", "pair_kl", "curve"}, "twin": {...}, "ratio"}}``, ``share`` being
  ``meter / scored_edges`` (None with the meter) and ``ratio`` ``edge_ratio`` of the two.
  ``judged_mlps``: the judged keys. ``edge_ratio_max``: the greatest ``ratio`` over them, or None
  if any is None or none is judged.
* **Recorded, across every block:** on both models, ``block_edge_scores``, ``block_curve`` and
  ``meter`` for the ``topk`` pair at blocks ``l`` and ``l + 1`` and for the ``staircase``'s sites
  ``l`` and ``l + 1``: ``block_pairs``, ``{"<run> <kind> <l>-><l + 1>": {"trained": {...},
  "twin": {...}}}`` as above.
* **Recorded, the Jacobian SAE:** on the trained models, the ``jsae`` at each of ``LAMBDAS``
  across every MLP, with the ``lam`` 0.0 pair beside it: its ``meter`` as above; ``fvu`` of each
  side on the held-out windows; and ``jacobian_stats`` of ``mlp_jacobian`` on the codes of every
  position of the meter's windows. ``jsae``: ``{"<run> mlp <l>": {"<lam>": {"meter", "share",
  "scored_edges", "pair_kl", "fvu_in", "fvu_out", "jacobian"}}}``, ``"<lam>"`` written as ``"0"``,
  ``"1"`` and ``"4"``.

Measurements: REQUIRED_MEASUREMENTS["G5.4"]. Pass needs ``edge_ratio_max <= BARS["edge_ratio"]``.
Round 2 read, trained against twin: induction's MLP 0, 606 edges against 58,391; its MLP 1, 252
against 22,363; two-hop's MLP 0, 263 against 3,944 (D42).
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g54_edges.py``. Takes --overwrite and --allow-dirty. It reads G4.1's runs, after
#   ``require_g41``, ``require_g51`` and ``zoo_run``'s checks; it writes dictionaries to the cache
#   and nothing else outside its report.
# * Every dictionary comes from ``_common.dictionary``, as in G5.2 and G5.3, so the TopK SAEs and
#   Staircase SAEs those gates trained come from the cache. The edge meter's passes are
#   ``dictionaries/edges.py``'s; the unintervened and ablated passes go through ``run_intervened``.
# * Each MLP's ``cost`` is the trained model's. Its twin pair is the ``jsae`` at ``lam`` 0.0 trained
#   on the twin's activations, with the twin's MLP weights.
# * Each reading's ``curve`` is ``[[edges, KL], ...]``, as ``mlp_curve`` and ``block_curve`` give
#   it. The Jacobian SAE's entries leave out the curve; its ``"0"`` entry is the TopK pair's reading
#   in ``edge_pairs``. ``fvu_in`` and ``fvu_out`` are over every position of the held-out windows.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G5.4"
REPORT_NAME = "g54_edges.json"


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    _common.require_g51()
    models = {name: _common.zoo_model(name, g41) for name in _common.ZOO}
    measurements = measure(models)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G5.4 rule holds."""
    value = m["edge_ratio_max"]
    return "pass" if _common.number(value) and value <= _common.BARS["edge_ratio"] else "fail"


def measure(models: dict[str, Any]) -> dict[str, Any]:
    edge_pairs: dict[str, Any] = {}
    block_pairs: dict[str, Any] = {}
    jsae: dict[str, Any] = {}
    for name, m in models.items():
        print(f"{GATE_ID}: {name} run {m.run_id}, final step {m.step:,}", flush=True)
        pairs = mlp_pairs(m)
        edge_pairs |= pairs
        block_pairs |= blocks(m)
        jsae |= jacobian_saes(m, pairs)
    judged = [key for key, entry in edge_pairs.items() if entry["judged"]]
    ratios = [edge_pairs[key]["ratio"] for key in judged]
    ratio_max = None if not ratios or any(r is None for r in ratios) else max(ratios)
    print(f"{GATE_ID}: judged MLPs {judged or 'none'}; the greatest ratio {ratio_max}", flush=True)
    return {
        "edge_pairs": edge_pairs,
        "judged_mlps": judged,
        "edge_ratio_max": ratio_max,
        "block_pairs": block_pairs,
        "jsae": jsae,
        "run_ids": {name: m.run_id for name, m in models.items()},
        "steps": {name: m.step for name, m in models.items()},
        "twin_seed": next(iter(models.values())).twin_seed,
        "meter_windows": {name: _common.METER_WINDOWS[name] for name in models},
        "bars": _common.BARS,
        "lambdas": list(_common.LAMBDAS),
    }


def meter_tokens(m: Any) -> np.ndarray:
    """The meter's tokens: what the model reads of the first ``METER_WINDOWS`` held-out windows."""
    return m.held_tokens[:_common.METER_WINDOWS[m.name]]


def mlp_pairs(m: Any) -> dict[str, Any]:
    """A TopK pair across every MLP, trained against twin, judged where ablating the MLP costs 0.1 nats."""
    from nablaframe.dictionaries import measure

    tokens = meter_tokens(m)
    with _common.ticking(f"{GATE_ID}: {m.name}'s MLP ablations", write=print):
        clean = _common.readout(_common.p3.unintervened(m.model, m.params, m.held_tokens, n_layers=m.n_layers)[0], m)
        costs = [_common.readout(measure.ablated_logits(m.model, m.params, m.held_tokens, f"blocks.{l}.hook_mlp_out",
                                                        n_layers=m.n_layers), m) - clean for l in range(m.n_layers)]
    out = {}
    for l in range(m.n_layers):
        key = f"{m.name} mlp {l}"
        cost = float(costs[l])
        trained = _common.mlp_pair(m, m.params, tokens, _common.dictionary(m, "jsae", l), l, f"{GATE_ID}: {key}")
        twin = _common.mlp_pair(m, m.twin, tokens, _common.dictionary(m, "jsae", l, twin=True), l,
                                f"{GATE_ID}: {key}, the twin")
        ratio = _common.edge_ratio(trained["meter"], twin["meter"], twin["scored_edges"])
        out[key] = {"cost": cost, "judged": cost >= _common.BARS["ablation_cost"], "trained": trained, "twin": twin,
                    "ratio": ratio}
        print(f"{GATE_ID}: {key}: ablation costs {cost:.4f} nats ({'judged' if out[key]['judged'] else 'recorded'});"
              f" {trained['meter']} edges against the twin's {twin['meter']}, ratio {ratio}", flush=True)
    return out


def blocks(m: Any) -> dict[str, Any]:
    """TopK and Staircase pairs across every block, trained and twin, recorded."""
    out = {}
    tokens = meter_tokens(m)
    for kind in ("topk", "staircase"):
        if kind == "topk":
            sides = {who: [(_common.dictionary(m, "topk", l, twin=who == "twin"), 0) for l in range(m.n_layers)]
                     for who in ("trained", "twin")}
        else:
            sides = {who: [(staircase, l) for l in range(m.n_layers)]
                     for who, staircase in (("trained", _common.dictionary(m, "staircase")),
                                            ("twin", _common.dictionary(m, "staircase", twin=True)))}
        for l in range(m.n_layers - 1):
            key = f"{m.name} {kind} {l}->{l + 1}"
            out[key] = {who: _common.block_pair(m, m.params if who == "trained" else m.twin, tokens, sides[who][l],
                                                sides[who][l + 1], f"{GATE_ID}: {key}{', the twin' if who == 'twin' else ''}")
                        for who in ("trained", "twin")}
    return out


def jacobian_saes(m: Any, pairs: dict[str, Any]) -> dict[str, Any]:
    """The Jacobian SAE at each ``lam`` across every MLP of the trained model, beside the TopK pair at ``lam`` 0."""
    from nablaframe.dictionaries import edges, measure
    from nablaframe.dictionaries.train import mlp_weights

    taps = [tap for l in range(m.n_layers) for tap in (f"blocks.{l}.ln2.hook_normalized", f"blocks.{l}.hook_mlp_out")]
    acts = _common.activations(m, m.params, m.held_tokens, taps, f"{GATE_ID}: {m.name}'s held-out activations")
    windows = _common.METER_WINDOWS[m.name]
    out: dict[str, Any] = {}
    for l in range(m.n_layers):
        key = f"{m.name} mlp {l}"
        x_in, x_out = (acts[tap] for tap in (f"blocks.{l}.ln2.hook_normalized", f"blocks.{l}.hook_mlp_out"))
        out[key] = {}
        for lam in (0.0, *_common.LAMBDAS):
            pair = _common.dictionary(m, "jsae", l, lam=lam)
            if lam == 0.0:
                reading = dict(pairs[key]["trained"])
            else:
                reading = _common.mlp_pair(m, m.params, meter_tokens(m), pair, l, f"{GATE_ID}: {key}, lam {lam:g}")
            del reading["curve"]
            with _common.ticking(f"{GATE_ID}: {key}, lam {lam:g}: FVU and the Jacobian", write=print):
                codes = _common.codes(pair, 0, {pair.sites[0][0]: x_in[:windows]})
                jac, _, _ = edges.mlp_jacobian((pair, 0), (pair, 1), mlp_weights(m.params, l), codes)
                reading |= {
                    "fvu_in": measure.fvu(pair, 0, _common.rows(x_in), _common.rows(x_in)),
                    "fvu_out": measure.fvu(pair, 1, _common.rows(x_out), _common.rows(x_out)),
                    "jacobian": edges.jacobian_stats(jac),
                }
            out[key][f"{lam:g}"] = reading
            print(f"{GATE_ID}: {key}, lam {lam:g}: meter {reading['meter']}, FVU in {reading['fvu_in']:.4f}, out"
                  f" {reading['fvu_out']:.4f}; Jacobian: 90% of its mass in {reading['jacobian']['entries_for_90']:.1f}"
                  f" entries, {reading['jacobian']['share_above_0.01']:.3f} above 0.01", flush=True)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
