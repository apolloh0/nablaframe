"""G5.5: every judged dictionary on the reference run, its meters and the meter's curves (recorded; D44).

Spec: docs/specs/phase-5.md, D44 and "The reference run"; helpers in ``_common.py``. Report:
``reports/phase5/g55_reference.json``. Run in nf-jax, on the Mac, after G5.1 passes, with G1.4's
run and the reference corpus in place.

On G1.4's run at its final checkpoint, read through Phase 1's helpers, the model reading each
window's first ``seq`` tokens. Every dictionary comes through ``cached``, under ``dictionary_key``
with that checkpoint's state hash, trained with ``settings("reference", ...)`` on
``reference_windows(train, validation, seq, "train")``: ``topk`` at every block, the
``staircase``, a ``transcoder`` at every block, and TopK pairs (the ``jsae`` at ``lam`` 0.0) across
blocks 1 and 2's MLPs. A dictionary's ``tokens``, for its key and for its training alike, are what the
model reads of the training windows, ``windows[:, :seq]``; G5.6 finds each shipped dictionary by
recomputing that key. Everything is measured on ``reference_windows(..., "held_out")``.

* ``meters``: as G5.2's, without a twin, the readout being every position. TinyStories has no
  ground truth, so nothing is judged.
* ``edge_pairs``: for the two TopK pairs, as G5.4's ``"trained"`` entries, the meter's tokens
  being the first ``METER_WINDOWS["reference"]`` held-out windows, each curve in full.
* ``train_seconds``: per dictionary, None when it came from the cache.

Measurements: REQUIRED_MEASUREMENTS["G5.5"]. Its status is always ``pass``: it records, not judges.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g55_reference.py``. Takes --overwrite and --allow-dirty. It reads G1.4's run through
#   ``p2.reference``, whose checks name every trace checkpoint's run, after ``require_g41`` and
#   ``require_g51``; it writes dictionaries to the cache and nothing else outside its report.
# * The checkpoint's state hash is its sidecar's ``state_sha256``, as for the zoo runs.
# * ``meters`` as G5.2's without the twin's two entries, the readout being every position the
#   model reads, 0 to ``seq - 1``, each predicting the next token of its held-out window.
# * ``edge_pairs``: ``{"reference mlp <l>": {"meter", "share", "scored_edges", "pair_kl",
#   "curve"}}``, each the TopK pair's reading as G5.4's ``"trained"`` entries hold it.
# * ``train_seconds`` as G5.2's, the TopK pairs named ``"jsae <output tap> lam 0"``.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G5.5"
REPORT_NAME = "g55_reference.json"
PAIR_BLOCKS = (1, 2)


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    _common.require_g41()
    _common.require_g51()
    m = _common.reference_model()
    if m.n_layers <= max(PAIR_BLOCKS):
        raise SystemExit(f"the reference model has {m.n_layers} blocks; the TopK pairs need blocks {PAIR_BLOCKS}."
                         " No report written.")
    measurements = measure(m)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """Always pass: G5.5 records, it doesn't judge (D44)."""
    return "pass"


def measure(m: Any) -> dict[str, Any]:
    layers = range(m.n_layers)
    print(f"{GATE_ID}: reference run {m.run_id}, final step {m.step:,}; {len(m.held_out)} held-out windows of"
          f" {m.config.seq} tokens", flush=True)
    shipped = [("topk", l) for l in layers] + [("staircase", None)] + [("transcoder", l) for l in layers]
    trained = {(kind, layer): _common.dictionary(m, kind, layer) for kind, layer in shipped}
    pairs = {l: _common.dictionary(m, "jsae", l) for l in PAIR_BLOCKS}
    taps = [tap for d in trained.values() for site in d.sites for tap in site]
    acts = _common.activations(m, m.params, m.held_tokens, taps, f"{GATE_ID}: held-out activations")
    meters = {}
    with _common.ticking(f"{GATE_ID}: meters", write=print):
        meter = _common.Meters(m, acts)
        for (kind, _), d in trained.items():
            for i, (_, writes) in enumerate(d.sites):
                name = _common.site_name(kind, writes)
                entry = meters[name] = meter.site(d, i)
                print(f"{GATE_ID}: {name}: FVU {entry['fvu']:.4f}, dead {entry['dead_share']:.3f}; loss"
                      f" {entry['clean_loss']:.4f}, spliced {entry['spliced_loss']:.4f}, ablated"
                      f" {entry['ablated_loss']:.4f}, recovered {_show(entry['recovered'])}", flush=True)
    tokens = m.held_tokens[:_common.METER_WINDOWS["reference"]]
    edge_pairs = {f"reference mlp {l}": _common.mlp_pair(m, m.params, tokens, pair, l, f"{GATE_ID}: reference mlp {l}")
                  for l, pair in pairs.items()}
    return {
        "meters": meters,
        "edge_pairs": edge_pairs,
        "train_seconds": m.train_seconds,
        "run_id": m.run_id,
        "step": m.step,
        "readout_positions": [m.readout[0], m.readout[-1]],
        "held_out_windows": int(len(m.held_out)),
        "held_out_sha256": _batch_hash(m.held_out),
        "meter_windows": _common.METER_WINDOWS["reference"],
        "pair_blocks": list(PAIR_BLOCKS),
    }


def _show(value: float | None) -> str:
    return "null" if value is None else f"{value:.4f}"


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
