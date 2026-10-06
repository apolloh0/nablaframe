"""G3.3: the J-lens experiment, the first answer to D21's open question. Recorded, not judged.

At the final checkpoint, for each block before the last and each of ``SCALE_FACTORS``: at
``EXPERIMENT_POSITIONS`` seeded positions in ``fit_range``, each with a seeded token (never
``<unk>`` or ``<eos>``), steer block l's output at that position along the J-lens direction
and along the logit-lens direction, scaled by that factor times ``steering_scales``' mean
residual norm. Record the change in the token's logit at the steered position and its mean
change over ``later_positions``. Measurements: ``layers``, ``scale_factors``, and
``results_by_layer`` with, per layer and scale, both directions' mean same-position and
later-position changes and their ratio.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase3/g33_jlens_experiment.py``. Takes --overwrite and --allow-dirty. It reads
#   G1.1's data, G1.4's final trace checkpoint and G2.9's J's for that step, refusing to start
#   (exit 1, no report) unless G2.9's report passes and every J is in its cache, as G2.10
#   checks them. It writes nothing outside its report.
# * The batch is G1.6's probe. Every pass is ``run_intervened``: one unintervened pass (an
#   empty spec) recording each block's output, then one pass per layer, scale, direction and
#   position, steering with a single ``add`` edit at ``blocks.<l>.hook_resid_post``, at that
#   one position, in every probe sequence at once.
# * Positions and tokens: ``default_rng(EXPERIMENT_SEED)`` (3; the probe, D17's fit sample and
#   D22's reference batch use 0, 1 and 2) draws ``EXPERIMENT_POSITIONS`` distinct positions
#   from ``fit_range(seq)``, sorted, then one token per position, uniformly from the
#   vocabulary without ``<unk>`` and ``<eos>``. The same positions and tokens serve every
#   layer and scale.
# * Directions: ``jlens_direction(J_l, wte, t)`` with G2.9's J for layer l, and
#   ``logit_lens_direction(wte, t)``, with the final step's ``wte``; both unit float32.
# * Scales: ``steering_scales`` of block l's output on the probe, from the unintervened pass:
#   each factor times the mean, over every sequence and position, of its L2 norm. The scale
#   is the edit's ``alpha``.
# * Changes, in float64 from the float32 logits: at the steered position p, the intervened
#   minus the unintervened logit of token t, averaged over the probe's sequences; later, the
#   same averaged over sequences and ``later_positions(p, seq)``. Per layer, scale and
#   direction, ``same_position`` and ``later_positions`` are those changes' means over the
#   positions, and ``ratio`` is ``later_positions / same_position`` (null when the
#   denominator is zero or the value isn't finite). ``changes_by_layer`` keeps every
#   position's two changes, in the order of ``positions``.
# * Also recorded: ``positions``, ``tokens``, ``experiment_seed``, the ``mean_residual_norm``
#   and ``scales`` per layer, and ``direction_cosine_mean`` per layer, the mean over the
#   drawn tokens of the cosine between the two directions.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G3.3"
REPORT_NAME = "g33_jlens_experiment.json"
EXPERIMENT_SEED = 3
DIRECTIONS = ("jlens", "logit_lens")
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    step = ref.steps[-1]
    jacobians = _common.jacobians(ref, p1.load_split("train"), [step])
    probe = _common.probe(p1.load_split("validation"), ref.config.seq)
    excluded = special_tokens()
    measurements = measure(ref.model, ref.params_for(step), jacobians(step), ref.config, probe, excluded)
    measurements["step"] = step
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """G3.3 records, not judges: always pass, as reports.py's rule requires."""
    return "pass"


def special_tokens() -> list[int]:
    """The ids of ``<unk>`` and ``<eos>`` in the run's tokenizer, which the experiment never steers toward."""
    from nablaframe.data.tokenizer import EOS, UNK

    vocab = list(p1.load_tokenizer().vocab)
    return sorted(vocab.index(token) for token in (UNK, EOS))


def draw(seq: int, vocab: int, excluded: list[int], *, seed: int = EXPERIMENT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """The seeded positions (sorted, distinct, in ``fit_range``) and one token per position."""
    count = _common.EXPERIMENT_POSITIONS
    if len(_common.fit_range(seq)) < count:
        raise SystemExit(f"the fit range of a {seq}-token window has fewer than {count} positions. No report written.")
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(_common.fit_range(seq), size=count, replace=False))
    allowed = np.setdiff1d(np.arange(vocab), np.asarray(excluded, dtype=np.int64))
    return positions, allowed[rng.integers(0, len(allowed), size=count)]


def measure(model: Any, params: Any, jacobians: dict[int, np.ndarray], config: Any, probe: np.ndarray,
            excluded: list[int]) -> dict[str, Any]:
    from nablaframe.interventions.jlens_experiment import (SCALE_FACTORS, jlens_direction, logit_lens_direction,
                                                           steering_scales)
    from nablaframe.interventions.spec import InterventionSpec, edit

    n_layers, seq = config.n_layers, config.seq
    layers = list(range(n_layers - 1))
    taps = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    base_logits, base = _common.unintervened(model, params, probe, n_layers=n_layers, taps=taps)
    base_logits = base_logits.astype(np.float64)
    wte = np.asarray(params.wte, dtype=np.float32)
    positions, tokens = draw(seq, config.vocab, excluded)
    objects = _common.Objects()

    results: dict[str, Any] = {}
    changes: dict[str, Any] = {}
    norms, scales_by_layer, cosines = {}, {}, {}
    progress = p1.Progress("steered passes", len(layers) * len(SCALE_FACTORS) * len(DIRECTIONS) * len(positions))
    for layer, tap in zip(layers, taps):
        scales = steering_scales(base[tap])
        norms[str(layer)] = float(np.linalg.norm(base[tap].astype(np.float64), axis=-1).mean())
        scales_by_layer[str(layer)] = [float(s) for s in scales]
        directions = {
            "jlens": [jlens_direction(jacobians[layer], wte, int(t)) for t in tokens],
            "logit_lens": [logit_lens_direction(wte, int(t)) for t in tokens],
        }
        cosines[str(layer)] = float(np.mean([float(np.dot(a.astype(np.float64), b.astype(np.float64)))
                                             for a, b in zip(directions["jlens"], directions["logit_lens"])]))
        results[str(layer)], changes[str(layer)] = {}, {}
        for factor, scale in zip(SCALE_FACTORS, scales):
            row, kept = {}, {}
            for name in DIRECTIONS:
                same, later = [], []
                for p, t, vector in zip(positions, tokens, directions[name]):
                    spec = InterventionSpec((edit(tap, [int(p)], "add", vector=objects.vector(vector),
                                                  alpha=float(scale)),))
                    logits, _ = _common.run(model, params, probe, spec, n_layers=n_layers, resolve=objects.resolve)
                    delta = logits[:, :, int(t)].astype(np.float64) - base_logits[:, :, int(t)]
                    same.append(float(delta[:, p].mean()))
                    later.append(float(delta[:, _common.later_positions(int(p), seq)].mean()))
                    progress.advance()
                same_mean, later_mean = float(np.mean(same)), float(np.mean(later))
                row[name] = {"same_position": p1.finite_or_none(same_mean),
                             "later_positions": p1.finite_or_none(later_mean),
                             "ratio": _ratio(later_mean, same_mean)}
                kept[name] = {"same_position": [p1.finite_or_none(v) for v in same],
                              "later_positions": [p1.finite_or_none(v) for v in later]}
            results[str(layer)][_factor(factor)] = row
            changes[str(layer)][_factor(factor)] = kept
            print(f"{GATE_ID}: block {layer}, scale {factor}: J-lens direction moves the token's logit by"
                  f" {_show(row['jlens']['same_position'])} at the steered position and"
                  f" {_show(row['jlens']['later_positions'])} later; the logit lens's by"
                  f" {_show(row['logit_lens']['same_position'])} and {_show(row['logit_lens']['later_positions'])}",
                  flush=True)
    return {
        "layers": layers,
        "scale_factors": list(SCALE_FACTORS),
        "results_by_layer": results,
        "changes_by_layer": changes,
        "positions": positions.tolist(),
        "tokens": tokens.tolist(),
        "excluded_tokens": list(excluded),
        "experiment_seed": EXPERIMENT_SEED,
        "later_window": _common.LATER_WINDOW,
        "mean_residual_norm": norms,
        "scales": scales_by_layer,
        "direction_cosine_mean": cosines,
        "probe_sha256": _batch_hash(probe),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return p1.finite_or_none(numerator / denominator)


def _factor(factor: float) -> str:
    """A scale factor as a report key: ``"0.25"``, ``"0.5"``, ``"1.0"``."""
    return repr(float(factor))


def _show(value: float | None) -> str:
    return "null" if value is None else f"{value:+.4f}"


def _batch_hash(tokens: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
