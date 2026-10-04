"""G2.3: the logit lens and its discrimination tests (D14).

* Planted, at decoder level: residual streams at size S carrying a chosen token's
  direction plus noise, through ``logit_lens`` with a standard-normal unembedding and an
  identity layer norm, as in tests/test_views_lens.py; the real final layer norm and
  unembedding are exercised by the last-layer KL check. ``planted_recovery`` is the
  fraction of positions whose top-1 lens token is the planted one.
* Random weights, at model level: a freshly initialized size-S model over the
  262,144-token evaluation slice, in windows of 256. ``random_accuracy_by_layer`` is
  ``next_token_accuracy`` per reading, ``random_max_accuracy`` its maximum,
  ``random_chance`` is ``1 / 4096`` and ``random_positions`` the positions scored.
* The meter on the reference run's probe at every trace step: ``meters`` maps each step
  to its ``lens_meter``; ``last_layer_kl_max`` is the largest last-reading ``kl_mean``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g23_lens.py``. Takes --overwrite and --allow-dirty. It reads G1.6's
#   store after G2.1 has added the logits, and writes nothing outside its report.
# * Planted: the construction of tests/test_views_lens.py at size S's shape (d_model
#   256, vocabulary 4,096, from G1.4's manifest) with ``PLANTED_POSITIONS`` (2,048, the
#   probe's count) positions. All draws come from ``default_rng(0)`` in the test's order:
#   a standard-normal unembedding, the targets, uniform over the vocabulary, then the
#   noise. Each residual is ``PLANTED_SCALE`` (4) times the target's embedding row plus
#   Gaussian noise with standard deviation ``PLANTED_NOISE`` (0.3), read through an
#   identity layer norm (weight 1, bias 0). The reference run's weights aren't used here:
#   with trained tied embeddings the lens also favours large-norm tokens, so a planted
#   direction can lose to a large-norm neighbour even though the lens is exact.
#   ``planted_misses`` lists the first 20 positions whose top-1 lens token isn't the
#   target.
# * Random weights: ``init_params(config, RANDOM_INIT_SEED)``, with seed 1, so the
#   model is not the reference run's own initialization. The evaluation slice is
#   ``eval_windows(validation, 262,144, 256)``, each window's first 256 tokens, in
#   batches of ``RANDOM_BATCH`` windows. Readings are taken as ``lens_readings`` takes
#   them: ``blocks.0.hook_resid_pre`` then each block's ``hook_resid_post``.
#   ``next_token_accuracy`` scores positions 0 to 254 of each window, so
#   ``random_positions`` is 255 per window. Accuracies are combined over batches by
#   correct counts. ``random_correct_by_layer`` and ``random_chance_count`` give the
#   counts behind them.
# * Meter: ``lens_meter(lens_readings(reader, step, params, n_layers), traced logits)``
#   at every trace step, with that step's parameters. ``meters`` is keyed by the step as
#   a string, since report keys are strings. A NaN or infinite meter value is recorded
#   as null; a null last-layer KL makes ``last_layer_kl_max`` null, which fails the gate.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.3"
REPORT_NAME = "g23_lens.json"
PLANTED_SEED = 0
PLANTED_POSITIONS = 2_048
PLANTED_SCALE = 4.0
PLANTED_NOISE = 0.3
RANDOM_INIT_SEED = 1
RANDOM_BATCH = 16
MIN_PLANTED_RECOVERY = 0.99
MAX_LAST_LAYER_KL = 1e-5
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    validation = p1.load_split("validation")
    probe = p1.probe(validation, ref.config.seq)
    reader = _common.open_reader(_common.g16_store(), ref.steps, probe)

    measurements = planted(ref.config.d_model, ref.config.vocab)
    measurements |= random_weights(ref.model, ref.config, validation)
    measurements |= meters(reader, ref.steps, ref.params_for, ref.config.n_layers)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.3 rule holds."""
    ok = (m["planted_recovery"] is not None and m["planted_recovery"] >= MIN_PLANTED_RECOVERY
          and m["random_max_accuracy"] is not None and m["random_max_accuracy"] <= 2 * m["random_chance"]
          and m["last_layer_kl_max"] is not None and m["last_layer_kl_max"] <= MAX_LAST_LAYER_KL)
    return "pass" if ok else "fail"


def planted(d: int, vocab: int, *, positions: int = PLANTED_POSITIONS) -> dict[str, Any]:
    """Decoder-level planted recovery, as in tests/test_views_lens.py, at the given shape."""
    from nablaframe.views.lens import logit_lens

    rng = np.random.default_rng(PLANTED_SEED)
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(1, positions))
    resid = (PLANTED_SCALE * wte[targets] + PLANTED_NOISE * rng.standard_normal((1, positions, d))).astype(np.float32)
    top1 = logit_lens(resid, np.ones(d, np.float32), np.zeros(d, np.float32), wte).argmax(-1)
    hits = top1 == targets
    misses = np.flatnonzero(~hits[0])
    recovery = float(hits.mean())
    print(f"{GATE_ID}: planted token recovered at {recovery:.4f} of {positions:,} positions", flush=True)
    return {
        "planted_recovery": recovery,
        "planted_positions": positions,
        "planted_scale": PLANTED_SCALE,
        "planted_noise": PLANTED_NOISE,
        "planted_d_model": d,
        "planted_vocab": vocab,
        "planted_seed": PLANTED_SEED,
        "planted_misses": [{"position": int(t), "target": int(targets[0, t]), "top1": int(top1[0, t])}
                           for t in misses[:20]],
    }


def random_weights(model: Any, config: Any, validation: np.ndarray, *, eval_tokens: int | None = None,
                   batch: int = RANDOM_BATCH) -> dict[str, Any]:
    """The lens of a freshly initialized model over the evaluation slice: next-token accuracy per reading."""
    from nablaframe.data.corpus import eval_windows
    from nablaframe.model.gpt2 import init_params
    from nablaframe.views.lens import lens_params, logit_lens, next_token_accuracy

    params = init_params(config, RANDOM_INIT_SEED)
    ln_w, ln_b, wte = lens_params(params)
    taps = ["blocks.0.hook_resid_pre"] + [f"blocks.{layer}.hook_resid_post" for layer in range(config.n_layers)]
    eval_tokens = p1.EVAL_TOKENS if eval_tokens is None else eval_tokens
    windows = eval_windows(np.asarray(validation), eval_tokens, config.seq)[:, :config.seq].astype(np.int32)
    if len(windows) == 0:
        raise SystemExit("the validation stream is too short for one evaluation window. No report written.")
    correct = np.zeros(len(taps), dtype=np.int64)
    scored = 0
    progress = p1.Progress("lens on random weights, windows", len(windows))
    for start in range(0, len(windows), batch):
        tokens = windows[start:start + batch]
        _, recorded = model.instrumented(params, tokens, taps)
        n = tokens.shape[0] * (tokens.shape[1] - 1)
        for i, tap in enumerate(taps):
            accuracy = next_token_accuracy(logit_lens(np.asarray(recorded[tap]), ln_w, ln_b, wte), tokens)
            correct[i] += int(round(accuracy * n))
        scored += n
        progress.advance(len(tokens))
    accuracy = [float(c / scored) for c in correct]
    chance = 1 / config.vocab
    print(f"{GATE_ID}: on random weights, the best reading's next-token accuracy is {max(accuracy):.3e}"
          f" against chance {chance:.3e}, over {scored:,} positions", flush=True)
    return {
        "random_accuracy_by_layer": dict(zip(taps, accuracy)),
        "random_correct_by_layer": dict(zip(taps, correct.tolist())),
        "random_max_accuracy": max(accuracy),
        "random_chance": chance,
        "random_bound": _common.random_accuracy_bound(config.vocab),
        "random_chance_count": _common.chance_count(config.vocab, scored),
        "random_positions": scored,
        "random_windows": int(len(windows)),
        "random_init_seed": RANDOM_INIT_SEED,
    }


def meters(reader: Any, steps: tuple[int, ...], params_for: Any, n_layers: int) -> dict[str, Any]:
    """The lens meter on the probe at every trace step, and the largest last-reading KL."""
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.lens import lens_meter, lens_readings

    by_step = {}
    progress = p1.Progress("lens meter, steps", len(steps))
    for step in steps:
        readings = lens_readings(reader, step, params_for(step), n_layers)
        meter = lens_meter(readings, reader.tensor(step, LOGITS))
        by_step[str(step)] = [{key: value if key == "layer" else p1.finite_or_none(value)
                               for key, value in reading.items()} for reading in meter]
        progress.advance()
    last = {step: meter[-1]["kl_mean"] for step, meter in by_step.items()}
    if any(kl is None for kl in last.values()):
        nonfinite = [int(step) for step, kl in last.items() if kl is None]
        print(f"{GATE_ID}: the last-layer KL is not finite at steps {nonfinite}", flush=True)
        return {"meters": by_step, "last_layer_kl_max": None, "last_layer_kl_max_step": None}
    worst = max(last, key=lambda step: last[step])
    print(f"{GATE_ID}: the largest last-layer KL is {last[worst]:.3e} nats, at step {worst}", flush=True)
    return {"meters": by_step, "last_layer_kl_max": last[worst], "last_layer_kl_max_step": int(worst)}


if __name__ == "__main__":
    raise SystemExit(main())
