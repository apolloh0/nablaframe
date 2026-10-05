"""G2.10: the J-lens view and its discrimination tests (D14, D17, D20).

Refuses to start unless G2.9's report exists and passes. Then:

* Planted, at decoder level, at size S: source directions that a J carries exactly to each
  target token's unembedding direction, drawn as a random orthogonal J first, each source
  then J.T @ wte[t], so J carries it exactly to its target's unembedding direction (the
  test's pseudo-inverse construction is exact only while the vocabulary fits in d_model),
  with a standard-normal unembedding and an identity layer norm. ``planted_recovery`` is the
  fraction of positions whose top-1 J-lens token is the target.
* ``identity_exact``: with J the identity, ``jlens_lens`` equals ``logit_lens`` bit for bit
  on the reference run's final-step residual streams.
* Random weights, at model level: a freshly initialized size-S model; each layer's J fitted
  with ``summed_jacobian`` on the first 16 windows of D17's fit sample; the J-lens over the
  262,144-token evaluation slice. ``random_accuracy_by_layer`` is ``next_token_accuracy`` per
  reading, ``random_max_accuracy`` its maximum, ``random_chance`` is ``1 / 4096``.
* The meter and the comparison on the reference run's probe at every trace step, with G2.9's
  J's: ``meters`` maps each step to its J-lens ``lens_meter``; ``last_reading_kl_max`` is the
  largest last-reading ``kl_mean``; ``comparison_by_step`` maps each step to
  ``lens_divergence`` against the logit lens: recorded, not judged.

Prints a progress line at least every 30 seconds.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g210_jlens_view.py``. Takes --overwrite and --allow-dirty. It reads G1.6's
#   store after G2.1 has added the logits, and G2.9's J's, and writes nothing outside its
#   report.
# * The confirmation, each failure an exit 1 with no report: G2.9's report must exist, pass
#   ``validate_report``, be G2.9's and say ``pass``. Then every J the meters need must be in
#   G2.9's cache, ``runs/phase2/jlens/fit/``, under ``jacobian_key`` with the trace
#   checkpoint sidecar's ``state_sha256``, ``batch_hash`` of D17's 128 fit windows and this
#   checkout's ``code_identity()``, for layers 0 to ``n_layers - 2`` at every trace step,
#   as G2.9 saved them. A missing J is a setup error, checked before anything is measured.
#   ``jacobians_for`` and ``fit_report_problem`` are shared with G2.6's rebuild.
# * Planted: the J-lens test's construction cannot carry every token exactly at size S. It
#   draws the source directions first and fits J to them by pseudo-inverse, which is exact
#   only while the vocabulary is no larger than d_model; at 4,096 tokens in 256 dimensions,
#   no linear map takes 4,096 independent draws to 4,096 others, and that J recovers about
#   59% of positions. So J is drawn first, as a random orthogonal matrix (the Q of a QR
#   decomposition of a standard-normal matrix), and each token's source direction is then
#   ``J.T`` times its unembedding row, which J carries exactly back to it.
#   ``planted_transport_max_abs_error`` is the largest float32 difference between J times a
#   source direction and its unembedding row. All draws come from ``default_rng(0)``: the
#   standard-normal unembedding (vocabulary by d_model, from G1.4's manifest), J's matrix,
#   the targets, uniform over the vocabulary, then the noise. Each residual is
#   ``PLANTED_SCALE`` (4) times the target's source direction plus Gaussian noise with
#   standard deviation ``PLANTED_NOISE`` (0.1), as in the test, at ``PLANTED_POSITIONS``
#   (2,048, the probe's count), read through an identity layer norm.
#   ``planted_logit_lens_recovery`` reads the same residuals with the logit lens, recorded
#   to show the planted structure is readable only through J. ``planted_misses`` lists the
#   first 20 positions whose top-1 J-lens token isn't the target.
# * ``identity_exact``: at the last trace step, every block's ``hook_resid_post`` through
#   ``jlens_lens`` with a float32 identity J and the step's real final layer norm and
#   unembedding, compared with ``logit_lens`` of the same residual stream by
#   ``numpy.array_equal``. ``identity_exact_by_layer`` gives each block.
# * Random weights: ``init_params(config, RANDOM_INIT_SEED)``, seed 1, as G2.3. Each J for
#   layers 0 to ``n_layers - 2`` is ``summed_jacobian(block_map(...))`` of that layer's
#   ``hook_resid_post`` on the first ``RANDOM_FIT_WINDOWS`` (16) windows of D17's fit sample,
#   computed as G2.9 computes them. The evaluation slice is G2.3's: ``eval_windows(validation,
#   262,144, seq)``, each window's first ``seq`` tokens, in batches of ``RANDOM_BATCH`` (16).
#   Readings are ``jlens_readings``': each block's ``hook_resid_post`` through its J, and the
#   last block's through the logit lens, its J being the identity. ``next_token_accuracy``
#   scores positions 0 to seq - 2 of each window; accuracies are combined over batches by
#   correct counts, which ``random_correct_by_layer`` and ``random_chance_count`` give.
# * Meter and comparison: at every trace step, with that step's parameters and G2.9's J's,
#   ``lens_meter(jlens_readings(...), traced logits)`` and ``lens_divergence`` of the J-lens
#   readings against ``logit_lens`` of the same block outputs. Both are keyed by the step as
#   a string, since report keys are strings. A NaN or infinite value is recorded as null; a
#   null last-reading KL makes ``last_reading_kl_max`` null, which fails the gate.

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import g27_jlens_prep as g27  # noqa: E402
import g29_jlens_fit as g29  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.10"
REPORT_NAME = "g210_jlens_view.json"
FIT_REPORT = "g29_jlens_fit.json"
PLANTED_SEED = 0
PLANTED_POSITIONS = 2_048
PLANTED_SCALE = 4.0
PLANTED_NOISE = 0.1
RANDOM_INIT_SEED = 1
RANDOM_FIT_WINDOWS = 16
RANDOM_BATCH = 16
MIN_PLANTED_RECOVERY = 0.99
MAX_LAST_READING_KL = 1e-5
TICK_SECONDS = 20.0
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    problem = fit_report_problem()
    if problem is not None:
        raise SystemExit(f"{problem}. G2.10 runs only after G2.9 passes. No report written.")
    p1.require_data()
    ref = _common.reference()
    validation, train = p1.load_split("validation"), p1.load_split("train")
    probe = p1.probe(validation, ref.config.seq)
    reader = _common.open_reader(_common.g16_store(), ref.steps, probe)
    jacobians = jacobians_for(ref, train)

    measurements = planted(ref.config.d_model, ref.config.vocab)
    measurements |= identity(reader, ref.steps[-1], ref.params_for(ref.steps[-1]), ref.config.n_layers)
    measurements |= random_weights(ref.model, ref.config, train, validation)
    measurements |= meters(reader, ref.steps, ref.params_for, jacobians, ref.config.n_layers)
    measurements["fit_sha256"] = jacobians.fit_sha256
    measurements["code_identity"] = jacobians.code
    measurements["fit"] = p1.rel(g27.fit_dir())
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.10 rule holds."""
    ok = (m["planted_recovery"] is not None and m["planted_recovery"] >= MIN_PLANTED_RECOVERY
          and m["identity_exact"] is True
          and m["random_max_accuracy"] is not None and m["random_max_accuracy"] <= 2 * m["random_chance"]
          and m["last_reading_kl_max"] is not None and m["last_reading_kl_max"] <= MAX_LAST_READING_KL)
    return "pass" if ok else "fail"


# ---------------------------------------------------------------------------
# G2.9's report and J's. G2.6's rebuild uses these too.
# ---------------------------------------------------------------------------


def fit_report_problem() -> str | None:
    """Why G2.9's report doesn't pass, as a sentence, or None when it exists, is valid and passes."""
    from nablaframe.reports import validate_report

    path = Path(_common.REPORT_DIR) / FIT_REPORT
    rel = p1.rel(path)
    if not path.exists():
        return f"{rel} is missing"
    try:
        fit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{rel} can't be read ({exc})"
    problems = validate_report(fit) if isinstance(fit, dict) else ["not a JSON object"]
    if problems:
        return f"{rel} is not a valid report: {'; '.join(problems)}"
    if fit["gate_id"] != "G2.9" or fit["status"] != "pass":
        return f"{rel} is {fit['gate_id']} with status {fit['status']}, not a passing G2.9"
    return None


class Jacobians:
    """Each trace step's J's from G2.9's cache: ``jacobians(step)`` is ``{layer: J}`` for layers 0 to n - 2."""

    def __init__(self, keys: dict[tuple[int, int], str], root: Path, fit_sha256: str, code: str) -> None:
        self.keys, self.root, self.fit_sha256, self.code = keys, root, fit_sha256, code

    def __call__(self, step: int) -> dict[int, np.ndarray]:
        from nablaframe.views.jlens import load_jacobian

        out = {}
        for (key_step, layer), key in self.keys.items():
            if key_step != step:
                continue
            J = load_jacobian(self.root, key)
            if J is None:
                raise SystemExit(f"G2.9's J for step {step}, layer {layer} is missing from {p1.rel(self.root)}."
                                 " Run G2.9 again. No report written.")
            out[layer] = J
        return out


def jacobians_for(ref: Any, train: np.ndarray) -> Jacobians:
    """G2.9's J's for G1.4's run, after checking every one is in the cache. SystemExit naming any missing."""
    from nablaframe.data.corpus import batch_hash
    from nablaframe.views.jlens import code_identity, jacobian_key

    root = g27.fit_dir()
    fit_sha, code = batch_hash(g27.fit_windows(train, ref.config.seq)), code_identity()
    keys = {}
    for step in ref.steps:
        state = g27.checkpoint_sha256(ref.checkpoint_dir / f"step-{step}.safetensors")
        for layer in range(ref.config.n_layers - 1):
            keys[(step, layer)] = jacobian_key(checkpoint_sha256=state, fit_sha256=fit_sha, layer=layer, code=code)
    missing = [pair for pair, key in keys.items() if not (Path(root) / f"{key}.npy").exists()]
    if missing:
        shown = ", ".join(f"step {step} layer {layer}" for step, layer in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise SystemExit(f"{len(missing)} of {len(keys)} J's for run {ref.manifest.run_id} are missing from"
                         f" {p1.rel(root)} ({shown}{more}), for this checkout's views/jlens.py at blob {code}."
                         " Run G2.9 first. No report written.")
    return Jacobians(keys, root, fit_sha, code)


# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------


def planted(d: int, vocab: int, *, positions: int = PLANTED_POSITIONS) -> dict[str, Any]:
    """Decoder-level planted transport: source directions J carries exactly to each target's unembedding row."""
    from nablaframe.views.jlens_view import jlens_lens
    from nablaframe.views.lens import logit_lens

    rng = np.random.default_rng(PLANTED_SEED)
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    J = q.astype(np.float32)
    source = wte.astype(np.float64) @ q  # row t is J.T @ wte[t], so J @ source[t] == wte[t]
    targets = rng.integers(0, vocab, size=(1, positions))
    resid = (PLANTED_SCALE * source[targets] + PLANTED_NOISE * rng.standard_normal((1, positions, d))).astype(np.float32)
    ones, zeros = np.ones(d, np.float32), np.zeros(d, np.float32)
    top1 = jlens_lens(resid, J, ones, zeros, wte).argmax(-1)
    logit_top1 = logit_lens(resid, ones, zeros, wte).argmax(-1)
    hits = top1 == targets
    misses = np.flatnonzero(~hits[0])
    recovery = float(hits.mean())
    transport_error = float(np.max(np.abs(source.astype(np.float32) @ J.T - wte)))
    print(f"{GATE_ID}: planted token recovered through J at {recovery:.4f} of {positions:,} positions"
          f" ({float((logit_top1 == targets).mean()):.4f} through the logit lens)", flush=True)
    return {
        "planted_recovery": recovery,
        "planted_logit_lens_recovery": float((logit_top1 == targets).mean()),
        "planted_transport_max_abs_error": transport_error,
        "planted_positions": positions,
        "planted_scale": PLANTED_SCALE,
        "planted_noise": PLANTED_NOISE,
        "planted_d_model": d,
        "planted_vocab": vocab,
        "planted_seed": PLANTED_SEED,
        "planted_misses": [{"position": int(t), "target": int(targets[0, t]), "top1": int(top1[0, t])}
                           for t in misses[:20]],
    }


def identity(reader: Any, step: int, params: Any, n_layers: int) -> dict[str, Any]:
    """Whether the J-lens with an identity J equals the logit lens bit for bit on every block output."""
    from nablaframe.views.jlens_view import jlens_lens
    from nablaframe.views.lens import lens_params, logit_lens

    ln_w, ln_b, wte = lens_params(params)
    eye = np.eye(wte.shape[1], dtype=np.float32)
    by_layer = {}
    for layer in range(n_layers):
        tap = f"blocks.{layer}.hook_resid_post"
        resid = reader.tensor(step, tap)
        by_layer[tap] = bool(np.array_equal(jlens_lens(resid, eye, ln_w, ln_b, wte), logit_lens(resid, ln_w, ln_b, wte)))
    exact = all(by_layer.values())
    print(f"{GATE_ID}: with J the identity, the J-lens " + ("equals" if exact else "differs from")
          + f" the logit lens bit for bit at step {step}", flush=True)
    return {"identity_exact": exact, "identity_exact_by_layer": by_layer, "identity_step": step}


def random_weights(model: Any, config: Any, train: np.ndarray, validation: np.ndarray, *,
                   eval_tokens: int | None = None, batch: int = RANDOM_BATCH) -> dict[str, Any]:
    """The J-lens of a freshly initialized model, with J's fitted to it, over the evaluation slice."""
    from nablaframe.data.corpus import batch_hash, eval_windows
    from nablaframe.model.gpt2 import init_params
    from nablaframe.views.jlens_view import jlens_lens
    from nablaframe.views.lens import lens_params, logit_lens, next_token_accuracy

    params = init_params(config, RANDOM_INIT_SEED)
    ln_w, ln_b, wte = lens_params(params)
    layers = list(range(config.n_layers - 1))
    fit = g27.fit_windows(train, config.seq)[:RANDOM_FIT_WINDOWS]
    xs = g29.residuals(model, params, fit, layers)
    jacobians = {}
    started = time.perf_counter()
    for layer in layers:
        with Heartbeat(f"fitting J for layer {layer} on random weights"):
            jacobians[layer] = g29.jacobian(model, params, xs[layer], layer)
        print(f"{GATE_ID}: random weights: J for layer {layer} fitted on {len(fit)} windows,"
              f" {time.perf_counter() - started:.1f} s so far", flush=True)
    del xs

    taps = [f"blocks.{layer}.hook_resid_post" for layer in range(config.n_layers)]
    eval_tokens = p1.EVAL_TOKENS if eval_tokens is None else eval_tokens
    windows = eval_windows(np.asarray(validation), eval_tokens, config.seq)[:, :config.seq].astype(np.int32)
    if len(windows) == 0:
        raise SystemExit("the validation stream is too short for one evaluation window. No report written.")
    correct = np.zeros(len(taps), dtype=np.int64)
    scored = 0
    progress = p1.Progress("J-lens on random weights, windows", len(windows))
    for start in range(0, len(windows), batch):
        tokens = windows[start:start + batch]
        _, recorded = model.instrumented(params, tokens, taps)
        n = tokens.shape[0] * (tokens.shape[1] - 1)
        for i, tap in enumerate(taps):
            resid = np.asarray(recorded[tap])
            reading = (logit_lens(resid, ln_w, ln_b, wte) if i == config.n_layers - 1  # the last block's J: identity
                       else jlens_lens(resid, jacobians[i], ln_w, ln_b, wte))
            correct[i] += int(round(next_token_accuracy(reading, tokens) * n))
        scored += n
        progress.advance(len(tokens))
    accuracy = [float(c / scored) for c in correct]
    chance = 1 / config.vocab
    print(f"{GATE_ID}: on random weights, the best J-lens reading's next-token accuracy is {max(accuracy):.3e}"
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
        "random_fit_windows": int(len(fit)),
        "random_fit_sha256": batch_hash(fit),
        "random_jacobian_norms": {str(layer): float(np.linalg.norm(J.astype(np.float64)))
                                  for layer, J in jacobians.items()},
    }


def meters(reader: Any, steps: tuple[int, ...], params_for: Callable[[int], Any],
           jacobians: Callable[[int], dict[int, np.ndarray]], n_layers: int) -> dict[str, Any]:
    """The J-lens meter and its comparison with the logit lens on the probe at every trace step."""
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.jlens_view import jlens_readings, lens_divergence
    from nablaframe.views.lens import lens_meter, lens_params, logit_lens

    meters_by_step, comparison_by_step = {}, {}
    progress = p1.Progress("J-lens meter, steps", len(steps))
    for step in steps:
        params = params_for(step)
        readings = jlens_readings(reader, step, params, jacobians(step), n_layers)
        ln_w, ln_b, wte = lens_params(params)
        logit = np.stack([logit_lens(reader.tensor(step, f"blocks.{layer}.hook_resid_post"), ln_w, ln_b, wte)
                          for layer in range(n_layers)])
        meters_by_step[str(step)] = _finite(lens_meter(readings, reader.tensor(step, LOGITS)))
        comparison_by_step[str(step)] = _finite(lens_divergence(readings, logit))
        del readings, logit
        progress.advance()
    last = {step: meter[-1]["kl_mean"] for step, meter in meters_by_step.items()}
    out: dict[str, Any] = {"meters": meters_by_step, "comparison_by_step": comparison_by_step}
    if any(kl is None for kl in last.values()):
        nonfinite = [int(step) for step, kl in last.items() if kl is None]
        print(f"{GATE_ID}: the last reading's KL is not finite at steps {nonfinite}", flush=True)
        return out | {"last_reading_kl_max": None, "last_reading_kl_max_step": None}
    worst = max(last, key=lambda step: last[step])
    final = comparison_by_step[str(steps[-1])]
    print(f"{GATE_ID}: the largest last-reading KL is {last[worst]:.3e} nats, at step {worst}; at step {steps[-1]},"
          " Jensen-Shannon divergence from the logit lens per reading: "
          + ", ".join("null" if row["js_mean"] is None else f"{row['js_mean']:.3e}" for row in final), flush=True)
    return out | {"last_reading_kl_max": last[worst], "last_reading_kl_max_step": int(worst)}


def _finite(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each row with its non-finite numbers as None, since reports allow only finite numbers."""
    return [{key: value if key == "layer" else p1.finite_or_none(value) for key, value in row.items()} for row in rows]


class Heartbeat:
    """Prints a progress line every ``interval_s`` from a background thread while one long step runs."""

    def __init__(self, label: str, interval_s: float = TICK_SECONDS) -> None:
        self.label, self.interval_s = label, interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Heartbeat:
        self.started = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            print(f"{GATE_ID}: {self.label}, {g29._clock(time.perf_counter() - self.started)} elapsed", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
