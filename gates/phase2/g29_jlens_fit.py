"""G2.9: the summed J for every layer and trace checkpoint, in the jax lane (D17). About 4 hours.

Refuses to start unless ``reports/phase2/g28_jlens_crosscheck.json`` exists and passes, so the
long computation runs only on a definition the reference has confirmed. For each of the 28
trace steps and each of layers 0 to 2 (layer 3's J is the identity, by definition), computes
``summed_jacobian`` on all 128 windows of D17's fit sample and saves it with
``save_jacobian`` under ``runs/phase2/jlens/fit/``, keyed by ``jacobian_key`` with the
checkpoint sidecar's state hash, the fit sample's token hash and ``code_identity()``. A J
already saved under its key is skipped, so an interrupted run resumes. Prints a progress
line at least every 30 seconds. At the end, recomputes one seeded (step, layer) and compares.

Measurements: ``jacobians_written`` (saved, now or before), ``jacobians_expected`` (84),
``recompute_identical``, ``seconds``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g29_jlens_fit.py``. Takes --overwrite and --allow-dirty. --overwrite only
#   allows replacing the report: the fit folder is never cleared, because a key never changes
#   meaning, so a J saved under it is always reusable, and clearing it would throw away hours.
# * The confirmation, each failure an exit 1 with no report: G2.8's report must exist, pass
#   ``validate_report``, be G2.8's, say ``pass``, name G1.4's run, and record the
#   ``code_identity`` this checkout's ``views/jlens.py`` has now. A J computed by code the
#   cross-check didn't run on is not a confirmed definition.
# * The fit sample is G2.7's ``fit_windows``: all 128 windows of ``batch_for_step(train, 1, 0,
#   128, seq)``, first ``seq`` tokens each. Its hash is ``batch_hash`` of those windows.
# * Per trace step with any J still to compute: the step's parameters, then
#   ``blocks.<l>.hook_resid_post`` for layers 0 to 2 from the instrumented pass, run on
#   ``RESIDUAL_BATCH`` (16) windows at a time to bound memory, then per layer
#   ``summed_jacobian(block_map(model, params, l, n_layers), xs)``, saved at once, so an
#   interruption loses at most the J in progress. Steps run in schedule order, layers 0 to 2.
# * Progress: one line per J saved, and a background line every ``TICK_SECONDS`` (20) while
#   a J is computing, since one J takes minutes. Each names the step, the layer, how many are
#   saved and an estimate of the time left from this run's mean time per J.
# * ``jacobians_written`` counts the expected keys with a saved J at the end, whenever saved.
#   ``jacobians_expected`` is the number of trace steps times ``n_layers - 1``: 84 for the
#   reference run. ``jacobians_computed`` and ``jacobians_skipped`` split this run's work.
# * The recompute: ``default_rng(0)`` picks one (step, layer) from the expected list in
#   schedule order. Its J is computed again exactly as above and compared with the saved
#   file, read with ``load_jacobian``: ``recompute_identical`` is true when dtype, shape and
#   every byte agree. It is false when that J was never saved.
# * ``seconds`` is this run's wall time, from the first J to the end of the recompute.

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import g27_jlens_prep as g27  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.9"
REPORT_NAME = "g29_jlens_fit.json"
CROSSCHECK_REPORT = "g28_jlens_crosscheck.json"
RESIDUAL_BATCH = 16
RECOMPUTE_SEED = 0
TICK_SECONDS = 20.0
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    require_crosscheck(ref.manifest.run_id)
    measurements = measure(ref, p1.load_split("train"), g27.fit_dir())
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.9 rule holds."""
    ok = m["jacobians_written"] == m["jacobians_expected"] and m["recompute_identical"] is True
    return "pass" if ok else "fail"


def require_crosscheck(run_id: str) -> None:
    """SystemExit unless G2.8's report passes, for this run, on the code that computes J here."""
    from nablaframe.reports import validate_report
    from nablaframe.views.jlens import code_identity

    path = Path(_common.REPORT_DIR) / CROSSCHECK_REPORT
    rel = p1.rel(path)
    if not path.exists():
        raise SystemExit(f"{rel} is missing. G2.9 runs only after G2.8 passes. No report written.")
    try:
        crosscheck = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{rel} can't be read ({exc}). No report written.") from None
    problems = validate_report(crosscheck) if isinstance(crosscheck, dict) else ["not a JSON object"]
    if problems:
        raise SystemExit(f"{rel} is not a valid report: {'; '.join(problems)}. No report written.")
    if crosscheck["gate_id"] != "G2.8" or crosscheck["status"] != "pass":
        raise SystemExit(f"{rel} is {crosscheck['gate_id']} with status {crosscheck['status']}, not a passing G2.8."
                         " G2.9 runs only after G2.8 passes. No report written.")
    m = crosscheck["measurements"]
    if m.get("run_id") != run_id:
        raise SystemExit(f"{rel} cross-checked run {m.get('run_id')}, not G1.4's run {run_id}. No report written.")
    if m.get("code_identity") != code_identity():
        raise SystemExit(f"{rel} confirmed views/jlens.py at blob {m.get('code_identity')}, but this checkout's is"
                         f" {code_identity()}. Run G2.7 and G2.8 again first. No report written.")


# ---------------------------------------------------------------------------


def measure(ref: Any, train: np.ndarray, root: Path) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.views.jlens import code_identity, jacobian_key, load_jacobian, save_jacobian

    config, model = ref.config, ref.model
    layers = tuple(range(config.n_layers - 1))
    windows = g27.fit_windows(train, config.seq)
    fit_sha, code = batch_hash(windows), code_identity()
    state = {step: g27.checkpoint_sha256(ref.checkpoint_dir / f"step-{step}.safetensors") for step in ref.steps}
    keys = {(step, layer): jacobian_key(checkpoint_sha256=state[step], fit_sha256=fit_sha, layer=layer, code=code)
            for step in ref.steps for layer in layers}
    expected = len(keys)

    def saved(key: str) -> bool:
        return (Path(root) / f"{key}.npy").exists()

    todo = [pair for pair, key in keys.items() if not saved(key)]
    skipped = expected - len(todo)
    print(f"{GATE_ID}: {expected} J's for {len(ref.steps)} steps and layers {list(layers)} on {len(windows)} windows;"
          f" {skipped} already saved in {p1.rel(root)}, {len(todo)} to compute", flush=True)

    started = time.perf_counter()
    ticker = Ticker(expected, skipped, started)
    computed, per_j = 0, []
    with ticker:
        for step in ref.steps:
            pending = [layer for layer in layers if (step, layer) in todo]
            if not pending:
                continue
            params = ref.params_for(step)
            xs = residuals(model, params, windows, pending)
            for layer in pending:
                ticker.begin(step, layer)
                j_started = time.perf_counter()
                J = jacobian(model, params, xs[layer], layer)
                save_jacobian(root, keys[(step, layer)], J)
                per_j.append(time.perf_counter() - j_started)
                computed += 1
                ticker.done(per_j[-1])
            del xs

    pairs = list(keys)
    pick = pairs[int(np.random.default_rng(RECOMPUTE_SEED).integers(len(pairs)))]
    step, layer = pick
    print(f"{GATE_ID}: recomputing step {step}, layer {layer}", flush=True)
    stored = load_jacobian(root, keys[pick])
    with Ticker(expected, expected, time.perf_counter(), label="recompute") as recompute_ticker:
        recompute_ticker.begin(step, layer)
        params = ref.params_for(step)
        again = jacobian(model, params, residuals(model, params, windows, [layer])[layer], layer)
    identical = bool(stored is not None and stored.dtype == again.dtype and stored.shape == again.shape
                     and stored.tobytes() == again.tobytes())
    seconds = time.perf_counter() - started
    written = sum(1 for key in keys.values() if saved(key))
    print(f"{GATE_ID}: {written} of {expected} J's saved; recompute "
          + ("identical" if identical else "differs") + f"; {seconds:.1f} s", flush=True)

    return {
        "jacobians_written": written,
        "jacobians_expected": expected,
        "recompute_identical": identical,
        "seconds": seconds,
        "jacobians_computed": computed,
        "jacobians_skipped": skipped,
        "seconds_per_jacobian_mean": float(np.mean(per_j)) if per_j else None,
        "seconds_per_jacobian_max": float(np.max(per_j)) if per_j else None,
        "recompute_step": step,
        "recompute_layer": layer,
        "recompute_seed": RECOMPUTE_SEED,
        "steps": list(ref.steps),
        "layers": list(layers),
        "windows": int(len(windows)),
        "seq": config.seq,
        "fit_sha256": fit_sha,
        "code_identity": code,
        "fit": p1.rel(root),
        "run_id": ref.manifest.run_id,
    }


def residuals(model: Any, params: Any, windows: np.ndarray, layers: list[int]) -> dict[int, np.ndarray]:
    """``blocks.<l>.hook_resid_post`` for every window, per layer, from the instrumented pass in batches."""
    taps = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    parts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for start in range(0, len(windows), RESIDUAL_BATCH):
        _, recorded = model.instrumented(params, windows[start:start + RESIDUAL_BATCH], taps)
        for layer, tap in zip(layers, taps):
            parts[layer].append(np.asarray(recorded[tap]))
    return {layer: np.concatenate(chunks) for layer, chunks in parts.items()}


def jacobian(model: Any, params: Any, xs: np.ndarray, layer: int) -> np.ndarray:
    from nablaframe.views.jlens import block_map, summed_jacobian

    return summed_jacobian(block_map(model, params, layer, model.config.n_layers), xs)


class Ticker:
    """Prints a progress line every ``interval_s`` from a background thread while J's compute."""

    def __init__(self, expected: int, saved: int, started: float, *, label: str = "fit",
                 interval_s: float = TICK_SECONDS) -> None:
        self.expected, self.saved, self.started = expected, saved, started
        self.label, self.interval_s = label, interval_s
        self.current: tuple[int, int] | None = None
        self.current_started = started
        self.durations: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Ticker:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def begin(self, step: int, layer: int) -> None:
        self.current, self.current_started = (step, layer), time.perf_counter()

    def done(self, seconds: float) -> None:
        self.durations.append(seconds)
        self.saved += 1
        step, layer = self.current
        self.current = None
        print(f"{GATE_ID}: saved J for step {step}, layer {layer} in {seconds:.1f} s; {self._state()}", flush=True)

    def _state(self) -> str:
        now = time.perf_counter()
        line = f"{self.saved}/{self.expected} saved, {_clock(now - self.started)} elapsed"
        if self.durations and self.saved < self.expected:
            line += f", about {_duration((self.expected - self.saved) * float(np.mean(self.durations)))} left"
        return line

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            current = self.current
            if current is None:
                print(f"{GATE_ID} {self.label}: preparing the next step; {self._state()}", flush=True)
                continue
            step, layer = current
            into = _clock(time.perf_counter() - self.current_started)
            print(f"{GATE_ID} {self.label}: step {step}, layer {layer}, {into} into this J; {self._state()}",
                  flush=True)


_duration = p1._duration


def _clock(seconds: float) -> str:
    """``seconds`` as h:mm:ss, or m:ss under an hour, so successive progress lines visibly advance."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
