"""G4.4: the lenses, the J-lens coordinate swap and direct logit attribution at model level. Recorded, not judged (D34).

Spec: docs/specs/phase-4.md, "Lenses", "J-lens coordinate swap" and "Direct logit attribution";
helpers in ``_common.py``. Report: ``reports/phase4/g44_lenses.json``. Run in nf-jax, on the
Mac, after G4.1 passes.

At each zoo model's final checkpoint, from ``instrumented`` passes on ``eval_windows``:

* **J:** for the two-hop model and its twin, each block's summed J but the last's, which is the
  identity: ``summed_jacobian(block_map(...), xs)`` on that block's ``hook_resid_post`` for
  ``fit_windows(task)``, over D17's valid positions for a sequence of 26, 16 to 24. Saved with
  ``save_jacobian`` under ``runs/phase4/two_hop/jlens/``, keyed by ``jacobian_key``, and reused
  when present.
* **Lens ranks:** at each of ``QUERY_POSITIONS`` and each block's output, ``logit_lens`` and
  ``jlens_lens`` readings, and the median over windows of ``token_ranks`` of the intermediate
  ``b`` and the answer ``c``, for the two-hop model and its twin: ``lens_ranks``, as
  ``{"trained"|"twin": {"logit_lens"|"jlens": {"<position>": {"<block>": {"intermediate",
  "answer"}}}}}``.
* **The coordinate swap:** at each block's output, ``jswap`` with that block's J of the
  intermediate ``b`` and the counterpart ``k``, the ``aligned_sources`` counterpart drawn with
  ``SOURCE_SEED``, at ``QUERY_POSITIONS``; at the last block J is the identity, so it's the logit
  lens's swap. ``jswap_by_layer``: per block, ``{"swap", "unchanged"}``, how often the answer
  becomes the counterpart's object ``d`` and how often it stays ``c``.
* **Direct logit attribution:** for both zoo models and their twins, ``attribution`` of the
  correct next token's logit, from each part ``parts_from_trace`` names, taken from the
  ``instrumented`` pass, averaged over windows and mechanism positions: ``attribution``, as
  ``{"induction"|"two_hop": {"trained"|"twin": {part: mean}}}``.

Measurements: REQUIRED_MEASUREMENTS["G4.4"]. Its status is always ``pass``. Prints a progress
line per J and per block.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g44_lenses.py``. Takes --overwrite and --allow-dirty. It reads G4.1's runs,
#   after ``require_g41`` and ``zoo_run``'s checks, and writes only its report and the J's.
# * Every pass is ``run_intervened``: the unintervened ones, which supply the residual streams
#   and the attribution's parts, with an empty spec. The model reads each window's first
#   ``seq`` tokens.
# * J: ``xs`` is block l's ``hook_resid_post`` on ``fit_windows(task)``, 64 sequences of 26.
#   ``summed_jacobian``'s default ``skip_first``, 16, gives D17's valid positions 16 to 24. The
#   key is ``jacobian_key`` with ``checkpoint_sha256`` the final checkpoint sidecar's
#   ``state_sha256`` for the trained model, and ``state_hash`` of the twin's params for the twin,
#   which has no checkpoint; ``fit_sha256`` is ``batch_hash(fit_windows(task))`` and ``code``
#   ``views/jlens.py``'s ``code_identity()``. ``jacobians`` records each key and whether it was
#   computed or reused.
# * Lens ranks: ``token_ranks`` over the whole vocabulary, separators included, at each query
#   position of each block's ``hook_resid_post``, read by ``logit_lens`` and by ``jlens_lens``
#   with that block's J (the last block's J-lens reading is its logit lens). The median over
#   the evaluation windows, as a plain float. ``b`` is ``task.parse``'s intermediate and ``c``
#   each window's answer.
# * The coordinate swap: the J-lens of the trained model only. ``jswap``'s tokens differ in
#   every window, so each window is its own pass: one ``jswap`` edit at positions 24 and 25 of
#   block l's ``hook_resid_post`` with ``a`` the window's ``b`` and ``b`` its counterpart ``k``;
#   at the last block J is the float32 identity. ``d`` is ``k``'s object in the window's own
#   facts. The answer is the argmax at position 25.
# * Attribution: ``parts_from_trace`` reads the ``instrumented`` pass's recorded taps, so the
#   parts are exactly the ones it names, plus ``attribution``'s ``ln_bias``. The target at a
#   mechanism position p is the window's token p + 1. ``attribution_error`` records, per model,
#   the largest gap between the summed contributions and the logit of the target.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.4"
REPORT_NAME = "g44_lenses.json"
ANSWER_POSITION = 25
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    runs = {name: _common.zoo_run(name, g41) for name in _common.NAMES}
    measurements = measure(runs)
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """Always pass: G4.4 records, never judges (D34)."""
    return "pass"


def measure(runs: dict[str, Any]) -> dict[str, Any]:
    from nablaframe.zoo.ground_truth import TWIN_SEED, twin_params
    from nablaframe.zoo.two_hop import QUERY_POSITIONS

    two = runs["two_hop"]
    params = {"trained": two.params_for(two.final), "twin": twin_params(two.config)}
    jacobians, records = {}, {}
    for label, p in params.items():
        jacobians[label], records[label] = fit_jacobians(two, p, label)
    windows = _common.eval_windows(two.task)
    ranks = {label: lens_ranks(two, p, jacobians[label], windows) for label, p in params.items()}
    swaps = coordinate_swap(two, params["trained"], jacobians["trained"], windows)
    contributions, errors = {}, {}
    for name, run in runs.items():
        contributions[name], errors[name] = {}, {}
        for label, p in (("trained", run.params_for(run.final)), ("twin", twin_params(run.config))):
            contributions[name][label], errors[name][label] = attribute(run, p, label)
    return {
        "lens_ranks": ranks,
        "jswap_by_layer": swaps,
        "attribution": contributions,
        "attribution_error": errors,
        "jacobians": records,
        "query_positions": list(QUERY_POSITIONS),
        "answer_position": ANSWER_POSITION,
        "source_seed": _common.SOURCE_SEED,
        "twin_seed": TWIN_SEED,
        "steps": {name: run.final for name, run in runs.items()},
        "run_ids": {name: run.run_id for name, run in runs.items()},
        "eval_windows": int(len(windows)),
        "eval_windows_sha256": _batch_hash(windows),
        "fit_windows_sha256": _batch_hash(_common.fit_windows(two.task)),
    }


def jlens_dir() -> Path:
    """``runs/phase4/two_hop/jlens/``: the two-hop model's and its twin's J's."""
    return _common.run_dir("two_hop") / "jlens"


def fit_jacobians(run: Any, params: Any, label: str) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Each block's summed J but the last's, loaded from ``jlens_dir()`` when present, else fitted and saved."""
    from nablaframe.train.checkpoint import state_hash
    from nablaframe.views.jlens import (block_map, code_identity, jacobian_key, load_jacobian, save_jacobian,
                                        summed_jacobian)

    n_layers = run.config.n_layers
    fit = _common.fit_windows(run.task)
    state = run.state_sha256(run.final) if label == "trained" else state_hash(params)
    fit_sha, code, root = _batch_hash(fit), code_identity(), jlens_dir()
    taps = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers - 1)]
    recorded = None
    jacobians, records = {}, []
    for layer in range(n_layers - 1):
        key = jacobian_key(checkpoint_sha256=state, fit_sha256=fit_sha, layer=layer, code=code)
        started = time.monotonic()
        J = load_jacobian(root, key)
        reused = J is not None
        if J is None:
            if recorded is None:
                _, recorded = _common.p3.unintervened(run.model, params, fit, n_layers=n_layers, taps=taps)
            J = summed_jacobian(block_map(run.model, params, layer, n_layers), recorded[taps[layer]])
            save_jacobian(root, key, J)
        seconds = time.monotonic() - started
        jacobians[layer] = J
        records.append({"layer": layer, "key": key, "reused": reused, "seconds": seconds})
        print(f"{GATE_ID}: two-hop {label}: block {layer}'s J {'reused' if reused else 'fitted'} in {seconds:.1f} s,"
              f" {p1.rel(root / f'{key}.npy')}", flush=True)
    return jacobians, records


def lens_ranks(run: Any, params: Any, jacobians: dict[int, np.ndarray], windows: np.ndarray) -> dict[str, Any]:
    """The intermediate's and the answer's median rank in both lenses, at each query position and block output."""
    from nablaframe.views.jlens_view import jlens_lens
    from nablaframe.views.lens import lens_params, logit_lens
    from nablaframe.zoo.two_hop import QUERY_POSITIONS, token_ranks

    task, n_layers = run.task, run.config.n_layers
    taps = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    _, recorded = _common.logits(run.model, params, task, windows, n_layers=n_layers, taps=taps)
    ln_w, ln_b, wte = lens_params(params)
    intermediate = np.array([task.parse(w).b for w in windows])
    answer = np.asarray(windows)[:, -1]
    out: dict[str, Any] = {"logit_lens": {}, "jlens": {}}
    for position in QUERY_POSITIONS:
        resid = [recorded[tap][:, position] for tap in taps]
        readings = {
            "logit_lens": np.stack([logit_lens(r, ln_w, ln_b, wte) for r in resid]),
            "jlens": np.stack([logit_lens(r, ln_w, ln_b, wte) if layer == n_layers - 1
                               else jlens_lens(r, jacobians[layer], ln_w, ln_b, wte)
                               for layer, r in enumerate(resid)]),
        }
        for lens, reading in readings.items():
            b_ranks, c_ranks = token_ranks(reading, intermediate), token_ranks(reading, answer)
            out[lens][str(position)] = {str(layer): {"intermediate": float(np.median(b_ranks[layer])),
                                                     "answer": float(np.median(c_ranks[layer]))}
                                        for layer in range(n_layers)}
    return out


def coordinate_swap(run: Any, params: Any, jacobians: dict[int, np.ndarray], windows: np.ndarray) -> dict[str, Any]:
    """``jswap`` of the intermediate and the counterpart at the query, window by window, at each block's output."""
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.zoo.two_hop import QUERY_POSITIONS, aligned_sources

    task, n_layers = run.task, run.config.n_layers
    _, _, counterparts = aligned_sources(task, windows, np.random.default_rng([_common.SOURCE_SEED, 0]))
    parsed = [task.parse(w) for w in windows]
    successor = np.array([p.facts[int(k)] for p, k in zip(parsed, counterparts)])
    answer = np.asarray(windows)[:, -1]
    tokens = _common.tokens(task, windows)
    out: dict[str, Any] = {}
    for layer in range(n_layers):
        J = jacobians[layer] if layer < n_layers - 1 else np.eye(run.config.d_model, dtype=np.float32)
        objects = _common.p3.Objects()
        digest = objects.vector(J)
        tap = f"blocks.{layer}.hook_resid_post"
        answers = []
        progress = p1.Progress(f"jswap after block {layer}, windows", len(windows))
        for i, p in enumerate(parsed):
            spec = InterventionSpec((edit(tap, list(QUERY_POSITIONS), "jswap", J=digest, a=p.b,
                                          b=int(counterparts[i])),))
            logits, _ = _common.p3.run(run.model, params, tokens[i:i + 1], spec, n_layers=n_layers,
                                       resolve=objects.resolve)
            answers.append(int(np.argmax(logits[0, ANSWER_POSITION])))
            progress.advance()
        answers = np.array(answers)
        out[str(layer)] = {"swap": float(np.mean(answers == successor)), "unchanged": float(np.mean(answers == answer))}
        print(f"{GATE_ID}: jswap after block {layer}: the answer becomes d in {out[str(layer)]['swap']:.4f} and stays"
              f" c in {out[str(layer)]['unchanged']:.4f} of {len(windows)} windows", flush=True)
    return out


class _Recorded:
    """One pass's recorded taps, read as ``parts_from_trace`` reads a trace."""

    def __init__(self, recorded: dict[str, np.ndarray]) -> None:
        self.recorded = recorded

    def tensor(self, step: int, tap: str) -> np.ndarray:
        return self.recorded[tap]


def attribute(run: Any, params: Any, label: str) -> tuple[dict[str, float], float]:
    """Each part's mean contribution to the correct next token's logit at the mechanism positions, and the error."""
    from nablaframe.views.attribution import attribution, parts_from_trace
    from nablaframe.views.lens import lens_params

    task, n_layers = run.task, run.config.n_layers
    windows = _common.eval_windows(task)
    taps = ["hook_embed", "hook_pos_embed", f"blocks.{n_layers - 1}.hook_resid_post"]
    taps += [f"blocks.{layer}.{part}" for layer in range(n_layers) for part in ("hook_attn_out", "hook_mlp_out")]
    logits, recorded = _common.logits(run.model, params, task, windows, n_layers=n_layers, taps=taps)
    parts, final = parts_from_trace(_Recorded(recorded), run.final, n_layers)
    positions = task.mechanism_positions
    targets = np.asarray(windows)[:, positions + 1]
    ln_w, ln_b, wte = lens_params(params)
    contributions = attribution({name: part[:, positions] for name, part in parts.items()}, final[:, positions],
                                ln_w, ln_b, wte, targets)
    at_positions = np.asarray(logits, np.float64)[:, positions]
    target_logits = np.take_along_axis(at_positions, targets[..., None], axis=-1)[..., 0]
    error = float(np.abs(sum(contributions.values()) - target_logits).max())
    means = {name: float(np.mean(values)) for name, values in contributions.items()}
    top = max(means, key=lambda name: abs(means[name]))
    print(f"{GATE_ID}: attribution, {run.name} {label}: largest mean part {top} {means[top]:+.3f};"
          f" decomposition error {error:.2e}", flush=True)
    return means, error


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
