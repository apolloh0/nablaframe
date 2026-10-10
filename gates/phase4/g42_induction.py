"""G4.2: the induction model's attention view, ablation, component swap and patch (judged; D36).

Spec: docs/specs/phase-4.md, "Discrimination tests at model level"; helpers in ``_common.py``.
Report: ``reports/phase4/g42_induction.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's induction run at its final checkpoint, on ``eval_windows``, the model reading each
window's first 63 tokens, at the task's mechanism positions:

* **Attention:** ``induction_scores`` of ``blocks.1.attn.hook_pattern``, the tap the attention
  view reads, from ``instrumented``: ``attention_by_head`` and ``best_head_attention``.
  ``induction_heads`` lists the block-1 heads at or above ``BARS["induction_attention"]``. The
  twin's (``twin_params``) best head, ``twin_best_attention``, beside ``uniform_baseline`` on the
  same windows and positions.
* **Ablation:** every head in ``induction_heads`` zeroed together through
  ``blocks.1.attn.hook_z``, one ``zero`` edit with ``head`` per head in a single spec:
  ``ablation_fraction_removed``, the ``fraction_removed`` of mechanism accuracy, null when no
  head qualifies. Each of the eight heads zeroed alone with ``head_zero``:
  ``fraction_removed_by_head``, keyed ``"<layer>.<head>"``.
* **Component swap:** mechanism accuracy at every G4.1 checkpoint; ``swap_step`` is
  ``last_at_chance`` of them, and ``swap_fraction_removed`` the ``fraction_removed`` by a ``swap``
  of ``blocks.1.attn`` from that step, ``params_at`` loading G4.1's checkpoints.
* **Patch:** on ``induction_patch_windows``, for each ``j`` in ``PATCH_TARGETS``, sources from
  ``new_token_sources(windows, j, numpy.random.default_rng(j), 64)``, as the feasibility run drew
  them, patched at position ``j`` at ``blocks.0.hook_resid_pre``, ``blocks.0.hook_resid_post`` and
  ``blocks.1.hook_resid_post``. Each records how often the argmax at ``j + PATCH_LAG`` is the
  source's token at ``j``: ``patch_followed_by_tap``, a list per tap with one rate per target;
  ``patch_after_block0_min`` and ``patch_after_block1_max``.

Measurements: REQUIRED_MEASUREMENTS["G4.2"], ``mechanism_accuracy`` and ``chance`` among them.
Pass needs every bar. G4.1's induction run retraces the feasibility run for seed 0 at batch 32, so
on the same Mac these numbers should reproduce that run's (docs/deltas/phase-4-delta-3.md,
"Feasibility").
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g42_induction.py``. Takes --overwrite and --allow-dirty. It reads G4.1's
#   induction run, after ``require_g41`` and ``zoo_run``'s checks, and writes nothing outside
#   its report.
# * Every pass is ``run_intervened``: the unintervened ones with an empty spec, the twin's
#   included. The model reads each window's first 63 tokens; accuracy is ``next_token_accuracy``
#   against the full windows, at the task's mechanism positions, 32 to 62.
# * ``induction_heads`` holds head indices within block 1, in order. ``fraction_removed_by_head``
#   covers both blocks' heads. A ``fraction_removed`` that raises ValueError, because the
#   unedited accuracy is less than ``MIN_ABOVE_CHANCE`` above chance, is recorded as null, and
#   a null judged fraction fails the report.
# * The swap: mechanism accuracy at every checkpoint in G4.1's schedule, from each checkpoint's
#   own unintervened pass; ``last_at_chance`` of them, or null (and a fail) when no checkpoint
#   qualifies. Its pass swaps ``blocks.1.attn`` into the final checkpoint, with ``params_at``
#   loading G4.1's checkpoints.
# * The patch: one spec per target and tap, a ``patch`` at position ``j`` from the source
#   batch's first 63 tokens, read as the argmax at ``j + PATCH_LAG`` against the source's token
#   at ``j``. Recorded beside it, ``unpatched_followed``: how often the unpatched model's argmax
#   there is the window's own token at ``j``, the copy the patch moves.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.2"
REPORT_NAME = "g42_induction.json"
NAME = "induction"
PATTERN_TAP = "blocks.1.attn.hook_pattern"
ZERO_TAP = "blocks.1.attn.hook_z"
SWAP_COMPONENT = "blocks.1.attn"
PATCH_TAPS = ("blocks.0.hook_resid_pre", "blocks.0.hook_resid_post", "blocks.1.hook_resid_post")
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    run = _common.zoo_run(NAME, g41)
    measurements = measure(run)
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G4.2 rule holds."""
    def number(key: str) -> bool:
        return isinstance(m[key], (int, float)) and not isinstance(m[key], bool)

    bars = _common.BARS
    ok = (number("best_head_attention") and m["best_head_attention"] >= bars["induction_attention"]
          and number("twin_best_attention") and number("uniform_baseline")
          and m["twin_best_attention"] <= bars["twin_attention_ratio"] * m["uniform_baseline"]
          and number("ablation_fraction_removed") and m["ablation_fraction_removed"] >= bars["ablation_fraction"]
          and number("swap_fraction_removed") and m["swap_fraction_removed"] >= bars["swap_fraction"]
          and number("patch_after_block0_min") and m["patch_after_block0_min"] >= bars["patch_followed"]
          and number("patch_after_block1_max") and m["patch_after_block1_max"] <= bars["patch_after_read"])
    return "pass" if ok else "fail"


def measure(run: Any) -> dict[str, Any]:
    from nablaframe.zoo.ground_truth import TWIN_SEED

    task, step = run.task, run.final
    windows = _common.eval_windows(task)
    chance = task.chance
    params = run.params_for(step)
    mechanism = task.mechanism_positions
    print(f"{GATE_ID}: induction run {run.run_id}, final step {step:,}, {len(windows)} evaluation windows",
          flush=True)

    measured: dict[str, Any] = {"chance": chance, "step": step, "run_id": run.run_id, "bars": _common.BARS}
    measured |= attention(run, params, windows)
    base = measured["mechanism_accuracy"]
    measured |= ablation(run, params, windows, base, measured["induction_heads"])
    measured |= component_swap(run, params, windows, base)
    measured |= patch(run, params)
    measured |= {
        "mechanism_positions": [int(mechanism[0]), int(mechanism[-1])],
        "eval_windows": int(len(windows)),
        "eval_windows_sha256": _batch_hash(windows),
        "twin_seed": TWIN_SEED,
    }
    return measured


def accuracy(logits: np.ndarray, windows: np.ndarray, task: Any) -> float:
    from nablaframe.zoo.ground_truth import next_token_accuracy

    return next_token_accuracy(logits, windows, task.mechanism_positions)


def attention(run: Any, params: Any, windows: np.ndarray) -> dict[str, Any]:
    """The attention view's test: block 1's induction scores, the trained model's and the twin's."""
    from nablaframe.zoo.ground_truth import twin_params
    from nablaframe.zoo.induction import induction_scores, uniform_baseline

    task, n_layers = run.task, run.config.n_layers
    mechanism = task.mechanism_positions
    logits, recorded = _common.logits(run.model, params, task, windows, n_layers=n_layers, taps=[PATTERN_TAP])
    scores = induction_scores(recorded[PATTERN_TAP], windows, mechanism)
    _, twin_recorded = _common.logits(run.model, twin_params(run.config), task, windows, n_layers=n_layers,
                                      taps=[PATTERN_TAP])
    twin_scores = induction_scores(twin_recorded[PATTERN_TAP], windows, mechanism)
    heads = [int(h) for h in np.flatnonzero(scores >= _common.BARS["induction_attention"])]
    uniform = uniform_baseline(windows, mechanism)
    mechanism_accuracy = accuracy(logits, windows, task)
    print(f"{GATE_ID}: mechanism accuracy {mechanism_accuracy:.4f}; block 1's attention to induction targets by head"
          f" {', '.join(f'{s:.3f}' for s in scores)}; induction heads {heads or 'none'}; the twin's best"
          f" {twin_scores.max():.4f} against a uniform {uniform:.4f}", flush=True)
    return {
        "mechanism_accuracy": mechanism_accuracy,
        "attention_by_head": [float(s) for s in scores],
        "best_head_attention": float(scores.max()),
        "induction_heads": heads,
        "twin_attention_by_head": [float(s) for s in twin_scores],
        "twin_best_attention": float(twin_scores.max()),
        "uniform_baseline": uniform,
    }


def _fraction(before: float, after: float, chance: float) -> float | None:
    try:
        return _common.fraction_removed(before, after, chance)
    except ValueError:
        return None


def ablation(run: Any, params: Any, windows: np.ndarray, base: float, heads: list[int]) -> dict[str, Any]:
    """Every induction head zeroed together, then each head of both blocks zeroed alone."""
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.zoo.induction import head_zero

    task, n_layers, chance = run.task, run.config.n_layers, run.task.chance

    def edited_accuracy(spec: Any) -> float:
        logits, _ = _common.p3.run(run.model, params, _common.tokens(task, windows), spec, n_layers=n_layers,
                                   resolve=_no_objects)
        return accuracy(logits, windows, task)

    together, removed = None, None
    if heads:
        together = edited_accuracy(InterventionSpec(tuple(edit(ZERO_TAP, "all", "zero", head=h) for h in heads)))
        removed = _fraction(base, together, chance)
    alone, by_head = {}, {}
    progress = p1.Progress("zeroing heads one at a time", n_layers * run.config.n_heads)
    for layer in range(n_layers):
        for head in range(run.config.n_heads):
            key = f"{layer}.{head}"
            alone[key] = edited_accuracy(head_zero(layer, head))
            by_head[key] = _fraction(base, alone[key], chance)
            progress.advance()
    shown = "no head meets the attention bar" if not heads else (
        f"zeroing heads {heads} of block 1 together leaves {together:.4f}, removing"
        f" {'null' if removed is None else f'{removed:.3f}'}")
    print(f"{GATE_ID}: {shown}; alone: "
          + ", ".join(f"{k} {'null' if v is None else f'{v:.3f}'}" for k, v in by_head.items()), flush=True)
    return {
        "ablation_fraction_removed": removed,
        "ablation_accuracy": together,
        "fraction_removed_by_head": by_head,
        "accuracy_by_head_zeroed": alone,
    }


def component_swap(run: Any, params: Any, windows: np.ndarray, base: float) -> dict[str, Any]:
    """Block 1's attention swapped in from the last checkpoint whose mechanism accuracy is at chance."""
    from nablaframe.interventions.spec import InterventionSpec, edit

    task, n_layers, chance = run.task, run.config.n_layers, run.task.chance
    curve = []
    progress = p1.Progress("mechanism accuracy at each checkpoint", len(run.steps))
    for step in run.steps:
        logits, _ = _common.logits(run.model, run.params_for(step), task, windows, n_layers=n_layers)
        curve.append(accuracy(logits, windows, task))
        progress.advance()
    try:
        swap_step = _common.last_at_chance(run.steps, curve, chance)
    except ValueError as exc:
        print(f"{GATE_ID}: no swap: {exc}", flush=True)
        swap_step = None
    swapped, removed = None, None
    if swap_step is not None:
        spec = InterventionSpec((edit(SWAP_COMPONENT, "all", "swap", step=swap_step),))
        logits, _ = _common.p3.run(run.model, params, _common.tokens(task, windows), spec, n_layers=n_layers,
                                   resolve=_no_objects, params_at=run.params_for)
        swapped = accuracy(logits, windows, task)
        removed = _fraction(base, swapped, chance)
        print(f"{GATE_ID}: block 1's attention from step {swap_step:,} leaves {swapped:.4f}, removing"
              f" {'null' if removed is None else f'{removed:.3f}'}", flush=True)
    return {
        "swap_step": swap_step,
        "swap_fraction_removed": removed,
        "swap_accuracy": swapped,
        "mechanism_accuracy_by_step": [{"step": s, "accuracy": a} for s, a in zip(run.steps, curve)],
    }


def patch(run: Any, params: Any) -> dict[str, Any]:
    """A new token at each copy target, patched in at three taps, read PATCH_LAG positions on."""
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.zoo.induction import new_token_sources

    task, n_layers = run.task, run.config.n_layers
    windows = _common.induction_patch_windows(task)
    tokens = _common.tokens(task, windows)
    base, _ = _common.p3.unintervened(run.model, params, tokens, n_layers=n_layers)
    followed: dict[str, list[float]] = {tap: [] for tap in PATCH_TAPS}
    unpatched, reads = [], []
    progress = p1.Progress("patches", len(_common.PATCH_TARGETS) * len(PATCH_TAPS))
    for j in _common.PATCH_TARGETS:
        q = j + _common.PATCH_LAG
        sources = new_token_sources(windows, j, np.random.default_rng(j), task.vocab_size)
        objects = _common.p3.Objects()
        source = objects.batch(_common.tokens(task, sources))
        unpatched.append(float(np.mean(np.argmax(base[:, q], axis=-1) == windows[:, j])))
        reads.append(q)
        for tap in PATCH_TAPS:
            spec = InterventionSpec((edit(tap, [j], "patch", source=source),))
            logits, _ = _common.p3.run(run.model, params, tokens, spec, n_layers=n_layers, resolve=objects.resolve)
            followed[tap].append(float(np.mean(np.argmax(logits[:, q], axis=-1) == sources[:, j])))
            progress.advance()
    after0, after1 = followed["blocks.0.hook_resid_post"], followed["blocks.1.hook_resid_post"]
    print(f"{GATE_ID}: the patch followed after block 0 {after0}, after block 1 {after1}", flush=True)
    return {
        "patch_followed_by_tap": followed,
        "patch_after_block0_min": min(after0),
        "patch_after_block1_max": max(after1),
        "patch_targets": list(_common.PATCH_TARGETS),
        "patch_read_positions": reads,
        "patch_period": _common.PATCH_PERIOD,
        "unpatched_followed": unpatched,
        "patch_windows": int(len(windows)),
        "patch_windows_sha256": _batch_hash(windows),
    }


def _no_objects(digest: str) -> np.ndarray:
    raise KeyError(f"this spec names no object, but {digest} was asked for")


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
