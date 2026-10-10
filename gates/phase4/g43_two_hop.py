"""G4.3: the two-hop aligned patch, the signature demo (D36). Judged on its distractor control; the swap and transplant profiles are recorded.

Spec: docs/specs/phase-4.md, "Patch, on the two-hop model"; helpers in ``_common.py``. Report:
``reports/phase4/g43_two_hop.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's two-hop run at its final checkpoint, on ``eval_windows``:
``aligned_sources(task, windows, numpy.random.default_rng([SOURCE_SEED, 0]))`` gives each window
its aligned source, its control and its counterpart. ``unpatched_accuracy``: how often the
unpatched model's answer, the argmax at position 25, is ``c``. Then for each site in
``TWO_HOP_SITES`` and each block's ``hook_resid_post``, two patches through ``run_intervened`` at
the site's positions, one from the sources and one from the controls, each read at position 25:

* ``swap_by_site`` and ``transplant_by_site``: ``patch_outcomes`` of the source patch, as
  ``{site: [rate per block]}``
* ``control_unchanged_by_site``: how often the control patch's answer equals the unpatched
  model's own answer, in the same shape; ``control_unchanged_min``, its minimum over every site
  and block
* ``best_swap``: ``{"site", "block", "rate"}`` for the largest swap rate, ties going to the
  earlier site in ``TWO_HOP_SITES`` order, then the earlier block

Measurements: REQUIRED_MEASUREMENTS["G4.3"]. Also recorded: how often each source patch leaves
the answer ``c``. Pass needs ``control_unchanged_min >= BARS["control_unchanged"]``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g43_two_hop.py``. Takes --overwrite and --allow-dirty. It reads G4.1's
#   two-hop run, after ``require_g41`` and ``zoo_run``'s checks, and writes nothing outside its
#   report.
# * Every pass is ``run_intervened``, the unpatched one with an empty spec. The model reads
#   each window's first 26 tokens; the answer is the argmax of the logits at position 25, ties
#   to the lowest id. Each patch is one ``patch`` edit at the site's positions of
#   ``blocks.<l>.hook_resid_post``, from the sources' (or the controls') first 26 tokens, which
#   line up window by window with the targets.
# * ``patch_outcomes`` reads the full target and source windows: a swap is the answer
#   becoming the counterpart's object in the target, a transplant the answer becoming ``d′``.
# * Sites and blocks keep ``TWO_HOP_SITES``' order and block order in every profile, each
#   profile ``{site: [rate for block 0, 1, 2, 3]}``. ``answer_kept_by_site`` is how often each
#   source patch leaves the answer ``c``.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.3"
REPORT_NAME = "g43_two_hop.json"
NAME = "two_hop"
ANSWER_POSITION = 25
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
    """pass exactly when reports.py's G4.3 rule holds."""
    value = m["control_unchanged_min"]
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value >= _common.BARS["control_unchanged"]
    return "pass" if ok else "fail"


def measure(run: Any) -> dict[str, Any]:
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.zoo.two_hop import aligned_sources, patch_outcomes

    task, n_layers, step = run.task, run.config.n_layers, run.final
    params = run.params_for(step)
    windows = _common.eval_windows(task)
    sources, controls, counterparts = aligned_sources(task, windows, np.random.default_rng([_common.SOURCE_SEED, 0]))
    tokens = _common.tokens(task, windows)
    objects = _common.p3.Objects()
    source_hash = objects.batch(_common.tokens(task, sources))
    control_hash = objects.batch(_common.tokens(task, controls))
    answers = answer(_common.p3.unintervened(run.model, params, tokens, n_layers=n_layers)[0])
    c = windows[:, -1]
    unpatched_accuracy = float(np.mean(answers == c))
    print(f"{GATE_ID}: two-hop run {run.run_id}, final step {step:,}: the unpatched answer is c in"
          f" {unpatched_accuracy:.4f} of {len(windows)} windows", flush=True)

    swap: dict[str, list[float]] = {}
    transplant: dict[str, list[float]] = {}
    unchanged: dict[str, list[float]] = {}
    kept: dict[str, list[float]] = {}
    progress = p1.Progress("patches", 2 * len(_common.TWO_HOP_SITES) * n_layers)
    for site, positions in _common.TWO_HOP_SITES.items():
        for profile in (swap, transplant, unchanged, kept):
            profile[site] = []
        for layer in range(n_layers):
            tap = f"blocks.{layer}.hook_resid_post"
            patched = {}
            for label, digest in (("source", source_hash), ("control", control_hash)):
                spec = InterventionSpec((edit(tap, list(positions), "patch", source=digest),))
                logits, _ = _common.p3.run(run.model, params, tokens, spec, n_layers=n_layers,
                                           resolve=objects.resolve)
                patched[label] = answer(logits)
                progress.advance()
            outcomes = patch_outcomes(patched["source"], windows, sources, counterparts)
            swap[site].append(outcomes["swap_rate"])
            transplant[site].append(outcomes["transplant_rate"])
            kept[site].append(float(np.mean(patched["source"] == c)))
            unchanged[site].append(float(np.mean(patched["control"] == answers)))
        print(f"{GATE_ID}: {site}: swap {_rates(swap[site])}, transplant {_rates(transplant[site])},"
              f" control unchanged {_rates(unchanged[site])}", flush=True)

    best = None
    for site in _common.TWO_HOP_SITES:
        for layer, rate in enumerate(swap[site]):
            if best is None or rate > best["rate"]:
                best = {"site": site, "block": layer, "rate": rate}
    control_min = min(rate for rates in unchanged.values() for rate in rates)
    print(f"{GATE_ID}: the control leaves the answer unchanged at least {control_min:.4f} of the time; the largest"
          f" swap is {best['rate']:.4f}, at {best['site']} after block {best['block']}", flush=True)
    return {
        "unpatched_accuracy": unpatched_accuracy,
        "swap_by_site": swap,
        "transplant_by_site": transplant,
        "control_unchanged_by_site": unchanged,
        "control_unchanged_min": control_min,
        "best_swap": best,
        "answer_kept_by_site": kept,
        "sites": {site: list(positions) for site, positions in _common.TWO_HOP_SITES.items()},
        "blocks": list(range(n_layers)),
        "answer_position": ANSWER_POSITION,
        "source_seed": _common.SOURCE_SEED,
        "step": step,
        "run_id": run.run_id,
        "bars": _common.BARS,
        "eval_windows": int(len(windows)),
        "eval_windows_sha256": _batch_hash(windows),
        "sources_sha256": _batch_hash(sources),
        "controls_sha256": _batch_hash(controls),
    }


def answer(logits: np.ndarray) -> np.ndarray:
    """The argmax at the answer position, ties to the lowest id."""
    return np.argmax(np.asarray(logits)[:, ANSWER_POSITION], axis=-1)


def _rates(rates: list[float]) -> str:
    return "[" + ", ".join(f"{rate:.3f}" for rate in rates) + "]"


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
