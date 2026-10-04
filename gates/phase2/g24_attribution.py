"""G2.4: direct logit attribution (D14).

* Planted, at decoder level, at size S: two equally strong parts, one along each target
  token's direction and one exactly perpendicular to it, as in
  ``tests/test_views_attribution.py``. ``planted_aligned_credit`` is the fraction of
  positions where the aligned part's contribution is positive;
  ``planted_perpendicular_quiet`` the fraction where the perpendicular part's magnitude
  is at most 1e-3 of the aligned part's.
* The decomposition error on the reference run's probe at every trace step:
  ``decomposition_error_by_step`` and its maximum, ``decomposition_error_max``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g24_attribution.py``. Takes --overwrite and --allow-dirty. It reads
#   G1.6's store after G2.1 has added the logits, and writes nothing outside its report.
# * Planted: the construction of tests/test_views_attribution.py, with size S's shape
#   (d_model 256, vocabulary 4,096, from G1.4's manifest) and ``PLANTED_POSITIONS``
#   (2,048, the probe's count) positions. As in that test, the unembedding is standard
#   normal and the layer norm is the identity (weight 1, bias 0), all drawn with
#   ``default_rng(0)``: the aligned part is 6 times the target's embedding row, the
#   perpendicular part is a random vector, centered and made orthogonal to the centered
#   row, then scaled to the aligned part's norm, and a third part is small noise.
# * Decomposition error: ``attribution_from_trace(reader, step, params, probe,
#   n_layers)`` at every trace step, with that step's parameters. It checks the probe
#   against the events' batch hash. Keys are steps as strings. A step whose error is NaN
#   or infinite is recorded as null and listed in ``decomposition_error_nonfinite_steps``,
#   and the maximum is then null, which fails the gate: reports hold finite numbers only.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.4"
REPORT_NAME = "g24_attribution.json"
PLANTED_SEED = 0
PLANTED_POSITIONS = 2_048
QUIET_RATIO = 1e-3
MIN_PLANTED = 0.99
MAX_DECOMPOSITION_ERROR = 1e-3
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    probe = p1.probe(p1.load_split("validation"), ref.config.seq)
    reader = _common.open_reader(_common.g16_store(), ref.steps, probe)

    measurements = planted(ref.config.d_model, ref.config.vocab)
    measurements |= decomposition(reader, ref.steps, ref.params_for, probe, ref.config.n_layers)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.4 rule holds."""
    ok = (m["planted_aligned_credit"] is not None and m["planted_aligned_credit"] >= MIN_PLANTED
          and m["planted_perpendicular_quiet"] is not None and m["planted_perpendicular_quiet"] >= MIN_PLANTED
          and m["decomposition_error_max"] is not None
          and m["decomposition_error_max"] <= MAX_DECOMPOSITION_ERROR)
    return "pass" if ok else "fail"


def planted(d: int, vocab: int, *, positions: int = PLANTED_POSITIONS) -> dict[str, Any]:
    """Two equally strong parts, one along each target's direction and one perpendicular to it."""
    from nablaframe.views.attribution import attribution

    rng = np.random.default_rng(PLANTED_SEED)
    wte = rng.standard_normal((vocab, d)).astype(np.float32)
    targets = rng.integers(0, vocab, size=(1, positions))
    u = wte[targets].astype(np.float64)
    uc = u - u.mean(-1, keepdims=True)
    r = rng.standard_normal((1, positions, d))
    r -= r.mean(-1, keepdims=True)
    r -= (r * uc).sum(-1, keepdims=True) / (uc * uc).sum(-1, keepdims=True) * uc  # perpendicular to the target
    aligned = 6.0 * u
    perpendicular = r * (np.linalg.norm(aligned, axis=-1, keepdims=True) / np.linalg.norm(r, axis=-1, keepdims=True))
    parts = {"aligned": aligned.astype(np.float32), "perpendicular": perpendicular.astype(np.float32),
             "quiet": (0.01 * rng.standard_normal((1, positions, d))).astype(np.float32)}
    out = attribution(parts, sum(parts.values()), np.ones(d, np.float32), np.zeros(d, np.float32), wte, targets)
    aligned_credit = float((out["aligned"] > 0).mean())
    ratio = np.abs(out["perpendicular"]) / np.abs(out["aligned"])
    quiet = float((np.abs(out["perpendicular"]) <= QUIET_RATIO * np.abs(out["aligned"])).mean())
    print(f"{GATE_ID}: planted: aligned part credited at {aligned_credit:.4f} of positions; perpendicular part"
          f" quiet at {quiet:.4f}", flush=True)
    return {
        "planted_aligned_credit": aligned_credit,
        "planted_perpendicular_quiet": quiet,
        "planted_positions": positions,
        "planted_d_model": d,
        "planted_vocab": vocab,
        "planted_seed": PLANTED_SEED,
        "planted_quiet_ratio": QUIET_RATIO,
        "planted_perpendicular_ratio_max": float(ratio.max()),
        "planted_perpendicular_ratio_median": float(np.median(ratio)),
    }


def decomposition(reader: Any, steps: tuple[int, ...], params_for: Any, probe: np.ndarray,
                  n_layers: int) -> dict[str, Any]:
    """``attribution_from_trace``'s decomposition error at every trace step."""
    from nablaframe.views.attribution import attribution_from_trace

    by_step = {}
    progress = p1.Progress("attribution, steps", len(steps))
    for step in steps:
        _, error = attribution_from_trace(reader, step, params_for(step), probe, n_layers)
        by_step[str(step)] = p1.finite_or_none(error)
        progress.advance()
    nonfinite = [int(step) for step, error in by_step.items() if error is None]
    if nonfinite:
        print(f"{GATE_ID}: the decomposition error is not finite at steps {nonfinite}", flush=True)
        return {"decomposition_error_by_step": by_step, "decomposition_error_max": None,
                "decomposition_error_max_step": None, "decomposition_error_nonfinite_steps": nonfinite}
    worst = max(by_step, key=lambda step: by_step[step])
    print(f"{GATE_ID}: the largest decomposition error is {by_step[worst]:.3e}, at step {worst}", flush=True)
    return {"decomposition_error_by_step": by_step, "decomposition_error_max": by_step[worst],
            "decomposition_error_max_step": int(worst), "decomposition_error_nonfinite_steps": []}


if __name__ == "__main__":
    raise SystemExit(main())
