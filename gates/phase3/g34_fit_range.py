"""G3.4: both lenses' meters on D17's fit range only, at every trace step. Recorded, not judged.

The second question D21 left: how much the positions J's fit excludes weigh in the meter.
For every trace step, the logit lens's and the J-lens's ``lens_meter`` restricted to
``fit_range`` positions, beside the full-probe meters already recorded. No intervention is
needed. Measurement: ``meters_by_step``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase3/g34_fit_range.py``. Takes --overwrite and --allow-dirty. It reads G1.6's
#   store after G2.1 has added the logits (checked as the Phase 2 gates check it), G1.4's
#   trace checkpoints for each step's final layer norm and unembedding, and G2.9's J's,
#   refusing to start (exit 1, no report) unless G2.9's report passes and every J is in its
#   cache. It writes nothing outside its report.
# * No model pass runs: the residual streams and the model's logits are the trace's, as the
#   Phase 2 views decode them.
# * At each step: the logit lens's readings are ``lens_readings`` (the stream entering block 0,
#   then every block's output) and the J-lens's are ``jlens_readings`` with that step's J's.
#   Each lens's ``lens_meter`` against the traced logits is taken twice: on every probe
#   position (``full``, as G2.3 and G2.10 record it) and on ``fit_range(seq)`` only
#   (``fit_range``), positions 16 to T - 2 of every probe sequence.
# * ``meters_by_step`` maps each step, as a string, to ``{"logit_lens": {"full": [...],
#   "fit_range": [...]}, "jlens": {...}}``, each list one ``lens_meter`` row per reading. A
#   NaN or infinite value is recorded as null. ``positions_full`` and ``positions_fit_range``
#   count the positions each meter averages over.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G3.4"
REPORT_NAME = "g34_fit_range.json"
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    jacobians = _common.jacobians(ref, p1.load_split("train"), ref.steps)
    probe = _common.probe(p1.load_split("validation"), ref.config.seq)
    reader = _common.p2.open_reader(_common.p2.g16_store(), ref.steps, probe)
    measurements = measure(reader, ref.steps, ref.params_for, jacobians, ref.config.n_layers, ref.config.seq)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """G3.4 records, not judges: always pass, as reports.py's rule requires."""
    return "pass"


def measure(reader: Any, steps: tuple[int, ...], params_for: Any, jacobians: Any, n_layers: int,
            seq: int) -> dict[str, Any]:
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.jlens_view import jlens_readings
    from nablaframe.views.lens import lens_meter, lens_readings

    fit = _common.fit_range(seq)
    meters_by_step = {}
    progress = p1.Progress("fit-range meters, steps", len(steps))
    batch = None
    for step in steps:
        params = params_for(step)
        logits = reader.tensor(step, LOGITS)
        batch = logits.shape[0]
        readings = {"logit_lens": lens_readings(reader, step, params, n_layers),
                    "jlens": jlens_readings(reader, step, params, jacobians(step), n_layers)}
        meters_by_step[str(step)] = {
            lens: {"full": _finite(lens_meter(r, logits)),
                   "fit_range": _finite(lens_meter(r[:, :, fit], logits[:, fit]))}
            for lens, r in readings.items()
        }
        del readings
        progress.advance()
    final = meters_by_step[str(steps[-1])]
    for lens in ("logit_lens", "jlens"):
        print(f"{GATE_ID}: at step {steps[-1]}, the {lens.replace('_', ' ')}'s KL per reading, full probe against"
              " fit range: " + ", ".join(f"{_show(a['kl_mean'])}/{_show(b['kl_mean'])}"
                                          for a, b in zip(final[lens]["full"], final[lens]["fit_range"])), flush=True)
    return {
        "meters_by_step": meters_by_step,
        "fit_range": [int(fit[0]), int(fit[-1])],
        "positions_full": int(batch * seq),
        "positions_fit_range": int(batch * len(fit)),
        "steps": list(steps),
    }


def _finite(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each row with its non-finite numbers as None, since reports allow only finite numbers."""
    return [{key: value if key == "layer" else p1.finite_or_none(value) for key, value in row.items()} for row in rows]


def _show(value: float | None) -> str:
    return "null" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
