"""Gate G0.6: Mode probe.

Spec: docs/specs/phase-0.md, Gates table, G0.6 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: nablaframe.lanes.jax_ref.toy.ToyGPT under jit: logits only vs logits plus all taps, on 3 seeds.
Pass: Always passes: records bitwise equality or the measured max abs diff.
The result sets the oracle policy through a D-entry.

Writes reports/phase0/g06_mode_probe.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.6"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures:
#
# * For each seed, parameters come from ToyGPT.init(seed) and the tokens are a
#   BATCH x SEQ batch drawn uniformly from the vocabulary with numpy's
#   default_rng(TOKEN_SEED_OFFSET + seed).
# * Each seed compares ToyGPT.fast against ToyGPT.instrumented with every canonical
#   tap, through the oracle's own compare_logits under the bitwise policy, so
#   "bitwise equal" means what the oracle means by it (bit patterns, -0.0 != 0.0).
# * bitwise_equal is true only if all seeds are; max_abs_logit_diff is the largest
#   over the seeds, 0.0 when all are equal, and null if any logit was not finite.
# * This gate chooses no policy and pins no tolerance: that takes a D-entry.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

GATE_ID = "G0.6"
REPORT_PATH = "reports/phase0/g06_mode_probe.json"

ENV_NAME = "nf-jax"
SEEDS = (0, 1, 2)
BATCH = 8
TOKEN_SEED_OFFSET = 1000
VERSIONS = ("jax", "jaxlib", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)

    measurements = measure()
    measurements["versions"] = _common.package_versions(VERSIONS)
    report = _common.build_report(GATE_ID, "pass", measurements, ENV_NAME)
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def measure(seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    import jax
    import numpy as np

    from nablaframe.lanes.jax_ref import toy
    from nablaframe.oracle.mode_equality import compare_logits

    model = toy.ToyGPT()
    per_seed = []
    for seed in seeds:
        params = model.init(seed)
        tokens = np.random.default_rng(TOKEN_SEED_OFFSET + seed).integers(0, toy.VOCAB, size=(BATCH, toy.SEQ))
        fast = np.asarray(model.fast(params, tokens))
        instrumented, recorded = model.instrumented(params, tokens, toy.TAPS)
        if set(recorded) != set(toy.TAPS):
            raise RuntimeError(f"instrumented mode recorded {sorted(recorded)}, not every canonical tap")
        result = compare_logits(fast, np.asarray(instrumented), "bitwise")
        per_seed.append({
            "seed": seed,
            "bitwise_equal": result.passed,
            "max_abs_logit_diff": _common.finite_or_none(result.max_abs_diff),
            "n_mismatched": result.n_mismatched,
            "reason": result.reason,
        })

    diffs = [entry["max_abs_logit_diff"] for entry in per_seed]
    return {
        "bitwise_equal": all(entry["bitwise_equal"] for entry in per_seed),
        "max_abs_logit_diff": None if None in diffs else max(diffs),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "taps": len(toy.TAPS),
        "tokens_shape": [BATCH, toy.SEQ],
        "logits_dtype": str(fast.dtype),
        "model": {"n_layers": toy.N_LAYERS, "d_model": toy.D_MODEL, "n_heads": toy.N_HEADS,
                  "d_mlp": toy.D_MLP, "vocab": toy.VOCAB, "seq": toy.SEQ},
        "backend": jax.default_backend(),
    }


if __name__ == "__main__":
    sys.exit(main())
