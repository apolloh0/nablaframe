"""G3.1: D25's exactness identities at size S, on the reference run (G3.1 is what every effect's meter cites).

At the final trace checkpoint, on G1.6's probe, every identity bit for bit: a null edit (no
positions) and steering at zero at every editable tap; a self-swap of every component; a
full patch of the last block's output from a second batch of 8 validation windows; and a full
swap of every component from step 1,024. Measurements: ``identities_checked``,
``identities_exact``, and ``identities_by_kind`` mapping each kind to its count exact.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase3/g31_identities.py``. Takes --overwrite and --allow-dirty. It reads G1.1's
#   data and G1.4's trace checkpoints, and writes nothing outside its report.
# * Every pass, the unintervened ones included, is ``run_intervened`` (an empty spec for an
#   unintervened pass), requesting every canonical tap, ``tap_names(n_layers)``.
# * Bit for bit means equal dtype, equal shape and ``numpy.array_equal``, as the package's
#   own tests compare. An identity that reproduces "the unintervened pass" or "B's own pass"
#   (null edits, steering at zero, self-swaps, the full swap) must match its logits and every
#   recorded tap. The full patch must match the source's logits, which is what D25 asks of
#   it: the taps before the patched one still see the probe.
# * Null edits: one per editable tap (``editable_taps``), a ``zero`` edit at no positions.
# * Steering at zero: one per editable tap, an ``add`` at every position with ``alpha`` 0.0
#   and a nonzero vector of the tap's per-position shape, standard-normal from
#   ``default_rng(VECTOR_SEED)`` (0), drawn tap by tap in ``editable_taps`` order.
# * Self-swaps: every component (``embeddings``, ``ln_final``, and each block whole, its
#   attention and its MLP) swapped in from the final step itself.
# * The full patch: ``blocks.<n - 1>.hook_resid_post`` at every position from the patch
#   source, a second batch of 8 validation windows drawn as the probe is but with seed 1,
#   against the source's own unintervened logits.
# * The full swap: one spec swapping ``embeddings``, every block and ``ln_final`` in from step
#   ``SWAP_STEP`` (1,024), against that checkpoint's own unintervened pass.
# * Recorded beside them, to show no identity holds vacuously: ``swap_step_differs`` (step
#   1,024's logits differ from the final step's) and ``patch_source_differs`` (the source's
#   logits differ from the probe's). ``failures`` names each identity that isn't exact, with
#   the arrays that differ and the largest absolute difference among them.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G3.1"
REPORT_NAME = "g31_identities.json"
SWAP_STEP = 1_024
VECTOR_SEED = 0
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    if SWAP_STEP not in ref.steps:
        raise SystemExit(f"step {SWAP_STEP} is not one of G1.4's trace checkpoints. No report written.")
    validation = p1.load_split("validation")
    probe = _common.probe(validation, ref.config.seq)
    source = _common.patch_source(validation, ref.config.seq)
    measurements = measure(ref.model, _common.Checkpoints(ref), ref.steps[-1], ref.config.n_layers, probe, source)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G3.1 rule holds."""
    return "pass" if m["identities_exact"] == m["identities_checked"] else "fail"


def components(n_layers: int) -> list[str]:
    """Every swappable component: the embeddings, the final layer norm, and each block whole and by part."""
    blocks = [name for layer in range(n_layers)
              for name in (f"blocks.{layer}", f"blocks.{layer}.attn", f"blocks.{layer}.mlp")]
    return ["embeddings", "ln_final", *blocks]


def measure(model: Any, params_at: Any, step: int, n_layers: int, probe: np.ndarray, source: np.ndarray, *,
            swap_step: int = SWAP_STEP) -> dict[str, Any]:
    from nablaframe.interventions.apply import editable_taps
    from nablaframe.interventions.effects import IDENTITIES
    from nablaframe.interventions.spec import InterventionSpec, edit
    from nablaframe.trace.registry import tap_names

    taps = list(tap_names(n_layers))
    objects = _common.Objects()
    params = params_at(step)

    def run(tokens: np.ndarray, *edits: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return _common.run(model, params, tokens, InterventionSpec(tuple(edits)), n_layers=n_layers,
                           resolve=objects.resolve, params_at=params_at, taps=taps)

    base = _common.unintervened(model, params, probe, n_layers=n_layers, taps=taps)
    source_logits, _ = _common.unintervened(model, params, source, n_layers=n_layers)
    other = _common.unintervened(model, params_at(swap_step), probe, n_layers=n_layers, taps=taps)

    # Each identity: (kind, label, the edits, what it must reproduce, whether the taps must match too).
    rng = np.random.default_rng(VECTOR_SEED)
    checks: list[tuple[str, str, tuple[Any, ...], Any, bool]] = []
    for tap in editable_taps(n_layers):
        checks.append(("null edit", tap, (edit(tap, [], "zero"),), base, True))
    for tap in editable_taps(n_layers):
        vector = rng.standard_normal(base[1][tap].shape[2:]).astype(np.float32)
        checks.append(("steering at zero", tap,
                       (edit(tap, "all", "add", vector=objects.vector(vector), alpha=0.0),), base, True))
    for component in components(n_layers):
        checks.append(("self-swap", component, (edit(component, "all", "swap", step=step),), base, True))
    last = f"blocks.{n_layers - 1}.hook_resid_post"
    checks.append(("full patch", last, (edit(last, "all", "patch", source=objects.batch(source)),),
                   (source_logits, {}), False))
    every = ["embeddings", *(f"blocks.{layer}" for layer in range(n_layers)), "ln_final"]
    checks.append(("full swap", f"{', '.join(every)} from step {swap_step}",
                   tuple(edit(component, "all", "swap", step=swap_step) for component in every), other, True))

    exact_by_kind = {kind: 0 for kind in IDENTITIES}
    checked_by_kind = {kind: 0 for kind in IDENTITIES}
    failures = []
    progress = p1.Progress("identities", len(checks))
    for kind, label, edits, (want_logits, want_taps), with_taps in checks:
        logits, recorded = run(probe, *edits)
        differing = _differing({"logits": want_logits, **(want_taps if with_taps else {})},
                               {"logits": logits, **(recorded if with_taps else {})})
        checked_by_kind[kind] += 1
        if differing:
            failures.append({"kind": kind, "identity": label, "differing": sorted(differing),
                             "max_abs_diff": p1.finite_or_none(max(differing.values()))})
        else:
            exact_by_kind[kind] += 1
        progress.advance()

    checked, exact = sum(checked_by_kind.values()), sum(exact_by_kind.values())
    print(f"{GATE_ID}: {exact} of {checked} identities bit for bit: "
          + ", ".join(f"{kind} {exact_by_kind[kind]}/{checked_by_kind[kind]}" for kind in IDENTITIES), flush=True)
    for failure in failures[:10]:
        print(f"{GATE_ID}: not exact: {failure['kind']} at {failure['identity']}: {', '.join(failure['differing'])}",
              flush=True)
    return {
        "identities_checked": checked,
        "identities_exact": exact,
        "identities_by_kind": exact_by_kind,
        "identities_checked_by_kind": checked_by_kind,
        "failures": failures,
        "step": step,
        "swap_step": swap_step,
        "swap_step_differs": not np.array_equal(other[0], base[0]),
        "patch_source_differs": not np.array_equal(source_logits, base[0]),
        "taps_compared": len(taps),
        "vector_seed": VECTOR_SEED,
        "probe_sha256": _batch_hash(probe),
        "patch_source_sha256": _batch_hash(source),
    }


def _differing(want: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> dict[str, float]:
    """The names whose arrays aren't bit for bit equal, each with its largest absolute difference."""
    out = {}
    for name, expected in want.items():
        actual = got.get(name)
        if actual is None or actual.dtype != expected.dtype or actual.shape != expected.shape:
            out[name] = float("inf")
        elif not np.array_equal(actual, expected):
            out[name] = float(np.nanmax(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    return out


def _batch_hash(tokens: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
