"""G3.2: determinism and caching of intervention results (D24).

A seeded sample of 12 specs, covering every operation, run on the reference run's final
checkpoint and probe through the command line's code twice, and once more after deleting the
results folder. Measurements: ``sample_size``, ``recompute_identical`` (how many gave
byte-identical results every time) and ``cache_keys_distinct`` (every spec, checkpoint and
batch gave a different key, and the same ones the same key).
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase3/g32_determinism.py``. Takes --overwrite and --allow-dirty. Its working
#   folder is runs/phase3/g32_determinism/, started fresh; --overwrite lets it replace one.
# * The command line's code is ``nablaframe.interventions.cli.main``, called in this process
#   with the files it reads: the config from G1.4's manifest, the probe as ``tokens.npy``,
#   ``objects/`` holding what the specs' hashes name, one JSON per spec, and G1.4's trace
#   checkpoints folder. Each run asks for the last block's output as a tap, so the payload's
#   tap changes are covered too. Its passes go through ``run_intervened``.
# * The sample: ``default_rng(SAMPLE_SEED)`` (0) shuffles every operation twice over, then
#   draws each spec's tap (an editable tap, or a component for ``swap``), its positions
#   (``"all"``, a sorted list of one to four positions, or a non-empty range; ``"all"`` for a
#   swap) and its arguments: D22's reference batch for ``mean`` and ``resample`` with a drawn
#   seed, the patch source (8 validation windows drawn as the probe is, with seed 1) for
#   ``patch``, a standard-normal vector of the tap's per-position shape and an ``alpha`` in
#   [-2, 2) for ``add``, and a trace step before the last for ``swap``. A spec whose hash
#   repeats an earlier one is drawn again, so the 12 are distinct.
# * The runs: every spec once into ``out/``, every spec again into the same folder, then
#   ``out/results/`` is deleted and every spec runs a third time. After each run the
#   payload's bytes and its cached result's bytes are kept. A spec counts toward
#   ``recompute_identical`` when all three runs exit 0, its result file is the one named by
#   the key below, and both files' bytes are the same after every run. The second run
#   meets its own cached result, and the command line refuses to overwrite a cached result
#   with different bytes, so a nondeterministic second pass shows up as a failed run.
#   ``runs_failed`` lists any run that exited 1, with its error line.
# * ``cache_keys_distinct`` holds when all three of these hold:
#   - ``keys_distinct``: ``intervention_key`` over every combination of the 12 specs, every
#     trace checkpoint's ``state_hash``, two batches (the probe and the patch source) and two
#     code identities (``apply.code_identity()``, and the blob hash of apply.py with one
#     byte appended, standing for an edit to the module) gives as many keys as combinations.
#   - ``keys_stable``: computing that grid again, from each spec read back from its JSON file
#     and each checkpoint loaded again, gives the same keys in the same places.
#   - ``cli_keys_match``: every run's result file is named by the grid's key for its spec,
#     the final checkpoint, the probe and this checkout's code, and by nothing else.

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G3.2"
REPORT_NAME = "g32_determinism.json"
WORK_NAME = "g32_determinism"
SAMPLE_SIZE = 12
SAMPLE_SEED = 0
RUNS = 3
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    validation, train = p1.load_split("validation"), p1.load_split("train")
    probe = _common.probe(validation, ref.config.seq)
    source = _common.patch_source(validation, ref.config.seq)
    reference = _common.reference_batch(train, ref.config.seq)
    work = _common.fresh_dir(Path(_common.RUN_DIR) / WORK_NAME, overwrite=args.overwrite)
    measurements = measure(ref, probe, source, reference, work)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G3.2 rule holds."""
    return "pass" if m["recompute_identical"] == m["sample_size"] and m["cache_keys_distinct"] is True else "fail"


def sample_specs(config: Any, steps: tuple[int, ...], objects: Any, reference: np.ndarray, source: np.ndarray, *,
                 size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[Any]:
    """``size`` distinct seeded specs of one edit each, every operation twice over (see the notes above)."""
    from nablaframe.interventions.apply import editable_taps
    from nablaframe.interventions.spec import OPERATIONS, InterventionSpec, edit, spec_hash

    rng = np.random.default_rng(seed)
    taps = list(editable_taps(config.n_layers))
    components = ["embeddings", "ln_final", *(f"blocks.{layer}{part}" for layer in range(config.n_layers)
                                              for part in ("", ".attn", ".mlp"))]
    ops = [OPERATIONS[i] for i in rng.permutation(np.arange(size) % len(OPERATIONS))]
    reference_hash, source_hash = objects.batch(reference), objects.batch(source)
    specs, hashes = [], set()
    for op in ops:
        while True:
            if op == "swap":
                tap, positions = components[int(rng.integers(len(components)))], "all"
            else:
                tap, positions = taps[int(rng.integers(len(taps)))], _positions(rng, config.seq)
            if op == "zero":
                args: dict[str, Any] = {}
            elif op == "mean":
                args = {"reference": reference_hash}
            elif op == "resample":
                args = {"reference": reference_hash, "seed": int(rng.integers(0, 2**16))}
            elif op == "patch":
                args = {"source": source_hash}
            elif op == "add":
                vector = rng.standard_normal(_per_position_shape(tap, config)).astype(np.float32)
                args = {"vector": objects.vector(vector), "alpha": float(rng.uniform(-2.0, 2.0))}
            else:
                args = {"step": int(steps[int(rng.integers(len(steps) - 1))])}
            spec = InterventionSpec((edit(tap, positions, op, **args),))
            if spec_hash(spec) not in hashes:
                break
        hashes.add(spec_hash(spec))
        specs.append(spec)
    return specs


def _positions(rng: np.random.Generator, seq: int) -> Any:
    form = int(rng.integers(3))
    if form == 0:
        return "all"
    if form == 1:
        count = int(rng.integers(1, 5))
        return sorted(int(p) for p in rng.choice(seq, size=count, replace=False))
    start = int(rng.integers(0, seq))
    return {"range": [start, int(rng.integers(start + 1, seq + 1))]}


def _per_position_shape(tap: str, config: Any) -> tuple[int, ...]:
    """A tap's shape after its (Batch, Pos) axes, from the registry's tap table."""
    from nablaframe.trace.registry import tap_spec

    spec = tap_spec(tap, batch=1, seq=config.seq, d_model=config.d_model, n_heads=config.n_heads,
                    d_head=config.d_head)
    return tuple(axis.size for axis in spec.axes[2:])


def measure(ref: Any, probe: np.ndarray, source: np.ndarray, reference: np.ndarray, work: Path) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.interventions.apply import code_identity
    from nablaframe.interventions.spec import spec_from_json, spec_hash, spec_json
    from nablaframe.train.checkpoint import state_hash

    config, step = ref.config, ref.steps[-1]
    objects = _common.Objects()
    specs = sample_specs(config, ref.steps, objects, reference, source)
    inputs = _common.cli_inputs(work, config, probe, objects)
    spec_dir = work / "specs"
    spec_paths = [_common.write_spec(spec_dir / f"{i:02d}-{spec_hash(spec)}.json", spec)
                  for i, spec in enumerate(specs)]
    print(f"{GATE_ID}: {len(specs)} specs drawn: " + ", ".join(spec.edits[0].op for spec in specs), flush=True)

    # The key grid: every spec, checkpoint, batch and code identity, computed twice.
    code = code_identity()
    apply_py = Path(sys.modules[code_identity.__module__].__file__).read_bytes() + b"\n"
    codes = (code, hashlib.sha1(b"blob %d\0" % len(apply_py) + apply_py).hexdigest())
    batches = (batch_hash(probe), batch_hash(source))
    progress = p1.Progress("checkpoint state hashes", 2 * len(ref.steps))

    def grid(spec_hashes: list[str]) -> dict[tuple[int, int, int, int], str]:
        from nablaframe.interventions.effects import intervention_key

        out = {}
        for s, step_ in enumerate(ref.steps):
            state = state_hash(ref.params_for(step_))
            progress.advance()
            for i, digest in enumerate(spec_hashes):
                for b, batch in enumerate(batches):
                    for c, code_ in enumerate(codes):
                        out[(i, s, b, c)] = intervention_key(checkpoint_sha256=state, batch_hash=batch,
                                                             spec_hash=digest, code=code_)
        return out

    keys = grid([spec_hash(spec) for spec in specs])
    again = grid([spec_hash(spec_from_json(json.loads(path.read_text(encoding="utf-8")))) for path in spec_paths])
    keys_distinct = len(set(keys.values())) == len(keys)
    keys_stable = keys == again
    expected = [keys[(i, ref.steps.index(step), 0, 0)] for i in range(len(specs))]

    # Three runs of every spec through the command line: twice, then again after deleting results/.
    out = work / "out"
    tap = f"blocks.{config.n_layers - 1}.hook_resid_post"
    snapshots: list[list[dict[str, Any]]] = [[] for _ in specs]
    failed = []
    progress = p1.Progress("command-line runs", RUNS * len(specs))
    for run in range(RUNS):
        if run == RUNS - 1 and (out / "results").exists():
            shutil.rmtree(out / "results")
        for i, (spec, path) in enumerate(zip(specs, spec_paths)):
            before = set(_results(out))
            code_, error = _common.run_cli(inputs, ref.checkpoint_dir, step, path, out, taps=[tap])
            if code_ != 0:
                failed.append({"run": run + 1, "spec": i, "spec_hash": spec_hash(spec), "error": error})
            payload = out / f"intervention-{spec_hash(spec)}-step-{step}.json"
            result = out / "results" / f"{expected[i]}.npy"
            snapshots[i].append({
                "exit": code_,
                "payload": payload.read_bytes() if payload.exists() else None,
                "result": result.read_bytes() if result.exists() else None,
                "new_results": sorted(set(_results(out)) - before),
            })
            progress.advance()
        print(f"{GATE_ID}: run {run + 1} of {RUNS} done", flush=True)

    identical = 0
    by_spec = []
    cli_keys_match = True
    for i, (spec, runs) in enumerate(zip(specs, snapshots)):
        same = (all(r["exit"] == 0 and r["payload"] is not None and r["result"] is not None for r in runs)
                and len({r["payload"] for r in runs}) == 1 and len({r["result"] for r in runs}) == 1)
        identical += same
        # The first and third runs write exactly the expected key; the second finds it cached and writes none.
        named = [r["new_results"] for r in runs]
        matches = named[0] == [expected[i]] and named[1] == [] and named[2] == [expected[i]]
        cli_keys_match = cli_keys_match and matches
        by_spec.append({"spec_hash": spec_hash(spec), "spec": spec_json(spec), "identical": same,
                        "key": expected[i], "key_matches": matches})

    distinct = keys_distinct and keys_stable and cli_keys_match
    print(f"{GATE_ID}: {identical} of {len(specs)} specs byte-identical every time; keys distinct over"
          f" {len(keys):,} combinations: {keys_distinct}; stable: {keys_stable}; the command line's keys match:"
          f" {cli_keys_match}", flush=True)
    return {
        "sample_size": len(specs),
        "recompute_identical": identical,
        "cache_keys_distinct": distinct,
        "keys_distinct": keys_distinct,
        "keys_stable": keys_stable,
        "cli_keys_match": cli_keys_match,
        "keys_checked": len(keys),
        "key_grid": {"specs": len(specs), "checkpoints": len(ref.steps), "batches": len(batches),
                     "codes": len(codes)},
        "operations": sorted({spec.edits[0].op for spec in specs}),
        "specs": by_spec,
        "runs": RUNS,
        "runs_failed": failed,
        "sample_seed": SAMPLE_SEED,
        "step": step,
        "code_identity": code,
        "work_dir": p1.rel(work),
    }


def _results(out: Path) -> list[str]:
    folder = out / "results"
    return sorted(path.stem for path in folder.glob("*.npy")) if folder.is_dir() else []


if __name__ == "__main__":
    raise SystemExit(main())
