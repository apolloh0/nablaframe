"""G3.5: the viewer's compare mode and its offline rule (D27).

Runs the command line twice for 6 example specs (one per operation) on the reference run's
final checkpoint and probe, into two folders; the second stays at ``runs/phase3/interventions/``
for the viewer. Measurements: ``builds_identical`` (every file byte-identical),
``external_urls`` (as G2.6 counts them) and ``payloads``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase3/g35_viewer.py``. Takes --overwrite and --allow-dirty. --overwrite also lets
#   it replace both build folders and its inputs folder.
# * The command line's code is ``nablaframe.interventions.cli.main``, called in this process,
#   so every pass goes through ``run_intervened``. Its inputs are written once, to
#   runs/phase3/g35_viewer/inputs/: the config from G1.4's manifest, the probe as
#   ``tokens.npy``, ``objects/`` and one JSON per spec. Its checkpoints are G1.4's trace
#   checkpoints. The first build goes to runs/phase3/g35_viewer/first-build/ and stays there,
#   so a difference can be inspected; the second to runs/phase3/interventions/. Each folder
#   starts empty, and the same six runs, in the same order, fill each.
# * The examples, with m = min(1, n_layers - 1) (block 1 at size S) and T the window length:
#   - ``zero``: ``blocks.0.hook_attn_out`` at every position.
#   - ``mean``: ``blocks.<m>.hook_mlp_out`` at every position, from D22's reference batch.
#   - ``resample``: ``blocks.<m>.hook_resid_mid`` on ``{"range": [T/4, T/2]}``, seed 0.
#   - ``patch``: ``blocks.<m>.hook_resid_post`` on ``{"range": [0, T/2]}`` from the patch
#     source, the second batch of 8 validation windows G3.1 patches from.
#   - ``add``: ``blocks.0.hook_resid_post`` at position T/4, along the logit-lens direction of
#     a token drawn with ``default_rng(0)`` from the vocabulary without ``<unk>`` and
#     ``<eos>``, with ``alpha`` once block 0's mean residual-stream norm on the probe
#     (``steering_scales``, from one unintervened ``run_intervened`` pass).
#   - ``swap``: ``blocks.<m>.attn`` from step 1,024, D26's own example.
#   Each run asks for the edited tap (for the swap, ``blocks.<m>.hook_attn_out``) and the
#   last block's output as taps, so every payload carries tap changes.
# * ``builds_identical``: all twelve runs exited 0, each build holds the six payloads, and both
#   folders hold the same files, ``results/`` included, by relative path, with equal bytes.
#   ``files_differing`` names any that aren't; ``runs_failed`` any run that exited 1, with
#   its error line.
# * ``external_urls`` counts every match of ``EXTERNAL_URL``, copied from tests/test_viewer.py
#   as G2.6 copies it, in every file directly in viewer/, as that test reads them.
#   ``external_url_matches`` lists them with their file.
# * ``payloads``: the payload files in the second build. ``index_entries`` counts its
#   index.json's entries, and ``examples`` names each example's operation, spec hash, file
#   and taps.

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import g31_identities as g31  # noqa: E402
import g33_jlens_experiment as g33  # noqa: E402

p1 = _common.p1

GATE_ID = "G3.5"
REPORT_NAME = "g35_viewer.json"
WORK_NAME = "g35_viewer"
VIEWS_NAME = "interventions"
ADD_TOKEN_SEED = 0
RESAMPLE_SEED = 0
# The offline rule's pattern, as tests/test_viewer.py states it.
EXTERNAL_URL = re.compile(r"https?://|src=[\"']//|import\s*\(\s*[\"']//")
PAYLOAD = re.compile(r"intervention-[0-9a-f]{64}-step-[0-9]+\.json")
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    if g31.SWAP_STEP not in ref.steps:
        raise SystemExit(f"step {g31.SWAP_STEP} is not one of G1.4's trace checkpoints. No report written.")
    validation, train = p1.load_split("validation"), p1.load_split("train")
    probe = _common.probe(validation, ref.config.seq)
    source = _common.patch_source(validation, ref.config.seq)
    reference = _common.reference_batch(train, ref.config.seq)
    excluded = g33.special_tokens()
    work = Path(_common.RUN_DIR) / WORK_NAME
    inputs_dir = _common.fresh_dir(work / "inputs", overwrite=args.overwrite)
    first = _common.fresh_dir(work / "first-build", overwrite=args.overwrite)
    second = _common.fresh_dir(Path(_common.RUN_DIR) / VIEWS_NAME, overwrite=args.overwrite)
    measurements = measure(ref, probe, source, reference, excluded, inputs_dir, first, second)
    measurements |= offline(Path(p1.REPO_ROOT) / "viewer")
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G3.5 rule holds."""
    return "pass" if m["builds_identical"] is True and m["external_urls"] == 0 else "fail"


def examples(model: Any, params: Any, config: Any, probe: np.ndarray, source: np.ndarray, reference: np.ndarray,
             excluded: list[int], objects: Any) -> list[tuple[Any, list[str]]]:
    """The six example specs, one per operation, each with the taps its run asks for."""
    from nablaframe.interventions.jlens_experiment import logit_lens_direction, steering_scales
    from nablaframe.interventions.spec import OPERATIONS, InterventionSpec, edit

    n, seq = config.n_layers, config.seq
    m = min(1, n - 1)
    last = f"blocks.{n - 1}.hook_resid_post"
    quarter, half = seq // 4, seq // 2
    block0 = "blocks.0.hook_resid_post"
    _, base = _common.unintervened(model, params, probe, n_layers=n, taps=[block0])
    alpha = steering_scales(base[block0])[-1]  # SCALE_FACTORS ends at once the mean norm
    allowed = np.setdiff1d(np.arange(config.vocab), np.asarray(excluded, dtype=np.int64))
    token = int(allowed[np.random.default_rng(ADD_TOKEN_SEED).integers(0, len(allowed))])
    vector = logit_lens_direction(np.asarray(params.wte, dtype=np.float32), token)
    edits = {
        "zero": (edit("blocks.0.hook_attn_out", "all", "zero"), "blocks.0.hook_attn_out"),
        "mean": (edit(f"blocks.{m}.hook_mlp_out", "all", "mean", reference=objects.batch(reference)),
                 f"blocks.{m}.hook_mlp_out"),
        "resample": (edit(f"blocks.{m}.hook_resid_mid", {"range": [quarter, half]}, "resample",
                          reference=objects.batch(reference), seed=RESAMPLE_SEED), f"blocks.{m}.hook_resid_mid"),
        "patch": (edit(f"blocks.{m}.hook_resid_post", {"range": [0, half]}, "patch", source=objects.batch(source)),
                  f"blocks.{m}.hook_resid_post"),
        "add": (edit(block0, [quarter], "add", vector=objects.vector(vector), alpha=float(alpha)), block0),
        "swap": (edit(f"blocks.{m}.attn", "all", "swap", step=g31.SWAP_STEP), f"blocks.{m}.hook_attn_out"),
    }
    out = []
    for op in OPERATIONS:
        e, tap = edits[op]
        out.append((InterventionSpec((e,)), list(dict.fromkeys([tap, last]))))
    return out


def measure(ref: Any, probe: np.ndarray, source: np.ndarray, reference: np.ndarray, excluded: list[int],
            inputs_dir: Path, first: Path, second: Path) -> dict[str, Any]:
    from nablaframe.interventions.spec import spec_hash

    config, step = ref.config, ref.steps[-1]
    objects = _common.Objects()
    chosen = examples(ref.model, ref.params_for(step), config, probe, source, reference, excluded, objects)
    inputs = _common.cli_inputs(inputs_dir, config, probe, objects)
    paths = [_common.write_spec(inputs_dir / "specs" / f"{spec.edits[0].op}-{spec_hash(spec)}.json", spec)
             for spec, _ in chosen]

    failed = []
    progress = p1.Progress("command-line runs", 2 * len(chosen))
    for label, out in (("first build", first), ("second build", second)):
        for (spec, taps), path in zip(chosen, paths):
            code, error = _common.run_cli(inputs, ref.checkpoint_dir, step, path, out, taps=taps)
            if code != 0:
                failed.append({"build": label, "op": spec.edits[0].op, "spec_hash": spec_hash(spec), "error": error})
            progress.advance()
        print(f"{GATE_ID}: {label}: {len(_payloads(out))} payloads in {p1.rel(out)}", flush=True)

    compared = _common.compare_trees(first, second)
    payloads = _payloads(second)
    identical = (compared["identical"] and not failed and len(payloads) == len(chosen)
                 and len(_payloads(first)) == len(chosen))
    index = second / "index.json"
    entries = json.loads(index.read_text(encoding="utf-8"))["interventions"] if index.exists() else []
    print(f"{GATE_ID}: the two builds are {'byte-identical' if identical else 'different'} over"
          f" {compared['files_compared']} files", flush=True)
    return {
        "builds_identical": identical,
        "payloads": len(payloads),
        "payload_files": payloads,
        "index_entries": len(entries),
        "files_compared": compared["files_compared"],
        "files_differing": compared["files_differing"],
        "runs_failed": failed,
        "examples": [{"op": spec.edits[0].op, "spec_hash": spec_hash(spec),
                      "file": f"intervention-{spec_hash(spec)}-step-{step}.json", "taps": taps}
                     for spec, taps in chosen],
        "step": step,
        "interventions_dir": p1.rel(second),
        "first_build_dir": p1.rel(first),
    }


def offline(viewer: Path) -> dict[str, Any]:
    """Matches of the offline rule's pattern in the viewer's files, as G2.6 counts them."""
    matches = []
    for path in sorted(Path(viewer).glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            matches += [{"file": path.name, "match": match.group(0)} for match in EXTERNAL_URL.finditer(text)]
    files = sorted(path.name for path in Path(viewer).glob("*") if path.is_file())
    print(f"{GATE_ID}: {len(matches)} external references in {', '.join(files) or 'no files'}", flush=True)
    return {"external_urls": len(matches), "external_url_matches": matches, "viewer_files": files}


def _payloads(folder: Path) -> list[str]:
    return sorted(path.name for path in Path(folder).glob("*.json") if PAYLOAD.fullmatch(path.name))


if __name__ == "__main__":
    raise SystemExit(main())
