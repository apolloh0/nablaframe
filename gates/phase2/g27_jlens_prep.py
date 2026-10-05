"""G2.7: the J-lens cross-check's inputs, in the jax lane (D17). Runs before G2.8.

At the final checkpoint (step 12,208), for layer 0 (the longest path, through blocks 1 to 3):

* Exports the checkpoint with ``export_hf`` to ``runs/phase2/jlens/hf-step-12208/``, with the
  run's tokenizer, ``<eos>`` and ``<unk>``, and reloads it to compare every tensor with its
  parameter: ``export_matches_params``.
* Computes ``summed_jacobian`` for ``block_map(..., layer=0)`` on the first 8 windows of
  D17's fit sample (``batch_for_step(train, 1, 0, 128, 256)``), saves it with
  ``save_jacobian`` under ``runs/phase2/jlens/crosscheck/``, writes those 8 windows' token
  ids beside it for G2.8, and recomputes it once: ``jacobian_recompute_identical``.

Measurements: ``step``, ``layer``, ``windows`` (8), ``export_matches_params``,
``jacobian_recompute_identical``.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g27_jlens_prep.py``. Takes --overwrite and --allow-dirty. --overwrite
#   also starts its two working folders over; without it, existing folders are refused.
# * The run's tokenizer is G1.1's ``data/tinystories/tokenizer.json``, our word-level
#   vocabulary format, after checking its SHA-256 is the manifest's ``tokenizer_sha256``.
#   ``export_hf`` builds a Hugging Face tokenizer from it as the export's ``tokenizer.json``.
#   ``eos_id`` is ``<eos>``'s index in its vocabulary.
# * ``export_matches_params``: ``model.safetensors`` is read back with
#   ``safetensors.numpy.load_file`` and must hold exactly GPT-2's tensor names for this
#   shape (no ``lm_head.weight``: the unembedding is tied), each float32, with its
#   parameter's shape and the same bytes. The name map is written out here, not taken from
#   the exporter, so the comparison doesn't share the code it checks. ``export_mismatches``
#   lists the first 20 differences.
# * ``tokenizer_roundtrip``: the export's ``tokenizer.json``, loaded with the ``tokenizers``
#   library's ``Tokenizer.from_file``, encodes the run tokenizer's ``decode`` of each of the 8
#   windows, without added special tokens, back to exactly that window's ids. Recorded, not
#   judged. ``tokenizer_roundtrip_mismatches`` lists each window that doesn't, with its first
#   differing position and both lengths.
# * The fit sample is ``fit_windows``: ``batch_for_step(train, 1, 0, 128, seq)``, each
#   window's first ``seq`` tokens, as in G2.5; 256 at size S. G2.7 takes its first 8.
#   The residual stream is ``blocks.0.hook_resid_post`` from the instrumented pass on those
#   windows. The J is ``summed_jacobian(block_map(model, params, 0, n_layers), xs)``.
# * The J is saved under ``jacobian_key`` with the checkpoint sidecar's ``state_sha256``,
#   ``batch_hash`` of the 8 windows as the fit sample's hash, the layer and
#   ``code_identity()``. ``tokens.npy`` holds the windows (int32, (8, seq)).
#   ``inputs.json`` names everything G2.8 needs: the run, step, layer, windows, both hashes,
#   the code identity, the key, the export's folder and its ``model.safetensors`` SHA-256.
#   It is written last, so its presence means the other outputs are complete.
# * The recompute runs the instrumented pass and ``summed_jacobian`` again from the
#   parameters and compares the result with the saved file, read back with
#   ``load_jacobian``: ``jacobian_recompute_identical`` is true when dtype, shape and every
#   byte agree.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.7"
REPORT_NAME = "g27_jlens_prep.json"
LAYER = 0
WINDOWS = 8
# D17's fit sample: batch_for_step(train, FIT_SEED, FIT_STEP, FIT_WINDOWS, seq).
FIT_SEED = 1
FIT_STEP = 0
FIT_WINDOWS = 128
MAX_MISMATCHES = 20
INPUTS_NAME = "inputs.json"
TOKENS_NAME = "tokens.npy"
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy", "tokenizers")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    tokenizer_file = p1.tokenizer_path()
    if p1.sha256_file(tokenizer_file) != ref.manifest.tokenizer_sha256:
        raise SystemExit(f"{p1.rel(tokenizer_file)} is not the tokenizer G1.4's run recorded. No report written.")
    step = ref.steps[-1]
    export = _common.fresh_dir(export_dir(step), overwrite=args.overwrite)
    crosscheck = _common.fresh_dir(crosscheck_dir(), overwrite=args.overwrite)
    measurements = measure(ref, step, p1.load_split("train"), tokenizer_file, export, crosscheck)
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.7 rule holds."""
    return "pass" if m["export_matches_params"] is True and m["jacobian_recompute_identical"] is True else "fail"


# Where G2.7's outputs live, read from _common's globals when called. G2.8 and G2.9 use these.


def jlens_dir() -> Path:
    return Path(_common.RUN_DIR) / "jlens"


def export_dir(step: int) -> Path:
    return jlens_dir() / f"hf-step-{step}"


def crosscheck_dir() -> Path:
    return jlens_dir() / "crosscheck"


def fit_dir() -> Path:
    return jlens_dir() / "fit"


def fit_windows(train: np.ndarray, seq: int) -> np.ndarray:
    """D17's fit sample: 128 training windows drawn with seed 1, each window's first ``seq`` tokens, int32."""
    from nablaframe.data.corpus import batch_for_step

    return batch_for_step(np.asarray(train), FIT_SEED, FIT_STEP, FIT_WINDOWS, seq)[:, :seq].astype(np.int32)


def checkpoint_sha256(checkpoint: Path) -> str:
    """The ``state_sha256`` recorded in a checkpoint's sidecar."""
    return json.loads(Path(checkpoint).with_suffix(".json").read_text(encoding="utf-8"))["state_sha256"]


def load_inputs() -> dict[str, Any]:
    """G2.7's ``inputs.json``, after checking the files it names exist. SystemExit if anything is missing."""
    path = crosscheck_dir() / INPUTS_NAME
    if not path.exists():
        raise SystemExit(f"{p1.rel(path)} is missing. Run G2.7 first. No report written.")
    inputs = json.loads(path.read_text(encoding="utf-8"))
    needed = [crosscheck_dir() / TOKENS_NAME, crosscheck_dir() / f"{inputs['jacobian_key']}.npy",
              Path(p1.REPO_ROOT) / inputs["export"] / "model.safetensors"]
    missing = [p1.rel(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"G2.7's outputs are incomplete: {', '.join(missing)} missing. Run G2.7 again with"
                         " --overwrite. No report written.")
    return inputs


# ---------------------------------------------------------------------------


def measure(ref: Any, step: int, train: np.ndarray, tokenizer_file: Path, export: Path,
            crosscheck: Path, *, layer: int = LAYER, windows: int = WINDOWS) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.data.tokenizer import EOS, UNK
    from nablaframe.model.hf_export import export_hf
    from nablaframe.views.jlens import code_identity, jacobian_key, load_jacobian, save_jacobian

    config, model = ref.config, ref.model
    params = ref.params_for(step)
    tokenizer = p1.load_tokenizer()

    # The export, and every tensor read back.
    export_hf(params, config, tokenizer_file, export, eos_token=EOS, eos_id=tokenizer.vocab.index(EOS),
              unk_token=UNK)
    mismatches = export_mismatches(params, config, export / "model.safetensors")
    print(f"{GATE_ID}: export to {p1.rel(export)}: "
          + ("every tensor equals its parameter" if not mismatches else f"{len(mismatches)} mismatches"), flush=True)

    # The summed J on the first windows of the fit sample, saved, then recomputed.
    tokens = fit_windows(train, config.seq)[:windows]
    fit_sha = batch_hash(tokens)
    roundtrip = tokenizer_roundtrip_mismatches(export / "tokenizer.json", tokenizer, tokens)
    print(f"{GATE_ID}: the export's tokenizer encodes the decoded text of {windows - len(roundtrip)} of {windows}"
          " windows back to their ids", flush=True)
    state_sha = checkpoint_sha256(ref.checkpoint_dir / f"step-{step}.safetensors")
    code = code_identity()
    key = jacobian_key(checkpoint_sha256=state_sha, fit_sha256=fit_sha, layer=layer, code=code)

    started = time.perf_counter()
    J = jacobian(model, params, tokens, layer)
    seconds = time.perf_counter() - started
    saved = save_jacobian(crosscheck, key, J)
    print(f"{GATE_ID}: summed J for layer {layer} on {windows} windows in {seconds:.1f} s, saved to {p1.rel(saved)}",
          flush=True)
    started = time.perf_counter()
    again = jacobian(model, params, tokens, layer)
    recompute_seconds = time.perf_counter() - started
    stored = load_jacobian(crosscheck, key)
    identical = bool(stored is not None and stored.dtype == again.dtype and stored.shape == again.shape
                     and stored.tobytes() == again.tobytes())
    print(f"{GATE_ID}: recomputed in {recompute_seconds:.1f} s: " + ("identical" if identical else "differs"),
          flush=True)

    p1.write_atomic(crosscheck / TOKENS_NAME, _npy(tokens))
    inputs = {
        "gate_id": GATE_ID, "run_id": ref.manifest.run_id, "step": step, "layer": layer, "windows": windows,
        "seq": config.seq, "n_layers": config.n_layers, "checkpoint_sha256": state_sha, "fit_sha256": fit_sha,
        "code_identity": code, "jacobian_key": key, "tokens": TOKENS_NAME, "export": p1.rel(export),
        "export_model_sha256": p1.sha256_file(export / "model.safetensors"),
    }
    p1.write_atomic(crosscheck / INPUTS_NAME, (json.dumps(inputs, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    return {
        "step": step,
        "layer": layer,
        "windows": windows,
        "export_matches_params": not mismatches,
        "jacobian_recompute_identical": identical,
        "export_mismatches": mismatches[:MAX_MISMATCHES],
        "tokenizer_roundtrip": not roundtrip,
        "tokenizer_roundtrip_mismatches": roundtrip,
        "jacobian_seconds": seconds,
        "recompute_seconds": recompute_seconds,
        "jacobian_norm": float(np.linalg.norm(J.astype(np.float64))),
        "jacobian_max_abs_difference": (None if stored is None or stored.shape != again.shape else
                                        float(np.max(np.abs(stored.astype(np.float64) - again.astype(np.float64))))),
        "jacobian_key": key,
        "checkpoint_sha256": state_sha,
        "fit_sha256": fit_sha,
        "code_identity": code,
        "fit_seed": FIT_SEED,
        "fit_windows_total": FIT_WINDOWS,
        "seq": config.seq,
        "export": p1.rel(export),
        "crosscheck": p1.rel(crosscheck),
        "run_id": ref.manifest.run_id,
    }


def jacobian(model: Any, params: Any, tokens: np.ndarray, layer: int) -> np.ndarray:
    """The summed J of ``blocks.<layer>.hook_resid_post`` for ``tokens``, from the instrumented pass."""
    from nablaframe.views.jlens import block_map, summed_jacobian

    tap = f"blocks.{layer}.hook_resid_post"
    _, recorded = model.instrumented(params, tokens, [tap])
    return summed_jacobian(block_map(model, params, layer, model.config.n_layers), np.asarray(recorded[tap]))


def tokenizer_roundtrip_mismatches(path: Path, tokenizer: Any, windows: np.ndarray) -> list[dict[str, Any]]:
    """Each window whose decoded text the Hugging Face tokenizer at ``path`` doesn't encode back to its ids."""
    from tokenizers import Tokenizer as HFTokenizer

    hf = HFTokenizer.from_file(str(path))
    mismatches = []
    for i, window in enumerate(np.asarray(windows).tolist()):
        ids = hf.encode(tokenizer.decode(window), add_special_tokens=False).ids
        if ids != window:
            first = next((t for t, (a, b) in enumerate(zip(ids, window)) if a != b), min(len(ids), len(window)))
            mismatches.append({"window": i, "first_difference": first, "encoded_length": len(ids),
                               "window_length": len(window)})
    return mismatches


def expected_tensors(params: Any, config: Any) -> dict[str, np.ndarray]:
    """GPT-2's tensor names for this shape, each mapped to the parameter it must equal."""
    tensors = {"transformer.wte.weight": params.wte, "transformer.wpe.weight": params.wpe,
               "transformer.ln_f.weight": params.ln_f.weight, "transformer.ln_f.bias": params.ln_f.bias}
    for i, block in enumerate(params.blocks):
        for name, layer in (("ln_1", block.ln_1), ("attn.c_attn", block.attn.c_attn),
                            ("attn.c_proj", block.attn.c_proj), ("ln_2", block.ln_2),
                            ("mlp.c_fc", block.mlp.c_fc), ("mlp.c_proj", block.mlp.c_proj)):
            tensors[f"transformer.h.{i}.{name}.weight"] = layer.weight
            tensors[f"transformer.h.{i}.{name}.bias"] = layer.bias
    if len(params.blocks) != config.n_layers:
        raise ValueError(f"params have {len(params.blocks)} blocks, the config {config.n_layers}")
    return {name: np.asarray(value) for name, value in tensors.items()}


def export_mismatches(params: Any, config: Any, path: Path) -> list[str]:
    """Every way the saved tensors differ from the parameters, as sentences; empty when they agree bit for bit."""
    from safetensors.numpy import load_file

    loaded = load_file(str(path))
    expected = expected_tensors(params, config)
    problems = [f"{name}: missing" for name in sorted(set(expected) - set(loaded))]
    problems += [f"{name}: not a parameter" for name in sorted(set(loaded) - set(expected))]
    for name in sorted(set(expected) & set(loaded)):
        want, got = expected[name], loaded[name]
        if got.dtype != np.float32 or want.dtype != np.float32:
            problems.append(f"{name}: saved {got.dtype}, parameter {want.dtype}")
        elif got.shape != want.shape:
            problems.append(f"{name}: saved shape {list(got.shape)}, parameter {list(want.shape)}")
        elif np.ascontiguousarray(got).tobytes() != np.ascontiguousarray(want).tobytes():
            differ = int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32)))
            problems.append(f"{name}: {differ} of {got.size} values differ")
    return problems


def _npy(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
