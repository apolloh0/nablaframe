"""G2.8: the J-lens cross-check against anthropics/jacobian-lens, in the torch lane (D17).

Run with ``UV_PROJECT_ENVIRONMENT=.venv-torch uv run --extra torch``. Refuses to start without
G2.7's outputs. Loads G2.7's export through the reference's own loader, then:

* Computes the reference's J for layer 0 with ``jlens.fitting.jacobian_for_prompt`` on the same
  8 windows, with ``max_seq_len=256``, ``skip_first=16``, and the final block as target, and
  averages the per-window J's equally, as its ``fit`` does. The reference reads text, so it
  must see our exact token ids: either through an adapter that supplies them, or through
  decoded text whose re-tokenization is checked id for id. ``tokens_identical`` records it.
* ``jacobian_relative_difference``: the relative Frobenius difference between the reference's
  J, in float32, never through its float16 save, and G2.7's.
* ``export_logit_max_abs_diff``: the export's logits on G1.6's probe, through the Hugging
  Face model, against the traced logits at step 12,208.

Measurements: ``windows``, ``tokens_identical``, ``export_logit_max_abs_diff``,
``jacobian_relative_difference``.
"""

# How it measures:
#
# * Run in nf-torch: ``UV_PROJECT_ENVIRONMENT=.venv-torch uv run --extra torch python
#   gates/phase2/g28_jlens_crosscheck.py``. Takes --overwrite and --allow-dirty. It writes
#   nothing outside its report. The report's ``env_name`` is ``nf-torch``.
# * Hugging Face offline mode is set first thing, as in G1.2: everything is read from
#   G2.7's export folder.
# * Setup checks, each an exit 1 with no report: G2.7's ``inputs.json`` and the files it names
#   exist; it names G1.4's run and its last trace step; the export's ``model.safetensors``
#   still has the SHA-256 G2.7 recorded; the installed ``jlens`` is D19's pinned commit, read
#   from its ``direct_url.json``; G1.6's store holds logits at that step on the probe.
# * Loading: ``GPT2LMHeadModel.from_pretrained`` on the export folder, float32, with eager
#   attention, then ``jlens.hf.from_hf``, the reference's loader. The export's
#   ``tokenizer.json`` is the run's word-level vocabulary (G2.7 copies it), not a Hugging Face
#   tokenizer file, so ``from_hf`` gets our ``Tokenizer``, which provides the ``decode`` its
#   protocol asks for; fitting never calls the tokenizer.
# * Tokens, by the adapter route: ``TokenIds`` wraps the reference's ``HFLensModel`` and
#   answers ``encode(name, max_length)`` with G2.7's ids for window ``name``, truncated to
#   ``max_length`` as the Hugging Face tokenizer would truncate, and records every
#   ``input_ids`` row the reference then passes to ``forward``. ``tokens_identical`` is true
#   when every row the reference ran equals that window's ids, id for id at full length, and
#   G2.7's ids are the first 8 windows of D17's fit sample, drawn again here from G1.1's
#   training stream. ``tokens_match_fit_sample`` and ``tokens_seen_identical`` give the two
#   parts.
# * The reference's J: ``jacobian_for_prompt(adapter, name, [0], target_layer=n_layers - 1,
#   max_seq_len=256, skip_first=16)`` per window, with its default ``dim_batch`` (8); the
#   per-window float32 J's are summed in float32 and divided by the count, as ``fit`` does.
#   ``jacobian_relative_difference`` is ``||J_reference - J_G2.7||_F / ||J_reference||_F``,
#   in float64 from the two float32 matrices. G2.7's J is read with ``load_jacobian``.
# * Logits: the Hugging Face model's ``logits`` on the probe (8 x 256), under
#   ``torch.no_grad``, before ``from_hf`` touches the model, against the ``logits`` tensor G2.1
#   recorded at that step. ``export_lm_head_tied`` records whether ``lm_head`` shares
#   ``wte``'s storage after loading.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import g27_jlens_prep as g27  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.8"
REPORT_NAME = "g28_jlens_crosscheck.json"
ENV_NAME = "nf-torch"
MAX_SEQ_LEN = 256
SKIP_FIRST = 16
REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
VERSIONS = ("torch", "transformers", "safetensors", "numpy", "jlens")


def main(argv: list[str] | None = None) -> int:
    # huggingface_hub reads these once, when first imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    manifest = p1.load_g14_manifest()
    # The model's shape from the manifest itself: GPT2Config's module imports JAX.
    seq, n_layers = manifest.model_config["seq"], manifest.model_config["n_layers"]
    inputs = g27.load_inputs()
    step = manifest.checkpoint_schedule[-1]
    if inputs["run_id"] != manifest.run_id or inputs["step"] != step:
        raise SystemExit(f"G2.7's outputs are for run {inputs['run_id']} step {inputs['step']}, not G1.4's run"
                         f" {manifest.run_id} step {step}. Run G2.7 again with --overwrite. No report written.")
    export = Path(p1.REPO_ROOT) / inputs["export"]
    if p1.sha256_file(export / "model.safetensors") != inputs["export_model_sha256"]:
        raise SystemExit(f"{p1.rel(export / 'model.safetensors')} changed after G2.7 wrote it. Run G2.7 again with"
                         " --overwrite. No report written.")
    commit = installed_jlens_commit()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"jlens is installed from commit {commit}, not D19's {REFERENCE_COMMIT}. Sync the torch lane"
                         " with `UV_PROJECT_ENVIRONMENT=.venv-torch uv sync --extra torch`. No report written.")
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.jlens import load_jacobian

    probe = p1.probe(p1.load_split("validation"), seq)
    reader = _common.open_reader(_common.g16_store(), (step,), probe)
    tokens = np.load(g27.crosscheck_dir() / inputs["tokens"], allow_pickle=False)
    ours = load_jacobian(g27.crosscheck_dir(), inputs["jacobian_key"])
    fit = g27.fit_windows(p1.load_split("train"), seq)[:inputs["windows"]]
    measurements = measure(export, tokens, ours, fit, probe, reader.tensor(step, LOGITS),
                           layer=inputs["layer"], n_layers=n_layers)
    measurements.update(step=step, run_id=manifest.run_id, code_identity=inputs["code_identity"],
                        jacobian_key=inputs["jacobian_key"], export=inputs["export"], reference_commit=commit,
                        versions=p1.phase0.package_versions(VERSIONS))
    return write(status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.8 rule holds."""
    logits, J = m["export_logit_max_abs_diff"], m["jacobian_relative_difference"]
    ok = (m["tokens_identical"] is True and logits is not None and logits <= 1e-4
          and J is not None and J <= 1e-4)
    return "pass" if ok else "fail"


def write(status: str, measurements: dict[str, Any], path: Path, *, overwrite: bool) -> int:
    """Phase 1's ``write``, with this lane's name in the envelope."""
    from nablaframe.reports import write_report

    report = p1.envelope(GATE_ID, ENV_NAME)
    report["status"] = status
    report["measurements"] = p1.plain(measurements)
    write_report(report, path, overwrite=overwrite)
    print(f"{GATE_ID}: {status}. Report written to {p1.rel(path)}", flush=True)
    return 0


def installed_jlens_commit() -> str | None:
    """The git commit jlens was installed from, as its ``direct_url.json`` records it, or None."""
    from importlib import metadata

    try:
        text = metadata.distribution("jlens").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id") if text else None


# ---------------------------------------------------------------------------


class TokenIds:
    """The reference's ``LensModel``, fed our token ids: ``encode`` looks up a window by name.

    Everything else is the wrapped ``HFLensModel``'s. ``seen`` collects, per window, every
    ``input_ids`` row ``forward`` received while that window was being computed.
    """

    def __init__(self, lens_model: Any, windows: dict[str, np.ndarray]) -> None:
        self._model = lens_model
        self._windows = windows
        self._current: str | None = None
        self.seen: dict[str, list[np.ndarray]] = {}
        self.n_layers, self.d_model = lens_model.n_layers, lens_model.d_model
        self.layers, self.tokenizer = lens_model.layers, lens_model.tokenizer

    def encode(self, text: str, *, max_length: int = 512) -> Any:
        import torch

        self._current = text
        ids = torch.from_numpy(np.asarray(self._windows[text], dtype=np.int64)[None, :max_length].copy())
        return ids.to(self._model.input_device)

    def forward(self, input_ids: Any) -> Any:
        self.seen.setdefault(self._current, []).extend(row for row in input_ids.detach().cpu().numpy())
        return self._model.forward(input_ids)

    def unembed(self, residual: Any) -> Any:
        return self._model.unembed(residual)


def measure(export: Path, tokens: np.ndarray, ours: np.ndarray | None, fit: np.ndarray, probe: np.ndarray,
            traced_logits: np.ndarray, *, layer: int, n_layers: int) -> dict[str, Any]:
    import torch
    from jlens.fitting import jacobian_for_prompt
    from jlens.hf import from_hf
    from transformers import GPT2LMHeadModel

    from nablaframe.data.tokenizer import Tokenizer

    hf_model = GPT2LMHeadModel.from_pretrained(str(export), dtype=torch.float32, attn_implementation="eager")
    hf_model.eval()
    tied = hf_model.lm_head.weight.data_ptr() == hf_model.transformer.wte.weight.data_ptr()

    # The export's logits on the probe, against the trace.
    with torch.no_grad():
        logits = hf_model(input_ids=torch.from_numpy(np.asarray(probe, dtype=np.int64))).logits.float().numpy()
    traced = np.asarray(traced_logits)
    logit_diff = (float(np.max(np.abs(logits.astype(np.float64) - traced.astype(np.float64))))
                  if logits.shape == traced.shape and np.all(np.isfinite(logits)) else None)
    print(f"{GATE_ID}: export's logits on the probe: largest difference from the trace {logit_diff}", flush=True)

    # The reference's J, from our token ids.
    tokenizer = Tokenizer.from_json((export / "tokenizer.json").read_text(encoding="utf-8"))
    lens_model = from_hf(hf_model, tokenizer)
    names = [str(i) for i in range(len(tokens))]
    adapter = TokenIds(lens_model, dict(zip(names, tokens)))
    per_window, seq_lens, n_valid = [], [], []
    started = time.perf_counter()
    progress = p1.Progress("reference J, windows", len(names))
    for name in names:
        J, seq_len, valid = jacobian_for_prompt(adapter, name, [layer], target_layer=n_layers - 1,
                                                max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST)
        per_window.append(J[layer])
        seq_lens.append(seq_len)
        n_valid.append(valid)
        progress.advance()
    total = torch.zeros_like(per_window[0])
    for J in per_window:
        total += J
    reference = (total / len(per_window)).numpy()
    seconds = time.perf_counter() - started
    print(f"{GATE_ID}: reference J for layer {layer} on {len(names)} windows in {seconds:.1f} s", flush=True)

    seen_identical = all(
        len(adapter.seen.get(name, [])) > 0
        and all(row.shape == window.shape and np.array_equal(row, window) for row in adapter.seen[name])
        for name, window in zip(names, tokens))
    matches_fit = bool(tokens.shape == fit.shape and np.array_equal(tokens, fit))
    tokens_identical = bool(seen_identical and matches_fit)

    if ours is not None and ours.shape == reference.shape:
        ref64, ours64 = reference.astype(np.float64), ours.astype(np.float64)
        relative = float(np.linalg.norm(ref64 - ours64) / np.linalg.norm(ref64))
        max_abs = float(np.max(np.abs(ref64 - ours64)))
    else:
        relative = max_abs = None
    print(f"{GATE_ID}: tokens identical: {tokens_identical}; J relative difference {relative}", flush=True)

    return {
        "windows": len(names),
        "tokens_identical": tokens_identical,
        "export_logit_max_abs_diff": p1.finite_or_none(logit_diff),
        "jacobian_relative_difference": p1.finite_or_none(relative),
        "tokens_seen_identical": bool(seen_identical),
        "tokens_match_fit_sample": matches_fit,
        "tokens_route": "adapter",
        "jacobian_max_abs_difference": p1.finite_or_none(max_abs),
        "reference_norm": float(np.linalg.norm(reference.astype(np.float64))),
        "ours_norm": None if ours is None else float(np.linalg.norm(ours.astype(np.float64))),
        "reference_seq_lens": seq_lens,
        "reference_valid_positions": n_valid,
        "reference_seconds": seconds,
        "layer": layer,
        "target_layer": n_layers - 1,
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST,
        "export_lm_head_tied": bool(tied),
        "probe_shape": list(np.shape(probe)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
