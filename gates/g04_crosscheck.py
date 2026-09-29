"""Gate G0.4: Cross-check.

Spec: docs/specs/phase-0.md, Gates table, G0.4 row.
Environment: nf-torch. Run on apolloh0's Mac only; never in the cloud.

Measures: The G0.1 checkpoint, exported to Hugging Face format with
levanter.main.export_lm_to_hf, loaded through TransformerLens TransformerBridge (without compatibility mode, so numerics stay raw Hugging Face), logits compared on a fixed 8x128 batch.
Pass: Max abs logit diff <= 1e-3, and identical top-1 tokens wherever the top-2 margin exceeds 1e-3.
On failure, diagnose the export or load path before D0.

Writes reports/phase0/g04_crosscheck.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.4"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures:
#
# * Levanter lives in nf-jax and TransformerLens in nf-torch, and the two never share
#   an environment (D1). So this script runs in nf-torch and starts itself once in
#   nf-jax (--levanter-side, with the interpreter from --jax-python, by default
#   .venv-jax/bin/python) to export G0.1's final checkpoint with export_lm_to_hf and
#   compute Levanter's logits on the fixed batch from the same checkpoint.
# * The fixed batch is 8x128 token ids drawn uniformly from the vocabulary with
#   numpy's default_rng(BATCH_SEED). The ids are saved with Levanter's logits and
#   fed unchanged to the bridge, with no BOS prepended.
# * The bridge comes from TransformerBridge.boot_transformers on the exported
#   directory, float32 on CPU, and enable_compatibility_mode() is never called (D2).
# * positions_compared is every position in the batch; max_abs_logit_diff is over
#   all of them. top1_agreement follows reports.py: positions whose Levanter top-2
#   margin exceeds 1e-3, and positions_near_tie counts the rest.
# * An export, bridge-load or bridge-forward error is the path under test failing:
#   the report says fail, with the stage and the error, and nulls for what could not
#   be measured. Missing prerequisites (G0.1's checkpoint, the nf-jax interpreter) or
#   an error computing Levanter's own logits mean the gate could not measure: no
#   report, exit code 1.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import _levanter  # noqa: E402
import g01_cpu_train  # noqa: E402

GATE_ID = "G0.4"
REPORT_PATH = "reports/phase0/g04_crosscheck.json"

ENV_NAME = "nf-torch"
WORK_DIR = _common.RUNS_DIR / "g04_crosscheck"
DEFAULT_JAX_PYTHON = _common.REPO_ROOT / ".venv-jax" / "bin" / "python"
BATCH, SEQ = 8, 128
BATCH_SEED = 0
MAX_ABS_LOGIT_DIFF = 1e-3
TIE_MARGIN = 1e-3
VERSIONS = ("torch", "transformer-lens", "transformers", "safetensors", "numpy")
JAX_VERSIONS = ("jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "transformers", "numpy")


def main(argv: list[str] | None = None) -> int:
    parser = _common.arg_parser(__doc__)
    parser.add_argument("--jax-python", type=Path, default=DEFAULT_JAX_PYTHON, help="the nf-jax interpreter")
    parser.add_argument("--levanter-side", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.levanter_side is not None:
        return levanter_side(args.levanter_side)

    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)
    if not g01_cpu_train.checkpoint_dir().exists():
        print(f"{GATE_ID} needs G0.1's checkpoints in {_common.rel(g01_cpu_train.checkpoint_dir())}."
              " Run G0.1 first. No report written.", file=sys.stderr)
        return 1
    if not args.jax_python.exists():
        print(f"{GATE_ID} needs the nf-jax interpreter at {args.jax_python} (--jax-python). No report written.",
              file=sys.stderr)
        return 1
    work = _common.fresh_dir(WORK_DIR, overwrite=args.overwrite)

    print(f"{GATE_ID}: exporting and computing Levanter logits in nf-jax, log in {work / 'levanter_side.log'}",
          flush=True)
    with open(work / "levanter_side.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(args.jax_python), str(Path(__file__).resolve()), "--levanter-side", str(work)],
            cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    side_file = work / "levanter_side.json"
    side = json.loads(side_file.read_text(encoding="utf-8")) if side_file.exists() else None
    if side is None or (side["error"] is not None and side["stage"] != "export"):
        detail = side["error"] if side and side["error"] else _tail(work / "levanter_side.log")
        print(f"{GATE_ID} could not measure (nf-jax side exit code {proc.returncode}):\n{detail}\nNo report written.",
              file=sys.stderr)
        return 1

    if side["error"] is not None:
        measurements = _unmeasured(error_stage="export", error=side["error"])
    else:
        with np.load(work / "levanter_logits.npz") as data:
            tokens, levanter_logits = data["tokens"], data["logits"]
        try:
            stage = "bridge_load"
            import torch
            from transformer_lens.model_bridge import TransformerBridge

            bridge = TransformerBridge.boot_transformers(str(work / "hf"), device="cpu", dtype=torch.float32)
            stage = "bridge_forward"
            with torch.no_grad():
                out = bridge(torch.as_tensor(tokens, dtype=torch.long), return_type="logits")
            bridge_logits = out.detach().cpu().numpy()
        except Exception:  # noqa: BLE001  the load path is what this gate tests
            measurements = _unmeasured(error_stage=stage, error=traceback.format_exc()[-4000:])
        else:
            measurements = compare(levanter_logits, bridge_logits)

    measurements.update(
        checkpoint=side["checkpoint"],
        hf_export_dir=_common.rel(work / "hf"),
        batch_shape=[BATCH, SEQ],
        batch_seed=BATCH_SEED,
        tie_margin=TIE_MARGIN,
        compatibility_mode=False,
        versions=_common.package_versions(VERSIONS),
        jax_side_versions=side["versions"],
    )
    report = _common.build_report(GATE_ID, status(measurements), measurements, ENV_NAME)
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def compare(levanter_logits: np.ndarray, bridge_logits: np.ndarray) -> dict[str, Any]:
    """The four required measurements, per reports.py. Pure numpy."""
    if levanter_logits.shape != bridge_logits.shape:
        return _unmeasured(
            error_stage="compare",
            error=f"logit shapes differ: Levanter {list(levanter_logits.shape)}, bridge {list(bridge_logits.shape)}",
        )
    lev = levanter_logits.astype(np.float64).reshape(-1, levanter_logits.shape[-1])
    hf = bridge_logits.astype(np.float64).reshape(-1, bridge_logits.shape[-1])
    positions = lev.shape[0]
    if not (np.all(np.isfinite(lev)) and np.all(np.isfinite(hf))):
        return _unmeasured(error_stage="compare", error="non-finite logits", positions_compared=positions)

    top2 = np.sort(lev, axis=-1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) > TIE_MARGIN
    agree = np.argmax(lev, axis=-1) == np.argmax(hf, axis=-1)
    n_decisive = int(np.count_nonzero(decisive))
    return {
        "max_abs_logit_diff": float(np.max(np.abs(lev - hf))),
        "top1_agreement": float(np.count_nonzero(agree & decisive)) / n_decisive if n_decisive else None,
        "positions_compared": int(positions),
        "positions_near_tie": int(positions - n_decisive),
        "top1_disagreements_near_tie": int(np.count_nonzero(~agree & ~decisive)),
        "error_stage": None,
        "error": None,
    }


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G0.4 rule holds."""
    diff, agreement = m["max_abs_logit_diff"], m["top1_agreement"]
    ok = diff is not None and diff <= MAX_ABS_LOGIT_DIFF and agreement is not None and agreement == 1.0
    return "pass" if ok else "fail"


def _unmeasured(*, error_stage: str, error: str, positions_compared: int | None = None) -> dict[str, Any]:
    return {
        "max_abs_logit_diff": None,
        "top1_agreement": None,
        "positions_compared": positions_compared,
        "positions_near_tie": None,
        "error_stage": error_stage,
        "error": error,
    }


def _tail(path: Path, lines: int = 30) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) if path.exists() else ""


# ---------------------------------------------------------------------------
# nf-jax side
# ---------------------------------------------------------------------------


def levanter_side(work: Path) -> int:
    """Export G0.1's final checkpoint and compute Levanter's logits on the fixed batch. Writes levanter_side.json."""
    side: dict[str, Any] = {"stage": "load_config", "error": None, "checkpoint": None,
                            "versions": _common.package_versions(JAX_VERSIONS)}
    try:
        from levanter.checkpoint import latest_checkpoint_path
        from levanter.main import export_lm_to_hf

        g01_work = g01_cpu_train.WORK_DIR
        config = _levanter.parse_config(_levanter.draccus_args({
            **_levanter.OVERRIDES, **_levanter.run_overrides(g01_work, g01_cpu_train.RUN_ID, _common.DATA_CACHE_DIR)
        }))

        side["stage"] = "find_checkpoint"
        checkpoint = latest_checkpoint_path(str(g01_cpu_train.checkpoint_dir()))
        side["checkpoint"] = _common.rel(Path(checkpoint))
        final = f"step-{_levanter.NUM_TRAIN_STEPS - 1}"
        if Path(checkpoint).name != final:
            raise RuntimeError(f"G0.1's latest checkpoint is {checkpoint}, not its final one ({final})")

        side["stage"] = "export"
        try:
            export_lm_to_hf.main(export_lm_to_hf.ConvertLmConfig(
                trainer=config.trainer,
                checkpoint_path=checkpoint,
                output_dir=str(work / "hf"),
                model=config.model,
                use_cpu=True,
            ))
        except Exception:  # noqa: BLE001  the export path is what this gate tests
            side["error"] = traceback.format_exc()[-4000:]
            return 0

        side["stage"] = "levanter_logits"
        tokens, logits = levanter_logits(config, checkpoint)
        np.savez(work / "levanter_logits.npz", tokens=tokens, logits=logits)
        side["stage"] = "done"
    except Exception:  # noqa: BLE001
        side["error"] = traceback.format_exc()[-4000:]
        return 1
    finally:
        (work / "levanter_side.json").write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return 0


def levanter_logits(config: Any, checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    """The fixed batch, and Levanter's float32 logits for it from ``checkpoint``, shaped (batch, pos, vocab)."""
    import equinox as eqx
    import haliax as hax
    import jax.numpy as jnp
    from levanter.layers.attention_mask import AttentionMask
    from levanter.utils.jax_utils import local_cpu_mesh

    if SEQ > config.model.max_seq_len:
        # JAX clamps out-of-range gathers, so Levanter would not fail on its own.
        raise ValueError(f"the fixed batch has {SEQ} positions; the model has {config.model.max_seq_len}")
    with local_cpu_mesh():
        model = _levanter.load_trained_model(config, checkpoint)
        vocab = model.Vocab.size
        tokens = np.random.default_rng(BATCH_SEED).integers(0, vocab, size=(BATCH, SEQ)).astype(np.int32)
        Batch, Pos = hax.Axis("batch", BATCH), config.model.max_Pos.resize(SEQ)
        named = hax.named(jnp.asarray(tokens), (Batch, Pos))
        logits = eqx.filter_jit(lambda m, t: m(t, attn_mask=AttentionMask.causal()))(model, named)
        logits = logits.rearrange((Batch, Pos, model.Vocab)).array
    return tokens, np.asarray(logits, dtype=np.float32)


if __name__ == "__main__":
    sys.exit(main())
