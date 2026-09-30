"""Gate G1.2: Parity with Levanter.

Spec: docs/specs/phase-1.md, Gates table, G1.2 row. Report: reports/phase1/g12_parity.json.

Builds a Levanter GPT-2 at size S with the vocabulary in
``data/tinystories/tokenizer.json``, seeded, and exports it to Hugging Face
format with Levanter's own converter, the path G0.4 used. Loads the export with
``load_hf_gpt2`` and compares logits with Levanter's on the probe batch:
``probe_batch(validation, EVAL_TOKENS, PROBE_SEED, PROBE_SEQUENCES, 256)``.

* ``n_params``: the loaded params' element count. ``n_params_expected``:
  ``GPT2Config.from_size("S", vocab).n_params()``.
* ``max_abs_logit_diff`` over every position and vocabulary entry.
* ``top1_agreement`` over positions where Levanter's top-2 margin exceeds 1e-3;
  ``positions_near_tie`` counts the rest.

Measurements: REQUIRED_MEASUREMENTS["G1.2"].


Every gate: runs on the Mac only, in nf-jax
(``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/<script>``);
refuses a dirty tree unless ``--allow-dirty``; refuses to overwrite its report unless
``--overwrite``; writes one schema 2 report with ``write_report``, measurements
passed through ``plain``; prints progress at least every 30 s during anything
longer than a minute; returns exit code 0 whenever it wrote a valid report,
whatever the gate's status, and non-zero only when it could not produce one.
"""

# How it measures:
#
# * Levanter's model is ``Gpt2Config(...).build(Vocab, key=PRNGKey(PARAM_SEED))``
#   with size S's shape: 4 layers, hidden 256, 4 heads, MLP scale 4, 256 positions.
#   Levanter's defaults set the GELU variant, layer-norm epsilon and attention
#   scaling, which is what this gate checks our model against.
# * The export is ``Gpt2Config.hf_checkpoint_converter().save_pretrained``, the
#   converter ``export_lm_to_hf`` calls, without a trained checkpoint to load.
#   Levanter's converter loads a tokenizer when it is built; the gate gives it our
#   vocabulary as a word-level Hugging Face tokenizer in a local folder, through
#   ``Gpt2Config.tokenizer``, so the export never needs the GPT-2 tokenizer from
#   the Hub. The tokenizer is not saved with the export.
# * Levanter's logits come from the model it built, under ``local_cpu_mesh`` with a
#   causal mask, as in G0.4. Ours come from ``GPT2Model.fast`` on the loaded export.
# * An export or load error is the path under test failing: the report says fail,
#   with the stage and the error, and nulls for what could not be measured. An
#   error building Levanter's model or computing its logits means the gate could
#   not measure: no report, exit code 1.

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

GATE_ID = "G1.2"
REPORT_PATH = "reports/phase1/g12_parity.json"

PARAM_SEED = 0
MAX_ABS_LOGIT_DIFF = 1e-4
TIE_MARGIN = 1e-3
VERSIONS = ("jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "transformers", "tokenizers",
            "safetensors", "numpy")


class CouldNotMeasure(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    report = _common.start(args, REPORT_PATH)
    _common.require_data()
    work = _common.fresh_dir(_common.RUN_DIR / "g12_parity", overwrite=args.overwrite)
    tokenizer = _common.load_tokenizer()
    config = _common.model_config(len(tokenizer.vocab))
    probe = _common.probe(_common.load_split("validation"), config.seq)
    try:
        measurements = measure(config, tokenizer.vocab, probe, work)
    except CouldNotMeasure as exc:
        print(f"{GATE_ID} could not measure: {exc}\nNo report written.", file=sys.stderr)
        return 1
    measurements.update(tokenizer_sha256=tokenizer.sha256, versions=_common.phase0.package_versions(VERSIONS))
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G1.2 rule holds."""
    diff, agreement = m["max_abs_logit_diff"], m["top1_agreement"]
    ok = (m["n_params"] is not None and m["n_params"] == m["n_params_expected"]
          and diff is not None and diff <= MAX_ABS_LOGIT_DIFF and agreement is not None and agreement == 1.0)
    return "pass" if ok else "fail"


def measure(config: Any, vocab: tuple[str, ...], probe: np.ndarray, work: Path) -> dict[str, Any]:
    """Build and export Levanter's model into ``work``, load it into ours, and compare logits on ``probe``."""
    from nablaframe.model.gpt2 import GPT2Model, load_hf_gpt2

    if len(vocab) != config.vocab:
        raise CouldNotMeasure(f"the tokenizer has {len(vocab)} entries, the config {config.vocab}")
    try:
        _stage("building Levanter's model and its logits on the probe batch")
        levanter_model, levanter_config, levanter_n_params = build_levanter(config, vocab, work / "hf_tokenizer")
        levanter_logits = levanter_forward(levanter_model, levanter_config, probe)
    except Exception as exc:  # noqa: BLE001  Levanter's own side failing is not a parity result
        raise CouldNotMeasure(traceback.format_exc()[-4000:]) from exc

    base = {"n_params_expected": config.n_params(), "levanter_n_params": levanter_n_params,
            "probe_shape": list(probe.shape), "probe_batch_hash": _batch_hash(probe), "param_seed": PARAM_SEED,
            "tie_margin": TIE_MARGIN, "hf_export_dir": _common.rel(work / "hf")}
    stage = "export"
    try:
        _stage(f"exporting to Hugging Face format in {_common.rel(work / 'hf')}")
        export(levanter_model, levanter_config, work / "hf")
        stage = "load"
        _stage("loading the export into our model")
        params = load_hf_gpt2(work / "hf", config)
        stage = "forward"
        _stage("comparing logits")
        ours = np.asarray(GPT2Model(config).fast(params, probe))
    except Exception:  # noqa: BLE001  the export and load path is what this gate tests
        return {**base, "n_params": None, "max_abs_logit_diff": None, "top1_agreement": None,
                "positions_near_tie": None, "positions_compared": None,
                "error_stage": stage, "error": _common.scrub(traceback.format_exc()[-4000:])}
    return {**base, "n_params": n_params(params), **compare(levanter_logits, ours), "error_stage": None, "error": None}


def _stage(text: str) -> None:
    """One line as each stage starts: at size S a stage can run over a minute on CPU with nothing else printed."""
    print(f"{GATE_ID}: {text}", flush=True)


def n_params(params: Any) -> int:
    import jax

    return int(sum(np.size(leaf) for leaf in jax.tree.leaves(params)))


def compare(levanter_logits: np.ndarray, ours: np.ndarray) -> dict[str, Any]:
    """Max abs logit diff everywhere, and top-1 agreement where Levanter's top-2 margin exceeds TIE_MARGIN."""
    if levanter_logits.shape != ours.shape:
        raise CouldNotMeasure(f"logit shapes differ: Levanter {list(levanter_logits.shape)}, ours {list(ours.shape)}")
    lev = levanter_logits.astype(np.float64).reshape(-1, levanter_logits.shape[-1])
    our = ours.astype(np.float64).reshape(-1, ours.shape[-1])
    positions = lev.shape[0]
    if not (np.all(np.isfinite(lev)) and np.all(np.isfinite(our))):
        return {"max_abs_logit_diff": None, "top1_agreement": None, "positions_near_tie": None,
                "positions_compared": positions, "nonfinite_logits": True}
    top2 = np.sort(lev, axis=-1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) > TIE_MARGIN
    agree = np.argmax(lev, axis=-1) == np.argmax(our, axis=-1)
    n_decisive = int(np.count_nonzero(decisive))
    return {
        "max_abs_logit_diff": float(np.max(np.abs(lev - our))),
        "top1_agreement": float(np.count_nonzero(agree & decisive)) / n_decisive if n_decisive else None,
        "positions_near_tie": int(positions - n_decisive),
        "positions_compared": int(positions),
        "top1_disagreements_near_tie": int(np.count_nonzero(~agree & ~decisive)),
        "nonfinite_logits": False,
    }


def hf_tokenizer(vocab: tuple[str, ...], folder: Path) -> Path:
    """Our vocabulary as a word-level Hugging Face tokenizer, saved to ``folder`` for Levanter's converter."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    from nablaframe.data.tokenizer import EOS, UNK

    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(vocab)}, unk_token=UNK))
    backend.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=UNK, eos_token=EOS)
    if len(fast) != len(vocab):
        raise ValueError(f"the Hugging Face tokenizer has {len(fast)} entries, our vocabulary {len(vocab)}")
    fast.save_pretrained(str(folder))
    return folder


def build_levanter(config: Any, vocab: tuple[str, ...], tokenizer_dir: Path) -> tuple[Any, Any, int]:
    """Levanter's GPT-2 with our model's shape, seeded: the model, its config and its parameter count."""
    import equinox as eqx
    import haliax as hax
    import jax
    from levanter.models.gpt2 import Gpt2Config
    from levanter.utils.jax_utils import local_cpu_mesh

    if config.d_mlp % config.d_model:
        raise ValueError(f"d_mlp {config.d_mlp} is not a multiple of d_model {config.d_model}")
    levanter_config = Gpt2Config(
        max_seq_len=config.seq, hidden_dim=config.d_model, num_heads=config.n_heads, num_layers=config.n_layers,
        mlp_scale=config.d_mlp // config.d_model, tokenizer=str(hf_tokenizer(vocab, tokenizer_dir)),
    )
    with local_cpu_mesh():
        model = levanter_config.build(hax.Axis("vocab", config.vocab), key=jax.random.PRNGKey(PARAM_SEED))
    count = int(sum(np.size(leaf) for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))))
    return model, levanter_config, count


def levanter_forward(model: Any, levanter_config: Any, tokens: np.ndarray) -> np.ndarray:
    """Levanter's float32 logits for ``tokens``, shaped (batch, pos, vocab)."""
    import equinox as eqx
    import haliax as hax
    import jax.numpy as jnp
    from levanter.layers.attention_mask import AttentionMask
    from levanter.utils.jax_utils import local_cpu_mesh

    batch, pos = tokens.shape
    if pos > levanter_config.max_seq_len:
        # JAX clamps out-of-range gathers, so Levanter would not fail on its own.
        raise ValueError(f"the probe has {pos} positions; the model has {levanter_config.max_seq_len}")
    with local_cpu_mesh():
        Batch, Pos = hax.Axis("batch", batch), levanter_config.max_Pos.resize(pos)
        named = hax.named(jnp.asarray(tokens, dtype=jnp.int32), (Batch, Pos))
        logits = eqx.filter_jit(lambda m, t: m(t, attn_mask=AttentionMask.causal()))(model, named)
        logits = logits.rearrange((Batch, Pos, model.Vocab)).array
    return np.asarray(logits, dtype=np.float32)


def export(model: Any, levanter_config: Any, out: Path) -> None:
    """Levanter's own Hugging Face converter, as export_lm_to_hf uses it."""
    from levanter.utils.jax_utils import local_cpu_mesh

    with local_cpu_mesh():
        levanter_config.hf_checkpoint_converter().save_pretrained(model, str(out), save_tokenizer=False)


def _batch_hash(batch: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(batch)


if __name__ == "__main__":
    sys.exit(main())
