"""Gate G0.3: Throughput.

Spec: docs/specs/phase-0.md, Gates table, G0.3 row.
Environment: nf-jax. Run on apolloh0's Mac only; never in the cloud.

Measures: Training tokens/s on CPU at size S (4 layers, d=256, batch 32x256) and size M (4 layers, d=512, batch 64x256).
Pass: S >= 25,000 tokens/s. M is recorded only.
On failure, revisit scale targets.

Writes reports/phase0/g03_throughput.json with nablaframe.reports.write_report. The report's
measurements must include REQUIRED_MEASUREMENTS["G0.3"], and its status must obey
the status rules in nablaframe/reports.py. Returns the process exit code: 0 when a
valid report was written, whatever the gate's status; non-zero only when the gate
could not produce a report.
"""

# How it measures:
#
# * The model is Levanter's GPT-2 (Haliax and Equinox, the Phase 1 JAX stack) at
#   each size, with Levanter's defaults for everything the spec leaves open: head
#   size 64, MLP 4x d, gradient checkpointing on, float32. The optimizer is
#   Levanter's default AdamConfig, as in G0.1.
# * One training step is forward, backward and optimizer update under one
#   eqx.filter_jit, on a fixed batch of random tokens, so data loading is excluded.
# * WARMUP_STEPS steps (compilation included) are discarded, then TIMED_STEPS steps
#   are timed back to back, blocking on the result of each. tokens/s is
#   batch x sequence length x TIMED_STEPS over their total time.
# * The spec leaves the vocabulary open (4,096 or 8,192). Each size is measured at
#   both, and tokens_per_s_S and tokens_per_s_M use GATE_VOCAB. See the PR.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

GATE_ID = "G0.3"
REPORT_PATH = "reports/phase0/g03_throughput.json"

ENV_NAME = "nf-jax"
MIN_TOKENS_PER_S_S = 25_000
VOCAB_SIZES = (4096, 8192)
GATE_VOCAB = 8192
WARMUP_STEPS = 3
TIMED_STEPS = 20
DATA_SEED = 0
MODEL_SEED = 0
VERSIONS = ("jax", "jaxlib", "marin-levanter", "marin-haliax", "equinox", "optax", "numpy")


class Size(NamedTuple):
    name: str
    n_layers: int
    d_model: int
    batch: int
    seq: int

    @property
    def n_heads(self) -> int:
        return self.d_model // 64


SIZES = (Size("S", 4, 256, 32, 256), Size("M", 4, 512, 64, 256))


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    _common.require_macos()
    report_file = _common.report_path(REPORT_PATH)
    _common.check_writable(report_file, overwrite=args.overwrite)

    import jax

    if jax.default_backend() != "cpu":
        print(f"{GATE_ID} measures CPU, but JAX's default backend is {jax.default_backend()}.", file=sys.stderr)
        return 1

    runs = {}
    for size in SIZES:
        for vocab in VOCAB_SIZES:
            print(f"{GATE_ID}: size {size.name}, vocab {vocab}", flush=True)
            runs[f"{size.name}_vocab{vocab}"] = measure_size(size, vocab)

    measurements = {
        "tokens_per_s_S": runs[f"S_vocab{GATE_VOCAB}"]["tokens_per_s"],
        "tokens_per_s_M": runs[f"M_vocab{GATE_VOCAB}"]["tokens_per_s"],
        "gate_vocab": GATE_VOCAB,
        "runs": runs,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "versions": _common.package_versions(VERSIONS),
    }
    report = _common.build_report(GATE_ID, status(measurements), measurements, ENV_NAME)
    return _common.emit_report(report, report_file, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G0.3 rule holds."""
    s = m["tokens_per_s_S"]
    return "pass" if s is not None and s >= MIN_TOKENS_PER_S_S else "fail"


def measure_size(size: Size, vocab: int, *, warmup: int = WARMUP_STEPS, timed: int = TIMED_STEPS) -> dict[str, Any]:
    """Train ``warmup + timed`` steps of Levanter's GPT-2 at ``size`` and time the last ``timed``."""
    import equinox as eqx
    import haliax as hax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from levanter.layers.attention_mask import AttentionMask
    from levanter.models.gpt2 import Gpt2Config
    from levanter.models.lm_model import LmExample
    from levanter.optim.config import AdamConfig

    config = Gpt2Config(
        hidden_dim=size.d_model, num_layers=size.n_layers, num_heads=size.n_heads, max_seq_len=size.seq
    )
    Vocab = hax.Axis("vocab", vocab)
    Batch = hax.Axis("batch", size.batch)
    Pos = config.max_Pos.resize(size.seq)

    model = config.build(Vocab, key=jax.random.PRNGKey(MODEL_SEED))
    optimizer = AdamConfig().build(warmup + timed)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    rng = np.random.default_rng(DATA_SEED)
    tokens = hax.named(jnp.asarray(rng.integers(0, vocab, size=(size.batch, size.seq)), dtype=jnp.int32), (Batch, Pos))
    loss_weight = hax.broadcast_to(LmExample.causal_loss_mask(Pos).astype(jnp.float32), (Batch, Pos))
    example = LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=AttentionMask.causal())

    @eqx.filter_jit
    def train_step(model, opt_state, example):
        def loss_fn(m):
            return m.compute_next_token_loss(example, key=None).scalar()

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    step_seconds, losses = [], []
    for _ in range(warmup + timed):
        start = time.perf_counter()
        model, opt_state, loss = train_step(model, opt_state, example)
        loss = float(jax.block_until_ready(loss))
        step_seconds.append(time.perf_counter() - start)
        losses.append(loss)

    timed_seconds = sum(step_seconds[warmup:])
    tokens_per_step = size.batch * size.seq
    n_params = sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    return {
        "tokens_per_s": tokens_per_step * timed / timed_seconds,
        "n_layers": size.n_layers,
        "d_model": size.d_model,
        "n_heads": size.n_heads,
        "d_mlp": size.d_model * config.mlp_scale,
        "vocab": vocab,
        "batch": size.batch,
        "seq": size.seq,
        "n_params": n_params,
        "gradient_checkpointing": config.gradient_checkpointing,
        "step_seconds": [round(s, 6) for s in step_seconds],
        "losses_finite": all(np.isfinite(losses)),
    }


if __name__ == "__main__":
    sys.exit(main())
