"""G2.5: the J-lens cost probe (D15). Always passes; its numbers set D15's method.

At the final checkpoint, for the residual stream entering block 2, with ``f`` the map to
the last block's output through the model's intervention hook:

* The exact reference on causal prefixes. The model is causal, so the same-position
  Jacobian at position t depends only on positions up to t. The exact computation on each
  sequence's first ``EXACT_PREFIX`` positions is therefore exact for those positions, at
  ``exact_jacobian_bytes(32, 256)`` per sequence rather than 17 GB. ``exact_positions`` is
  the number of positions it covers (64 sequences: 2,048), ``exact_seconds`` its time.
* The estimator on those same prefix positions, like for like: ``estimator_errors`` maps
  draws (1, 8, 32) to the relative Frobenius error against the exact reference, and
  ``estimator_seconds`` to the times.
* Convergence on full-length sequences from the training stream with seed 1:
  ``fit_sizes`` is ``FIT_SIZES``, ``fit_change_per_doubling`` the relative change in J
  between consecutive sizes, ``fit_seconds`` the times, and ``projected_seconds`` the
  projected time for 4 layers and 28 checkpoints at the largest size.
* Also the summed J, as anthropics/jacobian-lens computes it: for each output dimension,
  one backward pass with a one-hot cotangent at every position, averaged over source
  positions, on the largest fit sample at full length. `summed_seconds` is its time, and
  `summed_vs_same_position` the relative Frobenius difference from the same-position
  estimate at that size. Both are recorded, not judged.
* ``reference_method``: how ``anthropics/jacobian-lens`` computes J, in a sentence, and
  ``crosscheck_feasible``: whether it can run on our Hugging Face export, both read from
  its source by the implementing thread and written into this module.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g25_jlens_cost.py``. Takes --overwrite and --allow-dirty. It reads
#   G1.4's last trace checkpoint and G1.1's training stream, and writes nothing outside
#   its report. It needs no trace store.
# * ``f`` replaces ``blocks.2.hook_resid_pre`` with its argument through
#   ``GPT2Model.instrumented``'s ``interventions``, and returns the last block's
#   ``hook_resid_post`` for that one sequence. Nothing after block 2 reads the tokens
#   except through the residual stream, so the tokens ``f`` passes are zeros. ``f`` is
#   wrapped in ``jax.jit`` once per sequence length, so the exact reference and the
#   estimator each compile once and then reuse the program. Every time includes its
#   compilation.
# * The sample: ``batch_for_step(train, FIT_SEED, 0, max(FIT_SIZES) // seq, seq)``, the
#   data pipeline's own windows drawn with seed 1 in place of data seed 0, each window's
#   first ``seq`` tokens: 128 windows of 256 at size S. Every position counts. The fit
#   sizes take the first 8, 32 and 128 windows, so each sample contains the one before.
#   The exact reference takes the first 64 windows' first 32 tokens, so it covers 2,048
#   positions, as many as the smallest fit size. Its residual streams come from the
#   instrumented pass on those 32-token prefixes.
# * Exact: ``exact_same_position_jacobian(f, x)`` per prefix, averaged over the 64
#   sequences, which weights every position equally, as the estimator does.
# * Estimator: ``estimate_same_position_jacobian(f, xs, draws, ESTIMATOR_SEED)`` on the
#   same 64 prefixes, at 1, 8 and 32 draws. Error is ``||estimate - exact||_F /
#   ||exact||_F``. Maps are keyed by the draw count as a string.
# * Fit: the estimator at ``FIT_DRAWS`` (1) draw on each fit size's full-length windows.
#   Consecutive fit sizes are 4 times apart, not 2: ``fit_change_per_doubling`` holds,
#   for each consecutive pair "<smaller>-><larger>", ``||J_larger - J_smaller||_F /
#   ||J_larger||_F``, as this docstring defines it, between consecutive sizes.
# * Summed J: ``summed_jacobian(f, xs)`` on the largest fit size's windows, the same
#   residual streams the last fit used. Per sequence, one forward pass (``jax.vjp``) and
#   then one backward pass per output dimension i, with the cotangent 1 at dimension i
#   of every position's output and 0 elsewhere. Row i of that sequence's J is the
#   resulting gradient averaged over all source positions; J is the mean over sequences,
#   in float64. Unlike the reference, no position is skipped, the source is the residual
#   stream entering block 2 rather than a block's output, and the target is the last
#   block's output, as for every other J here. ``summed_vs_same_position`` is
#   ``||J_summed - J_same||_F / ||J_same||_F``, with J_same the largest fit. The time
#   includes compilation, as every other time here does.
# * Projection: one forward-mode pass per direction runs only the blocks after the
#   lensed layer, so a layer's cost is taken as proportional to the number of blocks
#   it runs through. ``projected_seconds`` is the largest fit size's time, times the
#   number of trace checkpoints, times the blocks run for all 4 layers (4 + 3 + 2 + 1)
#   over the blocks run here (2). ``projected_seconds_flat`` assumes every layer costs
#   what layer 2 costs, for comparison.
# * Memory: the exact reference holds each prefix's full Jacobian, which is
#   ``exact_jacobian_bytes(32, 256)`` (268 MB) in float32, and 537 MB once cast to
#   float64, plus JAX's forward-mode intermediates for 8,192 tangents at once.
#
# What anthropics/jacobian-lens does, read from its source at commit
# 581d398613e5602a5af361e1c34d3a92ea82ba8e (2026-07-02), files jlens/fitting.py,
# jlens/hf.py, jlens/hooks.py, jlens/protocol.py and jlens/lens.py:
#
# * ``jacobian_for_prompt`` runs one forward pass on the prompt replicated ``dim_batch``
#   times, keeps the graph, and runs ceil(d_model / dim_batch) backward passes. Batch
#   element b carries a one-hot cotangent on output dimension (start + b) at every valid
#   target position at once, so the gradient at source position p is
#   sum over p' >= p of d h_final[p'] / d h_l[p]: summed over the current and later
#   target positions, not the same-position Jacobian this spec defines. Rows of J_l are
#   then the mean of that gradient over valid source positions. Its module docstring
#   names the per-position estimator, d h_final[p] / d h_l[p] averaged over p, as giving
#   "a slightly different J_l", but the repository doesn't implement it.
# * Valid positions exclude the first 16 (``SKIP_FIRST_N_POSITIONS``, as attention
#   sinks) and the last; this spec counts every position in the fit windows.
# * h_l is the output of block l, captured by a forward hook on ``layers[l]``, and the
#   target defaults to the last block's output (``target_layer = n_layers - 1``), so
#   source layers are 0 to n_layers - 2. This spec's J_l is taken from the residual
#   stream entering block l, which is block l - 1's output for l >= 1, and the
#   embeddings for l = 0, which the reference doesn't fit.
# * ``fit`` averages the per-prompt J_l over prompts with equal weight per prompt, in
#   float32, from text prompts truncated to ``max_seq_len`` (default 128) tokens.
#   The lens is then ``unembed(J_l @ h)``: final norm then LM head.
# * Running it on our Hugging Face export: ``from_hf`` finds GPT-2's layout
#   (``transformer.h``, ``ln_f``, ``wte``, ``lm_head``), which is the layout
#   ``load_hf_gpt2`` reads and Levanter's converter writes. Its dependencies are torch,
#   huggingface_hub, transformers >= 5.5 and numpy, which the nf-torch lane meets: built
#   from uv.lock on Linux it has torch 2.14.0 and transformers 5.17.0. Checked in a
#   scratch folder, not in any gate: in that lane, with jlens on the path from its clone,
#   ``from_hf`` and ``fit`` ran on a freshly initialized GPT-2 with size S's shape saved
#   by ``save_pretrained`` with a word-level tokenizer, and ``load_hf_gpt2`` read the same
#   file. Gaps, none of which the reference's code needs changed for:
#   jlens is in no lane, so adding it needs apolloh0's ruling; no tool exports the reference
#   run's checkpoints to Hugging Face format (G1.2 exports a fresh Levanter model), though
#   our parameter names map one to one; and ``fit`` takes text, so the export needs our
#   word-level tokenizer saved beside it, as G1.2 builds it, or a small ``LensModel``
#   that takes token ids. A cross-check compares like with like only if one side computes
#   the other's estimator: the reference's summed, hook-based J_l is not this spec's.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.5"
REPORT_NAME = "g25_jlens_cost.json"
LAYER = 2
ESTIMATOR_DRAWS = (1, 8, 32)
ESTIMATOR_SEED = 0
FIT_DRAWS = 1
FIT_SEED = 1
REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
REFERENCE_METHOD = (
    "For each output dimension, a one-hot cotangent is placed at every valid target position at once and"
    " backpropagated with torch autograd from the last block's output to block l's output (forward hooks), giving"
    " at each source position p the sum over p' >= p of d h_final[p'] / d h_l[p]; J_l's rows are that gradient"
    " averaged over source positions 16 to T - 2, then averaged over prompts.")
CROSSCHECK_FEASIBLE = True
CROSSCHECK_NOTES = (
    "from_hf recognises the GPT-2 layout our export uses (transformer.h, ln_f, wte, lm_head)",
    "its dependencies (torch, huggingface_hub, transformers>=5.5, numpy) are met by the nf-torch lane; jlens"
    " itself is in no lane, so adding it needs a ruling",
    "from_hf and fit ran in the nf-torch lane on a size-S GPT-2 saved with save_pretrained and a word-level"
    " tokenizer, a file load_hf_gpt2 also reads (checked in a scratch folder by the implementing thread)",
    "no tool exports the reference run's checkpoints to Hugging Face format yet; G1.2 exports a fresh Levanter"
    " model, and our parameter names map one to one onto GPT-2's",
    "fit takes text prompts, so the export needs our word-level tokenizer saved with it, or a LensModel that"
    " takes token ids",
    "it computes a different J_l: summed over current and later target positions, from block l's output,"
    " skipping the first 16 positions and the last; a cross-check must compute the same estimator on both sides",
)
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    step = ref.steps[-1]
    measurements = measure(ref.model, ref.params_for(step), p1.load_split("train"), n_checkpoints=len(ref.steps))
    measurements["step"] = step
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, "pass", measurements, report, overwrite=args.overwrite)


def measure(model: Any, params: Any, train: np.ndarray, *, n_checkpoints: int, layer: int = LAYER,
            exact_prefix: int = _common.EXACT_PREFIX, fit_sizes: tuple[int, ...] = _common.FIT_SIZES,
            estimator_draws: tuple[int, ...] = ESTIMATOR_DRAWS) -> dict[str, Any]:
    from nablaframe.data.corpus import batch_for_step
    from nablaframe.views.jlens_probe import estimate_same_position_jacobian, exact_same_position_jacobian

    config = model.config
    seq, d, n_layers = config.seq, config.d_model, config.n_layers
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} is outside a {n_layers}-layer model")
    if any(size % seq for size in fit_sizes) or any(b <= a for a, b in zip(fit_sizes, fit_sizes[1:])):
        raise ValueError(f"fit sizes {fit_sizes} must increase and be multiples of the window length {seq}")
    exact_sequences = fit_sizes[0] // exact_prefix
    windows = batch_for_step(np.asarray(train), FIT_SEED, 0, fit_sizes[-1] // seq, seq)[:, :seq].astype(np.int32)
    if exact_sequences > len(windows) or exact_prefix > seq:
        raise ValueError(f"the exact reference needs {exact_sequences} windows of {exact_prefix} tokens")
    source, target = f"blocks.{layer}.hook_resid_pre", f"blocks.{n_layers - 1}.hook_resid_post"
    m: dict[str, Any] = {
        "layer": layer, "source_tap": source, "target_tap": target,
        "exact_prefix": exact_prefix, "exact_sequences": exact_sequences,
        "exact_bytes_per_sequence": _common.exact_jacobian_bytes(exact_prefix, d),
        "full_sequence_bytes": _common.exact_jacobian_bytes(seq, d),
        "fit_seed": FIT_SEED, "fit_draws": FIT_DRAWS, "estimator_seed": ESTIMATOR_SEED,
    }

    # The exact reference on causal prefixes, and the estimator on the same positions.
    prefixes = windows[:exact_sequences, :exact_prefix]
    xs = residual(model, params, prefixes, source)
    f = mapping(model, params, exact_prefix, source, target)
    started = time.perf_counter()
    exact = np.zeros((d, d), dtype=np.float64)
    progress = p1.Progress("exact Jacobian, prefixes", exact_sequences)
    for x in xs:
        exact += exact_same_position_jacobian(f, x)
        progress.advance()
    exact /= exact_sequences
    m["exact_seconds"] = time.perf_counter() - started
    m["exact_positions"] = exact_sequences * exact_prefix
    m["exact_norm"] = float(np.linalg.norm(exact))
    print(f"{GATE_ID}: exact reference over {m['exact_positions']:,} positions in {m['exact_seconds']:.1f} s",
          flush=True)

    errors, seconds = {}, {}
    for draws in estimator_draws:
        started = time.perf_counter()
        estimate = estimate_same_position_jacobian(f, xs, draws, ESTIMATOR_SEED)
        seconds[str(draws)] = time.perf_counter() - started
        errors[str(draws)] = relative_error(estimate, exact)
        print(f"{GATE_ID}: estimator at {draws} draws: relative error {errors[str(draws)]:.4f},"
              f" {seconds[str(draws)]:.1f} s", flush=True)
    m["estimator_draws"] = list(estimator_draws)
    m["estimator_errors"] = errors
    m["estimator_seconds"] = seconds

    # Convergence over fit sizes, on full-length windows.
    f = mapping(model, params, seq, source, target)
    fits, fit_seconds = [], {}
    for size in fit_sizes:
        xs = residual(model, params, windows[:size // seq], source)
        started = time.perf_counter()
        fits.append(estimate_same_position_jacobian(f, xs, FIT_DRAWS, ESTIMATOR_SEED))
        fit_seconds[str(size)] = time.perf_counter() - started
        print(f"{GATE_ID}: fit at {size:,} positions in {fit_seconds[str(size)]:.1f} s", flush=True)
    change = {f"{a}->{b}": relative_error(fits[i], fits[i + 1])
              for i, (a, b) in enumerate(zip(fit_sizes, fit_sizes[1:]))}
    m["fit_sizes"] = list(fit_sizes)
    m["fit_seconds"] = fit_seconds
    m["fit_change_per_doubling"] = change
    m["fit_vs_exact_errors"] = {str(size): relative_error(fit, exact) for size, fit in zip(fit_sizes, fits)}
    print(f"{GATE_ID}: change between consecutive fit sizes: "
          + ", ".join(f"{pair} {value:.4f}" for pair, value in change.items()), flush=True)

    # The reference implementation's definition: summed over target positions.
    started = time.perf_counter()
    summed = summed_jacobian(f, xs)
    m["summed_seconds"] = time.perf_counter() - started
    m["summed_positions"] = fit_sizes[-1]
    m["summed_vs_same_position"] = relative_error(summed, fits[-1])
    print(f"{GATE_ID}: summed J at {fit_sizes[-1]:,} positions in {m['summed_seconds']:.1f} s; relative"
          f" difference from the same-position estimate {m['summed_vs_same_position']:.4f}", flush=True)

    # The projected cost for every layer at every trace checkpoint.
    largest = fit_seconds[str(fit_sizes[-1])]
    blocks_here = n_layers - layer
    blocks_all = sum(n_layers - each for each in range(n_layers))
    m["projected_seconds"] = largest * n_checkpoints * blocks_all / blocks_here
    m["projected_seconds_flat"] = largest * n_checkpoints * n_layers
    m["projected_layers"] = n_layers
    m["projected_checkpoints"] = n_checkpoints
    m["projected_basis"] = (f"the {fit_sizes[-1]:,}-position fit's time, times {n_checkpoints} checkpoints, times"
                            f" {blocks_all} blocks run over all {n_layers} layers, over the {blocks_here} run here")
    print(f"{GATE_ID}: projected {m['projected_seconds'] / 3600:.2f} h for {n_layers} layers and {n_checkpoints}"
          " checkpoints", flush=True)

    m["reference_method"] = REFERENCE_METHOD
    m["reference_commit"] = REFERENCE_COMMIT
    m["crosscheck_feasible"] = CROSSCHECK_FEASIBLE
    m["crosscheck_notes"] = list(CROSSCHECK_NOTES)
    return m


def residual(model: Any, params: Any, tokens: np.ndarray, tap: str) -> np.ndarray:
    """The traced residual stream at ``tap`` for ``tokens``, from the instrumented pass."""
    _, recorded = model.instrumented(params, tokens, [tap])
    return np.asarray(recorded[tap])


def mapping(model: Any, params: Any, length: int, source: str, target: str) -> Any:
    """``f``: one sequence's residual stream at ``source`` (length, d) to ``target``, jit-compiled once."""
    import jax
    import jax.numpy as jnp

    tokens = jnp.zeros((1, length), dtype=jnp.int32)

    def f(x: Any) -> Any:
        _, recorded = model.instrumented(params, tokens, [target], {source: lambda _value: x[None]})
        return recorded[target][0]

    return jax.jit(f)


def summed_jacobian(f: Any, xs: np.ndarray) -> np.ndarray:
    """J as anthropics/jacobian-lens defines it: summed over target positions, averaged over source positions.

    For each sequence in ``xs`` (N, T, d), one forward pass, then for each output dimension i
    one backward pass with a one-hot cotangent at dimension i of every position. Row i is the
    gradient averaged over source positions. Returns the mean over sequences, (d, d) float64,
    with entry [i, j] the summed derivative of output dimension i by input dimension j.
    """
    import jax
    import jax.numpy as jnp

    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"xs must be sequences of shape (N, T, d), got {xs.shape}")
    n, _, d = xs.shape

    @jax.jit
    def per_sequence(x: Any) -> Any:
        out, pull = jax.vjp(f, x)

        def row(i: Any) -> Any:
            cotangent = jnp.zeros_like(out).at[:, i].set(1.0)
            return pull(cotangent)[0].mean(axis=0)

        return jax.lax.map(row, jnp.arange(out.shape[-1]))

    total = np.zeros((d, d), dtype=np.float64)
    progress = p1.Progress("summed J, sequences", n)
    for x in xs:
        total += np.asarray(per_sequence(jnp.asarray(x)), dtype=np.float64)
        progress.advance()
    return total / n


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    """``||estimate - reference||_F / ||reference||_F``."""
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


if __name__ == "__main__":
    raise SystemExit(main())
