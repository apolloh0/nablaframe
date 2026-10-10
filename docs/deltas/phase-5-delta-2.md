# Phase 5, delta 2: measuring a dictionary on a model

Date: 2026-10-10. Spec: [../specs/phase-5.md](../specs/phase-5.md). Rulings: D40 and D41.

This delta ships:

- **A stub for `dictionaries/measure.py`:** the meters, selectivity, readability and the patches that D41's tests are built from. Its docstring is its contract.
- **11 tests** in one new file.
- **`train.py`'s contract, settled.** PR #62 made five choices the docstring left open, and they were ruled on in review. The apply script writes them into the docstring. The code already does what they say, so no thread is needed for them.
- **Delta 2's tests,** listed in the spec's Tests section.

The edge-sparsity meter comes in delta 3, and the gates after it.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase5-delta2.zip
python3 apply_phase5_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `11 failed, 231 passed`: the measurement tests.

## What `measure.py` holds, and what it leaves to the gates

Every function is a measurement on arrays or on one model, testable on the tiny model. The gates choose the windows, sites, positions and variables, and hold the results to the bars.

| Function | Gives | For |
| --- | --- | --- |
| `activations` | each tap's values on the unintervened pass, in chunks | training data, and every measurement's inputs |
| `fvu`, `dead_share` | the fraction of variance unexplained; the share of a site's latents that never fire | the meters |
| `readout_loss` | the task's loss and accuracy at the readout positions | the meters |
| `spliced_logits`, `ablated_logits` | the pass with the reconstruction spliced in; the pass with the site mean-ablated | the meters |
| `f1_table`, `mean_best_f1` | each latent's F1 against each value of a variable; the mean of each value's best | selective latents |
| `assigned` | the latents whose best F1, over every variable at a position, is for this one | the single-latent patch |
| `probe_accuracy` | a ridge probe's accuracy | readability |
| `patch_predictions` | the predictions after the residual `patch`, the whole code's `dpatch`, and the assigned latents' `dpatch`, from the same sources | the faithful code and the single-latent patch |

Nothing in it caches, so the cache key question PR #61 raised doesn't arise here.

### Settled in the contract

These follow the feasibility program the bars were set from:

- **A value counts toward selectivity only if it occurs at least five times** at the position. A latent can match a rarer value perfectly by chance.
- **The probe fits on even rows and tests on odd ones.** It centres on the fitting rows, and its ridge is scaled by the Gram matrix's mean diagonal.
- **The mean ablation is per position, over the windows being measured:** `run_intervened`'s `mean` edit, with those windows as its reference.
- **The splice is a measurement, not an edit,** so it isn't an operation in the spec. It replaces the site's output tap at every position with the reconstruction of the input tap's clean value. That value enters the compiled program as an argument, as PR #38 requires.

## Contract changes

**`src/nablaframe/dictionaries/train.py`** (four passages of the docstring, written by the apply script)

- **Data and refusals:** "`data` holds, per site in order, `(inputs, outputs)`: finite float32 arrays shaped (N, d), with one N and one d for every site and both sides, since each encoder starts as its decoder transposed; for a `topk`, `staircase` or `jsae` site they must be equal, and no site's inputs or outputs may be all zero. A `jsae` needs `mlp_weights` of its block, and nothing else takes them. Anything else, a malformed `key` or a `progress` that isn't callable among it, raises ValueError before anything trains."
- **Loss:** "per site, the squared error of the scaled output summed over features and averaged over rows, divided by `d_out`; plus `aux_weight` times AuxK's: among the site's `active_width` latents that haven't fired on that site for `dead_after` steps, the `min(4k, m // 2)` largest pre-activations, clipped at zero and decoded without the decoder bias, reconstruct the site's error, held constant, their squared error normalised as the main term's. For a `jsae` with `lam > 0`, plus `lam` times the mean absolute entry of `pair_jacobian` on the batch's first `jac_batch` rows, differentiated through the codes as well as the weights."
- **Learning rate:** "Adam (optax defaults). With `W = min(warmup, steps // 10)` and `D = int(steps * decay_fraction)`, the learning rate at step `s`, counted from 0, is `lr · min(s / W, (steps − s) / D, 1)`, a term left out when its `W` or `D` is 0: it rises from 0, holds, and falls to `lr / D` at the last step. After each step every `W_dec` row is renormalised to unit length."
- **Progress:** "`progress(step, steps, losses)`, when given, is called once after each step `⌈i · steps / 10⌉` for `i` from 1 to 10, steps counted from 1, `losses` holding each site's reconstruction term."

The learning rate is the one choice with a visible consequence. The first step's rate is 0, where the main model's warmup (D9) starts at `lr / warmup_steps`. D9's reason doesn't arise here: it kept the step-1 trace checkpoint from duplicating step 0, and a dictionary keeps no checkpoint per step. A step at rate 0 is one step of 2,000 or 3,000, and the schedule matches the reference implementation the delta 1 tests were tuned on.

**`src/nablaframe/dictionaries/measure.py`:** its docstring is its contract.

## Threads

Paste into the Project:

```
Start one thread now, without proposing it first.

Thread "[P5] Dictionary measurements", on Sonnet at high effort:
Implement src/nablaframe/dictionaries/measure.py against its docstring and docs/specs/phase-5.md (D41). Change only that file. Every intervened pass goes through run_intervened, except the splice, which the docstring describes; its replacement enters the compiled program as an argument, as PR #38 established. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dictionaries_measure.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stub:** the 11 new tests fail, and all 231 existing tests pass, `train.py`'s with its new docstring.
- **On a private reference implementation, not shipped:** all 242 pass with no skips.
- **Planted bugs:** all 23 are caught by the test aimed at them. They include:
  - activations that drop the last short chunk
  - a fraction of variance unexplained that isn't centred, or that reconstructs the output instead of the input
  - dead latents counted past a Staircase site's share, or a zero code counted as firing
  - a loss scored against the position's own token, ties going to the higher id, or a log-softmax in float32
  - a splice that reconstructs the transcoder's output tap, or splices nothing
  - an ablation that averages over positions instead of windows
  - F1 with a zero code counted as active, a minimum count that must be exceeded rather than reached, or one denominator for every latent
  - latents assigned to the last variable on a tie, or a floor that must be exceeded
  - a probe that isn't centred, fits every row, or leaves its ridge unscaled
  - one position's latents used at every position, the code patched at every position, the prediction read one position early, or latents for positions that weren't patched

  One of them escaped the first draft: one position's latents used at both. The tiny model's predictions didn't move enough to tell. The test now uses 64 windows and a dictionary with a large output scale, and checks that it can tell the two apart.

**Not verified:** anything on the Mac, and any measurement of a trained dictionary on a zoo model. That's G5.2 and G5.3.
