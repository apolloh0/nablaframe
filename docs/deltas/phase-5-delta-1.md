# Phase 5, delta 1: the dictionaries package and `dpatch`

Date: 2026-10-10. Spec: [../specs/phase-5.md](../specs/phase-5.md). Rulings: D40, D43 and D45.

This delta ships:

- **Stubs for a new package,** `dictionaries/`:
  - `core.py`: the dictionary record, encoding and decoding, content hashes, cache keys and files
  - `train.py`: training every kind, and the Jacobian SAE's Jacobian through its MLP

  Each docstring is its contract. `__init__.py` is complete.
- **27 tests** in three new files.
- **Contract text** for `interventions/spec.py` and `apply.py`, which gain `dpatch`. The apply script writes it into both docstrings.
- **Delta 1's tests,** listed in the spec's Tests section.

The meters, the edge meter and the gates come in later deltas.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase5-delta1.zip
python3 apply_phase5_delta1.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `21 failed, 204 passed, 6 errors`:

- the dictionary core: 9 failed
- training: 11 failed
- `dpatch`: 1 failed, and 6 errors from the fixture that builds its dictionaries

## Contract changes

**`src/nablaframe/interventions/spec.py`** (added to the module docstring)

- "`OPERATIONS` gains `dpatch` (D45), taking `dictionary` and `source` (content hashes) and, in its other form, `latents` as well: a list of distinct non-negative ints, possibly empty, stored sorted, so two specs that list the same latents in another order hash equal."

**`src/nablaframe/interventions/apply.py`** (added to the module docstring)

- "`dpatch` (D45) applies at one of its dictionary's output taps, which picks the site. For its `dictionary`, `resolve` returns a `Dictionary`, checked with `content_hash`. Each of these raises ValueError before anything runs: any other tap, a transcoder's input tap among them; a resolved value that isn't a `Dictionary` or doesn't hash to its name; a `source` batch shaped differently from the tokens; a latent at or beyond the site's `active_width`. The codes `f_target` and `f_source` are `encode` of the target and source batches' values at the site's input tap, one row per sequence and position, from `model.instrumented` with the post-swap parameters and none of the spec's other edits. At the selected positions it adds `decode_delta` of `f_source − f_target`, with every latent not in `latents` zeroed when `latents` is given, so the target's error term is kept. A tap whose values aren't shaped (batch, pos, the dictionary's width there) raises ValueError. Where the change is zero at every selected position, as with a source equal to the target or an empty `latents`, it adds no operation at all, so it reproduces the unintervened pass bit for bit (D25)."

**`src/nablaframe/dictionaries/core.py`** and **`train.py`:** each docstring is its contract.

### Settled in the contracts

D45 left these open. Each follows the nearest existing rule:

- **`dpatch` reads both codes without the spec's other edits,** as `patch` reads its source's values. A `dpatch` after an upstream edit therefore patches toward the source's code from the target's unedited code. The faithful-code and single-latent tests use one edit, where the two readings agree.
- **A latent must be one the site can use.** A Staircase site uses only its share of the latents, so a latent beyond `active_width` is refused rather than ignored.
- **Training is spelled out to the draw:** the key split, the initial weights, the batch rows, AuxK's dead latents and the learning-rate schedule. Two implementations can differ in floating point, but on one backend the same settings and data give the same bytes, as G5.1 requires.

## Threads

Paste into the Project, in two waves. Wave 2 builds on the core, so it waits for wave 1 to merge.

**Wave 1:**

```
Start one thread now, without proposing it first.

Thread "[P5] Dictionary core", on Sonnet at high effort:
Implement src/nablaframe/dictionaries/core.py against its docstring and docs/specs/phase-5.md (D40). Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dictionaries_core.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once the core has merged, has two threads, which share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P5] Dictionary training", on Opus at high effort:
Implement src/nablaframe/dictionaries/train.py against its docstring and docs/specs/phase-5.md (D40, D43). Change only that file. On one backend, the same settings and data must give byte-identical arrays. The planted-feature tests are the evidence that training works: if one fails, diagnose the trainer and say what you found in the PR; never loosen a test. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dictionaries_train.py` passes with no skips, and every test that passed before still does.

Thread "[P5] dpatch", on Opus at high effort:
Implement the dpatch bullets in the module docstrings of src/nablaframe/interventions/spec.py and src/nablaframe/interventions/apply.py, quoted under "Contract changes" in docs/deltas/phase-5-delta-1.md, against docs/specs/phase-5.md (D45). Change only those two files. Every operand enters the compiled program as an argument, as PR #38 established. Both D25 identities are bit for bit; if one holds only approximately, stop and say so in the PR. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dpatch.py tests/test_interventions_apply.py tests/test_interventions_spec.py tests/test_jswap.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** the 27 new tests fail (21 failures, 6 errors), and all 204 existing tests pass.
- **After wave 1:** with the core implemented, its 9 tests pass, and the 18 training and `dpatch` tests still fail on their own stubs, not on the core.
- **On a private reference implementation, not shipped:** all 231 pass with no skips.
- **The training bars have margin.** The planted-feature tests were run on the reference across data seeds and initialisation seeds, and each bar was then set with margin:

  | Test | Measured | Bar |
  | --- | --- | --- |
  | TopK: fraction of variance unexplained, held out | 0.04 to 0.06 | below 0.1 |
  | TopK: each planted feature's best cosine | at least 0.977 | above 0.9 |
  | Transcoder: fraction of variance unexplained | 0.06 to 0.08 | below 0.1 |
  | Transcoder: output directions found | 23 or 24 of 24 | at least 22 |
  | Staircase: fraction of variance unexplained, either block | 0.02 to 0.07 | below 0.1 |
  | Jacobian SAE: mean absolute Jacobian, λ 1 over λ 0 | 0.16 to 0.19 | below 0.5 |
  | Jacobian SAE: fraction of variance unexplained, either site | at most 0.28 | below 0.4 |

  The transcoder's bar counts output directions found rather than requiring all 24, because one seed in nine missed one.
- **Planted bugs:** all 51 are caught by the test aimed at them. They include:
  - settings and sites that should be refused: `k` above the width, a penalty on a kind other than the Jacobian SAE, a Staircase out of block order, a transcoder writing another block
  - arrays left writable or non-finite, and scales left unchecked
  - encoding without the clip at zero, without the input scale, or past a Staircase site's share; decoding with the bias in the change, or without the output scale
  - a content hash without the key or the settings; a cache key without the code, or blind to batch order; a file loaded under another hash's name
  - training whose scales lack the √d, whose decoder rows aren't renormalised, or which reports progress at every step
  - training with AuxK off, no decay, the penalty ignored, the seed ignored, one key for both units, a fresh batch generator every step, an unscaled transcoder encoder or `b_pre` at zero
  - the pair Jacobian transposed or unmasked, and GELU without its cubic term
  - `dpatch` latents unsorted, repeated or floats
  - `dpatch` at any tap, a latent bound off by one, an unchecked dictionary hash or source batch, or a zero change that still runs
  - `dpatch` ignoring `latents`, moving the code the wrong way, reading codes at the output tap or from before the swaps, replacing instead of adding, or decoding with the bias

  Four of them escaped the first draft of the tests, which were strengthened before shipping: the seed reaching the batches but not the weights, no decay, the Jacobian's mask on output latents that don't fire, and a source batch of the wrong size failing only later, by accident.

**Not verified:** anything on the Mac, and any dictionary trained on a zoo model through this package. Feasibility trained them in its own program. The package's first run on the zoo is G5.1.
