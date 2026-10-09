# Phase 4, delta 1: the zoo's tasks and measurements, `jswap`, and the trainer's batch source

Date: 2026-10-08. Spec: [../specs/phase-4.md](../specs/phase-4.md). Rulings: D29 to D35.

This delta ships:

- **Stubs for four new packages:** `zoo/tasks.py`, `ground_truth.py`, `induction.py` and `two_hop.py`.
- **15 tests.**
- **Contract text** for three existing modules:
  - the trainer, which gains its batch source
  - `interventions/spec.py` and `apply.py`, which gain `jswap` and one-head zeroing

The Phase 4 gates come in delta 2.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase4-delta1.zip
python3 apply_phase4_delta1.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `15 failed, 159 passed`.

## Contract changes

**`src/nablaframe/train/loop.py`** (added to `run_until`'s docstring)

- "`run_until` gains a keyword `batch_source=None`. When given, step k's batch is `batch_source(k)`, which must be an int32 array shaped (batch, seq + 1), or ValueError before that step trains, and `stream` may then be None. When omitted, the loop is unchanged, and the reference run trains bit for bit as before (D29)."

**`src/nablaframe/interventions/spec.py`** (added to the module docstring)

- "`OPERATIONS` gains `jswap`, taking `J` (a content hash) and the token ids `a` and `b` (non-negative ints) (D33). `zero` takes either no arguments or exactly `head`, a non-negative int: zeroing one attention head's output."

**`src/nablaframe/interventions/apply.py`** (added to the module docstring)

- "`jswap` applies at residual-stream taps only, `hook_resid_pre`, `hook_resid_mid` or `hook_resid_post`. J must be shaped (d_model, d_model), and both tokens must be below the vocabulary size. Anything else raises ValueError before anything runs. With u = Jᵀw for each token's unembedding row w, it adds (β − α)(u_a − u_b) at the selected positions, where α u_a + β u_b is the residual's least-squares component in their span. Its operand enters the compiled program as an argument, as every operand does. With a == b it adds no operation at all, so it reproduces the unintervened pass bit for bit (D25)."
- "A `zero` with `head` applies only at `attn.hook_z`. It zeroes that head's slice at the selected positions and leaves every other head untouched. A head at or beyond the tap's head count raises ValueError."

## Threads

Paste into the Project, in two waves. Each wave's two threads share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P4] Zoo tasks and the trainer's batch source", on Opus at high effort:
Implement src/nablaframe/zoo/tasks.py against its docstring and docs/specs/phase-4.md (D29, D32), and the "src/nablaframe/train/loop.py" part of "Contract changes" in docs/deltas/phase-4-delta-1.md, written into run_until's docstring verbatim. Change only those two files. The reference run must train bit for bit as before: run tests/test_train.py and tests/test_train_d9.py as well as the new tests. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_tasks.py tests/test_train.py tests/test_train_d9.py` passes with no skips, and every test that passed before still does.

Thread "[P4] jswap and one-head zeroing", on Opus at high effort:
Implement the "src/nablaframe/interventions/spec.py" and "src/nablaframe/interventions/apply.py" parts of "Contract changes" in docs/deltas/phase-4-delta-1.md, written into each module's docstring verbatim. Change only those two files. Every operand enters the compiled program as an argument, as PR #38 established. Every D25 identity is bit for bit; if one holds only approximately, stop and say so in the PR. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_jswap.py tests/test_interventions_apply.py tests/test_interventions_spec.py` passes with no skips, and every test that passed before still does.
```

Once both have merged:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P4] Ground truth and induction", on Sonnet at high effort:
Implement src/nablaframe/zoo/ground_truth.py and src/nablaframe/zoo/induction.py against their docstrings and docs/specs/phase-4.md. Change only those two files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_ground_truth.py tests/test_zoo_induction.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Two-hop measurements", on Sonnet at high effort:
Implement src/nablaframe/zoo/two_hop.py against its docstring and docs/specs/phase-4.md. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_two_hop.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** 15 new cases fail and all 159 existing tests pass.
- **On a private reference implementation, not shipped:** all 174 pass. That includes the trainer's default path bit for bit unchanged, `jswap`'s exact self-swap, its coordinate exchange checked in float64, and one-head zeroing leaving the other heads untouched.
- **Planted bugs:** 15 of 16 are caught by the test aimed at them. The caught ones include:
  - an ignored or unchecked batch source
  - an induction window whose second half is drawn fresh, or whose first half repeats tokens
  - mechanism positions off by one
  - two-hop facts never shuffled, and a broken chain whose answer appears in the facts
  - accuracy ties going to the highest id
  - induction targets after the first occurrence rather than the most recent
  - formation-step ties going to the later step
  - `jswap` moving the wrong way
  - one-head zeroing that zeroes every head
  - `head` allowed on any operation
  - patch swaps judged on the source's facts
  - rank ties going to the higher id

**The one miss can't be wrong.** It was a self-swap that still runs its operation. With `a == b`, the two J-lens directions are identical, so the pseudo-inverse's rows are equal, and both the coefficient vector and the direction difference come out exactly zero. Running the operation therefore adds exact zeros, and the pass stays bit for bit identical. The contract's "no operation at all" is the cleaner design, but the identity holds either way.

**Not verified:** anything on the Mac, and any zoo model actually trained. That's G4.1, in delta 2.
