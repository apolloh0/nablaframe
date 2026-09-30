# Phase 1, delta 2: gate stubs and D9

Date: 2026-09-30. Spec: [../specs/phase-1.md](../specs/phase-1.md). Follows [delta 1](phase-1-delta-1.md) and PRs #6 to #9.

This delta ships stubs for the six Phase 1 gates and their summary, a shared helper, 11 new test cases, and the proposed ruling D9 with the one-line trainer change it asks for.

## Apply

Unzip at the repo root, then run the edit script from the repo root:

```
python3 apply_phase1_delta2.py
git diff
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

With the trace engine not yet merged, expect `10 failed, 78 passed`: 5 engine tests, 2 D9 tests and 3 gate-helper tests fail; the 6 gate-declaration checks pass on the stubs by design. If the engine has merged, expect `5 failed, 83 passed`.

## Proposed ruling

The edit script adds this after D8:

```
| D9 | 2026-09-30 | Warmup starts at lr / warmup_steps rather than 0, so update 1 moves the parameters and a step-n checkpoint holds n updates that each changed the model. This departs from optax's and Levanter's default schedule | PR #9 found that with warmup from 0, params after update 1 are bit-identical to step 0, so the step-1 trace checkpoint would duplicate step 0 and Phase 2's scrubber would open on two identical frames. D7 made the trainer ours; parity with Levanter covers only the forward pass. Tests: `tests/test_train_d9.py` |
```

## Contract change for the D9 thread

In `src/nablaframe/train/loop.py`, write into the docstring verbatim, then implement:

- "`learning_rate(train_config, step)` returns, as a float, the learning rate the optimizer uses at that step: linear warmup from `lr / warmup_steps` at step 0 to `lr` at `warmup_steps`, then cosine decay to `lr * min_lr_ratio` at `total_steps` (D9). The optimizer builds its schedule from the same function, so there is exactly one schedule."

Also update the `_optimizer` docstring, which now says the warmup rises from 0.

## The gate stubs

`gates/phase1/` holds `_common.py`, `g11_tokenizer.py` to `g16_trace.py`, and `summarize.py`. Each docstring is the contract:

- **Measurements.** Exactly what each gate measures and how, including every required measurement.
- **Progress.** Every gate prints progress at least every 30 seconds during anything longer than a minute: a `Heartbeat` for training, and a `Progress` line for tokenizing, encoding and tracing. Any gate that writes to a log file prints its `scripts/watch.py` command first.
- **Clean tree.** Gates refuse to start with uncommitted changes to tracked files, unless `--allow-dirty` is passed, because a report must be reproducible from its commit. Untracked files, such as earlier reports, don't count.
- **G1.4 can resume.** It records a manifest whose `run_id` goes into every checkpoint, and keeps trace and resume checkpoints in separate folders, since both are named `step-<n>.safetensors`. After an interruption, `--resume` continues from the latest resume checkpoint.

The tests in `tests/test_phase1_gates.py` pin what can be checked off the Mac:

- each script's gate id and report path
- the 28-step trace schedule
- the storage formula: 98,566,144 bytes per checkpoint and 2,759,852,032 in total
- the commit envelope
- the conversion of numpy values to plain JSON types

## Threads

The D9 change touches only the trainer, so it can start now, alongside the engine thread:

```
Start thread "[P1] D9 warmup", on Sonnet at medium effort:
Implement the "Contract change for the D9 thread" section of docs/deltas/phase-1-delta-2.md. D9 is ruled in docs/decisions.md. Change only src/nablaframe/train/loop.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_train.py tests/test_train_d9.py` passes with no skips.
```

After the engine and D9 have merged:

```
Start thread "[P1] Phase 1 gates", on Opus at high effort:
Implement every file in gates/phase1/ against its docstring and docs/specs/phase-1.md. You may read and reuse gates/_common.py but not change it. Change only files in gates/phase1/. The gates run on apolloh0's Mac only and Hugging Face is blocked from the cloud, so never run a gate for results and never write under reports/. Exercise each gate's logic on a small synthetic corpus and a tiny model config in a scratch folder, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips: 88 of 88.
```

## Gate runs on the Mac, after the gates thread merges

In order, since each gate reads the previous ones' outputs. Every command names its extra:

```
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g11_tokenizer.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g12_parity.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g13_determinism.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g14_reference.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g15_oracle.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/g16_trace.py
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/summarize.py
```

Rough times:

- **G1.1:** the dataset download, then tokenizing. It prints progress as it goes.
- **G1.2 and G1.5:** minutes.
- **G1.3:** about 20 minutes, more if it needs the single-threaded retry.
- **G1.4:** about 1.8 to 2.3 hours.
- **G1.6:** minutes, writing about 2.8 GB of traces.

## Verification record

**Verified** from a fresh clone of my mirror of `main` after PRs #6 to #9, with the jax lane built from the lockfile:

- **On the stubs:** 10 failed and 78 passed, with a clean collection.
- **On a private reference** (D9 applied to the merged trainer, plus the helper functions): all new tests pass, and every existing test except the unimplemented engine's still passes. That includes the trainer's bit-identical resume under D9's schedule.

**Not verified:** anything on the Mac, the TinyStories download, and the gate logic itself, which the gates thread exercises on synthetic data.
