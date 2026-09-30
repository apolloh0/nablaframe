# Phase 1, delta 1: stubs, tests and contract

Date: 2026-09-29. Spec: [../specs/phase-1.md](../specs/phase-1.md).

This delta ships the final Phase 1 spec, stubs for work packages 2 to 5, 28 tests (31 cases), the contract text for work package 1's changes to three existing files, the proposed rulings D7 and D8, and the Phase 0 doc fixes. Gate stubs for work package 6 come in a later delta, once the code they drive exists.

## Apply

Unzip at the repo root, then run the edit script from the repo root:

```
python3 apply_phase1_delta1.py
git diff
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `30 failed, 47 passed`: the 46 existing tests, one new test that pins Phase 0's reports as still valid, and the 30 new cases the threads implement. Plain `uv run pytest` gives `11 failed, 47 passed, 3 skipped`, the skips being the JAX test modules. Commit and push, then start the threads below.

## Proposed rulings

The edit script adds these after D6:

```
| D7 | 2026-09-29 | Nablaframe owns its reference model (GPT-2 in JAX, taps built in), training loop, batch order and checkpoints. Levanter stays a dependency as the parity reference (G1.2) and for Hugging Face export; the Phase 0 gate scripts keep using its trainer | Taps can't be added to Levanter's model without patching, and the model you read should be code you own. Batch k depending only on the seed and k makes replay exact by construction. Costs: determinism is proven again (G1.3), and continuity is kept by logit parity with Levanter (G1.2) |
| D8 | 2026-09-29 | The vocabulary is 4,096 if G1.1 measures an unknown-word rate of at most 1.0% on the TinyStories validation split at 4,096, otherwise 8,192. The rule is fixed before the measurement | 4,096 is 24% faster at size S (G0.3: 15,255 against 12,260 tokens/s) and gives embeddings 25% of the parameters instead of 39%; the J-lens can't name a word mapped to `<unk>` |
```

## Contract changes for work package 1

The carry-over thread writes each bullet into the named file's docstring, verbatim, and implements it.

**`src/nablaframe/trace/store.py`**

- Module docstring, replacing the one-writer paragraph from delta 2: "One writer per store at a time, enforced by a lock. A handle takes an exclusive `fcntl.flock` on `root/.lock` at its first append or supersede, rereads the log once it holds the lock, and keeps the lock until `close()`. If another handle holds it, the write raises StoreError at once. Opening and reading never lock."
- `__init__`: "`mode="w"` is the default. `TraceStore(root, mode="r")` requires the store to exist (FileNotFoundError otherwise), never creates anything, and raises StoreError on append or supersede. `close()` releases the lock, and TraceStore is a context manager."
- `append`, replacing the shard-writing sentence: "Shards are written through the module-level `save_file` to `shards/<event_id>.safetensors.tmp`, then renamed into place with `os.replace`, so a partial shard never carries its final name. A leftover `.tmp` file is ignored and overwritten. An unreadable file under a shard's final name can then only mean corruption, and raises IntegrityError."

**`src/nablaframe/reports.py`**

- New paragraph: "Schema version 2 adds two required fields: `git_commit`, 40 lowercase hex characters, and `git_dirty`, a bool. Version 1 reports have neither and stay valid. Phase 1 gates write version 2 only; a Phase 1 report on version 1 is invalid."
- New paragraph: "`GATE_IDS` stays the Phase 0 tuple, because Phase 0's tests and summary loop over it. Phase 1's gates are `PHASE1_GATE_IDS = ("G1.1", "G1.2", "G1.3", "G1.4", "G1.5", "G1.6")`, and the validator accepts both."
- G0.2 status rule, added sentence: "A null hash was not measured and never counts as equal to another hash." The same rule applies to G1.3.
- Required measurements for Phase 1:
  - G1.1: `vocab_selected`, `unk_rate_4096`, `unk_rate_8192`, `unk_rate_4096_cased`, `builds_identical`, `train_tokens`, `validation_tokens`, `stories_dropped`
  - G1.2: `n_params`, `n_params_expected`, `max_abs_logit_diff`, `top1_agreement`, `positions_near_tie`
  - G1.3: the same six as G0.2
  - G1.4: `steps`, `tokens`, `nonfinite_seen`, `checkpoints_written`, `checkpoints_scheduled`, `final_val_loss`, `bigram_val_loss`
  - G1.5: `bitwise_equal`, `steps_checked`, `defect_detected`
  - G1.6: `events_recorded`, `sample_size`, `sample_matches`, `retrace_new_events`, `store_bytes`, `store_bytes_expected`
- Status rules for Phase 1:
  - G1.1 pass requires `builds_identical` is true and `vocab_selected` is 4,096 exactly when `unk_rate_4096 <= 0.01`, otherwise 8,192 (D8).
  - G1.2 pass requires `n_params == n_params_expected`, `max_abs_logit_diff <= 1e-4` and `top1_agreement == 1.0`.
  - G1.3 follows G0.2's rules, including `conditional_pass` and the hash-agreement rule under every status. `conditional_pass` is allowed only for G0.2 and G1.3.
  - G1.4 pass requires `nonfinite_seen` is false, `checkpoints_written == checkpoints_scheduled` and `final_val_loss < bigram_val_loss`.
  - G1.5 pass requires `bitwise_equal` and `defect_detected` both true.
  - G1.6 pass requires `sample_matches == sample_size`, `retrace_new_events == 0` and `store_bytes` within 10% of `store_bytes_expected`.

**`src/nablaframe/oracle/mode_equality.py`**

- `compare_logits`, added sentence: "When the fast and instrumented logits differ in dtype, the result is a failure, not a ValueError: instrumentation that changes the dtype is a finding, not caller misuse. D3's dtype check applies only when both share a dtype." This states behavior merged in PR #4; no code changes.

## Threads

Paste into the Project. The model and data threads share no files and have no dependencies, so they go first:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P1] Reference model", on Opus at high effort:
Implement src/nablaframe/model/gpt2.py against its docstring and docs/specs/phase-1.md. Read Levanter's GPT-2 at the pinned version for the GELU variant, layer-norm epsilon and attention scaling, and say which you found. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_model.py` passes with no skips.

Thread "[P1] Data", on Sonnet at high effort:
Implement src/nablaframe/data/tokenizer.py and src/nablaframe/data/corpus.py against their docstrings and docs/specs/phase-1.md. Standard library and numpy only. Change only those two files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_data.py` passes.
```

Then, as slots free up:

```
Start thread "[P1] Carry-overs", on Sonnet at high effort:
Implement the "Contract changes for work package 1" section of docs/deltas/phase-1-delta-1.md. Write each bullet into the named file's docstring verbatim, then implement it. Change only src/nablaframe/trace/store.py, src/nablaframe/reports.py and src/nablaframe/oracle/mode_equality.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase1_store_reports.py tests/test_store.py tests/test_store_recovery.py tests/test_reports.py tests/test_reports_rules.py tests/test_oracle.py tests/test_oracle_d3.py` passes.
```

```
Start thread "[P1] Trainer", on Opus at high effort, after the model and data threads have merged:
Implement src/nablaframe/train/loop.py, checkpoint.py and heartbeat.py against their docstrings and docs/specs/phase-1.md. Change only those three files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_train.py` passes with no skips.
```

```
Start thread "[P1] Trace engine", on Opus at high effort, after the carry-overs and the trainer have merged:
Implement src/nablaframe/trace/engine.py against its docstring and docs/specs/phase-1.md. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips: 77 of 77.
```

## Verification record

**Verified** in a Linux sandbox on an exact copy of `main` after PR #5, with the lockfile's jax 0.11.2, equinox 0.13.8 and optax 0.2.6:

- **On the stubs:** 30 new cases fail and 47 pass: the original 46, plus the pin on Phase 0's reports. In a JAX-free base environment: 11 failed, 47 passed, 3 skipped by name.
- **On a private reference implementation, not shipped:** all 77 pass.
- **Planted bugs:** each of 15 is caught by the test aimed at it. The first run missed one, a store that doesn't reread its log after taking the lock, and `test_second_writer_is_refused` was tightened.

**Not verified:** anything on the Mac; Levanter's exact GPT-2 hyperparameters, which the model thread reads; the TinyStories download.

**Found while building the reference, and now part of the contract:**

- **Scalars in checkpoints.** safetensors stores zero-dimensional arrays with shape `(1,)`, so loading must restore template shapes, or a resumed optimizer drifts.
- **Keep `GATE_IDS` as Phase 0's tuple.** Extending it would have broken Phase 0's committed report test and its summary script.
- **One compiled program for recomputation.** It must match the one used to record.
