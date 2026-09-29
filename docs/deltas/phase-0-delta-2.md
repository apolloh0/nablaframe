# Phase 0, delta 2: contract fixes

Date: 2026-09-28. Follows [delta 1](phase-0-delta-1.md) and PRs #1 to #3.

Reviewing PRs #1 to #3 turned up gaps in the delta 1 contract, not in the implementations. This delta closes them before the gate scripts are written. It ships four new test files, one ruled correction to an existing test, and the contract text a fix thread implements.

## Apply

Unzip at the repo root, then run the edit script from the repo root and review:

```
python3 apply_delta2_edits.py
git diff
uv run pytest
```

Expect `15 failed, 31 passed` from pytest. The 15 failures are the new contract rules, which the fix thread implements. Commit, push, then send the fix thread its task below.

## New files

| Path | Covers |
| --- | --- |
| `tests/test_schema_rules.py` | Canonical dtype names (pins today's behavior), axis shape, event `kind` and `mode` |
| `tests/test_oracle_d3.py` | D3: tolerance dtype applies only to its own dtype; source reports are gate report paths |
| `tests/test_reports_rules.py` | G0.1 needs the full run; G0.2's flag agrees with its hashes; `write_report` makes folders; nulls allowed, numpy scalars not |
| `tests/test_store_recovery.py` | Retrying an interrupted append; an orphan shard with other content |

## Proposed rulings

The edit script adds these to the Phase decisions table:

```
| D3 | 2026-09-28 | A tolerance record applies only to logits of its own dtype: the canonical policy raises ValueError otherwise. Its source report must match `reports/phase<N>/<name>.json` | A tolerance measured on float32 logits means nothing on float16 ones; the pattern merged in PR #2 accepted `reports/../x.json`. Discrimination tests: `tests/test_oracle_d3.py`, which fails on the merged oracle |
| D4 | 2026-09-28 | Test defect: two G0.2 fixtures in `tests/test_reports.py` claimed `identical: false` while their three hashes agreed. Corrected to differing hashes | The fixtures contradicted themselves; the new G0.2 consistency rule rightly rejects them. The corrected test passes on the validator merged in PR #2 |
```

## Contract changes

The fix thread writes each bullet into the named file's docstring, verbatim, and implements it.

**`src/nablaframe/trace/schema.py`**

- TapSpec: "`dtype` must be the canonical numpy name, that is `np.dtype(dtype).name == dtype`, so `"float32"` is accepted and `"f4"` is not." Already true; this states it.
- TapSpec: "`axes` must be a tuple of `Axis`, each with a non-empty str name. Anything else raises ValueError."
- TraceEvent: "Construction raises ValueError when `kind` is not in `EVENT_KINDS` or `mode` is not in `MODES`. `tap` is not checked: checkpoint markers have none, and valid names depend on layer count."

**`src/nablaframe/oracle/tolerance.py` and `mode_equality.py`** (D3)

- `SOURCE_REPORT_PATTERN = r"^reports/phase[0-9]+/[A-Za-z0-9_-]+\.json$"`, with the comment "A source report is a committed gate report: `reports/phase<N>/<name>.json`, nothing else (D3)."
- `compare_logits`: "Under `"instrumented_canonical"`, a tolerance applies only to logits of its own dtype: if the logits' dtype name differs from `tolerance.dtype`, raise ValueError (D3)."

**`src/nablaframe/reports.py`**

- STATUS RULES, G0.1: "G0.1 pass requires nonfinite_seen is False, loss_drop_fraction >= 0.30 and steps >= 1000."
- STATUS RULES, new line: "G0.2, under every status: identical must be True exactly when hash_run_a, hash_run_b and hash_resumed are all equal."
- Measurements, new paragraph: "Null is allowed for anything a gate could not measure; every number must be finite. Values must be plain JSON types: convert numpy scalars with int(), float() or bool()."
- Measurements, new paragraph: "G0.4's `top1_agreement` is the fraction of positions whose top-1 token agrees, counted only over positions where the Levanter logits' top-2 margin exceeds 1e-3. `positions_near_tie` counts the positions left out."
- `write_report`: "Creates missing parent folders."

**`src/nablaframe/trace/store.py`**

- Module docstring: "One writer per store at a time. Two open handles each keep their own index and can record the same event twice. Phase 1 adds locking before anything writes traces concurrently."
- `append`: replace the sentence about an existing shard with: "If the shard file already exists but no logged event references it, it is an orphan from an interrupted append. If its tensor hashes to this event's `sha256`, reuse it and append the log line. Otherwise raise IntegrityError and leave the file in place for inspection."

## Implementation threads

**Contract fixes**, next. Paste into the Project:

```
Start thread "[P0] Contract fixes", on Opus at high effort:
Implement the "Contract changes" section of docs/deltas/phase-0-delta-2.md. Write each bullet into the named file's docstring verbatim, then implement it. Change only src/nablaframe/trace/schema.py, src/nablaframe/trace/store.py, src/nablaframe/oracle/tolerance.py, src/nablaframe/oracle/mode_equality.py and src/nablaframe/reports.py. D3 is ruled in docs/decisions.md. Done when `uv run pytest` passes in full: 46 of 46.
```

**Gate scripts**, after the contract fixes merge. This replaces the thread 4 message in the delta 1 notes:

```
Start thread "[P0] Gate scripts", on Opus at high effort:
Implement gates/g01_cpu_train.py to gates/g06_mode_probe.py, gates/summarize.py and src/nablaframe/lanes/jax_ref/toy.py against their docstrings, src/nablaframe/reports.py and docs/specs/phase-0.md. Change only those files. Never run a gate for results and never commit anything under reports/. Convert numpy scalars to Python int, float and bool before putting them in measurements. If metaljax fails to load, G0.5 still writes a fail report: runs false, null for anything it couldn't measure, plus the macOS version and the load error in measurements. Done when the full test suite passes.
```

## Verification record

**Verified** in a Linux sandbox, against an exact copy of `main` at `141626c`:

- On `main` as merged: 15 new cases fail, one for each contract change, and 5 pass, pinning today's behavior. The original 26 still pass.
- With a private reference fix, not shipped: all 46 pass.
- The D4 correction passes on `main` as merged, so it changes no working behavior.

**Not verified:** the fix thread's implementation, until its PR arrives.
