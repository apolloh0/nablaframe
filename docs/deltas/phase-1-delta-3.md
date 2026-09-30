# Phase 1, delta 3: engine rules and the public store lock

Date: 2026-09-30. Spec: [../specs/phase-1.md](../specs/phase-1.md). Follows [delta 2](phase-1-delta-2.md) and PRs #10 and #11.

Reviewing PR #10 showed two gaps in the trace engine's contract, both reproduced by probes. The engine also reached for the store's private lock method, because the store offers no public one. This delta closes all three before G1.6 runs, fixes spec prose that still described an early design, and pins PR #11's ruling on zero warmup. It ships 4 new test cases and contract text for two existing files. It needs no new D-entry: neither rule changes the oracle's policy or a tolerance.

## Apply

Unzip at the repo root, then run the edit script from the repo root:

```
python3 apply_phase1_delta3.py
git diff
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

The 3 engine-rule tests fail until the thread below lands, and the zero-warmup test passes already. If the gates thread hasn't merged, its 3 helper tests also fail: `6 failed, 86 passed`. If it has: `3 failed, 89 passed`.

## Contract changes

One thread writes each bullet into the named file's docstring, verbatim, and implements it.

**`src/nablaframe/trace/store.py`**

- `acquire()`: "Takes the writer lock now, exactly as the first append or supersede would: rereads the log once it holds the lock, and does nothing if this handle holds it already. Raises StoreError if another handle holds it, or if the store is read-only."

**`src/nablaframe/trace/engine.py`**

- `open`: "Takes the store's writer lock with `TraceStore.acquire()` before writing `manifest.json`, so the run is the store's one writer from the start." This replaces the call to the private `_acquire_lock()`.
- `record`, new rule: "Before writing anything, `record` checks every requested tap. If the store's current events already hold one with the same run, step, kind, mode, tap and batch hash but a different `sha256`, it raises `NotReproducible` and writes no event for any tap. An event with identical content is still skipped. Deliberate corrections go through the store's `supersede`, never through `record`."
- `record` and `rematerialize`, new rule: "Each reads the checkpoint's sidecar, `step-<n>.json`, and raises ValueError before loading any tensor unless its `run_id` equals the manifest's. A sidecar with no `run_id` is refused too, since a trace needs provenance."
- New public exception: `class NotReproducible(Exception)`, with the docstring "Recomputing a recorded tap gave a different tensor; nothing was written."

## The thread

It shares no files with the gates thread, so both can run at once:

```
Start thread "[P1] Engine rules", on Opus at high effort:
Implement the "Contract changes" section of docs/deltas/phase-1-delta-3.md. Write each bullet into the named file's docstring verbatim, then implement it. Change only src/nablaframe/trace/store.py and src/nablaframe/trace/engine.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_engine_rules.py tests/test_engine.py tests/test_store.py tests/test_store_recovery.py tests/test_phase1_store_reports.py` passes with no skips.
```

## Verification record

**Verified** on my mirror of `main` after PR #11, with the jax lane built from the lockfile:

- **On `main` as merged:** each engine-rule test fails for the reason it targets. `NotReproducible` doesn't exist, a foreign checkpoint is traced without complaint, and the store has no `acquire`. The zero-warmup test passes.
- **On a private reference** (the rules and the public lock applied to the merged code): every new test passes, and every existing test still does, except the 3 gate-helper stubs awaiting the gates thread.

**How the tests guard the rules:**

- **Checking every tap before writing.** The conflicting-retrace test puts a new tap ahead of the conflicting one. An engine that appended tap by tap would write the first event before noticing, and the test's check that the log is unchanged would catch it.
- **Both entry points refuse a foreign checkpoint.** The provenance test checks `record` and `rematerialize` separately, plus a sidecar with no `run_id`.
