# Phase 2, delta 2: the view builder, the payload contract, the gate stubs, corrections

Date: 2026-10-03. Spec: [../specs/phase-2.md](../specs/phase-2.md).

This delta ships:

- **The view builder** (`views/build.py`), whose docstring is the payload contract the viewer reads.
- **Stubs for the six Phase 2 gates** and their summary.
- **Contract text** for `reports.py` and `readouts.py`.
- **A stronger planted test for attribution,** plus a test for its probe check.
- **Spec corrections.**

## Apply

This delta changes two existing files, `tests/test_views_attribution.py` and `viewer/app.js`, so unzip with `-o`, which overwrites. Neither file has been changed since delta 1. From the repo root:

```
unzip -o nablaframe-phase2-delta2.zip
python3 apply_phase2_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

`git status` should show 3 modified files (`docs/specs/phase-2.md`, `tests/test_views_attribution.py`, `viewer/app.js`) and the new files. With PR #21's probe check merged, expect `9 failed, 112 passed`:

- the viewer's hash slice, as before
- 4 builder tests
- 3 gate-helper tests
- 1 report-rules test

Without PR #21 merged, the attribution tests add to that count.

## Contract changes

**`src/nablaframe/reports.py`**

- "`PHASE2_GATE_IDS = ("G2.1", "G2.2", "G2.3", "G2.4", "G2.5", "G2.6")`. The validator accepts them, and a Phase 2 report must use schema version 2."
- Status rules, in the docstring's STATUS RULES list:
  - "G2.1 pass requires `logits_events == logits_events_expected`, `other_events_written == 0`, `sample_matches == sample_size` and `logits_bytes` within 1% of `logits_bytes_expected`."
  - "G2.2 pass requires `sample_matches == sample_size` and `builds_identical` is true."
  - "G2.3 pass requires `planted_recovery >= 0.99`, `random_max_accuracy <= 2 * random_chance` and `last_layer_kl_max <= 1e-5`."
  - "G2.4 pass requires `planted_aligned_credit >= 0.99`, `planted_perpendicular_quiet >= 0.99` and `decomposition_error_max <= 1e-3`."
  - "G2.5's status is always `pass`, as G0.6's is."
  - "G2.6 pass requires `builds_identical` is true and `external_urls == 0`."
- Required measurements, in the docstring's list:
  - "G2.1: `logits_events`, `logits_events_expected`, `other_events_written`, `sample_size`, `sample_matches`, `logits_bytes`, `logits_bytes_expected`"
  - "G2.2: `sample_size`, `sample_matches`, `builds_identical`"
  - "G2.3: `planted_recovery`, `random_max_accuracy`, `random_chance`, `last_layer_kl_max`"
  - "G2.4: `planted_aligned_credit`, `planted_perpendicular_quiet`, `decomposition_error_max`"
  - "G2.5: `exact_positions`, `estimator_errors`, `fit_sizes`, `fit_change_per_doubling`, `projected_seconds`, `crosscheck_feasible`"
  - "G2.6: `builds_identical`, `external_urls`, `export_bytes`"

**`src/nablaframe/views/readouts.py`**

- "`event(step, tap)` returns the current event for that step and tap, under the same rules as `tensor`: KeyError if there is none, ValueError if there are several."

## Threads

Paste into the Project. The first two share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P2] Phase 2 report rules", on Sonnet at high effort:
Implement the "src/nablaframe/reports.py" part of "Contract changes" in docs/deltas/phase-2-delta-2.md. Write each bullet into the docstring's matching list verbatim, then implement it, following how the Phase 1 rules are written. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase2_reports.py tests/test_phase1_store_reports.py tests/test_reports.py tests/test_reports_rules.py` passes with no skips, and every test that passed before still does.

Thread "[P2] View builder", on Sonnet at high effort:
Implement src/nablaframe/views/build.py against its docstring and docs/specs/phase-2.md, and the "src/nablaframe/views/readouts.py" part of "Contract changes" in docs/deltas/phase-2-delta-2.md, written into that file's docstring verbatim. Change only those two files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_build.py tests/test_views_readouts.py` passes with no skips, and every test that passed before still does.
```

Then, once the builder has merged, the viewer. And once both of the above have merged, the gates:

```
Start thread "[P2] Viewer", on Opus at high effort:
Implement src/nablaframe/views/shards.py against its docstring, and viewer/index.html and viewer/app.js against the contract in app.js's header, reading the payloads described in src/nablaframe/views/build.py. Plain JavaScript, no build step, no library or request outside the repo. Change only those three files. Exercise the page in a scratch folder on a tiny traced run built with build_views, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_viewer.py` passes with no skips, and every test that passed before still does.
```

```
Start thread "[P2] Phase 2 gates", on Opus at high effort:
Implement every file in gates/phase2/ against its docstring and docs/specs/phase-2.md. You may read and reuse gates/_common.py and gates/phase1/_common.py but not change them. Change only files in gates/phase2/. The gates run on apolloh0's Mac only, so never run a gate for results and never write under reports/. For G2.5, read anthropics/jacobian-lens on GitHub and write what its docstring asks into the module. Exercise each gate's logic on a tiny model in a scratch folder, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips, apart from the viewer's test if the viewer hasn't merged yet.
```

## Corrections in this delta

- **Package dependencies.** The lens needs readouts, attribution needs readouts and the lens, and the caches and the J-lens probe need nothing. The spec's table and paragraph now say so. Delta 1's notes stand as a record.
- **G2.1 counts totals.** It counts total logits events and bytes rather than new ones, so a rerun after success passes again.
- **Fixed bounds instead of "measured tolerances".** G2.3's last-layer bound is now 1e-5 nats. G2.4's criterion is the stronger planted test.
- **G2.5's exact reference uses causal prefixes:** 268 MB per sequence at size S, rather than 17 GB.
- **The builder can read an event's hash.** `TraceReader.event` is public, because the per-token loss meter states the logits event's hash, and a builder reaching into private attributes would be fragile.

## Verification record

**Verified** from a clone of my mirror of `main`, with the jax lane built from the lockfile:

- **On the stubs:** 10 fail and 111 pass, with PR #21 applied as first pushed, before its probe check. Collection is clean.
- **On a private reference implementation, not shipped:** all 121 pass. The reference covers the builder, `TraceReader.event`, the Phase 2 report rules, the gate helpers and the probe check.
- **Planted bugs:** each of 17 is caught by the test aimed at it, as a test failure rather than a collection error. Two came from tightening my own tests first:
  - **The missing-measurement check now deletes G2.5's `projected_seconds`.** The measurement it used before also feeds a status rule, so the report failed either way and the check couldn't test the required list.
  - **The schema-1 check now removes the commit fields,** as Phase 1's test does. With them left in, the report failed for a different reason.

**Found while building the reference:**

- **The viewer needed a defined payload format,** now in `build.py`'s docstring.
- **The builder needed a public way to read an event's hash,** now `TraceReader.event`.

**Not verified:** anything on the Mac, the viewer page itself, and every gate's logic at size S.
