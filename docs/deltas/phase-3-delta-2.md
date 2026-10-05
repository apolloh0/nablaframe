# Phase 3, delta 2: the command line, the compare mode, and the Phase 3 gates

Date: 2026-10-05. Spec: [../specs/phase-3.md](../specs/phase-3.md). Rulings: D22 to D27.

This delta ships:

- **The command line's stub,** `interventions/cli.py`, with the compare payload. It pairs both passes and adds tap changes beside the effects.
- **Stubs for G3.1 to G3.5,** their summary and their helpers.
- **6 tests.**
- **Contract text** for `reports.py` and the viewer's compare mode.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase3-delta2.zip
python3 apply_phase3_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `5 failed, 154 passed`:

- 3 command-line tests
- the position-helper test
- the report-rules test

## Contract changes

**`src/nablaframe/reports.py`**

- "`PHASE3_GATE_IDS = ("G3.1", "G3.2", "G3.3", "G3.4", "G3.5")`. The validator accepts them, and a Phase 3 report must use schema version 2."
- Status rules:
  - "G3.1 pass requires `identities_exact == identities_checked`."
  - "G3.2 pass requires `recompute_identical == sample_size` and `cache_keys_distinct` true."
  - "G3.3's and G3.4's status is always `pass`: they record, not judge."
  - "G3.5 pass requires `builds_identical` true and `external_urls == 0`."
- Required measurements:
  - "G3.1: `identities_checked`, `identities_exact`, `identities_by_kind`"
  - "G3.2: `sample_size`, `recompute_identical`, `cache_keys_distinct`"
  - "G3.3: `layers`, `scale_factors`, `results_by_layer`"
  - "G3.4: `meters_by_step`"
  - "G3.5: `builds_identical`, `external_urls`, `payloads`"

**`viewer/app.js`** (added to the contract in its header)

- "When `../runs/phase3/interventions/index.json` exists, or the folder `?interventions=` names, a compare panel lists its interventions by spec hash and step. Choosing one shows:
  - the spec as readable lines, one per edit
  - the probe text colored by per-token KL and by the next token's logit change, with a switch between them
  - each requested tap's change per position
  - the effect's exactness meter, naming its identities and citing G3.1

  A payload whose step differs from the timeline's current step says so, rather than mixing two steps' numbers. The panel computes nothing: every number comes from the payload."

## Threads

Paste into the Project, in two waves. Each wave's two threads share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P3] Phase 3 report rules", on Sonnet at medium effort:
Implement the "src/nablaframe/reports.py" part of "Contract changes" in docs/deltas/phase-3-delta-2.md, written into the docstring's matching lists verbatim, following the Phase 2 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase3_reports.py tests/test_phase2_reports.py` passes with no skips, and every test that passed before still does.

Thread "[P3] Command line", on Opus at high effort:
Implement src/nablaframe/interventions/cli.py against its docstring and docs/specs/phase-3.md. Change only that file. Every pass goes through run_intervened, never through hooks of your own. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_interventions_cli.py` passes with no skips, and every test that passed before still does.
```

Once both have merged:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P3] Viewer compare mode", on Opus at high effort:
Implement the "viewer/app.js" part of "Contract changes" in docs/deltas/phase-3-delta-2.md: add its bullet to the contract in app.js's header verbatim, then implement it in viewer/app.js and viewer/index.html. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder on a tiny traced run, with payloads written by the command line for one spec per operation, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full, apart from the gate-helper test if the gates haven't merged yet.

Thread "[P3] Phase 3 gates", on Opus at high effort:
Implement every file in gates/phase3/ against its docstring and docs/specs/phase-3.md. You may read and reuse gates/_common.py, gates/phase1/_common.py and gates/phase2/_common.py but not change them. Change only files in gates/phase3/. Every pass goes through run_intervened. Never run a gate for results and never write under reports/. Exercise each gate on a tiny model in a scratch folder and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** 5 fail and 154 pass. The constants test passes already, as a pin.
- **On a private reference implementation, not shipped:** all 159 pass.
- **Planted bugs:** each of 9 is caught by the test aimed at it. They include:
  - tap changes as an L1 norm, or signed rather than a norm
  - a cache key that ignores the code
  - a payload missing its step
  - a command line that writes before checking its inputs
  - a later window that starts at the steered position
  - a fit range that includes the last position
  - G3.3 allowed to fail
  - G3.1's identity rule dropped

**Carried from delta 1's threads into this delta's contracts:**

- **Every pass goes through `run_intervened`** (PR #38's constant-folding finding).
- **A swapped embedding also swaps the output layer** (PR #38's note on tied weights).
- **The tap change comes with the compare payload** (PR #39's open question).

**Not verified:** anything on the Mac, and every gate at size S.
