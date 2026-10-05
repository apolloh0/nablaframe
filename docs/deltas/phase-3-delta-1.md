# Phase 3, delta 1: specs, edits, effects, and the J-lens experiment's directions

Date: 2026-10-05. Spec: [../specs/phase-3.md](../specs/phase-3.md). Rulings: D22 to D27.

This delta ships:

- **Stubs for four packages:** `interventions/spec.py`, `apply.py`, `effects.py` and `jlens_experiment.py`.
- **16 tests,** with a shared helper, `tests/interventions_fixtures.py`.
- **Contract text** for the J-lens layout fix in `viewer/app.js`.

The command line, the viewer's compare mode and the Phase 3 gates come in delta 2, once results have a format to export.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase3-delta1.zip
python3 apply_phase3_delta1.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `16 failed, 137 passed`. Without JAX, plain `uv run pytest` gives 8 failed, and the JAX modules are skipped by name.

## Contract changes

**`viewer/app.js`** (added to the contract in its header)

- "In the J-lens table, the "vs logit lens" columns come directly after each reading's name, before the top tokens, so the comparison D21 records is visible without scrolling sideways."

## Threads

Paste into the Project, in three waves. Each wave's two threads share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P3] Intervention specs", on Sonnet at high effort:
Implement src/nablaframe/interventions/spec.py against its docstring and docs/specs/phase-3.md (D22). Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_interventions_spec.py` passes with no skips, and every test that passed before still does.

Thread "[P3] J-lens experiment directions", on Sonnet at medium effort:
Implement src/nablaframe/interventions/jlens_experiment.py against its docstring. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_jlens_experiment.py` passes with no skips, and every test that passed before still does.
```

Once the specs have merged:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P3] Applying interventions", on Opus at high effort:
Implement src/nablaframe/interventions/apply.py against its docstring and docs/specs/phase-3.md (D23, D25, D26). Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_interventions_apply.py` passes with no skips, and every test that passed before still does. Every D25 identity is a bit-for-bit equality; if one holds only approximately, stop and say so in the PR rather than loosening anything.

Thread "[P3] J-lens layout fix", on Sonnet at high effort:
Implement the "viewer/app.js" part of "Contract changes" in docs/deltas/phase-3-delta-1.md: add its bullet to the contract in app.js's header verbatim, then implement it in viewer/app.js and, if needed, viewer/index.html. Change only those two files. Exercise the page in a scratch folder on a tiny traced run built with build_views and jacobians_for, at a 1,280-pixel window, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full, apart from the intervention tests still waiting on their threads.
```

Once applying has merged, since the effects tests read `apply.code_identity()`:

```
Start thread "[P3] Effects and their cache", on Sonnet at high effort:
Implement src/nablaframe/interventions/effects.py against its docstring and docs/specs/phase-3.md ("Effects and their meters", D24). Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_interventions_effects.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** 16 new cases fail and all 137 existing tests pass.
- **On a private reference implementation, not shipped:** all 153 pass, including all five D25 identities bit for bit at the tiny size.
- **Planted bugs:** each of 18 is caught by the test aimed at it. They include:
  - edits along the batch axis
  - a mean over positions as well as sequences
  - a resample that ignores its seed
  - an unchecked resolver
  - an editable attention pattern
  - an embeddings swap that leaves the positional embedding behind
  - steering that ignores its scale
  - KL the wrong way round
  - a loss change with the wrong sign
  - a meter that drops an identity
  - a cache key that ignores the batch
  - an untransposed J-lens direction
  - steering scales taken from the largest norm

**Found while building the reference:**

- **A null edit is an edit at no positions,** so the identity tests the plumbing without the compiled-program question a self-patch would raise.
- **The model's hook re-wraps an edited tap's value in its canonical axis order,** even for an edit that changes nothing. The null edit stayed bit for bit exact through that, at every tap tested.
- **The J-lens cache's `save_jacobian` refuses non-square arrays,** rightly. So `effects.py` keeps its own result cache, with the same rules.

**Not verified:** anything on the Mac, and the identities at size S. That's G3.1, in delta 2.
