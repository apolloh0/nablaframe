# Phase 2, delta 4 (3b): the J-lens view

Date: 2026-10-04. Spec: [../specs/phase-2.md](../specs/phase-2.md). Rulings: D17 and D20.

This delta ships:

- **`views/jlens_view.py`:** the J-lens view, its meter, and its comparison with the logit lens. The comparison is the measurement D17 promised, of how far the two lenses' readings actually differ.
- **Contract text** for six existing files:
  - the builder carries the J-lens and the actual next token's rank
  - the export writes a Hugging Face tokenizer instead of a copy of ours
  - G2.10's report rules
  - G2.6 rebuilds with the J-lens
  - the summary covers G2.10
  - the viewer gains a J-lens panel, the next token's rank, and a log-scale lens-KL chart
- **G2.10's stub** and **9 new or changed test cases.**

## Apply

This delta changes one existing file, `tests/test_hf_export.py`, so unzip with `-o`, which overwrites. That file hasn't changed since delta 3a. From the repo root:

```
unzip -o nablaframe-phase2-delta4.zip
python3 apply_phase2_delta4.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `9 failed, 128 passed`:

- 5 J-lens view tests
- 2 builder tests
- the export test, which now expects a Hugging Face tokenizer
- G2.10's report rules

## Contract changes

**`src/nablaframe/views/build.py`**

- "`build_views` takes a keyword `jacobians_for=None`. When given, each step also writes `jlens_payload(reader, step, params_for(step), jacobians_for(step), probe, n_layers)` as `jlens-step-<step>`, and the index's `views` is `VIEWS` followed by `"jlens"`. Without it, nothing changes."
- "The `logit_lens` payload gains `target_rank` and `target_prob`, from `jlens_view.target_ranks(readings, probe)`."

**`src/nablaframe/model/hf_export.py`** (replaces the `tokenizer.json` bullet)

- "`tokenizer_path` is our tokenizer's file (`{"vocab": [...]}`). `tokenizer.json` is a Hugging Face tokenizer built from it. It has a WordLevel model over our vocabulary, with `unk_token` as its unknown token, and a normalizer of NFKD, accent stripping and lowercasing. Its pre-tokenizer splits on whitespace, then isolates our token pattern `[a-z]+(?:'[a-z]+)*|\S`. `unk_token` and `eos_token` are added as special tokens, matched before splitting. Encoding text our tokenizer decoded reproduces its ids exactly. For raw text, NFKD and accent stripping approximate our ASCII folding, which also rewrites some punctuation, so raw-text ids are not promised identical."

**`src/nablaframe/reports.py`**

- "`PHASE2_GATE_IDS` gains G2.10."
- "G2.10 pass requires `planted_recovery >= 0.99`, `identity_exact` true, `random_max_accuracy <= 2 * random_chance` and `last_reading_kl_max <= 1e-5`."
- "G2.10: `planted_recovery`, `identity_exact`, `random_max_accuracy`, `random_chance`, `last_reading_kl_max`, `comparison_by_step`"

**`gates/phase2/g26_viewer.py`**

- "If G2.9's report exists and passes, both builds pass `jacobians_for`, loading each step's J's for layers 0 to 2 from G2.9's cache with `load_jacobian` under `jacobian_key`; a missing J is a setup error. The report records `jlens_included`."

**`gates/phase2/summarize.py`**

- "The summary covers G2.1 to G2.10 and requires all ten reports. Its H2 evidence gains the J-lens view: G2.10's discrimination numbers, and from `comparison_by_step` the final step's Jensen-Shannon divergence and top-1 agreement with the logit lens per layer."

**`viewer/app.js`** (added to the contract in its header)

- "When the index lists `jlens`, a J-lens panel sits beside the logit lens. It shows the same table of top tokens per reading, its meter and badge, and per reading its Jensen-Shannon divergence and top-1 agreement with the logit lens."
- "Both lens tables show, per reading, the actual next token's rank and probability from `target_rank` and `target_prob`, so the next token is visible even outside the top 5."
- "The layer-structure lens-KL chart uses a log scale, so readings near zero stay distinguishable."

## Threads

Paste into the Project. The first two share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P2] J-lens view and builder", on Opus at high effort:
Implement src/nablaframe/views/jlens_view.py against its docstring, then the "src/nablaframe/views/build.py" part of "Contract changes" in docs/deltas/phase-2-delta-4.md, written into build.py's docstring verbatim. Change only those two files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_jlens_view.py tests/test_views_build_jlens.py tests/test_views_build.py` passes with no skips, and every test that passed before still does.

Thread "[P2] Export tokenizer", on Sonnet at high effort:
Implement the "src/nablaframe/model/hf_export.py" part of "Contract changes" in docs/deltas/phase-2-delta-4.md, replacing the docstring's tokenizer.json bullet with it verbatim. Change only that file. In a scratch folder, load an export with transformers' AutoTokenizer in the torch lane and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_hf_export.py` passes with no skips, and every test that passed before still does.
```

Then G2.10's report rules, as soon as a slot frees:

```
Start thread "[P2] G2.10 report rules", on Sonnet at medium effort:
Implement the "src/nablaframe/reports.py" part of "Contract changes" in docs/deltas/phase-2-delta-4.md, written into the docstring's matching lists verbatim, following the G2.7 to G2.9 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase2_reports_jlens_view.py tests/test_phase2_reports_jlens.py tests/test_phase2_reports.py` passes with no skips, and every test that passed before still does.
```

Then, once all three have merged, the last two together:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P2] Viewer J-lens panel", on Opus at high effort:
Implement the "viewer/app.js" part of "Contract changes" in docs/deltas/phase-2-delta-4.md: add its bullets to the contract in app.js's header verbatim, then implement them in viewer/app.js and viewer/index.html. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder on a tiny traced run built with build_views and jacobians_for, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.

Thread "[P2] G2.10 and the J-lens rebuild", on Opus at high effort:
Implement gates/phase2/g210_jlens_view.py against its docstring, and the "gates/phase2/g26_viewer.py" and "gates/phase2/summarize.py" parts of "Contract changes" in docs/deltas/phase-2-delta-4.md. You may read and reuse gates/phase1/_common.py and gates/phase2/_common.py but not change them. Change only those three files. Never run a gate for results and never write under reports/. Exercise all three on a tiny model in a scratch folder and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** 9 new or changed cases fail and all 128 others pass.
- **On a private reference implementation, not shipped:** all 137 pass.
- **Planted bugs:** each of 14 is caught by the test aimed at it. They include:
  - J applied untransposed
  - the last reading through the previous block's J
  - the J-lens reading a block's input instead of its output
  - KL in place of Jensen-Shannon
  - 1-based ranks
  - the current token as the target
  - a missing probe check
  - a comparison against the embedding reading
  - the builder adding the J-lens without J's
  - a tokenizer without special tokens, without lowercasing, or splitting on whitespace alone
  - two loosened report rules

**Found while building the reference:**

- **Our `decode` writes `<eos>` and `<unk>` literally.** A Hugging Face tokenizer must match them as special tokens before splitting, or every story boundary breaks the round trip.
- **The merged G2.3 rule helper's arguments run `(key, factor, other)`.** My first reference used another order and broke every gate's validation, which the existing tests caught at once.

**Not verified:** anything on the Mac, the viewer page itself, and G2.10 at size S.
