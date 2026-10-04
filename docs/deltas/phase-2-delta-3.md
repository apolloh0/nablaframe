# Phase 2, delta 3a: the summed Jacobian, the Hugging Face export, and the J-lens gates

Date: 2026-10-03. Spec: [../specs/phase-2.md](../specs/phase-2.md). Rulings: D17; D18 and D19, proposed with this delta.

This delta builds what the J-lens cross-check needs, so it can run before the four-hour computation. It ships:

- **`views/jlens.py`:** the summed Jacobian exactly as `anthropics/jacobian-lens` computes it, its block map, and its cache, with D18's code identity.
- **`model/hf_export.py`:** a Hugging Face GPT-2 export of our parameters.
- **Stubs for G2.7, G2.8 and G2.9,** whose order enforces cross-check first.
- **Contract text** for `reports.py` and `gates/phase2/summarize.py`.
- **The guarded `jlens` dependency (D19).**

Delta 3b, the J-lens view and its comparison with the logit lens, follows once G2.8 has passed.

## Apply

From the repo root:

```
unzip -n nablaframe-phase2-delta3.zip
python3 apply_phase2_delta3.py
git diff
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

The script first records every lane's resolution, adds `jlens` to the torch lane and runs `uv lock`. Unless the JAX and Metal lanes resolve identically and the torch lane only gains packages, it restores `pyproject.toml` and `uv.lock` and writes nothing else. Then it adds D18 and D19 and the spec's edits.

Expect `8 failed, 121 passed`:

- 5 summed-Jacobian tests
- 2 export tests
- 1 report-rules test

## Contract changes

**`src/nablaframe/reports.py`**

- "`PHASE2_GATE_IDS` gains G2.7, G2.8 and G2.9."
- Status rules:
  - "G2.7 pass requires `export_matches_params` and `jacobian_recompute_identical` both true."
  - "G2.8 pass requires `tokens_identical` true, `export_logit_max_abs_diff <= 1e-4` and `jacobian_relative_difference <= 1e-4`."
  - "G2.9 pass requires `jacobians_written == jacobians_expected` and `recompute_identical` true."
- Required measurements:
  - "G2.7: `step`, `layer`, `windows`, `export_matches_params`, `jacobian_recompute_identical`"
  - "G2.8: `windows`, `tokens_identical`, `export_logit_max_abs_diff`, `jacobian_relative_difference`"
  - "G2.9: `jacobians_written`, `jacobians_expected`, `recompute_identical`, `seconds`"

**`gates/phase2/summarize.py`**

- "The summary covers G2.1 to G2.9 and requires all nine reports. Its H2 evidence gains a J-lens part: G2.8's agreement (tokens identical, J's relative difference, the export's logit difference) and G2.9's count and time."

## Threads

Paste into the Project. The first two share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P2] Summed Jacobian", on Opus at high effort:
Implement src/nablaframe/views/jlens.py against its docstring, docs/specs/phase-2.md ("The J-lens") and D17. Read jlens/fitting.py in anthropics/jacobian-lens at commit 581d398 and match its behaviour exactly; if it and the docstring disagree, say so in the PR rather than choosing. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_jlens.py` passes with no skips, and every test that passed before still does.

Thread "[P2] Hugging Face export", on Sonnet at high effort:
Implement src/nablaframe/model/hf_export.py against its docstring. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_hf_export.py` passes with no skips, and every test that passed before still does.
```

Then the report rules, which can start as soon as a slot frees:

```
Start thread "[P2] J-lens report rules", on Sonnet at medium effort:
Implement the "src/nablaframe/reports.py" part of "Contract changes" in docs/deltas/phase-2-delta-3.md, written into the docstring's matching lists verbatim, following how the G2.1 to G2.6 rules are written. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase2_reports_jlens.py tests/test_phase2_reports.py` passes with no skips, and every test that passed before still does.
```

Last, once all three have merged:

```
Start thread "[P2] J-lens gates", on Opus at high effort:
Implement gates/phase2/g27_jlens_prep.py, g28_jlens_crosscheck.py and g29_jlens_fit.py against their docstrings, and the "gates/phase2/summarize.py" part of "Contract changes" in docs/deltas/phase-2-delta-3.md. You may read and reuse gates/phase1/_common.py and gates/phase2/_common.py but not change them. Change only those four files. G2.8 runs in the torch lane (`UV_PROJECT_ENVIRONMENT=.venv-torch uv run --extra torch`), where jlens is installed. Never run a gate for results and never write under reports/. Exercise all three gates on a tiny model in a scratch folder, including G2.8 against the real jlens, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Verification record

**Verified** from a clone of my mirror of `main`:

- **On the stubs:** 8 new cases fail and all 121 existing tests pass.
- **On a private reference implementation, not shipped:** all 129 pass.
- **Planted bugs:** each of 16 is caught by the test aimed at it. They include:
  - the last position counted as valid
  - cotangents at every position
  - an average over every source position
  - the same-position J in place of the summed one
  - the block map replacing a block's input instead of its output
  - a content hash in place of the git blob hash
  - a cache key that ignores the layer
  - `lm_head` written by the export
  - a transposed `c_attn`
  - exact GELU
  - GPT-2's token ids
  - a timestamp in the config
  - three loosened report rules
- **The guard,** run by the apply script on a copy of `main`: the JAX and Metal lanes resolve identically, and the torch lane only gains `jlens`. The new lockfile hash is `9c2155edb31c`.

**Read from the reference implementation at `581d398`:**

- One mask, positions 16 to T − 2, places the cotangents and selects the source positions averaged.
- Each prompt's J is averaged over its valid positions, then prompts are averaged.
- The unembedding is the final norm followed by `lm_head`.
- `jacobian_for_prompt` tokenizes text itself and truncates to 128 tokens by default, so G2.8 passes `max_seq_len=256` and must prove the tokens are ours.
- A saved lens is float16 by default, so G2.8 compares in memory.

**Not verified:** anything on the Mac, and G2.8 against the real `jlens`. That's the gates thread's scratch run, then your gate run.
