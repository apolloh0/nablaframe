# Phase 4, delta 4: model-level badges, and both demos in the viewer

Date: 2026-10-10. Spec: [../specs/phase-4.md](../specs/phase-4.md). Rulings: D29 to D38.

The zoo's gates all passed. apolloh0's first look at both zoo runs in the viewer found four gaps between those results and H4's question:

- **The timeline was empty.** It plots a validation-loss curve, and a zoo run has none, so the loss bump the induction demo is named for wasn't on the scrubber.
- **The badges still said model-level tests wait for Phase 4.**
- **Two-hop's answer position was blank.** The probe holds the tokens the model reads, so its last position has no next token, and two-hop's answer is read there.
- **The compare panel couldn't show the two-hop demo.** It shows an edit's KL and logit change, not what the model predicts, so a patched answer couldn't be seen to change.

This delta closes them, as D38 rules. It ships:

- **Contract changes** to `views/readouts.py`, `views/attribution.py`, `views/jlens_view.py`, `views/build.py`, `interventions/cli.py`, `reports.py`, G4.5, the Phase 4 summary and the viewer.
- **A new module,** `views/badges.py`, which generates the badges from the reports. Its docstring is its contract.
- **A new gate, G4.6,** which writes the badges and the zoo's compare payloads. Its stub's docstring is its contract.
- **9 tests** in four new files, and an edit to one committed test.
- **D38 and spec edits:** G4.5's and G4.6's gate rows, work package 7, and this delta's tests.

## Apply

First push the Phase 4 reports and summary, so `main` carries them. Every file in the zip is new. From the repo root:

```
git push
unzip -n nablaframe-phase4-delta4.zip
python3 apply_phase4_delta4.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `10 failed, 194 passed`:

- the badges: 4
- the last position and the curves: 2
- the compare payload's top tokens: 2
- G4.6's report rules: 1, plus the edited `test_phase4_gate_ids`

## Contract changes

**`src/nablaframe/views/readouts.py`** (adds to `per_token_loss`'s bullet)

- "`per_token_loss` takes a keyword `next_token=None`: one int per probe sequence, the token that follows its last position in the window. When given, the last position is scored against it like any other; without it, the last position stays NaN. Anything but a 1-D integer array with one entry per sequence raises ValueError. A helper here, `checked_next_token(next_token, probe)`, returns it as an int64 array, or None, or raises that ValueError, and the other views use it."

**`src/nablaframe/views/attribution.py`** (adds to `attribution_from_trace`'s bullet)

- "`attribution_from_trace` takes the same keyword `next_token=None`. When given, the last position's target is `next_token[b]`, its contributions are computed like any other's, and `error` covers it too."

**`src/nablaframe/views/jlens_view.py`** (adds to the `target_ranks` and `jlens_payload` bullets)

- "`target_ranks` and `jlens_payload` take the same keyword `next_token=None`. When given, the last position has a rank and a probability like any other, under the same tie rule; without it, both stay null there."

**`src/nablaframe/views/build.py`** (adds to the `view_payloads` and `build_views` bullets)

- "`view_payloads` takes `next_token=None` and passes it to `per_token_loss`, `target_ranks` and `attribution_from_trace`. `build_views` takes `next_token=None` and `curves=None`. It passes `next_token` to every payload, the J-lens's included. When it's given, the index gains `"next_token"`, a list of ints. When `curves` is given, the index gains `"curves"`: a list of `{"name", "unit", "points"}`, `name` and `unit` non-empty strings, each point `{"step", "value"}` with an int step and a finite value or null. Anything else raises ValueError before a file is written. Without either keyword, the index has exactly its earlier keys and every file is byte-identical to before, so earlier runs' builds don't change."

**`src/nablaframe/interventions/cli.py`** (adds a bullet, and to `compare_payload`'s)

- "`top_tokens(logits, k=5)`: `(tokens, probs)`, nested lists (Batch, Pos, k): each position's k most probable tokens, best first, ties to the lower id, and their softmax probabilities in float64. The same rule as the logit lens payload's top tokens."
- "`compare_payload` gains `"top_tokens"` and `"top_probs"`, each `{"unintervened": ..., "intervened": ...}`, from `top_tokens` of each pass's logits. So the viewer can show what each pass predicts."

**`src/nablaframe/reports.py`**

- "`PHASE4_GATE_IDS` gains G4.6."
- Status rule: "G4.6 pass requires `builds_identical` true and `subjects_missing == 0`."
- Required measurements: "G4.6: `badges`, `subjects_missing`, `builds_identical`, `payloads`"

**`gates/phase4/g45_formation_viewer.py`** (replaces the docstring's "Views" bullet)

- "**Views:** each zoo model's views built with `build_views`, with the task's vocabulary, `next_token` the probe windows' last tokens, and `curves`, into `runs/phase4/<name>/views/`, then again into a scratch folder, compared file by file: `builds_identical`. For induction the curves are the formation curves, `"mechanism positions' loss"` in `"nats"` and `"best block-1 head's attention to induction targets"` as a `"fraction"`. For two-hop there is one curve, `"answer position's loss"` in `"nats"`: at each checkpoint, the mean over the probe of `-log softmax(logits[b, 25])[window[b, 26]]` from the traced logits, in float64. `external_urls` counts external references in the viewer as G2.6 does. Also recorded: `curves`, each model's curve names, and `answer_loss_by_step`."

**`gates/phase4/summarize.py`** (adds to its docstring's list)

- "the summary covers six reports. G4.6's row gives `subjects_missing` and whether the builds matched. A section lists each badge's status and facts, with each fact's gate."

**`viewer/app.js`** (added to the contract in its header)

- "When the index has `curves`, the timeline draws each as its own row on the step axis. Each row has its name and unit, a line through its points on its own vertical scale, with nulls left as gaps, and its value at the current step. With neither `curves` nor `val_loss_curve`, it says so."
- "When the index has `next_token`, each probe sequence's last position is scored like any other. The probe text colors it by its loss, and the lens tables outline its next token, decoded with the index's vocab."
- "Model-level badges: it reads `../runs/phase4/badges.json`, or the file `?badges=` names, which G4.6 writes.
  - Each view's panel shows its subject's badge under its D14 badge or meter: the probe text `per_token_loss`; the lens panels `logit_lens` and `jlens`; then `attribution`, `attention_pattern`, `residual_norm` and `layer_structure`. The compare panel shows the badge of each operation in the chosen spec.
  - A badge shows its status, each fact as its label, its value in its unit and its gate, and its notes.
  - Units: a fraction as a percentage; a fraction removed to three decimals; a rank, 0-based in the file, as #n counted from 1, so a median of 50 reads #51, followed by "of <the tokens fact>" when the badge has one; a step as an integer; a logit signed, to three decimals.
  - A missing file at the default path hides the badges. A named file that is missing or unreadable says so. Nothing is computed: every value is the file's."
- "Compare mode: when the payload has `top_tokens` and `top_probs`, the panel shows both passes' top five tokens at the chosen position, with their probabilities, decoded with the index's vocab, side by side. A payload without them says so."

**`src/nablaframe/views/badges.py`** and **`gates/phase4/g46_badges_demos.py`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave 1** has four threads, which share no files:

```
Start four threads now, without proposing them first. They share no files.

Thread "[P4] Last-position targets and timeline curves", on Sonnet at high effort:
Implement the "views/readouts.py", "views/attribution.py", "views/jlens_view.py" and "views/build.py" parts of "Contract changes" in docs/deltas/phase-4-delta-4.md, adding each bullet to its module's docstring verbatim. Change only those four files. Without the new keywords, every output must stay byte-identical to before. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_last_target.py tests/test_views_build.py tests/test_views_build_jlens.py tests/test_views_readouts.py tests/test_views_attribution.py tests/test_jlens_view.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Compare payload top tokens", on Sonnet at medium effort:
Implement the "interventions/cli.py" part of "Contract changes" in docs/deltas/phase-4-delta-4.md, adding its bullets to the module docstring verbatim. Change only src/nablaframe/interventions/cli.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_compare_top_tokens.py tests/test_interventions_cli.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Model-level badges", on Sonnet at high effort:
Implement src/nablaframe/views/badges.py against its docstring, docs/specs/phase-4.md and D38. Change only that file. Every value on a badge must come from a report; nothing is typed in. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_badges.py` passes with no skips, and every test that passed before still does.

Thread "[P4] G4.6 report rules", on Sonnet at medium effort:
Implement the "reports.py" part of "Contract changes" in docs/deltas/phase-4-delta-4.md, written into the docstring's matching lists verbatim, following the Phase 4 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase4_reports_g46.py tests/test_phase4_reports.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once all four have merged, has two threads, which share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P4] G4.5's curves and G4.6", on Opus at high effort:
Implement the "g45_formation_viewer.py" and "summarize.py" parts of "Contract changes" in docs/deltas/phase-4-delta-4.md, writing each into its docstring verbatim, and implement gates/phase4/g46_badges_demos.py against its docstring. Change only files in gates/phase4/, and in G4.5 only what its views and their curves need. Every intervened pass goes through run_intervened or the command line's main. Never run a gate for results and never write under reports/. Exercise G4.5 and G4.6 in a scratch folder with shrunk settings and fixture reports, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.

Thread "[P4] Viewer: curves, badges and top tokens", on Opus at high effort:
Implement the "viewer/app.js" part of "Contract changes" in docs/deltas/phase-4-delta-4.md: add its bullets to the contract in app.js's header verbatim, then implement them in viewer/app.js and viewer/index.html. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder on a tiny traced run built with curves and next tokens, a badges.json from views/badges.py on fixture reports, and a compare payload with top tokens, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

Once wave 2 has merged, on the Mac. G4.5 runs again so its views carry the curves and the last-position targets:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase4/g45_formation_viewer.py --overwrite
git add reports/phase4 && git commit -m "Phase 4 report: G4.5, with curves and last-position targets"
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase4/g46_badges_demos.py
git add reports/phase4 && git commit -m "Phase 4 report: G4.6"
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase4/summarize.py
git add reports/phase4 && git commit -m "Phase 4 summary"
```

G4.6 prints the viewer's address for each zoo run.

## Verification record

**Verified** on a clone of `main` (`3a29202`):

- **On the stubs:** 10 fail and 194 pass.
- **On a private reference implementation, not shipped:** all 204 pass. It covers the views' last position and curves, the compare top tokens, the badges and G4.6's rules.
- **Earlier runs' views don't change.** Built without the new keywords, the fixture run's six view files are byte-identical to `main`'s.
- **The badges, generated from your five reports** by the reference, read as below. Every number is the report's own:

  | Badge | Status | Facts |
  | --- | --- | --- |
  | Logit lens | recorded | the patch swaps the intermediate in 85.2% at the query after block 2. There, at `⇒`, the lens ranks the intermediate 50 (the twin 28.5); at position 24, 27 (twin 26). The answer ranks 0 after block 3, of 51 tokens |
  | J-lens | recorded | the same reading: 50 at `⇒` (twin 29), 26 at position 24 (twin 27). Its coordinate swap moves the answer 2.3% and leaves it 62.9% unchanged. A note says J's fit leaves out `⇒` |
  | Attribution | recorded | induction: block 1's MLP +9.95, its attention +3.35, while zeroing the heads removes 1.001. Two-hop: block 3's attention +11.13, its MLP +2.61. A note says direct contribution isn't necessity |
  | Attention pattern | judged: pass | best head 98.6%, the twin's 4.8%, uniform 4.6% |
  | Zero, swap, patch | judged: pass | heads together remove 1.001; swap from step 50 removes 0.929; patch followed 100% after block 0 and 0% after block 1, control unchanged 99.6%, largest swap 85.2% |
  | Coordinate swap | recorded | 2.3%, against the aligned patch's 85.2% |
  | Mean, resample, add | not tested | no Phase 4 test exercises them |
  | Probe text, residual norm, layer structure | readout | exact readouts, checked by hash |

- **Planted bugs:** 22 of 25 are caught by the test aimed at them. The other three are invisible to the tests:
  - Two still raise the ValueError the contract asks for, through another route. A wrong-length `next_token` fails in the concatenation, and a NaN curve value fails in `canonical_json`.
  - The third leaves the last position out of the attribution's decomposition error. It changes nothing unless that position happens to hold the largest float error.

  The 22 caught include:
  - the last position scored against the wrong token, or every target shifted by one
  - ranks that ignore the tie rule
  - new index keys written when not asked for
  - the J-lens built without the next token
  - top tokens with ties to the higher id, or the two passes swapped
  - attribution parts ranked by signed value
  - the lens badges fixed at block 2 instead of G4.3's best
  - the patch badge judged by G4.2 alone
  - a missing, invalid or misfiled report accepted
  - the twin's ranks left out
  - G4.6 passing without its badge count

**Not verified:** the viewer and G4.6, which wave 2 builds, and anything on the Mac.
