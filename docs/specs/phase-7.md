# Phase 7 spec: crystallize-this, with modular addition in the zoo

Status: proposed with C11 and D54 to D57. Charter: [../charter.md](../charter.md).

## Purpose

Phase 6's gestures say what a component's features stand for, which edges connect them, and where in training they hold (D53). None of them says whether that is all the component does. Phase 7 builds the gesture that asks it:

- **crystallize-this:** replace a component with an explicit function of what it reads, and measure how far the model's prediction moves. The charter's principle applies as written: crystallization is demonstrated over an envelope, never proven, and the gap is always shown.

Modular addition joins the zoo for it. Its known mechanism is a function in closed form, sums of cosines at a few key frequencies (D31), so the gesture has a closed form to recover as well as a table.

Two feasibility rounds in the cloud workspace shaped the definitions and the bars (`docs/notebook.md`, findings 23 to 32):

- **The meter:** a crystallization can't be judged by accuracy or loss. A table keyed by the answer predicts better than a model that hasn't learned yet. It is judged by the KL from the model's own prediction, beside a mean ablation's.
- **Counterfactuals:** a low KL can't tell "the component computes f" from "f happens to equal the answer". Inputs where the two part ways can.
- **Components:** a block's attention output, MLP output or whole write. Not a single head: induction's copy is spread over four.
- **Boundaries:** modular addition's MLP doesn't compute the sum alone, but the block's write does.
- **The specimen's width:** it decides the algorithm. At width 32 the model learns the clock, and the block's write crystallizes into five frequencies.
- **Names:** Phase 6's names can't hold modular addition's sum, which is spread over five frequencies (C11).

Human gate H7 asks two questions:

- Does crystallize-this replace each judged component with its explicit function at a small KL, while every wrong function fails, the twin shows nothing, and on counterfactual inputs the crystallization follows what the component reads, not the answer?
- Does modular addition's block write at `=` crystallize into a few frequencies of `(a + b) mod p`, with its loss delta, its envelope across training and its gap shown, and with the record saying why Phase 6's names fall short?

## Rulings

Proposed with this spec; committing them is the ruling.

**C11: crystallization cites the gestures where they apply.** In the charter's Phases section, "Phase 7's crystallization comes after the gestures: a component is replaced by an explicit function only once the gestures can say what it computes." becomes "Phase 7's crystallization comes after the gestures. A component is replaced by an explicit function only once the gestures have been tried on it: where they can say what it computes, the crystallization cites them, and where they can't, its own tests stand alone and its record says why." apolloh0 chose this on 2026-10-23.

**D54: modular addition joins the zoo.**

- **The task:** `ModularAdditionTask(p=113, train_fraction=0.3, split_seed=0)`.
  - **Vocabulary:** `"0"` to `"112"`, then `"="`: 114 tokens.
  - **A window:** `a b = c`, four tokens, with `c = (a + b) mod p`. The model reads three (`seq` 3), and position 2, the `=`, predicts the answer.
  - **The split:** the `p²` = 12,769 pairs, `a`-major, split once. `numpy.random.default_rng(split_seed).permutation(p²)`; the first `round(train_fraction · p²)` = 3,831 train and the other 8,938 are held out, each kept in ascending order.
  - **`batch_for_step(seed, step, batch)`:** `batch` training pairs drawn with replacement by `default_rng([seed, step])`.
  - **Positions:** mechanism position 2; control position 0, which predicts `b` from `a` and can't beat chance; chance `1 / p`.
- **The model:** the reference GPT-2 with LayerNorm and tied embeddings (D29), 1 block, `d_model` 32, 4 heads, `d_mlp` 128, `seq` 3, vocabulary 114: 16,512 parameters.
- **Training:** the trainer unchanged. Loss over every position; AdamW with weight decay 1.0 on matrices; `lr` 1e-3, warmup 10, cosine to 0.1 of `lr` over 10,000 steps; batch 512; data and parameter seed 0.
- **Checkpoints:** every 100 steps to 3,000, then every 500 to 10,000: 45 in all.
- **The twin:** `init_params(config, seed=1)`, as D30's twins.
- **Ground truth (D30):**
  - **Required:** a held-out pair's answer can't be predicted except by computing the sum.
  - **The trained model succeeds:** held-out accuracy at position 2.
  - **Controls fail:** position 0, and the twin.
  - **The mechanism:** measured on the weights and logits, never on a view. The embedding's non-constant energy concentrates in a few frequencies, and the logits follow `Σₖ αₖ cos(2πk(a + b − c)/p)` over those frequencies.
- **Width 32:** apolloh0 chose it on 2026-10-23. At width 128 the model gave each number its own direction instead of learning the clock. Its top 5 frequencies hold 10.4% of the energy, which matches its twin's 10.2%, and the five-frequency form misses (KL 1.89).

**D55: `set` joins the operations.** It replaces a tap's value, at the edit's positions in every sequence, with given vectors. A crystallization writes a different vector into every sequence, and no existing operation can do that: `patch` needs a source batch, and `add` adds one vector everywhere.

- **Arguments:** `values`, a float32 array by content hash (`array_hash`), shaped (batch, the edit's positions, `d_model`).
- **Taps:** those shaped (Batch, Pos, Embed): the embeddings, every block's residual taps, `hook_attn_out`, `hook_mlp_out`, and the layer-norm outputs.
- **Identities:** values equal to the unintervened tap's reproduce the pass bit for bit, as does a `set` at no positions (D25).
- **Refusals, before anything runs:** values of the wrong shape or dtype, non-finite values, and a tap of another shape.
- **Its values enter the compiled program as arguments** (PR #38).

**D56: crystallize-this replaces a component with an explicit function of what it reads, and measures how far the model's prediction moves.**

- **The request:** a run, a block `l`, a part, a position class, a function the run declares, a form, and the held-out windows and seed (D47).
- **The part:**
  - `attention`: `blocks.l.hook_attn_out` at the class's positions.
  - `mlp`: `blocks.l.hook_mlp_out` there.
  - `write`: both. `set` puts the attention output to the function's value and `zero` clears the MLP's output, so the residual after the block is the residual before it plus the function's value.
  - The component's output `Y` is the part's tap, or for `write` the two summed, on the unintervened pass.
- **The function:** declared by the run, like a description (D49). It maps a window and a position to an integer key, from tokens at or before that position. It names:
  - what the component reads: descriptions at positions
  - the declared relation, where one of them is read at the component's own position (D53)
  - its wrong functions, as controls
  - optionally, a counterfactual: inputs where the key the component reads and the answer's key differ
- **The form:**
  - **`table`:** `T[v]`, the mean of `Y` over the fitting rows whose key is `v`, taken in float64. A key the fitting rows never show gets their overall mean, and is counted.
  - **`additive`:** `Y ≈ Σⱼ Tⱼ[keyⱼ]` over a declared tuple of keys, fitted by least squares on one-hot designs (`numpy.linalg.lstsq`, minimum norm).
  - **`fourier`,** declared only by modular addition: `Y ≈ μ + Σ_{k∈K} cos(2πkv/p)·Uₖ + sin(2πkv/p)·Vₖ`, fitted by least squares. `K` is the fewest frequencies holding at least 0.75 of the embedding's non-constant energy at the request's checkpoint, at most 8; the record lists them.
- **Fitting and reading:** the form is fitted on the run's fitting windows and read on its held-out windows, at the class's positions where the key is defined. Modular addition fits on the training pairs and reads the held-out pairs.
- **Measured, on the held-out rows:**
  - **KL:** from the model's prediction to the crystallized model's, averaged over rows. This is the meter.
  - **The mean ablation's KL:** `Y` replaced by its fitting mean. This is the scale, with the share of it removed, `1 − KL / KL_ablation`.
  - **Agreement:** the share of rows whose top token is the model's.
  - **The loss delta:** the crystallized model's loss on the true next token minus the model's (the charter's), beside both accuracies. Recorded, never judged: a table keyed by the answer can beat the model.
  - **R²** of the function's value against `Y`.
  - **Each wrong function's KL,** with the same form fitted on the same rows.
  - **The counterfactual,** where the function declares one: the KL from the counterfactual model and the agreement with it, keyed by what the component reads and keyed by the answer.
  - **The twin:** everything above on the run's twin, fitted on the twin.
- **The label:**
  - **holds:** KL at most 0.25 and the ablation's KL at least 1.
  - **doesn't hold:** the ablation's KL at least 1 but the KL above 0.25.
  - **inert:** the ablation's KL below 1, so the component barely moves the prediction there.
  - The record holds labels, never a pass or a fail (D47).
- **Across training:** at every checkpoint of the run, the form is fitted again at that checkpoint, since a table is read off the model it replaces. `fourier` keeps `K` from the request's checkpoint, as generalize-this keeps the final dictionaries (D52).
  - **Each row:** the KL, the ablation's KL, the label, and the model's accuracy at the class.
  - **The envelope:** the checkpoints labelled holds, recorded as runs of consecutive checkpoints.
  - **The gap:** the checkpoints where the model's accuracy is at least 0.9 but the crystallization doesn't hold, shown, never hidden.
  - **Before the envelope:** the checkpoints where it holds before accuracy reaches 0.9, shown too.
- **The meter, shown first:** the label, the KL beside the ablation's, agreement and the loss delta.

**D57: what is judged, and what each crystallization cites.**

- **Judged:**
  - induction's block 1 attention at `mechanism`, as a `table` by the copied token
  - two-hop's block 3 write at `query`, as a `table` by the hop's object
  - modular addition's block 0 write at `answer`, as a `table` by the sum and in `fourier`
- **Recorded:**
  - modular addition's MLP, `additive` in the sum and both operands, with its counterfactual
  - its attention, `additive` in the operands
  - two-hop's block 3 attention alone
- **Citations (C11):** a crystallization cites the judged name records for its key's description at its part's output site, and the judged edge record for anything it reads at its own position.
  - **Induction** cites G6.2's `token(p + 1)` after block 1 at `mechanism`. It reads across positions, which needs no edge (D50).
  - **Two-hop** cites G6.2's `token(p + 1)` after block 3 at `query`, and G6.3's hop, computed, for `b` read at the same position.
  - **Modular addition** cites G7.2's recorded names, which don't hold its sum, and its record says so: the sum is written in a few frequencies, which no set of four latents carries. A cited record's key must be in its run's records.
- **Why these:** each is a component the task requires (D30), and each was measured in feasibility. The MLP's same-position input has no edge to cite, so it stays recorded (D53), with its counterfactual beside it.

## What each run declares

The functions live with each run's task, in `zoo/functions.py`, as descriptions do (D29).

| Run | Component | Key | Reads | Wrong functions | Counterfactual |
| --- | --- | --- | --- | --- | --- |
| Induction | block 1 attention at `mechanism` (32 to 62) | the copied token `w[q − P + 1]`, `P` the window's period | `token(p)` at `q − P + 1`, across positions | the current token `w[q]`, the previous token `w[q − 1]` | broken windows |
| Two-hop | block 3 write at `query` (25) | the hop's object: the object of `b`'s fact | `b` at 25 after block 2, the same position, through the hop (G6.3, computed); the facts, across positions | `a`, `b` | `b` patched at 25 |
| Modular addition | block 0 write at `answer` (2) | the sum `(a + b) mod p` | `a` and `b`, across positions | `(a − b) mod p`, `(a + b + 1) mod p`, `a`, `b` | none: the sum is the answer by the task's construction |
| Modular addition, recorded | block 0 MLP at `answer` | the sum, `a` and `b`, additive | the attention output at its own position: no edge | none | the attention output patched |
| Modular addition, recorded | block 0 attention at `answer` | `a` and `b`, additive | `a` and `b`, across positions | none | none |
| Two-hop, recorded | block 3 attention at `query` | the hop's object | as the write | `a`, `b` | none |

The counterfactuals, each drawn with a fixed seed:

- **Broken windows:** in each held-out induction window, one position `j` from 33 to 62 gets another token. At `q = j − 1` the model can't see the change: what the component reads is still the copied token `w[q − P + 1]`, while the answer becomes the new token at `j`.
- **`b` patched at 25:** two-hop's residual after block 2 at 25 is patched from G4.3's aligned source, which asks `a → k → d′`. Block 3 then reads `b = k` there: what it reads is the object of `k`'s fact in the target window, while the answer is the window's `c`.
- **The attention output patched:** modular addition's attention output at 2 is patched from another held-out pair with a different sum. The MLP then reads the other pair's sum and operands, while the answer is the window's sum.

Fitting and held-out windows:

- **The zoo runs:** each uses Phase 5's training and held-out windows (`zoo_windows`).
- **Modular addition:** fits on its training pairs and reads its held-out pairs.
- **Probe:** its 8 windows are `default_rng(EVAL_SEED).choice(held, 8, replace=False)`, in that order, with Phase 4's `EVAL_SEED` (2).

A declaration for the gestures joins `zoo/declarations.py` too, for G7.2's recorded names:

- **Classes:** `answer` (2) and `every` (0 to 2).
- **Planted:** `token(p + 1)` after block 0 at `answer`, recorded.
- **Predictions:** `token(p + 1)` = `w` at 2 gives `(2, w)`.
- **Relations:** none.

## Discrimination tests on the zoo

The bars come from the feasibility rounds, with margin; the measured value follows each one. All of it ran in the cloud workspace:

- **Induction and two-hop:** on stand-ins trained with G4.1's configs and seeds. Two-hop's stood at step 10,000 of 20,000.
- **Modular addition:** on the specimen of D54.
- **Window counts:** fewer than the gates will use, so the gates measure every number again.

**Crystallizations (judged).**

| Test | Bar | Induction | Two-hop | Modular addition: table, fourier |
| --- | --- | --- | --- | --- |
| KL from the model | at most 0.25 | 0.000 | 0.134 | 0.021, 0.019 |
| The mean ablation's KL | at least 1 | 6.94 | 4.58 | 4.14 |
| Each wrong function's KL | at least 1 | 6.73 (current token) | 4.61 (`a`), 4.62 (`b`) | 5.90 to 10.16 |
| Counterfactual, keyed by what it reads: agreement | at least 0.8 | 1.000 | 0.938 | none applies |
| Counterfactual, keyed by the answer: agreement | at most 0.1 | 0.000 | 0.008 | none applies |
| On the twin: every wrong function's KL within 0.05 of the right one's | yes | within 0.001 | within 0.001 | within 0.003 |
| The envelope holds at the final checkpoint, with step 0 outside it | yes | from step 900 | from step 8,000 | fourier from step 1,100; the table at 1,100 and from 1,300 |

Recorded beside them:

- **Modular addition's loss delta:** −0.033 nats as a table and −0.027 in the fourier form, against the model's held-out loss of 0.123. The crystallized write is a little more confident than the model's own.
- **Its fourier form:** frequencies 2, 36, 25, 44 and 18, holding 0.795 of the embedding's energy. With four frequencies the KL is 0.263; with five, 0.019. A second seed learned frequencies 48, 14, 2, 4 and 24 (0.822), and gives 0.027 against 4.03.
- **Its envelope:**
  - The fourier form holds from step 1,100 (KL 0.199, ablation 1.12), while held-out accuracy is only 0.061 there.
  - The model passes 0.9 at step 1,700 (0.926).
  - So the write takes the closed form six checkpoints before the model answers held-out pairs: the clock forms before it is read out sharply.
  - The table holds at 1,100, misses at 1,200 (KL 0.258), and holds from 1,300. That flicker is why the envelope's bar asks only for the final checkpoint and step 0, and records the runs.
- **Induction:**
  - Its crystallization holds from step 900 (KL 0.146, ablation 5.13), while accuracy reaches 0.9 at step 800 (0.951, KL 0.307). Step 800 is its gap.
  - Its counterfactual: the model copies at every broken position, and the answer-keyed table's KL is 13.47.
- **Two-hop:**
  - Block 3's write is inert at `⇒` through step 5,000, where the ablation's KL is only 0.35 while the model answers 69.6%. The earlier route doesn't run through it (H6).
  - It holds from step 8,000 (KL 0.234, ablation 4.12).
  - In its counterfactual the patched model answers `k`'s object 95.1% of the time. The answer-keyed table's KL is 9.33, against 0.346 for the one keyed by what block 3 reads.
- **Recorded crystallizations:**
  - **Modular addition's MLP:** additive in the sum and both operands, KL 0.019 (ablation 7.50, R² 0.971). By the sum alone, 1.546: in the narrow model the MLP writes operand terms as well, which the attention's own cancel.
  - **Its counterfactual:** keyed by what the MLP reads, KL 0.017 and agreement 1.000; keyed by the window, 8.08 and 0.008.
  - **Its attention:** additive in the operands, KL 0.094 (ablation 7.99).
  - **Two-hop's block 3 attention alone:** KL 0.175 (ablation 4.93).

**The specimen (judged, G7.1).**

| Test | Bar | Feasibility |
| --- | --- | --- |
| Held-out accuracy at 2, final checkpoint | at least 0.99 | 1.000 (seed 1 also 1.000) |
| The twin at 2 | at most twice chance | 0.0088, chance 0.0088 |
| Position 0, trained | at most twice chance | 0.0001 |
| The fewest frequencies holding 0.75 of the embedding's non-constant energy | at most 8 | 5 (seed 1: 5; the twin: 39) |
| The logits against those frequencies' cosines, R² on held-out pairs | at least 0.8 | 0.966 (seed 1: 0.966) |
| Retrained from step 0 | identical final state hash | as G4.1 |

Recorded: training and held-out accuracy at every checkpoint. Training passes 0.99 at step 1,800, as held-out accuracy does. At width 32 the model doesn't memorize first, though at width 128 it did: 100% training and 18.3% held-out accuracy at step 1,000.

**The names on modular addition (recorded, G7.2).** A TopK SAE at block 0's `hook_resid_post` with Phase 5's settings (width 256, `k` 16), trained on the training pairs' activations at every position: FVU 0.017 on the held-out pairs (twin 0.046). Then name-this-direction, unchanged, at `answer` on the held-out pairs:

- **`token(p + 1)`, the sum:**
  - Named for 31.0% of values, and as planted for 23.9% (mean F1 0.433, the set's 0.459).
  - Steering follows 0.8%, the same as its null (0.8%). The swapped output stays unchanged 99.4% of the time.
  - Ablating the set changes nothing (0.0%): the other active latents still carry the sum.
- **`a` and `b` at 2:** named as planted for no value.
- **The twin:** names 1.8% of values, none as planted.

This is what C11 rules on. The record shows it beside the crystallization.

## Records, the viewer and badges

- **Records built by G7.5,** at `runs/phase7/gestures/<run>/`:
  - every crystallization above, with its envelope and its twin
  - modular addition's name records for every value of `token(p + 1)`, `token(p − 2)` and `token(p − 1)` at `answer`
  - name records for the latents the features view shows active at its readout in the probe windows, up to 8 per window, as G6.7 built them
- **The gestures panel gains crystallize-this.** Each record shows its meter first, then:
  - **the function:** its key, what it reads, and the records it cites, each by key and opening it
  - **the form:** the keys the fitting rows showed; or for `fourier`, its frequencies, each with its share of the fitted coefficients
  - **the controls:** the wrong functions' KLs; the counterfactual, both ways; the twin
  - **the timeline:** while the record is chosen, rows for the KL, the ablation's KL and the model's accuracy, with the envelope shaded, the gap marked, and its checkpoints before accuracy reaches 0.9 marked apart
- **The modular-addition run in the viewer:**
  - views built as G4.5 and G5.6 build a zoo run's: per-token loss, logit lens, attribution, layer structure and the timeline
  - the trace of every checkpoint on its probe, from G7.1
  - its TopK SAE's features payload, with its meters
  - its gestures panel
- **Badges:**
  - crystallize-this gets a badge of its zoo results, from G7.2's, G7.3's and G7.4's reports, above every record of it (D38)
  - its facts include modular addition's recorded names, with the note that they don't hold the sum
- **D53's correction:** the open-this-up badge's twin line becomes "most of any judged target". The Phase 7 badges file carries every phase's badges, as G6.7's did, and G6.7 runs again with the correction.

## Gates

Each runs on the Mac, prints progress while it works, and writes one schema-2 report under `reports/phase7/`.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G7.1 The modular-addition specimen | trained, traced at every checkpoint, retrained; behaviour and the clock (D54) | every bar met | Diagnose training |
| G7.2 Its dictionary and names | the TopK SAE, trained and twin, its meters; names at `answer` | recorded, not judged | Not applicable |
| G7.3 Determinism and the operation | `set`'s identities at every residual-shaped tap of the three zoo models; request hashes over 64 requests; each form's record built twice from one request | identities bit for bit; distinct hashes; byte-identical records | Diagnose the operation or the record |
| G7.4 Crystallizations | every judged and recorded crystallization, trained and twin, its counterfactual and its envelope | every bar met | Diagnose the gesture |
| G7.5 Records, views and badges | modular addition's views; every record above built twice; badges from the reports | byte-identical, no external URL, every subject has a badge, and the records shown are the ones G7.4 measured | Diagnose the build |

G6.7 runs again after the badge correction, before G7.5.

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 1 | The modular-addition task, its declaration and every run's functions (D54, D56) | `zoo/tasks.py`, `zoo/declarations.py`, `zoo/functions.py` | None |
| 2 | `set`, with its identities (D55) | `interventions/spec.py`, `interventions/apply.py` | None |
| 3 | crystallize-this: forms, measurements, labels, the envelope, citations (D56, D57) | `gestures/crystallize.py`, `gestures/records.py`, `gestures/cli.py` | 1, 2 |
| 4 | The Phase 7 gates G7.1 to G7.4 and their report rules | `gates/phase7/`, `reports.py` | 1 to 3 |
| 5 | The panel's crystallize-this, its timeline rows, the badges and D53's correction, modular addition's views, and G7.5 | `views/`, `viewer/`, `gates/phase7/` | 4 |

## Tests

Written with the stubs, delta by delta:

- **Delta 1 (D54, D55):** `test_modadd_vocab_split_and_windows`, `test_modadd_batches_are_seeded_training_pairs`, `test_modadd_declaration`, `test_functions_keys_and_reads`, `test_functions_wrong_and_counterfactual`, `test_set_arguments`, `test_set_identities_are_exact`, `test_set_moves_only_its_positions`, `test_set_refusals`
- **Delta 2 (D56):** `test_table_form`, `test_unseen_keys`, `test_additive_form`, `test_fourier_form_and_its_frequencies`, `test_write_is_attention_set_and_mlp_zeroed`, `test_kl_and_the_ablation`, `test_loss_delta_is_recorded`, `test_wrong_functions`, `test_counterfactuals`, `test_labels`, `test_envelope_and_gap`, `test_citations`
- **Delta 3 (the gates):** `test_crystallize_requests_round_trip_and_hash`, `test_records_are_cached_and_repeatable`, `test_the_command_line`, `test_settings_and_bars_are_fixed`, `test_phase7_gate_ids`, `test_phase7_status_rules`
- **Delta 4 (panel and badges):** `test_crystallize_payloads`, `test_crystallize_badges_cite_the_reports`, `test_open_badge_twin_line`, `test_g75_status_rules`

## Out of scope

- **Single attention heads:** induction's copy is spread over all four of block 1's heads (attention from `q` to `q − P + 1` of 0.36 to 0.45 each). Replacing the best head, by the explicit pattern or by its mean, moves the KL by at most 0.001. Heads can be crystallized only where one alone matters.
- **Forms found by search:** only declared forms are fitted. Finding a closed form automatically would be a hypothesis no meter has checked.
- **The reference run:** it has no planted function (D44).
- **Names for distributed codes:** for example, the phase of the sum at a frequency, which might name modular addition's latents (notebook, H10).
- **Dictionary-space crystallization:** writing a name's sets at their mean codes instead of a table.
- **Modular addition at width 128:** recorded in the notebook as the counterexample.
- **The ledger** (Phase 8); **Tracr** and the architecture dial; **two-hop's earlier route** as an explanation.

## Settled while drafting

- **KL, not accuracy or loss:** a table keyed by the answer beat models that hadn't learned. Modular addition's MLP table at width 128, step 1,000, was right on 99.3% of held-out pairs where the model was right on 18.3%. Its KL, 3.63, was above the mean ablation's, 2.49 (finding 24).
- **Counterfactuals:**
  - **Induction:** in broken windows the model copies at every broken position; the table keyed by what it reads agrees every time, and the one keyed by the answer never does (finding 26).
  - **Modular addition's MLP:** keyed by what it reads, KL 0.017; keyed by the window, 8.08.
- **A block's write via `set` and `zero`:** the residual after the block becomes the residual before it plus the function's value. The tables were fitted to the write, attention plus MLP.
- **Refitting at every checkpoint:** a table is read off the model it replaces, so a checkpoint's crystallization is that checkpoint's. Only the fourier form's frequencies stay fixed.
- **Width 32 and the closed form as the headline,** with the table beside it: apolloh0's choices on 2026-10-23 (finding 23).
- **The boundary:** at width 32, modular addition's MLP alone, keyed by the sum, gives KL 1.546, but the block's write gives 0.021. The attention writes each operand, and the MLP writes the sum plus terms that cancel them (finding 27).
