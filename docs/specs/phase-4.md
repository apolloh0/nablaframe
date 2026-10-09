# Phase 4 spec: the ground-truth zoo and discrimination tests

Status: ready for stubs and tests once D29 to D35 are committed. Charter: [../charter.md](../charter.md). Decisions: [../decisions.md](../decisions.md).

## Purpose

Every view and tool so far has passed tests where we planted the answer in the decoder: a lens handed residuals built to contain a token, an attribution handed a known direction, an edit held to bit-for-bit identities. Every D14 badge ends the same way: "planted structure at model level waits for Phase 4's zoo".

Phase 4 builds that zoo: small models trained on tasks whose mechanism is known in advance, because the task requires it. Then every view and tool meets a model whose answer we know. The charter's two signature demos come out of it:

- **an induction head forming at the loss bump,** watched on the scrubber
- **a planted two-hop intermediate swapped, with the answer changing**

It also settles the question D28 carried forward: does either lens recover structure we know is there?

Human gate H4 asks: does every shipped view and tool now carry a model-level result on its badge, and do both signature demos reproduce? Only then can Phase 5's explanatory gestures be tested against something real.

## Rulings

Proposed with this spec; committing them is the ruling.

**D29: zoo tasks plug into the existing pipeline unchanged.** A task is a deterministic, seeded batch generator with the corpus's interface:

- `batch_for_step(step)` returns int32 windows of the task's length
- every batch has a `batch_hash`
- the task writes a tokenizer file in our format, so views can show its tokens

The trainer gains an optional batch source. Its default is the corpus, unchanged, and the reference run's tests and gates must still pass bit for bit. Everything else, the trace engine, the views and the interventions, applies to a zoo model exactly as to the reference run, with no task-specific code paths. Each zoo model is a run with its own manifest, checkpoints, trace and views. A task also declares which positions require its mechanism, so behaviour can be measured where it matters.

**D30: ground truth comes from the task's construction, never from our own views.** A mechanism counts as planted when three things hold:

- **the task requires it:** the positions that need it can't be predicted any other way
- **the trained model succeeds there:** measured from logits alone
- **a control fails there:** positions or inputs that don't allow it, and the model's random-weights twin

Views and tools are then tested against that ground truth. Defining ground truth with the views being tested would make every test circular.

**D31: zoo v1 is induction and in-context two-hop, the charter's two signature demos.** The other two ground-truth tasks wait for the phase whose question they answer:

- **Modular addition waits for crystallization** (Phase 5's crystallize-this). Its known mechanism is a function, sums of cosines at a few key frequencies, which is exactly what crystallization should recover.
- **Tracr-compiled models wait for the architecture dial.** Their weights assume no LayerNorm and ReLU MLPs. Translating them into our GPT-2 would change the very program they were compiled from, losing the exact ground truth that makes them worth having.

**D32: two-hop is in context, not memorized.** A window states six facts in shuffled order, then asks. In `d → k ; a → d ; m → q ; ... a ⇒ k`, the answer `k` is reached through the intermediate `d`. So:

- **the intermediate is a known token at a known position,** in every window
- **the model must compose:** distractor facts and shuffled order mean no single fact gives the answer

Two-hop over facts memorized in the weights is known to resist composition in small transformers. A model that only memorized each chain would have no intermediate to find, so parametric two-hop is recorded as a later question.

**D33: the J-lens coordinate swap joins the operations,** as `jswap`. The charter's manipulation scope lists it, and Phase 3 left it out. Its arguments are `J` (a content hash), and two token ids, `a` and `b`. At a residual-stream tap:

- **The directions:** with `u_a = Jᵀ w_a` and `u_b = Jᵀ w_b`, where `w` is the model's unembedding.
- **The swap:** the residual's component in their span, `α u_a + β u_b` by least squares, is exchanged, giving `r + (β − α)(u_a − u_b)`.
- **The identity:** `a == b` leaves `r` unchanged, and D25 requires it to reproduce the unintervened pass bit for bit.

On the two-hop model it asks the decisive question. If swapping the intermediate's J-lens coordinates for another entity's changes the answer to that entity's successor, the J-lens direction *is* the intermediate, causally.

**D34: model-level lens results are recorded and shown on the badge, not pass or fail.** The causal ground truth, from patching, locates where and when the intermediate exists. A lens can legitimately miss a representation that isn't aligned with the unembedding: that's a fact about the model, not a broken view. So each lens's badge gains its model-level result, what it recovered and what it didn't, beside the random-weights twin's. Tools that change the model, by contrast, are judged: an ablation or patch that doesn't produce the effect the ground truth predicts is a broken tool.

**D35: leave-a-batch-out replay waits for the self-learning ledger** (Phase 6). It's a training-time counterfactual, retraining without one batch, and its natural question is what a single update taught the model, which is the ledger's question. No zoo test needs it.

## The zoo

### Induction

- **Task:** random tokens from a vocabulary of 64, in windows of 64, each window's second half repeating its first. A window of 64 tokens trains a model of sequence length 63, since each window holds one more token than the model reads, as the reference run's windows do. Two-hop's 27 tokens train a sequence length of 26 the same way.
- **Mechanism positions:** 32 to 62, each predicting a token copied from the first half. **Control positions:** 0 to 30, unpredictable, with chance 1/64.
- **Model:** 2 blocks, d_model 64, 4 heads, d_mlp 256: two blocks, so a previous-token head and an induction head can compose.
- **Training:** 4,000 steps at batch 32, seed 0, with checkpoints every 50 steps through step 1,500 and every 250 after. Every checkpoint is traced, so the loss bump is resolved on the scrubber.

### In-context two-hop

- **Task:** a vocabulary of 48 entities, `e0` to `e47`, and the separators `→`, `;` and `⇒`. A window holds six facts, `x → y ;`, in shuffled order, then the query `a ⇒ c`: 27 tokens.
- **The facts:**
  - the chain `a → b` and `b → c`
  - four distractors
  - every subject distinct, so each lookup is unique
  - one distractor, `b′ → d`, plants a **counterpart** subject for the patch test
- **Mechanism position:** 25, the `⇒`, predicting the answer `c`.
- **Controls:**
  - the random-weights twin
  - **broken chains:** windows with the `b → c` fact removed, where the answer can't be known
- **Model:** 4 blocks, d_model 128, 4 heads, d_mlp 512: four blocks, matching the reference run, so the lenses have middle layers to read.
- **Training:** 20,000 steps at batch 64, seed 0, with checkpoints every 1,000 steps.
- **J:** each block's summed Jacobian, fitted as D17 fits the reference run's, on the task's own windows with seed 1, over positions 16 to 25, so the J-lens and `jswap` have their J.

## Discrimination tests at model level

The bars are set here, before any run, and are never moved after.

**Behaviour (judged).**

- **Induction:** accuracy at least 0.9 at mechanism positions; at most twice chance at control positions; at most twice chance for the random-weights twin.
- **Two-hop:** accuracy at least 0.9 on held-out windows; at most twice chance on broken chains; at most twice chance for the random-weights twin.

**Attention view, on the induction model (judged).**

- Some head in block 1 puts a mean of at least 0.5 of its attention on induction targets, at mechanism positions. A target is the token after the current token's previous occurrence.
- The random-weights twin's best head puts at most twice what uniform attention would.

**Ablation, on the induction model (judged).** Zeroing that head's output through `attn.hook_z`: Zeroing one head uses `zero` with an optional `head` argument, valid only at `attn.hook_z`. At no positions it's a null edit, held to D25 like every operation.

- removes at least half of the mechanism positions' accuracy above chance
- removes more than zeroing any other single head does

**Component swap, on the induction model (judged).** Swapping block 1's attention from the last checkpoint before the loss bump removes at least half of the accuracy above chance.

**Patch, on the two-hop model (judged): the signature demo.** At every block output, patch the query's two positions, 24 and 25, from a source window whose intermediate is the target's counterpart subject `b′`. Recorded per layer:

- **the swap rate:** how often the answer becomes `d`, the counterpart's successor in the target's own facts, meaning the intermediate moved
- **the transplant rate:** how often it becomes the source's own answer, meaning the answer moved

Pass:

- **the best layer's swap rate is at least 0.8**
- **a control:** a patch from a source window with the *same* intermediate leaves the answer unchanged at least 95% of the time

**Lenses, on the two-hop model (recorded, D34).** At positions 24 and 25, at each layer: the intermediate's and the answer's ranks in the logit lens and in the J-lens, beside the patch profile and the random-weights twin. This is D28's question, asked of a model where the answer is known.

**J-lens coordinate swap, on the two-hop model (recorded).** Swapping the intermediate's J-lens coordinates with the counterpart's, at each layer, at the query's positions: how often the answer becomes `d`.

**Direct logit attribution, on both models (recorded).** Which parts the attribution credits for the answer's logit at mechanism positions. Recorded, because an MLP may legitimately amplify what attention copies.

## The induction formation demo

The induction model's trace, built into views like the reference run's. On the scrubber, the mechanism positions' loss drops sharply between two checkpoints, and the induction head's attention to its targets rises across the same checkpoints. Recorded: the step of the steepest loss drop and the step of the steepest attention rise, side by side.

## Gates

Each runs on the Mac, prints progress while it works, and writes one schema-2 report under `reports/phase4/`.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G4.1 Zoo training | Both models trained, traced, and retrained to a seeded recompute; each model's behaviour bars | The recompute identical at the final checkpoint; every behaviour bar met | Diagnose the task or the model size; a size change is a ruling |
| G4.2 Induction | The attention view's, the ablation's and the component swap's tests | All three bars met | Diagnose the view or tool that failed |
| G4.3 Two-hop | The patch test, with its swap and transplant profiles | Both bars met | Diagnose the patch, or the model's composition |
| G4.4 Lenses at model level | Both lenses' ranks, the J-lens coordinate swap, direct logit attribution, and the zoo J's | Recorded, not judged | Not applicable |
| G4.5 Formation and viewer | The formation demo's two steps; both zoo models' views built twice | Builds byte-identical, no external URL | Diagnose the build |

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 1 | Task generators, their tokenizers and mechanism positions; the trainer's batch source (D29, D32) | `zoo/tasks.py`, `train/` | None |
| 2 | Behavioural ground truth: accuracy at mechanism positions and on controls (D30) | `zoo/ground_truth.py` | 1 |
| 3 | The J-lens coordinate swap, with its D25 identity (D33) | `interventions/spec.py`, `interventions/apply.py` | None |
| 4 | Induction scores, head ablation and the formation steps | `zoo/induction.py` | 2 |
| 5 | The two-hop patch profiles and lens ranks | `zoo/two_hop.py` | 2, 3 |
| 6 | Phase 4 gates and their report rules | `gates/phase4/`, `reports.py` | All |
| 7 | Model-level badges: each view's model-level result on its badge, after H4's runs (D34) | `views/build.py`, `viewer/` | 6 |

## Tests

Written with the stubs:

- **1:** `test_induction_windows`, `test_two_hop_windows`, `test_trainer_batch_source_leaves_the_default_unchanged`
- **2:** `test_next_token_accuracy`, `test_twin_params`
- **3:** `test_jswap_and_head_arguments`, `test_jswap_of_a_token_with_itself_is_exact`, `test_jswap_exchanges_the_coordinates`, `test_jswap_and_head_zero_where_they_apply`
- **4:** `test_induction_targets`, `test_induction_scores`, `test_formation_steps_and_head_ablation`
- **5:** `test_sources_carry_the_counterpart`, `test_patch_outcomes`, `test_token_ranks`

## Out of scope

- **Modular addition** (Phase 5) and **Tracr-compiled models** (the architecture dial), per D31.
- **Two-hop over memorized facts,** per D32.
- **Leave-a-batch-out replay** (Phase 6), per D35.
- **The architecture dial** itself: LayerNorm-free, weight-sparse and bilinear variants.
- **Dictionaries and gestures** (Phase 5).

## Settled while drafting

- **The bars:** as stated under "Discrimination tests", fixed before any run.
- **The model sizes:** induction 2 blocks at d_model 64; two-hop 4 blocks at d_model 128. If a model misses its behaviour bar, one size change is allowed, recorded as a ruling, before any other gate runs.
- **The two-hop format:** six facts, query last, 27 tokens. The counterpart fact makes the patch test tell an intermediate from an answer.
- **Patching both query positions:** the intermediate may live at the queried subject's position, 24, or at `⇒`, 25. Patching both doesn't assume which.
