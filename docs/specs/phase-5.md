# Phase 5 spec: dictionaries and their edge-sparsity meter

Status: ready for stubs and tests once D40 to D45 are committed. Charter: [../charter.md](../charter.md). Decisions: [../decisions.md](../decisions.md).

## Purpose

Phase 4 ended with a finding, not a fix. On the two-hop model, both lenses rank the intermediate last at `⇒` after blocks 1 and 2, exactly where patching shows the model holds it (D39). A view that reads activations through the unembedding can't see a variable the model keeps out of its output.

Phase 5 builds the other kind of decoder: dictionaries, which split an activation into a few active latents out of many. The charter lists four: the TopK SAE, the Staircase SAE, the transcoder and, as an experiment, the Jacobian SAE. They share one edge-sparsity meter, which says how few connections between two dictionaries' latents carry a computation.

Nothing ships on the reference run until it has been held to the zoo, where the answer is known, and to each model's random-weights twin, which has nothing to find. Two rounds of feasibility runs on the reference backend came first, as D37 requires of the zoo itself. The bars below come from them, with margin.

Human gate H5 asks: does every shipped dictionary carry its meters and a badge of zoo results, generated from the reports? Does the features view show the two-hop intermediate's latents at `⇒`, where patching finds it? Only then can Phase 6's gestures name a feature or assert an edge between two.

## Rulings

Proposed with this spec; committing them is the ruling.

**D40: a dictionary is a derived, content-addressed artifact, trained bit for bit.** The four kinds share one code path:

- **Encode:** subtract a pre-encoder bias, multiply by the encoder, add the encoder bias, and keep the `k` largest pre-activations, clipped at zero.
- **Decode:** multiply by a decoder whose rows have unit norm, and add the decoder bias.
- **Scale:** each site's inputs and outputs are divided by a constant recorded with the dictionary, the root mean square norm of its training activations over `√d_model`.

The kinds:

- **TopK SAE:** one tap, reconstructed.
- **Staircase SAE:** every block's `hook_resid_post` at once. The encoder and decoder are shared; block `l` uses the first `(l + 1) · width / n_layers` latents, with its own biases and scale.
- **Transcoder:** reads `blocks.l.ln2.hook_normalized` and reconstructs `blocks.l.hook_mlp_out`, standing in for the MLP.
- **Jacobian SAE (experimental):** a TopK SAE on each side of one MLP, trained together. Its penalty is `λ` times the mean absolute entry of the Jacobian between the two SAEs' active latents, through the MLP.

**Training:** Adam, with a linear warmup over the first 100 steps and a linear decay to zero over the last 30%. Latents silent for 100 steps are trained on the reconstruction error through AuxK: the largest `min(4k, width / 2)` of their pre-activations, weighted 1/32. Decoder rows are renormalised after every step. The same key trains the same bytes on the reference backend.

**Data:** activations recomputed through the model at its final checkpoint, which is exact under the run's determinism. The trace store's eight-window probe is far too small to train on. Training windows come from the run's own source with seed 6: a zoo task's generator, or the reference corpus's training split. Measurements use held-out windows: seed 7 for a zoo task, the reference run's evaluation slice for TinyStories. Every position trains; the dictionary isn't told where to look.

**Cache:** under D11 and D18. The key holds:

- the checkpoint's state hash
- the taps
- the kind and every setting: width, `k`, steps, batch, learning rate, seed and `λ`
- the training windows' batch hashes
- the code identity

A dictionary file holds its weights, biases, scales and key. It is never edited, only superseded by another key.

**D41: dictionaries are judged on the zoo, against their twins, by what the ground truth predicts.** Each shipped kind is trained on both zoo models and on each model's random-weights twin, with the same settings. Three tests are judged:

- **Meters:** reconstruction, and the task's loss with the reconstruction spliced in.
- **A faithful code:** wherever a Phase 4 patch moves a planted variable, the same patch made through the dictionary must do what the residual patch does. The target's code is replaced by the source's, and the target's reconstruction error is kept. A dictionary whose latents lose the causal variable to its error term would mislead every reading of its features.
- **Selective latents:** where the ground truth places a planted variable, single latents must pick it out, clearly better than the twin's dictionary can.

Recorded beside them, not judged:

- **The single-latent patch:** moving only the latents assigned to the variable.
- **Readability:** how well a probe reads the variable from the activation, and from the code.
- **Seed robustness:** the same results for two more dictionary seeds.

The single-latent patch asks whether a feature *is* a variable. That is the claim Phase 6's gestures will make, and Phase 5 measures how far it already holds.

**D42: the edge-sparsity meter is SCALAR's ablation curve, read as the edges that carry 90% of the effect.** Take a pair of dictionaries around a segment of the model.

1. **Score:** every upstream-to-downstream latent pair, by integrated gradients from the origin in five steps, then the root mean square over windows.
2. **Ablate:** for each of 14 log-spaced edge counts `n`, keep the top `n` edges. Compute each downstream latent from only its kept upstream latents, zeroing the others at every position. Splice the downstream decode in, and average the KL from the clean logits over positions.
3. **Read:** the meter is the fewest edges after which the KL stays within 10% of the way from one edge's KL to the floor. The floor is the lower of two KLs: the pair spliced in whole, and the curve's last point. The meter is also reported as a share of the scored edges.

Where it is judged and where recorded:

- **Across an MLP, judged:** the planted computation must look sparse beside its twin's.
- **Across a block, recorded:** masking upstream latents at every position also changes where attention looks. So a circuit routed by attention can look as dense as noise.

It applies to pairs of SAEs: TopK pairs, adjacent blocks of a Staircase SAE, and Jacobian SAE pairs. A transcoder replaces its MLP, so on a transcoder the meter would count the transcoder's own encoder. It isn't measured there in Phase 5.

**D43: the Jacobian SAE is built and recorded, not shipped.** Its penalty did what it penalises without doing what it's for. It shrank the Jacobian's entries, but it didn't concentrate them into fewer, and the meter found fewer edges at only one MLP of four. Meanwhile its reconstruction worsened. It stays experimental, as the charter says: a recorded result, no view.

**D44: the reference run gets every judged dictionary, with recorded meters and its zoo badge.** These train at G1.4's final checkpoint:

- TopK SAEs on every block's `hook_resid_post`
- the Staircase SAE
- transcoders on every MLP

TinyStories has no ground truth, so their meters are recorded, never judged, and shown with every view of them. Each dictionary's badge carries its zoo results, generated from the Phase 5 reports and never typed (D38). Dictionaries at earlier checkpoints, and features followed across training, are a later question.

**D45: a dictionary patch joins the operations, and a features view joins the viewer.**

- **`dpatch`:** its arguments are `dictionary` (a content hash), `source` (a batch hash, as `patch` takes) and optionally `latents` (a list of latent ids). At the dictionary's site and the edit's positions, the activation `x` becomes `x + (f_source − f_target) · W_dec`, times the site's scale. With `latents`, only those latents' change counts.
  - **A transcoder:** its codes come from its input tap and the edit lands on its output tap.
  - **Identities:** a source equal to the target, and an empty `latents`, reproduce the unintervened pass bit for bit (D25).
  - **Uses:** the faithful-code and single-latent tests run through `run_intervened` with it. So will Phase 6's gestures when they move a feature.
- **The features view:** for a chosen dictionary and site, at the final checkpoint. For the probe, each position's active latents and their values. For each latent:
  - how often it is active
  - its top-activating contexts from held-out windows
  - its decoder row's direct effect on the logits through the unembedding, labelled as that and not as a name

  The meters and the badge sit above it. It builds into the run's views like the others, byte-identically, with nothing outside the repo.

## The dictionaries

| Kind | Sites | Width | `k` | Training |
| --- | --- | --- | --- | --- |
| TopK SAE | each block's `hook_resid_post` | 8 × d_model | 16 | zoo: 3,000 steps at batch 4,096; reference: 2,000 at batch 2,048 |
| Staircase SAE | every block's `hook_resid_post`, together | 8 × d_model in all | 16 | as TopK |
| Transcoder | each block's MLP | 8 × d_model | 16 | as TopK |
| Jacobian SAE | across each block's MLP, `λ` of 1 and 4 | 8 × d_model a side | 16 | as TopK; the penalty on 1,024 of each batch |

All train at learning rate 5e-4 with dictionary seed 0. Training windows: two-hop 16,384, induction 8,192, reference 2,048 (524,288 tokens). Held-out windows: two-hop 2,048, induction 1,024, the reference run's first 64 evaluation windows. A 32 × width raised selectivity but left up to 76% of a site's latents dead, while 8 × clears every bar.

## Discrimination tests on the zoo

The readout is `⇒` (position 25) for two-hop, and the mechanism positions, 32 to 62, for induction. Each test runs on the trained model; a twin is trained with the same settings wherever a bar compares with it.

**Meters (judged, every kind, every site).** The fraction of variance unexplained is at most 0.1. Wherever a per-position mean ablation of the site costs at least 0.1 nats at the readout, splicing the reconstruction in recovers at least 90% of that cost. Elsewhere the change in nats is recorded: two-hop's MLPs in blocks 1 to 3 cost at most 0.012 nats to ablate, so a ratio there would only be noise. Also recorded: the share of dead latents, and the twin's reconstruction.

**Faithful code (judged).** Each test patches through `dpatch` and through the residual `patch`, from the same sources. The dictionary passes where its patch reaches at least 90% of the residual patch's rate. The test applies only where the residual patch moves the variable at least half the time.

- **Two-hop, TopK and Staircase:** at the query, from G4.3's aligned sources, after every block. The swap rate counts where it applies (blocks 1 and 2), and the transplant rate where it applies (block 3). The control source leaves the answer unchanged at least 95% of the time.
- **Induction, TopK and Staircase, after block 0:**
  - **Following:** G4.2's patch at copy targets 13, 16 and 19, a new token at `j`, read at `j + 19`.
  - **Breaking:** a new token at `j − 1` patched in at `j`, which removes the match the induction head looks for. Here the dictionary's patch must keep the original prediction no more than 0.05 more often than the residual patch does.
- **Induction, transcoder:** the same two tests at block 0's `hook_mlp_out`, where patching the MLP's output alone moves the prediction. Two-hop's MLP outputs carry no planted variable at the query, so there is nothing there to test.

**Selective latents (judged).** For each value of a planted variable, its best latent is the one whose activity best matches it by F1, over held-out windows at the variable's position. The mean of those best F1s is at least 0.5, and at least twice the twin's dictionary's at the same site.

- **Two-hop, TopK and Staircase:**
  - the intermediate `b` at `⇒`, after G4.3's best swap block (block 2)
  - the answer `c` at `⇒`, after the last block
- **Two-hop, transcoder:** `b` at `⇒`, in the code of the MLP of G4.3's best swap block.
- **Induction, all three kinds:**
  - the current token, after block 0 (from block 0's MLP input for the transcoder)
  - the predicted next token, after block 1 (from block 1's MLP input for the transcoder)

**Recorded beside them.**

- **The single-latent patch:** `dpatch` with `latents` limited to those whose best F1, over every variable at the position, is for the tested variable and at least 0.5.
- **Readability:** a ridge probe's accuracy for each variable from the activation and from the code.
- **Other variables:** the query subject `a` at 24 and 25, `b` at 24 and after block 1, the previous token at block 0. The last is not readable at all (a probe reaches 9% against 1.6% chance). Matching there doesn't run on a token identity, so nothing about it is judged.
- **Seed robustness:** TopK at dictionary seeds 1 and 2.

## The edge-sparsity meter

**Judged:** TopK pairs across every MLP whose mean ablation costs at least 0.1 nats at the readout: induction's blocks 0 and 1, two-hop's block 0. At each, the trained model's meter is at most a fifth of its twin's.

**Recorded:**

- the same pairs across every other MLP
- TopK and Staircase across every block of both zoo models and their twins
- Jacobian SAE pairs at `λ` of 1 and 4 across every MLP of both zoo models
- on the Jacobian, the median number of entries holding 90% of its absolute mass, the share of entries above 0.01, and the mean absolute entry

The meter scores edges over eight held-out windows: two for the reference run, whose windows are 256 tokens.

## The reference run

At G1.4's final checkpoint, these train with the settings above:

- TopK SAEs on every block
- the Staircase SAE
- transcoders on every MLP
- TopK pairs across blocks 1 and 2's MLPs, for the meter

Recorded:

- every meter
- the meter's curve across both MLPs
- each dictionary's training time

Nothing is judged: there is no planted structure to recover.

## Gates

Each runs on the Mac, prints progress while it works, and writes one schema-2 report under `reports/phase5/`. Dictionaries come from the cache when their key matches, so a gate retrains nothing another has trained.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G5.1 Determinism and the operation | Each kind trained twice from one key on the induction model; cache keys over 64 setting combinations; `dpatch`'s two identities at every site of both zoo models | Byte-identical files; distinct keys; both identities bit for bit | Diagnose the trainer or the operation |
| G5.2 Induction | TopK, Staircase and transcoder on the induction model and its twin: meters, faithful code, selective latents | Every bar met by every kind | Diagnose the dictionary that failed |
| G5.3 Two-hop | The same on the two-hop model and its twin | Every bar met by every kind | Diagnose the dictionary that failed |
| G5.4 Edge sparsity | The meter on every pair named above; the Jacobian SAE | Every judged MLP's meter at most a fifth of its twin's | Diagnose the meter |
| G5.5 Reference dictionaries | Every dictionary on the reference run, its meters and the meter's curves | Recorded, not judged | Not applicable |
| G5.6 Features view and badges | The features view on both zoo runs and the reference run, built twice; each dictionary's badge, generated from the reports; the reference run's and both zoo runs' views rebuilt with D39's wording | Builds byte-identical, no external URL, every shipped dictionary has a badge | Diagnose the build |

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 1 | The dictionary kinds, their training and cache (D40) | `dictionaries/` | None |
| 2 | `dpatch`, with its D25 identities (D45) | `interventions/spec.py`, `interventions/apply.py` | 1 |
| 3 | Meters, readability, selectivity and the faithful-code and single-latent patches (D41) | `dictionaries/measure.py` | 1, 2 |
| 4 | The edge-sparsity meter and the Jacobian SAE's penalty (D42, D43) | `dictionaries/edges.py` | 1 |
| 5 | Phase 5 gates and their report rules | `gates/phase5/`, `reports.py` | 1 to 4 |
| 6 | The features view and its payloads; dictionary badges from the reports; the D14 badges' new wording (D38, D39, D44, D45) | `views/`, `viewer/`, `gates/phase5/` | 5 |

## Tests

Written with the stubs, delta by delta:

- **Delta 1 (D40, D43, D45):** `test_settings_are_checked`, `test_sites_are_checked`, `test_a_dictionary_checks_its_arrays`, `test_encode_keeps_the_k_largest_pre_activations`, `test_a_staircase_site_uses_only_its_share`, `test_decode_delta_and_reconstruct`, `test_the_content_hash_covers_everything`, `test_the_cache_key`, `test_files_round_trip_and_refuse_tampering`, `test_training_is_deterministic`, `test_initialisation_follows_the_seed`, `test_every_setting_reaches_the_trainer`, `test_a_topk_sae_finds_planted_features`, `test_a_staircase_sae_trains_every_block`, `test_a_transcoder_learns_a_sparse_map`, `test_mlp_matches_the_model`, `test_the_pair_jacobian_is_the_derivative`, `test_the_penalty_shrinks_the_jacobian`, `test_bad_inputs_are_refused`, `test_code_identity_is_the_modules_blob_hash`, `test_dpatch_arguments`, `test_dpatch_identities_are_exact`, `test_dpatch_moves_the_code`, `test_dpatch_latents_move_only_theirs`, `test_dpatch_on_a_transcoder_and_a_staircase`, `test_dpatch_reads_the_codes_after_a_swap`, `test_dpatch_refusals`
- **Delta 2 (D41):** `test_activations_come_in_chunks`, `test_fvu`, `test_dead_share`, `test_readout_loss`, `test_spliced_logits`, `test_ablated_logits`, `test_f1_table`, `test_mean_best_f1`, `test_assigned`, `test_probe_accuracy`, `test_patch_predictions`

## Out of scope

- **Gestures** (Phase 6), and with them crystallize-this and modular addition (C9, D31).
- **Dictionaries at earlier checkpoints,** and features followed across training.
- **Cross-layer transcoders** and attribution graphs over transcoder features.
- **The edge meter across attention,** beyond recording it: separating where a head looks from what it moves is a later design.
- **The architecture dial,** which a Jacobian SAE across a whole block would need, as SCALAR's authors found.
- **The self-learning ledger** (Phase 7).

## Settled while drafting

- **The bars:** as stated above, set from feasibility with margin. Every kind clears them by at least 0.12 in F1, by at least a factor of three on the edge meter, and at no worse than 94% of the residual patch's rate on the faithful code.
- **Width and `k`:** 8 × d_model and 16, on the evidence above.
- **Selectivity at G4.3's best swap block, not every block.** After block 1 the intermediate is nearly as causally present (in feasibility, a residual swap of 0.887 against 0.896 after block 2) but spread over latents: single latents reach a mean best F1 of only 0.50 to 0.53. That is recorded as a fact about the model's representation, not held against the dictionary.
- **The meter's reading.** SCALAR's own summary, the area under the curve, ranked the models inconsistently. The curves aren't monotone, and their scales differ between a trained model and its twin by two to three orders of magnitude. The edges needed for 90% of the drop separated every MLP pair from its twin.
- **D39's carried sentence.** The D14 badges' last sentence, "Planted structure at model level waits for Phase 4's zoo.", becomes "Its model-level result is in the badge below (D38).". G5.6 rebuilds the reference run's and both zoo runs' views to carry it, and their bytes change.
