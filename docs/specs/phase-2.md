# Phase 2 spec: read-only views and the timeline scrubber

Status: views ruled at human gate H2 on 2026-10-03: see D16 in [../decisions.md](../decisions.md). The J-lens views follow in a later delta under D17. Charter: [../charter.md](../charter.md).

## Purpose

Phase 2 makes the reference run something you can look at. A timeline scrubber moves through its 28 trace checkpoints, and every panel updates with it: what each token cost the model, where each attention head looks, what each layer would predict if the model stopped there, and which components push the answer up or down.

Nothing in Phase 2 changes the model or its trace. Every view is a pure decode of the trace store and the checkpoints, and every view shows a meter saying how far to trust it.

Human gate H2 asks: are these views faithful enough for Phase 3 to manipulate the model through them? The evidence is gates G2.1 to G2.6, and your own session with the scrubber.

## Rulings

Proposed with this spec; committing them is the ruling.

**D11: views are decoders, and anything they derive is a cache.** A view is a pure function of the trace store, the manifest and the checkpoint files. Anything a view computes and keeps, such as lens readings or attributions, is a cache keyed by the SHA-256 of its inputs and the commit of the code that made it. Deleting a cache loses nothing, and rebuilding it gives identical bytes. The trace stays the only canonical record.

**D12: the engine also records the model's logits.** Today the trace stops at `ln_final.hook_normalized`. Per-token loss, and every lens's comparison against the model's own prediction, need logits. Recomputing them outside the model's compiled program isn't guaranteed bit-identical. Recording them makes those views exact readouts.

- The tap is named `logits`, with axes Batch, Pos, Vocab and a new site, `unembed`. It's the one tap name that isn't a TransformerLens hook, since TransformerLens has no hook on its output. That's a documented exception to C5.
- Retracing the reference run adds one event per checkpoint, 28 in all, and about 0.94 GB. The 980 existing events are untouched, because the engine skips identical content.

**D13: the viewer is a static page that reads the trace store directly.** It's one HTML file with plain JavaScript and no build step, served locally by `python3 -m http.server`. It never contacts the network.

- **Readouts come from the canonical trace,** with nothing copied out. The viewer parses `events.jsonl` and reads tensors straight from the safetensors shards.
- **It checks what it shows.** The browser computes each loaded tensor's SHA-256 with its built-in Web Crypto and compares it with the event's recorded hash. A readout's meter is that check, live, on every tensor displayed.
- **Derived views come from D11's caches,** written as small JSON files.

**D14: what "discrimination test" means in Phase 2.** The charter's rule is that a view ships only after it recovers planted structure and comes up empty on random weights. Phase 2 can test both halves, but not both at model level yet:

- **Planted structure, at the decoder level:** synthetic traced tensors with a known signal planted in them, such as a residual stream carrying a chosen token's direction. The view must recover it.
- **Empty on random weights, at the model level:** the same view run on a freshly initialized model must find nothing beyond chance.
- **Planted structure at the model level arrives with Phase 4's zoo.** Until then, each explanatory view carries a badge saying which levels it has passed.
- **Direct logit attribution is a decomposition, not a reading.** Its parts sum to the residual stream by construction, so "empty on random weights" doesn't apply to it. Its test is the decomposition error instead, plus planted recovery.

**D15: the J-lens views wait for a measured method.** The charter places the J-lens slice view in Phase 2, and this spec keeps its definition, its cost probe and its tests. But computing the lens means averaging Jacobians, and the obvious exact method is too slow on a CPU. Rough arithmetic puts it in the hundreds of trillions of operations for a single 256-token sequence, for one layer at one checkpoint, because it needs a separate pass for every direction at every position. A cheaper estimator exists (below), but whether it's accurate enough is an empirical question. G2.5 measures that. The J-lens views ship in a later Phase 2 delta, once a D-entry rules the method and the fit-sample size on G2.5's evidence.

## Views

All views cover the 8-sequence probe batch at each of the 28 trace checkpoints of run `b9258e47fef4`. The probe text is decoded with the run's tokenizer and shown alongside every panel.

| View | What it shows | Kind | Meter |
| --- | --- | --- | --- |
| Per-token loss | The loss on each probe token, as a heat map over the text | Readout of `logits` (D12) | Hash check |
| Attention patterns | Each head's pattern, for a chosen layer, head and sequence | Readout of `attn.hook_pattern` | Hash check |
| Residual stream | Norm of the residual stream per position and layer | Readout of the `hook_resid_*` taps | Hash check |
| Logit lens | Each layer's top predictions per position, read through `ln_final` and the tied unembedding | Explanation | KL from the model's own prediction, and top-1 agreement with it |
| Direct logit attribution | Each component's push on the logit of the actual next token, at a chosen position | Explanation | Decomposition error |
| Layer structure | Per-layer and per-head summaries across all 28 checkpoints: residual norms, lens KL, attention entropy | Readouts and logit lens | Each underlying view's meter |
| J-lens | Each block output's top predictions per position, read through D17's summed Jacobian, `ln_final` and the unembedding | Explanation | KL from the model's own prediction and top-1 agreement; divergence from the logit lens |

**Per-token loss** covers positions 0 to 254 of each probe window. Position 255 has no next token inside the window, so it's shown as blank, not guessed.

**The logit lens** applies `ln_final` and the unembedding to the residual stream at each layer: the embedding output, then each block's `hook_resid_post`. Its meter is computed per layer, over every probe position:

- **KL divergence** from the model's final distribution, taken from the traced logits.
- **Top-1 agreement** with the model's final prediction.

At the last layer, the lens is the model's own computation, so its KL should be essentially zero. That's a built-in check of the decoder. The tolerance comes from G2.3's measurement, not a guess.

**Direct logit attribution** splits the final residual stream into its parts: the embeddings, then each block's attention output and MLP output. Each part is projected onto the unembedding direction of the actual next token, holding `ln_final`'s scale fixed at its measured value for that position. The parts sum to the final residual stream, exactly in exact arithmetic and to float32 rounding in practice. The main approximation is the fixed layer-norm scale, so the meter is the gap between the summed attributions and the true logit, which captures both.

**The J-lens**, as D17 rules: the summed Jacobian, exactly as `anthropics/jacobian-lens` computes it. For each output dimension, one backward pass places a one-hot cotangent at every valid position (16 to T - 2) of the last block's output; the gradient at each valid source position of block l's output is averaged over those positions, then over the fit sample's sequences. The lens reads `ln_final` and the unembedding applied to J_l times block l's output. Layer 3's J is the identity. The fit sample:

- **The same positions at every checkpoint.** Redrawing it per checkpoint would make the lens appear to change over training partly because the sample changed. With fixed positions, every change the scrubber shows is the model's.
- **Drawn from the training stream with seed 1,** distinct from data seed 0 and probe seed 0, so it never touches the probe or the evaluation slice. Every position in the drawn windows counts.
- **Sized by D17:** 128 windows of 256 tokens, 32,768 positions. G2.5's convergence measurements put its error at most about 1.6%.

**The J-lens view** reads blocks 0 to 3's outputs; block 3's J is the identity, so its last reading is the logit lens's. Its meter is the logit lens's kind: KL from the model's own prediction and top-1 agreement. Beside it, per reading, the view shows its Jensen-Shannon divergence and top-1 agreement with the logit lens at the same residual stream: the measurement D17 promised, of how far the two lenses' readings actually differ. Both lens tables also give the actual next token's rank and probability, so it stays visible outside the top 5.

## The scrubber

One page:

- **A timeline** over the 28 checkpoints, drawn on G1.4's validation-loss curve, with the current checkpoint marked.
- **The probe text**, with per-token loss as color.
- **The panels above,** all following the timeline.

Clicking a token selects its position for the logit lens and the attribution view. Every panel shows its meter, and a view without one can't render: the export refuses to write a derived view without its meter.

**Progress.** Building the derived views over 28 checkpoints takes minutes, so the build prints a progress line every 30 seconds. The page shows loading progress per tensor.

## Gates

Each runs on the Mac and writes one schema-2 report under `reports/phase2/`.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G2.1 Logits in the trace | Records D12's `logits` tap at every trace step of run `b9258e47fef4`; recomputes a seeded sample of 20 logits events | 28 logits events in the store and 0 other events written; all 20 recompute bit for bit; logits shards within 1% of 939,524,096 bytes. Counting totals, not new events, lets a rerun after success pass again | Diagnose the engine |
| G2.2 Readout fidelity | Builds the readout views for all 28 checkpoints, twice; checks a seeded sample of displayed values against the trace | Every sampled value equals the traced tensor bit for bit; both builds are byte-identical | Diagnose the decoder |
| G2.3 Logit lens | Decoder-level planted recovery; the lens on a freshly initialized model, over the 262,144-token evaluation slice; the meter on the reference run's probe | Planted token recovered at 99% or more of positions; on random weights, next-token accuracy at most twice chance at every layer; last-layer KL at most 1e-5 nats at every trace step | Diagnose before shipping the view |
| G2.4 Direct logit attribution | Decoder-level planted recovery with two equally strong parts, one along each target's direction and one perpendicular to it; the decomposition error on the reference run | The aligned part credited at 99% or more of positions, the perpendicular part under 0.1% of the aligned part's credit at 99% or more; decomposition error at most 1e-3 at every trace step | Diagnose before shipping the view |
| G2.5 J-lens cost probe | At one checkpoint and one layer: the exact same-position Jacobian on causal prefixes (the model is causal, so on each sequence's first 32 positions the exact computation is exact for those positions, at 268 MB rather than 17 GB per sequence), against the estimator below at fit sizes of about 2,000, 8,000 and 32,000 positions; wall time for each. Also, read from its source: how the reference implementation in `anthropics/jacobian-lens` computes J, and whether it can run on our model's Hugging Face export | Always passes: records the error at each size, the change between consecutive sizes (4x apart; the measurement keeps its name, `fit_change_per_doubling`), the times, a projected cost for 4 layers and 28 checkpoints, and the cross-check's feasibility | Not applicable; the results set D15's method and size |
| G2.6 Viewer | Builds every derived view twice; checks the page references nothing outside the repo | Byte-identical builds; no external URL in the viewer's files | Diagnose the export |
| G2.7 J-lens inputs | At step 12,208, layer 0: exports the checkpoint to Hugging Face format and compares every tensor; computes the summed J on 8 fit windows and recomputes it | Export equals the parameters; J recomputes bit for bit | Diagnose the export or the computation |
| G2.8 J-lens cross-check | In the torch lane: the reference implementation's J on the same 8 windows, from the same token ids; the export's logits on the probe | Same tokens; J within 1e-4 relative; logits within 1e-4 of the trace | Diagnose the definition before G2.9 runs |
| G2.9 J-lens fit | The summed J for layers 0 to 2 at all 28 trace steps on the full fit sample, about 4 hours; refuses to start until G2.8 passes | All 84 written; one recomputes bit for bit | Diagnose the computation |
| G2.10 J-lens view | Decoder-level planted transport; the identity J against the logit lens; the J-lens on a freshly initialized model, over the evaluation slice; the meter and the comparison with the logit lens on the reference run's probe | Planted tokens recovered at 99% or more; the identity J equals the logit lens bit for bit; on random weights, next-token accuracy at most twice chance; last-reading KL at most 1e-5; the comparison recorded, not judged | Diagnose before shipping the view |

**Why G2.3's random-weights check uses the evaluation slice.** At a 4,096-word vocabulary, chance is 1 in 4,096. Over the probe's 2,048 positions, that's about half a correct guess, so a threshold of twice chance would rest on counts of 0 to 3. Over the evaluation slice, chance is about 64 correct guesses and twice chance about 128, a real measurement. The check runs a freshly initialized model rather than reading traced data, so it can use any tokens. The displayed meters stay on the probe, since they describe what's on screen.

**The estimator G2.5 tests.** Perturb the residual stream at every position at once, along one basis direction, flipping the sign independently and randomly per position. Then correlate each position's output change with its own sign. The other positions' contributions carry random signs and average away, which leaves an estimate of the same-position Jacobian at one forward-mode pass per direction, rather than one per direction per position.

## Carried over from Phase 1

- **G1.1 downloads the revision it records.** It fetches from the dataset's current `main` and records the revision it got, so a rerun after an upstream update would get different data, a different corpus hash and a different `run_id`. It should download from the recorded revision instead, so the raw files can be fetched again exactly as long as Hugging Face keeps that revision.
- **G1.2 runs offline or declares its network use.** Its export contacted the Hugging Face Hub, which the gates PR said it wouldn't. With Hugging Face's offline mode set, the export either works offline, or fails and names what it needs. Either way, the gate then states its network use.

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 0 | Phase 1 carry-overs | `gates/phase1/g11_tokenizer.py`, `gates/phase1/g12_parity.py` | None |
| 1 | Logits in the trace (D12) | `trace/schema.py` (new site), `trace/registry.py`, `trace/engine.py` | None |
| 2 | Readout views | `views/readouts.py` | 1 |
| 3 | Logit lens and its meter | `views/lens.py` | 1, 2 |
| 4 | Direct logit attribution | `views/attribution.py` | 1, 2, 3 |
| 5 | Caches and export (D11) | `views/cache.py`, `views/export.py` | None |
| 6 | Viewer | `views/shards.py`, `viewer/index.html`, `viewer/app.js` | 1, 2, 9 |
| 7 | J-lens probe | `views/jlens_probe.py` | None |
| 8 | Phase 2 gates | `gates/phase2/` | All |
| 9 | View builder | `views/build.py`, and `TraceReader.event` in `views/readouts.py` | 2, 3, 4, 5 |
| 10 | Summed Jacobian (D17) | `views/jlens.py` | None |
| 11 | Hugging Face export | `model/hf_export.py` | None |
| 12 | J-lens gates and their report rules | `gates/phase2/g27_jlens_prep.py` to `g29_jlens_fit.py`, `summarize.py`, `reports.py` | 10, 11 |
| 13 | J-lens view | `views/jlens_view.py` | 10 |
| 14 | Builder: the J-lens view and the next token's rank | `views/build.py` | 9, 13 |
| 15 | Export tokenizer | `model/hf_export.py` | 11 |
| 16 | Viewer: the J-lens panel, the next token's rank, the log-scale lens KL | `viewer/app.js`, `viewer/index.html` | 14 |
| 17 | G2.10, the J-lens rebuild in G2.6, the summary, their report rules | `gates/phase2/g210_jlens_view.py`, `g26_viewer.py`, `summarize.py`, `reports.py` | 13, 14 |

Packages 0, 1, 5 and 7 need nothing and can start at once. Package 2 needs 1; 3 needs 2; 4 needs 2 and 3; the builder, 9, needs 2 to 5; the viewer needs 9. The J-lens views themselves, and the torch-lane cross-check if G2.5 finds it feasible, come in a later Phase 2 delta after D15's method is ruled.

## Tests

Written with the stubs, after D11 to D15 are committed. Proposed names:

- **0:** `test_g11_downloads_by_revision`, where the download address for a recorded revision names that revision rather than `main`
- **1:** `test_logits_tap_is_registered`, `test_logits_event_matches_model_output`, `test_retrace_adds_only_logits`
- **2:** `test_readout_equals_traced_tensor`, `test_per_token_loss_leaves_last_position_blank`, `test_readout_build_is_deterministic`
- **3:** `test_lens_recovers_planted_token`, `test_lens_is_empty_on_random_weights`, `test_last_layer_lens_matches_logits`, `test_lens_meter_is_reported_per_layer`
- **4:** `test_attribution_parts_sum_to_residual`, `test_attribution_recovers_planted_component`, `test_attribution_error_is_reported`, `test_attribution_refuses_a_different_probe`
- **5:** `test_cache_key_covers_inputs_and_code`, `test_deleting_cache_then_rebuilding_is_identical`, `test_derived_view_without_meter_is_refused`
- **6:** `test_viewer_has_no_external_urls`, `test_viewer_hash_check_matches_tensor_sha256` (the page's hashing, cross-checked against `tensor_sha256` on fixed bytes)
- **7:** `test_estimator_matches_exact_jacobian_on_linear_model`, where the exact Jacobian is known in closed form
- **8:** `test_logits_bytes_formula`, `test_random_weights_bound`, `test_exact_reference_uses_causal_prefixes`, `test_phase2_status_rules`
- **9:** `test_build_writes_every_view_with_a_meter`, `test_build_payloads_match_the_views`, `test_build_is_deterministic`, `test_build_reports_progress`

## Out of scope

Interventions of any kind (Phase 3), model-level planted structure (Phase 4), dictionaries and J-space decomposition (Phase 5), comparing two runs side by side, and other probe batches.

## Settled while drafting

- **The J-lens fit sample:** fixed positions across all checkpoints, from the training stream with seed 1, sized by G2.5's convergence measurements.
- **Cross-checking the reference implementation:** yes, in the J-lens delta, as a torch-lane gate, if G2.5 finds it can run on our model. A lens can be wrongly defined and still agree with the model, which its meter can't catch but an independent implementation can.
- **The probe stays at 8 sequences.** It's ample for looking, and the displayed meters describe it. Statistical gate checks use the evaluation slice. Larger probes can be added later without disturbing the existing trace, since probe batches are kept apart by hash.
