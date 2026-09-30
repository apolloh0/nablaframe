# Phase 1 spec: the reference model, its trainer, and the trace engine

Status: draft for review. D7 and D8 must be ruled before stubs and tests are written. Charter: [../charter.md](../charter.md). Decisions: [../decisions.md](../decisions.md).

## Purpose

Phase 1 turns Phase 0's schema and oracle into a working instrument. At its end, one real model has been trained on TinyStories. Every tap at every trace checkpoint is recorded in the store, and any recorded tensor can be recomputed bit for bit from its manifest and checkpoint.

Human gate H1 asks one question: can Phase 2 build its views on this engine? The evidence is gates G1.1 to G1.6.

## Rulings needed first

**D7: who owns the model, the training loop and the checkpoints.**

| | Own them (recommended) | Keep Levanter's trainer |
| --- | --- | --- |
| Model | GPT-2 in Haliax and Equinox, taps built in | Our GPT-2 registered as a Levanter model |
| Training loop | A short optax loop we own | Levanter's `train_lm` |
| Batch order | A pure function of (seed, step) | Levanter's data loader |
| Checkpoints | Our format, written atomically | Levanter's |
| Levanter's role | Parity reference (G1.2) and Hugging Face export | Everything except the model |
| Cost | Determinism must be proven again (G1.3) | Couples our model to Levanter's internal interfaces, which change between dev builds, and keeps patching for anything the trainer doesn't expose |

The recommendation follows the charter: the model you read should be code you own. It also makes replay exact by construction, since batch *k* depends only on the seed and *k*. G1.2 keeps continuity with Phase 0: on identical weights, our model must reproduce Levanter's logits. The Phase 0 gate scripts are unaffected and keep using Levanter's trainer.

**D8: the vocabulary.** Proposed rule: use 4,096 if G1.1 measures an unknown-word rate of at most 1.0% on the TinyStories validation split at 4,096; otherwise use 8,192.

- For 4,096: 24% faster at size S (G0.3 measured 15,255 against 12,260 tokens/s), and tied embeddings take 25% of the parameters rather than 39%.
- For 8,192: the J-lens only names single-token concepts, so words mapped to `<unk>` are invisible to it. Above 1%, that loss starts to matter.

The rule is fixed now and applied by measurement in G1.1, so the choice can't drift toward whichever run looks better.

## The reference model

Levanter's GPT-2 architecture, pinned by G0.3's recorded parameter counts, which match this formula exactly at all four measured sizes: tied input and output embeddings, learned position embeddings, biases on every linear layer, pre-norm blocks with two layer norms each, and a final layer norm.

| Size | Layers | d_model | Heads | d_head | d_mlp | Seq | Params at 4,096 | Params at 8,192 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | 4 | 256 | 4 | 64 | 1,024 | 256 | 4,273,664 | 5,322,240 |
| M | 4 | 512 | 8 | 64 | 2,048 | 256 | 14,838,784 | 16,935,936 |

The GELU variant, layer-norm epsilon and attention scaling follow Levanter's GPT-2 at the pinned version. The implementing thread reads them from Levanter's source, and G1.2 verifies them.

The model lives in `src/nablaframe/model/gpt2.py`, float32 throughout, with named axes. It satisfies the oracle's `InstrumentedModel` protocol, with `params` being the Equinox module itself:

- `fast(params, tokens)`: logits only, `jit`-compiled.
- `instrumented(params, tokens, taps)`: logits plus the requested taps as extra outputs of a separately compiled function, never through host callbacks. Tap names come from `registry.tap_names(n_layers)`; any other name raises `KeyError`.
- `load_hf_gpt2(path)`: loads a Hugging Face GPT-2 export into our module, for G1.2.

## Data

**Tokenizer** (`src/nablaframe/data/tokenizer.py`). A word-level tokenizer built from the TinyStories train split:

- Text is NFKC-normalized and lowercased.
- Words match `[a-z]+(?:'[a-z]+)*`. Each digit and each other non-space character is its own token.
- Every story ends with `<eos>`.
- The vocabulary is `<unk>` (id 0), `<eos>` (id 1), then the most frequent types in the train split. Ties are broken alphabetically, so two builds are always identical.
- It's saved as canonical JSON, and its SHA-256 goes into `RunManifest.tokenizer_sha256`.

**Corpus** (`src/nablaframe/data/corpus.py`). Each split becomes one little-endian `uint16` token stream under `data/tinystories/`, which is not committed. The SHA-256 of the pair goes into `RunManifest.corpus_sha256`, and the exact dataset files and their hashes are recorded by G1.1.

**Batch order.** Batch *k* for seed *s* is `batch_size` windows of `seq + 1` tokens, with start offsets from `numpy.random.default_rng([s, k])`. It depends on nothing else, so a resumed run sees exactly the batches an uninterrupted one would. Numpy doesn't promise identical random streams across versions, but the lockfile pins numpy, and a changed stream would show up as a changed batch hash.

**Probe set.** Traces are recorded on a fixed probe batch: 8 windows of 256 tokens from the validation split, drawn with probe seed 0. Its `batch_hash` is the SHA-256 of its `int32` token bytes.

## Trainer and checkpoints

The trainer lives in `src/nablaframe/train/`. Proposed defaults, which a D-entry can change:

- AdamW with β = (0.9, 0.95), weight decay 0.1 on matrices only, and gradient clipping at 1.0.
- Learning rate 1e-3, 500 warmup steps, then cosine decay to 10%.
- Batch 32 × 256 tokens, float32.

**Checkpoints.** A checkpoint named `step-n` holds the state after exactly *n* updates, unlike Levanter's off-by-one. There are two kinds:

- *Resume* checkpoints hold parameters, optimizer state and step, every 500 steps.
- *Trace* checkpoints hold parameters only, at steps 0, 1, 2, 4, 8 and so on up to the run's length, at every multiple of 500, and at the final step.

Each checkpoint is one safetensors file keyed by pytree path, plus a JSON sidecar with the step, `run_id` and a state hash. Files are written under a temporary name and renamed into place, so a crash never leaves a partial checkpoint under its real name.

**Heartbeat.** The trainer prints one line to the terminal about every 30 seconds, never inside a timed measurement:

```
step 1,200/6,104 | loss 2.431 | 12,100 tokens/s | about 41 min left
```

## Trace engine

`src/nablaframe/trace/engine.py` records traces into the Phase 0 store:

- **One manifest per run.** `open_run(root, manifest)` writes `manifest.json` once. Reopening with an identical manifest is allowed; a different one raises.
- **The oracle runs first.** `record(step, params, probe, taps)` first runs `check_mode_equality` on the probe batch under the D6 policy. If it fails, it raises before any event is written. Then it appends one event per tap: kind `forward`, mode `instrumented`, the tap name, the step and the probe's batch hash.
- **Resumable.** Taps already recorded for this step and batch with identical content are skipped, so re-running a trace after a crash adds nothing. It never supersedes an event on its own.
- **Recompute on demand.** `rematerialize(event)` loads the trace checkpoint for `event.step`, runs the instrumented forward pass on the probe batch named by `event.batch_hash`, and returns the tensor. `verify(event)` checks that tensor's hash against `event.sha256`.

**Storage.** One trace checkpoint stores `4 × P × seq × (d × (7L + 3) + L × H × seq)` bytes, where P is the number of probe sequences. At size S with P = 8, that's about 99 MB. The reference run has 27 trace checkpoints (14 log-spaced, 12 multiples of 500, and the final step), so about 2.7 GB.

## Carried over from Phase 0

- **Atomic shard writes.** The store writes each shard to `<id>.safetensors.tmp` and renames it into place. An unreadable shard under its final name can then only mean corruption, and raises `IntegrityError`.
- **One writer per store.** `TraceStore(root)` takes an exclusive lock on `root/.lock`, and a second writer gets `StoreError` at once. `TraceStore(root, mode="r")` opens read-only, needs no lock, and refuses appends.
- **Commit hashes in reports.** Report schema version 2 adds `git_commit` (40 hex characters) and `git_dirty`. Version 1 reports stay valid, so Phase 0's record doesn't change. Phase 1 gates write version 2 only. `GATE_IDS` gains G1.1 to G1.6, with their required measurements and status rules.
- **Two docstring clarifications from PR #4.** In `compare_logits`: fast and instrumented logits of different dtypes give a failed result, not an error. In `reports.py`: a null hash never counts as equal.
- **Doc fixes.** Restore the charter link in `phase-0.md`'s status line. Mark Phase 0 done in the charter's phase table and link this spec. Add `--extra` to every lane command in `CLAUDE.md`. Add `data/` to `.gitignore`.

## Gates

Run on the Mac in this order. Each writes one version 2 report under `reports/phase1/`.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G1.1 Tokenizer and corpus | Builds the tokenizer twice at each of 4,096 and 8,192; unknown-word rate on validation for each; token counts; dataset file hashes | Both builds identical at each size; D8's rule selects a vocabulary | Fix the pipeline |
| G1.2 Parity | A seeded Levanter GPT-2 at size S and the D8 vocabulary, exported to Hugging Face format and loaded into ours; logits on the probe batch | Parameter count equals the table exactly; max abs logit diff ≤ 1e-4; identical top-1 wherever the top-2 margin exceeds 1e-3 | Diagnose before training anything |
| G1.3 Trainer determinism | Size S, 600 steps: two fresh runs, and a run killed after its step-300 checkpoint and resumed | All three final state hashes identical. Same single-threaded retry and conditional pass as G0.2 | Rule in a D-entry before G1.4 |
| G1.4 Reference run | Size S on TinyStories, about 50M tokens (6,104 steps), heartbeat on | No NaN or Inf; every scheduled checkpoint written; final validation loss below an add-one-smoothed bigram model fit on the train split | Diagnose; nothing downstream runs |
| G1.5 Oracle on the real model | Fast against instrumented logits on the probe batch, all taps, at steps 0, 3,000 and 6,104 of G1.4 | Bitwise equal everywhere (D6), and a tap adding 1e-3 at `blocks.0.hook_resid_post` fails | A D-entry on the policy, per D6 |
| G1.6 Trace and recompute | Traces every tap at every trace checkpoint of G1.4, recomputes a seeded sample of 50 events, then traces again | All 50 recomputed hashes match; the second trace writes zero events; store size within 10% of the formula | Diagnose the engine |

**Rough times,** not measurements:

- G1.1: minutes, plus the dataset download.
- G1.2: minutes.
- G1.3: about 20 minutes, more if it needs the single-threaded retry.
- G1.4: about 1 to 1.2 hours at the measured size-S speed.
- G1.5 and G1.6: minutes; they only run forward passes.

## Work packages

| # | Package | Files | Depends on | Suggested model |
| --- | --- | --- | --- | --- |
| 1 | Phase 0 carry-overs | `trace/store.py`, `reports.py`, `oracle/mode_equality.py` | None | Sonnet, high |
| 2 | Tokenizer and corpus | `data/` | None | Sonnet, high |
| 3 | Reference model | `model/` | None | Opus, high |
| 4 | Trainer and checkpoints | `train/` | 2, 3 | Opus, high |
| 5 | Trace engine | `trace/engine.py` | 1, 3, 4 | Opus, high |
| 6 | Phase 1 gates | `gates/phase1/` | All | Opus, high |

Packages 1, 2 and 3 share no files and can run in parallel. The doc fixes ship with the stubs, not as a thread.

## Tests

Written with the stubs, after D7 and D8 are ruled. Proposed names:

- **1:** `test_shard_write_is_atomic_under_crash`, `test_second_writer_is_refused`, `test_readonly_store_refuses_appends`, `test_report_v2_requires_git_commit`, `test_phase0_reports_still_validate`
- **2:** `test_tokenizer_build_is_deterministic`, `test_vocabulary_ties_break_alphabetically`, `test_unknown_words_map_to_unk`, `test_batch_depends_only_on_seed_and_step`, `test_splits_never_share_tokens`
- **3:** `test_param_count_matches_levanter`, `test_tap_names_match_registry`, `test_tap_shapes_match_tapspecs`, `test_fast_and_instrumented_logits_bitwise`, `test_mutating_tap_breaks_oracle_on_real_model`
- **4:** `test_train_step_is_deterministic`, `test_checkpoint_roundtrip_is_bitwise`, `test_resume_matches_uninterrupted`, `test_checkpoint_writes_are_atomic`, `test_step_n_holds_n_updates`
- **5:** `test_oracle_failure_writes_no_events`, `test_retrace_writes_nothing`, `test_rematerialized_tensor_matches_event`, `test_manifest_is_written_once`, `test_probe_batch_hash_is_stable`

## Out of scope

Views and the scrubber (Phase 2), interventions (Phase 3), the ground-truth zoo (Phase 4), dictionaries and lenses (Phase 5), size-M training runs, the Metal backend, and the PyTorch lane.

## Open questions

- [ ] Which TinyStories release: the original, or the GPT-4-only version 2? G1.1 records whichever is used; this question picks it.
- [ ] Lowercasing costs capitalization in decoded text. That's acceptable for an instrument, but say so if you'd rather keep case.
- [ ] Are the trainer defaults and the 50M-token reference run right for Phase 1? Only G1.4's outcome tests them.
