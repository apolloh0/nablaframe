# Phase 3 spec: manipulation

Status: ready for stubs and tests once D22 to D27 are committed. Charter: [../charter.md](../charter.md). Decisions: [../decisions.md](../decisions.md).

## Purpose

Phase 2 made the reference run visible. Phase 3 makes it answerable to experiment: ablate a component and see what breaks, patch an activation from one input into another, steer along a direction, or swap a component for its past self from earlier in training. Every manipulation is deterministic, cached and recomputable, and every tool is held to exactness identities before its effects are trusted.

Its first experiment comes from D21: does the J-lens read what a residual stream tells *later* positions? Steering makes that testable.

Human gate H3 asks: can Phase 4 build its ground-truth zoo on these tools, trusting that an effect they report is an effect of the edit and nothing else?

## Rulings

Proposed with this spec; committing them is the ruling.

**D22: an intervention is a declarative, hashable spec.** It is a list of edits, each with:

- a tap
- the positions it applies to
- an operation
- the operation's arguments

The operations:

- **`zero`:** replace with zeros.
- **`mean`:** replace with the reference batch's mean at that tap, taken per position.
- **`resample`:** take the value from a seeded draw of the reference batch's windows, at the same positions.
- **`patch`:** take the value at the same tap and positions from another input.
- **`add`:** steer by α times a vector.
- **`swap`:** see D26.

Positions are written as `"all"`, an explicit list, or a half-open range `{"range": [a, b]}`. Vectors and batches are referenced by content hash, never by value or name alone. The spec's identity is the SHA-256 of its canonical JSON, so two people describing the same edit get the same hash.

**D23: edits apply only at existing tap sites.** The trace is recomputed with every canonical tap requested at once, and adding a tap would change that compiled program. That would put at risk the bit-for-bit guarantee on all 1,008 recorded events, the property H1 ruled on. So Phase 3 edits at the taps that exist:

- each block's residual stream, before, between and after its two sublayers
- the per-head attention output, `attn.hook_z`
- the attention and MLP outputs
- the layer norms' outputs
- the embeddings

Neuron-level edits inside the MLP wait for a deliberate ruling on a new tap and a fresh trace.

**The reference batch,** used by `mean` and `resample`, is 64 windows of 256 tokens from the training stream drawn with seed 2. That keeps it disjoint from the probe (validation split), the evaluation slice (validation split), and D17's fit sample (training stream, seed 1). Mean-ablating with the probe's own mean would leak the probe into its own edit. The mean is taken per position because positional embeddings make positions genuinely different.

**D24: intervention results are derived caches, not trace.** An intervened pass is a computation on a checkpoint and a batch. Its outputs are cached under D11 and D18, keyed by:

- the checkpoint's state hash
- the batch hash
- the spec hash
- the computing module's code identity

Those outputs are the logits plus any requested taps. The canonical trace stays the unintervened reference run. No schema change is needed, and the `intervention` event kind stays unused for now.

**D25: exactness identities are the manipulation tools' discrimination tests.** This is D14's counterpart for tools that change the model rather than read it. Before any effect is reported, each tool must reproduce, bit for bit:

- **null edits:** an edit that returns its input reproduces the unintervened pass
- **α = 0 steering:** reproduces the unintervened pass
- **a self-swap:** a component swapped with itself reproduces the unintervened pass
- **a full patch:** patching every position of the last block's output from a source input reproduces the source's logits
- **a full swap:** swapping every component from checkpoint B reproduces B's own pass

These test the plumbing exactly. Tests of whether an effect means what it seems to need planted structure in a model, and arrive with Phase 4's zoo.

**D26: "swap" substitutes a component from another checkpoint of the same run.** The component can be:

- one block's attention
- its MLP
- the whole block
- the embeddings

It's time travel along the scrubber's timeline. Swapping block 1's attention from step 12,208 back to step 1,024, for example, asks which of the model's capabilities that component carries.

**D27: the viewer stays read-only (D13).** Interventions run in Python, through a small API and a command line, never in the browser. Their results reach the viewer as derived payloads. The viewer gains a compare mode: the unintervened and intervened passes side by side, with the effect's meter.

## Effects and their meters

An intervention's effect is measured against the unintervened pass on the same checkpoint and batch:

- **the change in the actual next token's logit and loss, per position**
- **the KL divergence of the intervened prediction from the unintervened one, per position**
- **the change in any requested tap**

The meter beside every effect is its exactness record. The payload names the D25 identity checks that passed for that operation, so a reader knows the plumbing was verified before looking at the effect.

## The first experiment: what the J-lens reads

D21 left two questions. Both get measured here, recorded and not judged.

**1. Does the J-lens read effects on later positions?** At block l's output, at position p, steer along one of two directions for a chosen token t:

- **the J-lens direction:** the residual direction that J_l carries most strongly onto t's unembedding, Jₗᵀ w_t
- **the logit-lens direction:** t's unembedding itself, w_t

Both directions are unit vectors, scaled by a quarter, a half and once the layer's mean residual-stream norm at that checkpoint, so steering stays in the range the model actually sees. Positions p are a seeded sample of 64 from the fit range, 16 to T − 2; tokens t are a seeded draw from the vocabulary, excluding `<unk>` and `<eos>`. Measure the change in t's logit at position p, and its mean change over the next 16 positions. If the J-lens reads what a residual tells later positions, as its summed definition suggests, its direction should move later positions more and the same position less than the logit lens's does.

**2. How much do the excluded positions weigh?** Both lenses' meters again, restricted to positions 16 to T − 2, the range J's fit uses. No intervention is needed: it's computed from the existing traces.

## The J-lens layout fix

Carried over from apolloh0's H2 session. At ordinary window widths, the J-lens table scrolls sideways, and its "vs logit lens" columns, the comparison D21 records, sit off-screen. Phase 3 moves them up front, beside each reading's name.

## Gates

Each runs on the Mac and writes one schema-2 report under `reports/phase3/`.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G3.1 Exactness identities | Every D25 identity, at size S, on the reference run's final checkpoint and probe | All bit for bit | Diagnose the tool before any effect is trusted |
| G3.2 Determinism and caching | A seeded sample of interventions, computed twice and after deleting the cache | Byte-identical results; cache keys change with spec, inputs and code, and with nothing else | Diagnose the cache |
| G3.3 The J-lens experiment | Question 1 above, at every block before the last, over a seeded sample of positions and tokens | Recorded, not judged | Not applicable |
| G3.4 The fit-range meters | Question 2 above, at every trace step | Recorded, not judged | Not applicable |
| G3.5 Viewer compare mode | Builds the compare payloads twice; checks the page references nothing outside the repo | Byte-identical builds; no external URL | Diagnose the export |

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 1 | Intervention specs and their hashing (D22) | `interventions/spec.py` | None |
| 2 | Applying edits at tap sites, including swap (D23, D26) | `interventions/apply.py` | 1 |
| 3 | Effects, their meters and caches (D24) | `interventions/effects.py` | 2 |
| 4 | The J-lens experiment | `interventions/jlens_experiment.py` | 3 |
| 5 | Viewer: compare mode and the J-lens layout fix (D27) | `views/build.py`, `viewer/` | 3 |
| 6 | Phase 3 gates and their report rules | `gates/phase3/`, `reports.py` | All |

## Out of scope

- **Neuron-level edits,** which need a new tap (D23).
- **Edits during training,** such as altering gradients or updates.
- **Planted structure at model level** (Phase 4).
- **Dictionaries and J-space** (Phase 5).
- **Any computation in the browser.**

## Tests

Written with the stubs. Proposed names:

- **1:** `test_spec_hash_is_canonical`, `test_spec_refers_to_vectors_and_batches_by_hash`, `test_positions_language`
- **2:** `test_null_edit_is_exact`, `test_zero_alpha_steering_is_exact`, `test_self_swap_is_exact`, `test_full_patch_reproduces_the_source`, `test_full_swap_reproduces_the_other_checkpoint`, `test_edits_only_at_existing_taps`
- **3:** `test_effects_against_the_unintervened_pass`, `test_effect_meter_names_the_identities_checked`, `test_intervention_cache_key_covers_spec_inputs_and_code`
- **4:** `test_jlens_direction_is_the_pullback_of_the_unembedding`, `test_steering_scales_follow_the_residual_norm`
- **5:** `test_compare_payload_has_a_meter`, `test_jlens_comparison_columns_come_first`

## Settled while drafting

- **The reference batch:** 64 training windows drawn with seed 2, disjoint from the probe, the evaluation slice and the J's fit sample; means taken per position.
- **Positions in a spec:** `"all"`, an explicit list, or a half-open range.
- **Steering scales:** unit directions at a quarter, a half and once the layer's mean residual-stream norm at that checkpoint.
