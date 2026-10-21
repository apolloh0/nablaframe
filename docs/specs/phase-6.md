# Phase 6 spec: gestures on dictionary features

Status: proposed with D47 to D52. Charter: [../charter.md](../charter.md).

## Purpose

Phase 5 built dictionaries that hold up against the zoo and its twins (D46). They split an activation into latents, but a latent isn't an explanation yet. At H5, three of the dictionaries' results showed why:

- **Contexts and largest effects misname latents:** they missed the intermediate entirely.
- **One latent carries only part of a variable:** 9% to 23% of what the whole code moves.
- **Edges across blocks look as dense as noise:** the edge meter can't see a circuit routed by attention.

Phase 6 builds the four gestures that turn latents into explanations, each with a faithfulness number:

- **name-this-direction:** what does this latent stand for, and does moving it move the model as the name says?
- **assert-this-edge:** does this feature drive that one, and is the edge computed or only carried?
- **open-this-up:** which upstream features make this one?
- **generalize-this:** across held-out windows, dictionary seeds and the training run, where does a name or an edge hold?

Three feasibility rounds on the reference backend shaped the definitions and the bars (`docs/notebook.md`, findings 9 to 22):

- **Sets:** a name needs a set of about four latents.
- **Effects:** a value's effect has to be read from that set, not from its most selective latent.
- **Parts:** parts must be ranked by absolute attribution.
- **Edges:** the residual stream carries anything put into it, so an edge between two sites at one position has to be checked for a carry.
- **Checkpoints:** a dictionary trained at the end of a run sees a variable only in its final form. Two-hop learned its task twice, and only the replayable run shows it.

Human gate H6 asks two questions:

- Does every gesture recover the zoo's planted structure, come up empty on the random-weights twin, and render with its meter?
- Does the two-hop intermediate's name show its set, the set's effect with `b` at the bottom, and its envelope across training, where the name appears at the transition the probe finds?

## Rulings

Proposed with this spec; committing them is the ruling.

**D47: gestures run in Python and are recorded for the viewer.** apolloh0 chose this on 2026-10-19. The viewer stays a pure decoder (D13, D27).

- **A request:** a gesture is asked for by a declarative request, a frozen dataclass with a JSON form, like an intervention spec (D22). Its identity is the SHA-256 of its canonical JSON.
- **A record:** running a request on the jax lane gives a record. It holds:
  - the request
  - every measurement, each beside its control
  - the inputs: each checkpoint's state hash, each dictionary's content hash, the windows' batch hashes
  - the code identity of the modules that computed it (D18)
- **A derived artifact:** a record is keyed by the request hash, its inputs and its code identity, and cached under D11 and D24. Deleting it loses nothing, and rebuilding it gives identical bytes.
- **Files:** `runs/phase6/gestures/<run>/gesture-<key>.json`, beside an `index.json` the viewer reads.
- **No verdicts:** a record holds measurements and the gesture's own labels (named or weak, computed or carried), never a pass or a fail. The gates judge the zoo results in their reports, and each gesture's badge carries them (D38).

**D48: `dsteer` joins the operations.** It moves chosen latents of a dictionary's code to chosen values. Neither existing operation can: `dpatch` needs a source batch, and `add` adds one vector to every sequence.

- **Arguments:** `dictionary` (a content hash), and three arrays, each by content hash:
  - `remove`: int32 rows `[sequence, position, latent]`
  - `add`: int32 rows `[sequence, position, latent]`
  - `amounts`: float32, one per `add` row
- **The edit:** at the dictionary's site, for each `remove` row it subtracts that latent's code there times its decoder row; for each `add` row it adds the amount times the decoder row. Both are scaled as `dpatch` scales them, so the target's error term is kept.
  - **Codes:** from the unintervened pass, as `dpatch` computes them (D45).
  - **A transcoder:** its codes come from its input tap, and the edit lands on its output tap.
- **Identities:** empty `remove` and `add`, removing only latents whose code is zero there, and adding only zero amounts each reproduce the unintervened pass bit for bit (D25).
- **Refusals, before anything runs:** a row outside the batch or the sequence, a position the edit doesn't select, a latent at or beyond the site's active width, a repeated row within `remove` or within `add`, `amounts` of the wrong length, or a non-finite amount.
- **Its values enter the compiled program as arguments** (PR #38).

**D49: name-this-direction names a latent only from declared descriptions, and holds the name with a set of four.**

A **description** maps a window and a position to a token, or to nothing:

- every run has `token(p − k)` for `k` from 0 to 3, and `token(p + 1)`
- a zoo task adds the variables it plants, each where it plants them

A **position class** is a set of positions a run declares. A description counts at a class only if it takes at least two values there, each at least 5 times. A constant names nothing.

Pointed at a latent, at a dictionary's site and a position class, the gesture takes six steps, all on held-out windows:

1. **Candidates:** the latent's F1 against each description's each value at the class's positions. The top 5 are recorded.
2. **Name:** the top candidate.
   - **Named:** its F1 is at least 0.5 (D41's floor).
   - **Weak:** below 0.5. Steps 3 to 6 still run, because a latent can belong to a working set without being selective alone. A value's largest-code latent at `query` after block 2 is in its set for 97.9% of values, but reaches the floor for only 39.6%.
3. **Set:** the 4 latents with the highest F1 for the name at the class, ties to the lower latent. The record says where the pointed latent ranks; it need not be in the set. Wherever a name is read again, on other windows, seeds or checkpoints, its F1 is the highest F1 among its set's latents.
4. **Coverage:** the share of the pointed latent's activation mass, over every held-out position, at which each description holds the name's value. The same for the set's summed codes.
5. **Effect:** the set's direct effect (D45's), weighted by its mean codes where the name holds. The record gives the value's rank from the top and from the bottom of the vocabulary. The pointed latent's own effect sits beside it, labelled as that latent's.
6. **Steering,** where the run declares a prediction for the description. At held-out instances where the name holds, `dsteer` removes the set's codes and adds another value `w`'s set at `w`'s mean codes. Those are the mean codes over held-out rows where the description is `w` at the class. Recorded:
   - **follows:** the share whose output, read where the run declares, becomes the prediction for `w`; and the share unchanged
   - **ablated:** the set removed and nothing added; the share whose output changes
   - **null:** 4 random latents active there are removed, and 4 random others added at the set's amounts; the share that follows, and the share unchanged. The removed latents exclude the value's top 8 by F1 and any latent active in at least 90% of the class's rows. The added ones exclude `w`'s top 8 and anything already active.

   Where the run declares no prediction, the record gives the KL between the steered and unsteered logits at the instance's position, and each pass's top tokens. It says no causal test applies.

The name's meter, shown first: follows beside the null's, then F1 and coverage. A name never comes from a latent's top contexts or largest effects alone (D46).

**D50: assert-this-edge swaps and ablates the upstream set, and labels an edge computed or carried.**

- **The request:** an upstream and a downstream description, each at a site and a class, and a relation the run declares. A relation pairs positions (the same one, an offset, or the task's map) and maps values (the same value, or the task's map `g`).
- **Instances:** held-out windows and position pairs where the upstream description is `v`, the downstream one is `g(v)`, and the downstream set is active.
- **Measured, each beside a null of random sets at the same amounts (D49's):**
  - **follows:** `dsteer` swaps the upstream set for `w`'s. The share where the downstream set for `g(w)` then outweighs the set for `g(v)`, by summed codes.
  - **kept:** the upstream set removed; the median share of the downstream set's mass left.
  - **output:** where the downstream description has a prediction, the share whose output becomes the prediction for `g(w)`.
  - **through the residual alone:** where both sites are at one position, follows again, with every block between them frozen at that position. Each block's attention and MLP outputs there are patched from the unintervened pass (`patch` with the target batch as its source).
  - **single latents:** follows and kept with only each name's best latent, beside the sets (D46).
- **The label:** an edge at one position is **carried** where through the residual alone follows at least half as often as follows, and **computed** otherwise. An edge across positions can only be computed. The residual stream passes on whatever is put into it. On the random-weights twin, the carry from block 1 to block 2 follows 51.5% of the time (null 20.8%), and 48.0% through the residual alone.
- **The edge's meter:** follows and kept, each beside its null, then the label.

**D51: open-this-up ranks upstream parts by absolute attribution and reads how many reproduce the target.**

- **The request:** a target, a latent or a name's set at a site and one position per instance, and an upstream dictionary at an earlier site.
- **Attribution:** integrated gradients, 5 steps at the midpoints from zero, of the target's pre-activation to every upstream code at every position up to the target's.
  - A latent's pre-activation is `(y / s_in − b_pre) · W_enc[:, l] + b_enc[l]`, before the top `k` are kept; a set's is the sum.
  - The upstream error term is held at its unintervened value, as the edge meter holds it (D42).
- **Parts** are (position, latent) pairs, ordered by absolute attribution, ties to the lower position and then the lower latent.
  - **Keeping `k` parts:** their codes stay and every other upstream code is zeroed, with the error kept.
  - **The relative error:** `|pre(kept) − pre(all)| / |pre(all) − pre(none)|`, for `k` of 1, 2, 4, 8, 16, 32, 64 and 128. An instance with `|pre(all) − pre(none)|` below 1e-6 is left out and counted.
  - **The reading:** the fewest parts with an error of at most 0.1, or "not within 128".
- **Recorded beside it:**
  - the same curve with parts in signed order, and in a random order of active parts
  - the top 8 parts, each with its position, latent, attribution and name (steps 1 and 2 of D49 at its position's class)
- **The meter:** the reading and its curve. Signed order overshoots: positive parts outrun the negative ones they no longer meet. In signed order, `b`'s set came within 0.1 for only 47.7% of instances by 128 parts; in absolute order, for all of them, at a median of 8.

**D52: generalize-this reads a name or an edge across held-out slices, dictionary seeds and the training run.**

- **Slices:** the held-out windows split into 8 disjoint slices, and each slice's F1 and follows. Recorded as their least and greatest.
- **Dictionary seeds 1 and 2** (D41's robustness seeds):
  - The name is derived again from its description and value at each seed's dictionary, never matched by latent id. Then its F1 and follows.
  - Beside them, the mean over the seed's set of each latent's highest decoder cosine with seed 0's set: 0.45 to 0.53 in feasibility. Names transfer; latent ids don't.
- **Checkpoints:** every checkpoint of the run, read through the final dictionaries. apolloh0 chose this on 2026-10-19; dictionaries trained per checkpoint wait (D44). At each:
  - the name's F1, with the final dictionary's set
  - follows, with that set at `w`'s mean codes at that checkpoint, and its null
  - beside them: a ridge probe for the description at the site and class from that checkpoint's activations (D41's probe), the residual patch where the task defines one (G4.2's, G4.3's), and the task's accuracy
- **The envelope:** the checkpoints where the name holds, with F1 and follows both at least 0.5.
- **The gap:** the checkpoints where the probe reads the variable at 0.9 or more but the name doesn't hold. It is shown, never hidden.
- **Before the envelope:** accuracy at those checkpoints is shown beside it. Two-hop answered 74.3% of windows at step 5,000, outside its envelope, while the probe read `b` at 0.366: the earlier route (notebook, H6).
- **An edge:** follows and kept at each slice, seed and checkpoint, and its label.

## What each run declares

Descriptions, classes, predictions and relations live with each run's task, so the gestures carry no task-specific code (D29).

| Run | Position classes | Planted names | Predictions | Relations |
| --- | --- | --- | --- | --- |
| Two-hop | `query` (25), `subjects` (0, 4, …, 20), `objects` (2, 6, …, 22), `every` (0 to 25) | `b` at `query` after block 2; `token(p + 1)`, the answer `c`, at `query` after block 3; `b` at `query` after block 1, recorded | `b` = `w` at `query`: the object of `w`'s fact, read at 25; `token(p + 1)` = `w` at `query`: `w`, read at 25 | hop: `b` after block 2 to `token(p + 1)` after block 3, same position, `g(v)` the object of `v`'s fact; carry, recorded: `b` after block 1 to `b` after block 2, same position; two back, recorded: `token(p)` at `objects` after block 0 to `token(p − 2)` at `subjects` after block 2, offset +2 |
| Induction | `mechanism` (32 to 62), `every` (0 to 62) | `token(p)` after block 0 at `every`; `token(p + 1)` after block 1 at `mechanism` | `token(p + 1)` = `w` at `p`: `w`, read at `p`; `token(p)` = `w` at a sole target `s`: `w`, read at `s + P − 1` | copy: `token(p)` at a sole target `s` after block 0 to `token(p + 1)` at `s + P − 1` after block 1 |
| Reference | `every` (0 to 255) | None | `token(p + 1)` = `w`: `w`, read at `p` | None |

Each instance's other value `w` is drawn with a fixed seed from the description's values at the class that have a set and that the window doesn't contain. Two-hop's `b` is the exception: its `w` is one of the window's distractor subjects, so that `w`'s fact gives a prediction.

A **sole target** `s` of an induction window with period `P` is a position whose token the induction head copies to `s + P − 1` from nowhere else: `1 ≤ s ≤ P` and `s + P − 1` is a mechanism position. With several earlier occurrences, attention splits between them, and changing one moves the prediction about 30% of the time (round 1).

## Discrimination tests on the zoo

Each test runs on the trained model and on its twin, with dictionaries from the Phase 5 cache: 512 instances per name and edge, 128 per open-this-up target. The bars come from feasibility round 3, with margin; its value follows each bar.

**Names (judged).** At two-hop's `b` after block 2 and `token(p + 1)` after block 3, and induction's `token(p)` after block 0 and `token(p + 1)` after block 1:

| Test | Bar | Two-hop `b`, `token(p + 1)` | Induction `token(p)`, `token(p + 1)` |
| --- | --- | --- | --- |
| Pointed at each value's best latent by F1: named, and named as planted | at least 0.7 of values | 0.812, 1.000 | 0.969, 1.000 |
| Follows | at least 0.8 | 0.939, 1.000 | 1.000, 0.951 |
| The null follows | at most 0.05 | 0.008, 0.000 | 0.000, 0.000 |
| On the twin: values named at all | at most 0.1 | 0.000, 0.000 | 0.000, 0.000 |
| On the twin: follows | at most 0.05 | 0.014, 0.012 | 0.018, 0.010 |

Recorded beside them:

- the same pointed at each value's largest-code latent (named as planted: 0.375, 1.000, 0.922, 0.906), and how often it is in its set (at least 0.938)
- the null's unchanged share (0.768 to 1.000)
- ablated (0.326 to 0.893)
- coverage
- the set's effect. `b`'s set ranks `b` last for every value, after block 2 and after block 1. The answer's and the next token's sets rank their token first for every value. Induction's current-token set ranks its own token at 46 on average, of 64 counting from 0 at the top: well below the middle.
- `b` after block 1: named as planted 0.625; follows 0.912, null 0.000

**Edges (judged).** The hop and the copy:

| Test | Bar | Hop | Copy |
| --- | --- | --- | --- |
| Follows | at least 0.8 | 0.939 | 1.000 |
| The null follows | at most 0.1 | 0.029 | 0.008 |
| Output follows | at least 0.8 | 0.934 | 1.000 |
| Kept | at most 0.4 | 0.211 | 0.000 |
| The null kept | at least 0.8 | 1.013 | 0.962 |
| Through the residual alone, follows; labelled computed | at most 0.1 | 0.025 | not applicable: across positions |
| On the twin: follows, less the null's | at most 0.1 | −0.032 | 0.000 |

Recorded:

- **the carry:** follows 0.988 (null 0.006), kept 0.015, and 0.918 through the residual alone, so it is labelled carried. On the twin: 0.515 (null 0.208), and 0.480 through the residual alone.
- **two back:** from block 0, first measured by G6.3. From block 1, round 3 found swapping the token's set barely moves the entity's set (0.068 with the upstream set chosen at `objects`, 0.000 at `every`), though ablating it removes about half (0.506 and 0.514 left); changing the token itself moves it every time (1.000).
- **single latents:** the hop 0.074, the copy 0.453, the carry 0.117.

**Open-this-up (judged).**

| Target, opened into | Reading, median | Within 128 parts | The planted set's first part in the top 8 | Planted |
| --- | --- | --- | --- | --- |
| Two-hop `b`'s set at `query` after block 2, into block 1 | at most 16 (8) | at least 0.8 (1.000) | at least 0.6 (1.000) | `b`'s set at `query` after block 1 |
| Two-hop `token(p + 1)`'s set at `query` after block 3, into block 2 | at most 64 (32) | at least 0.8 (0.930) | at least 0.6 (0.781) | `b`'s set at `query` after block 2 |
| Induction `token(p + 1)`'s set at `s + P − 1` after block 1, into block 0 | at most 64 (24) | at least 0.8 (0.992) | at least 0.6 (0.797) | `token(p)`'s set at `s` after block 0 |

- **Random order (judged):** within 128 parts for at most 0.1 of instances (0.016, 0.000, 0.000).
- **The twin, induction (judged):** the planted set's first part in the top 8 for at most 0.1 of instances (0.000; median rank 352).
- **The twin, two-hop (recorded):** readings 32 and 64. Within 128 for 0.898 and 0.664; the planted set in the top 8 for 0.867 and 0.648. On the twin, too, the same position's set one block earlier ranks first: the residual stream carries it. So a planted part at the same position can't tell a trained model from its twin, and the twin test is the induction target's, across positions.

**Generalize-this (judged).** For two-hop's `b` after block 2 and induction's `token(p + 1)` after block 1:

| Test | Bar | Two-hop | Induction |
| --- | --- | --- | --- |
| Every held-out slice: F1 | at least 0.5 | 0.780 to 0.825 | 0.787 to 0.794 |
| Every held-out slice: follows | at least 0.8 | 0.902 to 0.945 | 0.953 to 1.000 |
| Dictionary seeds 1 and 2, by name: F1 | at least 0.5 | 0.756, 0.832 | 0.747, 0.772 |
| Dictionary seeds 1 and 2, by name: follows | at least 0.8 | 0.920, 0.961 | 0.996, 0.959 |
| Step 0 is outside the envelope | yes | F1 0.026, follows 0.020 | F1 0.020, follows 0.014 |
| The envelope is one unbroken run to the final checkpoint | yes | from step 6,000 | from step 700 |
| It starts within two checkpoints of the probe's first reading of at least 0.9 | yes | both at step 6,000 | step 700; the probe at 750 |

Recorded: every checkpoint's F1, follows, null, probe, patch and accuracy; the decoder cosines across seeds; accuracy outside the envelope. Two-hop's follows reaches 0.25 at steps 1,000 and 2,000, but so does its null (0.20 and 0.21): an untrained model's answers land on the window's objects by chance.

## The reference run

At G1.4's final checkpoint, with the TopK SAEs on every block, everything is recorded and nothing judged (D44). Feasibility's figures, by block 0 to 3:

- **name-this-direction for 256 latents per block, drawn with a fixed seed:** the share named, by description, and their coverage. Feasibility: 30.1%, 27.0%, 18.4% and 8.6% named, mostly as `token(p)`; a named latent's median coverage 0.72 to 0.84.
- **steering for the names whose description is `token(p + 1)`**, the one with a prediction, up to 32 values and 8 instances each: 8 to 11 values named, following 3.2%, 6.3%, 8.0% and 22.7% (nulls at most 1.1%).
- **open-this-up for those names' sets, each into the block below:** within 128 parts for 25%, 45% and 91% of instances at blocks 1 to 3.
- **each gesture's time:** the whole set took 15 seconds.

## Records, the viewer and badges

- **Records built by G6.7:**
  - a name record for every value of every planted name
  - a name record for each latent the features view shows active at a zoo run's readout in the probe windows, at the planted names' sites, up to 8 per window
  - an edge record per relation
  - an open record per judged target
  - a generalize record per judged name
  - the reference run's recorded requests
- **The gestures panel:** for a run, its records by kind. Each record shows its meter first:
  - a name: named or weak, follows beside the null, F1, coverage, the set's effect at both ends, the top candidates
  - an edge: follows and kept beside their nulls, through the residual alone, and the label
  - open-this-up: the error curve and the top parts with their names
  - generalize-this: the envelope across checkpoints on the timeline, with the probe and accuracy beside it, the slices and the seeds
- **The features view links** each latent to its name record, where one exists.
- **Badges:** each gesture gets a badge of its zoo results, generated from the Phase 6 reports and shown above every record of that gesture (D38).

## Gates

Each runs on the Mac, prints progress while it works, and writes one schema-2 report under `reports/phase6/`. Dictionaries come from the Phase 5 cache; records from the Phase 6 cache when their key matches.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G6.1 Determinism and the operation | `dsteer`'s identities at every TopK site of both zoo models; request hashes over 64 requests; each kind of record built twice from one request | Identities bit for bit; distinct hashes; byte-identical records | Diagnose the operation or the record |
| G6.2 Names | Every planted name, trained and twin | Every bar met | Diagnose the gesture |
| G6.3 Edges | Every relation, trained and twin | Every bar met | Diagnose the gesture |
| G6.4 Open-this-up | Every judged target, trained and twin | Every bar met | Diagnose the gesture |
| G6.5 Generalize-this | Both judged names across slices, seeds and every checkpoint | Every bar met | Diagnose the gesture |
| G6.6 Reference run | The requests above | Recorded, not judged | Not applicable |
| G6.7 Records, panel and badges | Every record above built twice; badges from the reports | Byte-identical, no external URL, every gesture has a badge | Diagnose the build |

## Work packages

| # | Package | Files | Depends on |
| --- | --- | --- | --- |
| 1 | `dsteer`, with its identities (D48) | `interventions/spec.py`, `interventions/apply.py` | None |
| 2 | Requests, records and their cache; descriptions, classes, predictions and relations per run (D47) | `gestures/records.py`, `gestures/declare.py`, `zoo/` | None |
| 3 | name-this-direction and assert-this-edge (D49, D50) | `gestures/name.py`, `gestures/edge.py` | 1, 2 |
| 4 | open-this-up and generalize-this (D51, D52) | `gestures/open.py`, `gestures/generalize.py` | 1 to 3 |
| 5 | The command line, the Phase 6 gates and their report rules | `gestures/cli.py`, `gates/phase6/`, `reports.py` | 1 to 4 |
| 6 | The gestures panel, the features view's links and the badges | `views/`, `viewer/`, `gates/phase6/` | 5 |

## Tests

Written with the stubs, delta by delta:

- **Delta 1 (D47, D48):** `test_dsteer_arguments`, `test_dsteer_identities_are_exact`, `test_dsteer_moves_only_its_rows`, `test_dsteer_on_a_transcoder_and_a_staircase`, `test_dsteer_refusals`, `test_requests_round_trip_and_hash`, `test_request_refusals`, `test_the_record_key_covers_its_inputs`, `test_records_round_trip_and_refuse_tampering`, `test_descriptions_and_classes`, `test_a_constant_description_names_nothing`, `test_predictions_and_relations`, `test_sole_targets`
- **Delta 2 (D49, D50):** `test_candidates_rank_by_f1`, `test_named_and_weak`, `test_the_set_is_the_top_four`, `test_coverage`, `test_the_set_effect_at_both_ends`, `test_steering_follows_a_planted_variable`, `test_the_null_avoids_the_sets_and_shared_latents`, `test_ablation`, `test_no_prediction_records_the_kl`, `test_edge_follows_and_kept`, `test_edge_value_map`, `test_the_residual_alone_freezes_the_blocks_between`, `test_carried_and_computed`, `test_single_latents_beside_sets`
- **Delta 3 (D51, D52):** `test_the_pre_activation`, `test_attribution_matches_finite_differences`, `test_absolute_order_and_ties`, `test_the_error_curve_and_reading`, `test_signed_and_random_orders`, `test_no_span_is_left_out`, `test_top_parts_carry_their_names`, `test_slices`, `test_seeds_match_by_name`, `test_checkpoint_rows`, `test_the_envelope_and_the_gap`
- **Delta 4 (gates):** `test_settings_and_bars_are_fixed`, `test_instances_are_seeded`, `test_judgements`, `test_ticking`, `test_phase6_gate_ids`, `test_phase6_status_rules`
- **Delta 5 (panel and badges):** `test_gesture_payloads_are_plain_json_and_repeatable`, `test_the_index_lists_every_record`, `test_features_link_to_names`, `test_gesture_badges_cite_the_reports`, `test_bad_reports_are_refused`, `test_g67_status_rules`

## Out of scope

- **Crystallize-this and modular addition** (Phase 7, C10); **the ledger** (Phase 8).
- **Names from free text:** names come only from declared descriptions. Labelling a latent in words, by a person or a language model, would give a name no meter can check. On the reference run, declared descriptions name 8.6% to 30.1% of latents, and the record says so.
- **Dictionaries per checkpoint** (D44), and **model seeds**, which have no specimens yet.
- **Attention edges through the edge meter:** assert-this-edge tests edges by swapping sets instead (D42).
- **Two-hop's earlier route** (H6) as an explanation: generalize-this shows the accuracy before the envelope, and studying it is research for the notebook.

## Settled while drafting

- **Four latents to a set:** the top 4 swap 91.0% and 96.5%, against 5.9% and 7.8% for one, and 6 or 8 add at most 3.5 points (finding 9).
- **The effect from the set:** the set's effect ranks `b` last for every value; the latent best by F1 alone, for 25% (finding 10).
- **Weak names still steer:** a value's largest-code latent is in its set for at least 93.8% of values on every planted name, so a latent below the F1 floor gets its candidate's set and steering, labelled weak, instead of no answer (finding 17).
- **Absolute attribution:** finding 13.
- **Names, not ids, across seeds:** finding 15.
- **The carry label:** the residual stream carries a swapped set from block 1 to block 2 at `query` on the twin as well (finding 18). Without the residual-alone pass, a carry would pass as an edge.
- **Judged edges come from the task's construction:** the hop and the copy are required by the tasks (D30). The carry is the model's choice of where to hold `b`, so it is recorded with its label.
- **The induction `token(p)` name at `every`:** its steering instances are sole targets, which sit before the mechanism positions. Sets chosen at `every` and at `mechanism` gave the same results (named as planted 0.969 and 0.984; follows 1.000 and 1.000).
- **Through the final dictionaries, with a probe beside them:** apolloh0's choice, and the gap is what makes it honest (finding 11).
