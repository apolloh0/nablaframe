# Nablaframe research notebook

Status: working notes, started by apolloh0 at H5 on 2026-10-18 and updated as phases close. This notebook rules nothing: rulings live in [decisions.md](decisions.md), and the current phase's work in its spec. Charter: [charter.md](charter.md).

## Where things stand

Phase 6 is closed: H6 passed (D53, commit `6a918a4`). Its four gestures name latents, test edges, open targets and read names across training, each with a faithfulness number. Phase 7 builds crystallize-this, with modular addition joining the zoo. Two feasibility rounds in the cloud workspace on 2026-10-23 shaped the Phase 7 spec (C11, D54 to D57); their findings are under "Phase 7 feasibility". Phase 5 closed at H5 (D46, commit `8bef5da`), and the Phase 6 spec took its bars from the rounds under "Phase 6 feasibility". This notebook keeps what the zoo has shown, what it hasn't, and what to try next.

The zoo's two-hop model has 4 layers and 51 tokens: entities `e0` to `e47`, then `→`, `;` and `⇒`. A window lists six facts `s → o ;` at positions 0 to 23, then `a ⇒ c`, with `⇒` at position 25. To answer, the model finds `b` in `a → b`, then `c` in `b → c`.

The modular-addition model (Phase 7) has 1 block of width 32 and 114 tokens: the numbers `0` to `112`, then `=`. A window is `a b = c`, and position 2, the `=`, predicts `c = (a + b) mod 113`.

Each entry carries one of five tags:

- **Judged**: a gate held it to a bar.
- **Recorded**: measured and in a report, but not judged.
- **Feasibility**: measured by a feasibility round, an experiment outside the gates. It isn't in a report, so a gate measures it again before a decision cites it.
- **Hypothesis**: a reading of the results above that no intervention has tested.
- **Idea**: something to build or try.

Numbers come from `reports/phase4/` to `reports/phase6/` and `docs/decisions.md`, unless marked as read off the viewer or found in a feasibility round (the runs are listed under "Where the evidence lives"). To update: change a hypothesis's status in place, add new observations under the phase they came from, cite a report or decision for every number, and add a line to the log at the end.

## Established results (judged)

The tools agree with the zoo's planted answers, and every control fails where it should.

| Result | Numbers | Source |
| --- | --- | --- |
| The two-hop model solves its task | 99.6%; broken chains 1.2%, twin 0.8% | G4.1 |
| The induction model solves its task | 100.0%; controls 0.3%, twin 1.7% | G4.1 |
| Induction heads exist and are necessary | best block-1 head puts 98.6% of attention on targets (twin 4.8%); zeroing block 1's heads together removes 1.001 of the accuracy above chance | G4.2 |
| Patching finds where two-hop holds `b` | swaps the answer in 9.0%, 83.6%, 85.2% and 0.0% of windows after blocks 0 to 3; transplants it in 100% after block 3 | G4.3 |
| Dictionaries reconstruct the zoo | FVU at most 0.032 (twins 0.094 to 0.170); the splice recovers at least 99.2% | G5.2, G5.3 |
| A dictionary's code is as causal as the residual | code patch reaches at least 0.957 of the residual patch's rate; at `⇒` after block 2, 90.2% against 85.2% | G5.2, G5.3 |
| Single latents read the planted variables | `b` at `⇒` after block 2: F1 0.795 (twin 0.248); `c` after block 3: 0.866 (twin 0.123) | G5.3 |
| The zoo's MLP circuits are sparse | 602, 122 and 269 edges, against the twins' 56,632, 22,506 and 3,981 | G5.4 |

## Observations (recorded)

At `⇒` after block 2, one TopK latent per value of `b` carries the intermediate, written in a direction that lowers `b`'s own logit.

### 1. A latent for each `b` at `⇒`

In all eight probe windows, a latent whose smallest direct effect is that window's `b` is active at `⇒`. It has the largest code in seven windows and the second largest in one.

| Window | `b` | Latent | Code at `⇒` | Rank among active latents | Smallest effect |
| --- | --- | --- | --- | --- | --- |
| 0 | e40 | 867 | 3.235 | 1st | e40 (-4.33) |
| 1 | e37 | 559 | 1.443 | 2nd | e37 |
| 2 and 6 | e10 | 368 | 2.868 and 2.643 | 1st | e10 |
| 3 | e12 | 879 | 3.244 | 1st | e12 (-4.28) |
| 4 and 7 | e25 | 614 | 2.295 and 2.693 | 1st | e25 |
| 5 | e2 | 860 | 3.321 | 1st | e2 |

Windows that share a `b` share the latent. Five more latents (48, 50, 64, 90 and 984) are active in every window, whatever `b` is; their largest effects are the format tokens `⇒`, `;` and `→`. Source: G5.6's `query_latents` and the block-2 TopK payload.

### 2. `b` is stored upside down

Each of these latents ranks its own `b` lowest of the 51 tokens. For 867 and 879 the gap is wide: -4.33 and -4.28, with no other token below -0.96. Both lenses also rank `b` last at `⇒` after blocks 1 and 2, after ranking it near the top after block 0 (logit lens median rank 3, counting from 0; D39). Finding 10 checks the sign for all 48 values.

### 3. Why the lens misses `b`

In windows 0 and 3, the logit lens's top token at `⇒` after block 2 is the `b`-latent's largest effect: e42 (41.3%) for 867, e14 (27.1%) for 879. The lens reads the latent's positive side, while `b` sits at its negative end. Read off the viewer, in two windows only.

### 4. The same latents fire two tokens after their entity

None of the six latents' 48 top held-out contexts is at `⇒`. All sit on fact tokens, at positions 2, 4, 8, 12 and 20, with codes of 4.60 to 5.41, against 1.44 to 3.32 at `⇒`.

- **867 and 879:** in all 16 contexts the viewer shows, the entity is the object of the fact just ended, two tokens back. The latent fires on the next fact's subject (`e4 → e40 ;` then `e15`).
- **559 (e37):** all 8 contexts are at position 2. In the two windows whose variables pin that fact down, it is `e37 → c`, so the entity is again two tokens back, as the subject.
- **No position 16:** the subject positions 4, 8, 12 and 20 hold 18, 8, 3 and 11 of the 40 subject contexts; position 16, fact 5's subject, holds none.
- **Self-loops score high:** distractor facts whose object is their own subject (`e12 → e12`, `e40 → e40`) give 879's first and third contexts and 867's second.

### 5. One latent carries only part of a variable

Moving only `b`'s latents after block 2 swaps the answer in 23.0% (TopK) and 9.4% (Staircase) of windows, against 90.2% and 86.3% for the whole code. Induction's transcoder: its current-token latents are followed 12.5% to 16.8% of the time, against 77.3% to 80.9% (G5.2, G5.3). A value's top four latents carry nearly all of it (finding 9).

### 6. Two-hop leans on attention after block 0

Ablating two-hop's block-0 MLP costs 4.209 nats; the MLPs in blocks 1 to 3 cost 0.005, 0.004 and 0.012. Attribution credits block 3's attention with +11.13 at the answer, against +2.61 for its MLP (G4.4, G5.4).

### 7. A real language model is far denser

The zoo's judged MLPs need 0.4% to 1.8% of their scored edges; the reference run's two MLPs need 41.1% and 40.6% (G5.5). Its dictionaries leave 6% to 22% of the variance unexplained, against at most 3.2% on the zoo.

### 8. Departures from feasibility

- **Two-hop's blocks 0 to 1:** the TopK pair takes 1,203 edges against its twin's 22,138 (feasibility: 6,825 against 21,824). Induction's still matches its twin, 20,819 against 20,035.
- **Single latents:** the Staircase SAE's swap (9.4%) is below D41's 14.6% to 33%; the transcoder's following (12.5% to 16.8%) is below 24% to 100%.
- **Training time:** reference dictionaries took 112 to 372 seconds, not D44's 22 to 25 minutes.
- **Jacobian SAE:** at `λ` 4 its share of entries above 0.01 falls from 0.74–0.92 to 0.02–0.24, but 90% of the mass still takes 86 to 143 of 256 entries, and its output FVU rises up to sixfold (D43).

## Phase 6 feasibility (2026-10-19 and 2026-10-20)

Three rounds on the Mac measured what each gesture needs, at G4.1's checkpoints with the Phase 5 TopK SAEs. Everything here is tagged feasibility. A value's **set** is its top 4 latents by F1 at the variable's site and position. A **null** swaps 4 random active latents for 4 random others at the set's amounts. Both rounds first reproduced the gates: `b`'s mean best F1 at `⇒` after block 2 is 0.795 (twin 0.248), induction's current and next token 0.890 and 0.790, the code patch at `⇒` swaps 90.2%.

### 9. A name needs about four latents

Each value's top `k` latents by F1, swapped for another value's at their mean codes:

| Latents swapped | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| Two-hop, `b` at `⇒` after block 2: the answer swaps | 5.9% | 45.7% | 68.8% | 91.0% | 94.1% | 94.1% |
| Induction, the next token after block 1: the prediction follows | 7.8% | 66.4% | 91.8% | 96.5% | 100% | 100% |

On two-hop, the whole code swaps 90.2% and the residual 94.9%. Controls:

- **A third value:** swapping the set for a third entity's sends the answer to that entity's object in 94.5% of windows, and to the counterpart's in 1.2%.
- **Ablating the set:** the answer changes in 74.6% of two-hop windows and 36.7% of induction's.
- **The null:** two-hop's answer swaps in 1.2% and stays unchanged in 83.6%; induction's never follows and stays unchanged in 100%.

### 10. `b` is upside down in the code as a whole

For each of `b`'s 48 values at `⇒` after block 2, where does `b` rank in the direct effect?

- **The latent with the largest mean code:** lowest of the 51 tokens for 100% of values.
- **The set, weighted by its mean codes:** 100%.
- **The whole mean code, with or without the five shared latents:** 100%.
- **The latent best by F1:** lowest for 25.0%, highest for 43.8%.

The largest-code latent is the F1-best for only 8.3% of values, and D46's six latents were largest-code latents. So Observation 2 holds for the representation, but not for the most selective single latent. The logit lens ranks `b` 50 (median) on the residual itself. Induction is the other way up: its F1-best next-token latent ranks its token highest for 85.9% of values.

### 11. Two-hop learned its task twice

All 21 checkpoints were read through the final dictionaries, beside ridge probes, the logit lens and G4.3's patch at `⇒`. Selected rows:

| Step | Accuracy | Probe for `b` after block 2 | Probe for `c` after block 2 | Patch swap after block 2 | Lens rank of `b` after block 2 | Named F1 after block 2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.0% | 0.338 | 0.057 | 2.3% | 25 | 0.007 |
| 1,000 | 16.5% | 0.276 | 0.147 | 2.0% | 3 | 0.104 |
| 3,000 | 47.5% | 0.226 | 0.324 | 1.6% | 3 | 0.125 |
| 5,000 | 74.3% | 0.366 | 0.356 | 1.2% | 9 | 0.189 |
| 6,000 | 96.7% | 0.981 | 0.103 | 78.5% | 49 | 0.506 |
| 7,000 | 98.8% | 0.987 | 0.070 | 92.2% | 50 | 0.618 |
| 20,000 | 100% | 0.993 | 0.043 | 94.9% | 50 | 0.795 |

- **Before step 6,000:** the model answers up to 74% of windows without holding `b` at `⇒`. The probe stays near its untrained level (0.34), patching moves nothing, and the lens ranks `b` near the top. Meanwhile `c` grows readable at `⇒` after block 2 (0.36 by step 5,000).
- **At step 6,000, everything moves at once:** `b` becomes readable at `⇒` (0.98), patching it swaps the answer (78.5%), accuracy reaches 96.7%, and the lens drops `b` from rank 9 to 49. At the same time `c` fades from block 2 (0.10, and 0.04 at the end); after block 3 it reads 0.93.
- **A second run did the same:** the cloud stand-in used for dry runs has G4.1's config and seeds but trained on another processor, so it isn't bit-identical, and it stopped at step 10,000. In round 2's small dry run it answered 68% at step 5,000 with no patch swap (0 of 64 windows) and the lens ranking `b` 4. The patch then swapped 16% at step 6,000 and 89% at 8,000, while the lens rank went to 45.5 and 49.
- **Induction learns once:** its probe for the next token after block 1 tracks accuracy at every checkpoint (0.52, 0.83 and 0.95 against 54%, 84% and 95% at steps 550, 650 and 800).
- **The final dictionaries lag:** at step 6,000 they read `b` at F1 0.506 where the probe reads 0.981. A dictionary trained at the end sees a variable only in its final form.

### 12. Edges are carried by sets

- **Two-hop, `b`'s set after block 1 at `⇒`, swapped for the counterpart's:** block 2's set follows in 97.7% of 256 windows, no old set mass is left (median), and the answer swaps in 87.5%. The null: 0%, 87.9% left, 1.2%. Ablating the set leaves 1.8% of block 2's set mass, against 95.6% for the null.
- **Single latents carry part of an edge:** in round 1, ablating a planted upstream latent removed 14% to 54% of the downstream latent's code (nulls -3% to 10%). On induction, swapping the current token's top latent at the copied position after block 0 made the new next-token latent at the copy target outweigh the old one in 34.4% of instances; swapping its top 4, in 91.3%.
- **A failed edge:** swapping the token two back's set after block 1 (chosen over every position) for another entity's made the entity's set after block 2 follow in 0.6% (null 0.1%). It did remove about half the old set's mass (52.9% left, null 107.1%). Changing the input token itself made it follow in 99.8%. See H8.

### 13. Open-this-up works with parts ordered by absolute attribution

Integrated gradients attribute a target's pre-activation to every upstream (position, latent), with the upstream error term kept. Parts are kept in order, every other upstream code zeroed, and the error is |kept − full| / |full − none|. The median number of parts for an error of 0.1:

| Target | Opened into | By attribution | By absolute attribution | Random | Planted set's first rank |
| --- | --- | --- | --- | --- | --- |
| `b`'s latent at `⇒` after block 2 | block 1 | 8 | 8 | not by 128 | 0 |
| `b`'s set at `⇒` after block 2 | block 1 | not by 128 | 8 | not by 128 | 0 |
| The entity's set at a subject after block 2 | block 1 | 4 | 4 | not by 128 | 0 |
| Induction's next-token latent after block 1 | block 0 | 16 | 32 | not by 128 | 3 |
| Induction's next-token set after block 1 | block 0 | 16 | 16 | not by 128 | 3 |

Signed order never brings `b`'s set within 0.1, because the positive parts overshoot. Two-hop's planted parts sit at the same position one block earlier, so their rank 0 partly reflects the residual stream. Induction's, the current token's set at the copied position, sits elsewhere and ranks 3.

### 14. Two-hop's two-back copy is spread over heads

Seven heads, 1.1 to 1.3 and 2.0 to 2.3, put 41% to 47% of their attention from fact tokens on the token two back; 0.0 and 0.1 put 34% and 32%.

| Zeroed | Entity latent's code left | `b`'s set mass at `⇒` left | Accuracy |
| --- | --- | --- | --- |
| Head 1.1, 1.2 or 2.1 | 0.96 to 0.97 | 0.89 to 0.94 | 99.7% to 100% |
| Head 1.0 (looks one back, 48%) | 0.88 | 0.51 | 71.6% |
| Every head of block 0 | 0 | 0 | 0% |
| Every head of block 1 | 0.55 | 0.12 | 24.6% |
| Every head of block 2 | 0.70 | 0.76 | 98.0% |

Head 1.0 was meant as a null, the head in blocks 0 to 2 that looks two back least. It turned out to be the one head whose zeroing alone hurts.

### 15. Names transfer across dictionary seeds; latent ids don't

- **Names:** `b`'s mean best F1 is 0.756 and 0.832 at dictionary seeds 1 and 2; induction's next token 0.747 and 0.772. Over eight held-out slices, 0.768 to 0.801 and 0.784 to 0.793.
- **Ids:** a value's best latent at seed 1 or 2 has a mean decoder cosine of 0.38 to 0.40 with seed 0's, which is the closest seed-0 latent for only 31% to 42% of values.

### 16. A `b`-latent's activity is mostly elsewhere

The share of each `b`-latent's activation mass, over every position of the held-out windows, where a description holds:

| Description | Share |
| --- | --- |
| `b = v` at `⇒` | 3.6% |
| `v` is the token | 19.5% |
| `v` is one back | 7.0% |
| `v` is two back | 28.6% (10.1% at subjects, 17.9% at objects) |
| `v` is three back | 4.1% |
| `b = v` at `⇒`, or `v` two back | 32.3% |

- **Follow test:** when the token two back changes, the old latent halves in 99% of instances and the new one fires in 49% to 53%.
- **Self-loops (H3):** a mean code of 1.32 over 141 self-loop instances, against 0.75 over 10,099 others.
- **Position 16 (H4):** it holds 3.3% of the mass, beside 3.4% at positions 12 and 20.

### 17. Names come up empty on the twins, and weak names still steer

Round 3 ran name-this-direction as the spec defines it (D49), 512 instances per name. Pointed at each value's best latent by F1:

| Name | Named as planted | Follows | Null | Twin: named at all | Twin: follows |
| --- | --- | --- | --- | --- | --- |
| Two-hop `b` at `⇒` after block 2 | 0.812 | 0.939 | 0.008 | 0 | 0.014 |
| Two-hop `b` at `⇒` after block 1 | 0.625 | 0.912 | 0.000 | 0 | 0.021 |
| Two-hop answer at `⇒` after block 3 | 1.000 | 1.000 | 0.000 | 0 | 0.012 |
| Induction current token after block 0 | 0.969 | 1.000 | 0.000 | 0 | 0.018 |
| Induction next token after block 1 | 1.000 | 0.951 | 0.000 | 0 | 0.010 |

- **Weak names:** a value's largest-code latent, the one the features view lists first, is in its set for at least 93.8% of values on every name. But at `b` after block 2 it reaches F1 0.5 for only 39.6% of values. So the spec names such a latent weakly and still steers its set.
- **The sign again:** `b`'s set ranks `b` last for every value after block 1 as well as block 2. The answer's and the next token's sets rank their token first. Induction's current-token set ranks its own token at 46 of 64 on average, counting from 0 at the top.

### 18. The residual stream makes a carry look like an edge

Swapping `b`'s set after block 1 at `⇒`, block 2's set follows 98.8% of the time, and still 91.8% with block 2's attention and MLP frozen at `⇒`: block 2 only carries `b` there. On the twin the same swap is followed 51.5% of the time (null 20.8%), and 48.0% through the residual alone. The hop is different: swapping `b`'s set after block 2, the answer's set after block 3 follows 93.9% (null 2.9%), and only 2.5% with block 3 frozen. Induction's copy, across positions, follows 100% (null 0.8%); the twin's sets don't beat their nulls.

### 19. Two-hop copies the entity two back in two blocks

Swapping the token's set two places back after block 1 barely moves the entity's set after block 2: 6.8% with the set chosen at object positions, 0% with it chosen at every position. Removing that set leaves about half the entity's set mass (50.6%, 51.4%). Changing the token itself moves the entity's set every time (100%). Block 1's two-back heads have already copied the token from after block 0 by the time block 2's read it after block 1, which fits finding 14: zeroing either block's heads leaves part of the code. H8 is refuted for this edge.

### 20. Open-this-up on the twins

On induction's twin, the next token's set reaches a 0.1 error within 128 parts for only half the instances, and the planted current-token set ranks 352 (median), never in the top 8 (trained: 24 parts, rank 4, 79.7% in the top 8). On two-hop's twin, the same position's set one block earlier still ranks first (in the top 8 for 86.7%), because the residual stream carries it. A planted part at the same position doesn't separate trained from twin; one across positions does.

### 21. Envelopes with sets

Through the final dictionaries, the name holds (F1 and follows both at least 0.5) from step 6,000 on two-hop and from step 700 on induction, unbroken to the end. At step 6,000 the set follows 86.5% where the probe first reads `b` at 0.981. Induction's envelope starts one checkpoint before its probe reaches 0.9 (0.927 at step 750). Two-hop's follows reaches 25% at steps 1,000 and 2,000, but so does its null (20% to 21%): an untrained model's answers land on the window's objects. Across dictionary seeds, the same names follow 92.0% to 99.6%, with sets whose decoders match seed 0's at a cosine of 0.45 to 0.53.

### 22. The reference run

TopK SAEs at G1.4's final checkpoint, 256 latents sampled per block:

| Block | Named | Mostly as | `token(p + 1)` names | Their sets follow | Open-this-up within 128 parts |
| --- | --- | --- | --- | --- | --- |
| 0 | 30.1% | `token(p)` 93.5% | 8 | 3.2% | not measured |
| 1 | 27.0% | `token(p)` 92.8% | 8 | 6.3% | 25% of instances |
| 2 | 18.4% | `token(p)` 85.1% | 11 | 8.0% | 45% |
| 3 | 8.6% | `token(p)` 59.1%, `token(p − 1)` 18.2% | 11 | 22.7% | 91% |

A named latent's median coverage is 0.72 to 0.84. Token offsets name fewer latents with depth, and the next-token sets steer weakly until the last block. The whole section took 15 seconds.

## Phase 7 feasibility (2026-10-23)

Two rounds in the cloud workspace measured what crystallize-this needs. Everything here is tagged feasibility:

- **Induction and two-hop:** on stand-ins trained with G4.1's configs and seeds. Two-hop's stood at step 10,000 of 20,000.
- **Modular addition:** on models trained there with the repo's trainer.

A **crystallization** replaces a component's output, at its positions, by a function of what it reads. A **table** is the component's mean output for each value of the function's key, fitted on training windows. The **meter** is the KL from the model's prediction to the crystallized model's on held-out windows, beside a mean ablation's KL.

### 23. Width decides how modular addition is learned

`a b = c` with `c = (a + b) mod p`, `p` 113, 30% of the pairs trained. One block, weight decay 1.0, the trainer unchanged.

| Width | Held-out ≥ 0.99 from step | Final held-out | Embedding energy in its top 5 frequencies | Logits vs. those frequencies' cosines, R² |
| --- | --- | --- | --- | --- |
| 128 | 3,000 | 0.995 | 0.104 (twin 0.102) | 0.081 |
| 64 | 3,000 | 1.000 | 0.358 | 0.744 |
| 32 | 1,800 | 1.000 | 0.795 (frequencies 2, 36, 25, 44, 18) | 0.966 |

At width 128 each number gets its own direction. That model memorizes first, with 100% of training pairs and 18.3% held out at step 1,000. At width 32 the model learns the clock, and training and held-out accuracy rise together. A second seed at width 32 learned frequencies 48, 14, 2, 4 and 24, holding 0.822.

### 24. Accuracy and loss can't judge a crystallization

A table keyed by the answer predicts better than a model that hasn't learned. At width 128, step 1,000:

- **The table:** with the MLP replaced by a table by the sum, the model is right on 99.3% of held-out pairs, against 18.3% for the model itself.
- **The KL:** 3.63, above the mean ablation's 2.49.

On the induction stand-in at step 300, block 1's attention as a table by the copied token is right 68.2% of the time, against 6.5% for the model.

### 25. The meter

The KL from the model's own prediction, beside a mean ablation's, catches both cases. At the final checkpoint:

- **Modular addition, width 32:** the block's write at `=` as a table by the sum gives 0.021 against 4.14.
- **Induction:** block 1's attention as a table by the copied token gives 0.000 against 6.94.
- **Two-hop:** block 3's write at `⇒` as a table by the hop's object gives 0.134 against 4.58.
- **Wrong functions:**
  - induction's current token: 6.73
  - two-hop's `a`: 4.61; its `b`: 4.62
  - modular addition's `(a − b)`: 10.16; `(a + b + 1)`: 8.31; `a` or `b` alone: 5.90 to 5.94

### 26. Counterfactuals separate what a component reads from the answer

A function can equal the answer on every ordinary window, so a low KL alone can't say the component computes it. Inputs where the two part ways can:

- **Induction, broken windows:** one token at position `j` is changed, and the model is read at `j − 1`, where it can't see the change.
  - The model predicts the copied token every time, and the window's next token never.
  - Keyed by the copied token, the crystallization agrees with it every time (KL 0.000).
  - Keyed by the next token, it never does (KL 13.47).
- **Two-hop, `b` patched at `⇒` after block 2 from G4.3's aligned source:**
  - The model answers the counterpart `k`'s object 95.1% of the time.
  - Keyed by `k`'s object, what block 3 reads, the crystallization agrees 93.8% (KL 0.346).
  - Keyed by the window's answer, 0.8% (KL 9.33).
- **Modular addition, width 32, the attention output at `=` patched from another pair:**
  - The MLP, crystallized additively in the sum and both operands, agrees with the patched model every time when keyed by the other pair (KL 0.017).
  - Keyed by the window's own pair, it agrees 0.8% (KL 8.08).

### 27. Modular addition's block computes the sum, not its MLP alone

At width 32:

- **The attention output at `=`** is additive in the operands: `A[a] + B[b]`, KL 0.094 against 7.99.
- **The MLP alone,** as a table by the sum, gives 1.546 against 7.50. As the sum plus a term for each operand, it gives 0.019 (R² 0.971).
- **The block's write,** attention plus MLP, is a function of the sum alone: 0.021. The MLP's operand terms cancel the attention's.
- **At width 128,** the remainder isn't operand terms: the MLP's additive form is no better than its table, 0.334 each.

So the component has to be drawn where the function holds.

### 28. Five frequencies

At width 32 the block's write at `=`, fitted as cosines and sines of the sum at the embedding's five largest frequencies, gives KL 0.019 against 4.14. That is 11 vectors of 32 numbers. With four frequencies the KL is 0.263; with six, still 0.019. The second seed gives 0.027 against 4.03. The loss delta is −0.027 nats against the model's 0.123: the crystallized write is a little more confident than the model's own.

### 29. One head isn't a component

Induction's copy is spread over all four of block 1's heads, each putting 0.36 to 0.45 of its attention from `q` on `q − P + 1`. Replacing the best head's pattern by the explicit one-hot, or by its mean, moves the KL by at most 0.001: the other heads cover it.

### 30. Envelopes and gaps

The crystallization holds where its KL is at most 0.25 and the ablation's is at least 1:

- **Modular addition:**
  - The fourier form holds from step 1,100 (KL 0.199, ablation 1.12), while held-out accuracy is 0.061. Accuracy passes 0.9 at step 1,700: the clock forms before the model reads it out.
  - The table holds at 1,100, misses at 1,200 (0.258), and holds from 1,300.
- **Induction:** holds from step 900. Accuracy passes 0.9 at step 800 (0.951, KL 0.307), so step 800 is a gap.
- **Two-hop:**
  - Block 3's write at `⇒` barely matters through step 5,000: the ablation's KL is 0.35 there while the model answers 69.6%.
  - It holds from step 8,000.
- **Width 128's MLP table:** lowest at step 8,000 (0.106). It drifts back to 0.334 by step 25,000, while held-out accuracy slips from 1.000 to 0.995.

### 31. Phase 6's names can't hold the sum

On modular addition at width 32, with a TopK SAE at block 0 (Phase 5's settings, FVU 0.017, twin 0.046), name-this-direction, unchanged, at `=`:

- **The sum:**
  - Named as planted for 23.9% of values (named at all: 31.0%; mean set F1 0.459).
  - Steering follows 0.8%, as its null does. The swapped output is unchanged 99.4% of the time.
  - Ablating a value's set changes no prediction.
- **`a` and `b` at `=`:** named as planted for no value.
- **The twin:** names 1.8% of sums, none as planted.

The sum is written in five frequencies (finding 28), a code no set of four latents carries. C11 rules on it.

### 32. The twins show nothing

On every twin, the right function and the wrong ones all leave the twin's prediction within 0.005 nats, and within 0.003 of each other. The mean ablation does the same: induction 0.0035, two-hop 0.0016, modular addition 0.0041. A twin's components don't steer its readout, so there is nothing to crystallize, and no function stands out.

## Hypotheses to test

Readings of the results above. Feasibility tested H1, H3, H4 and H8, and added H6 to H9. Phase 6's gestures can test most of the rest. Status is one of Open, Testing, Supported or Refuted; a status set by feasibility says so.

| # | Hypothesis | Evidence so far | Test | Status |
| --- | --- | --- | --- | --- |
| H1 | Entity latents copy the entity two tokens back, probably through an attention head that looks two positions back | 16 of 16 contexts for 867 and 879; both pinned contexts for 559 (Observation 4). Changing the token two back makes the entity's set follow in 99.8%, and the old latent halves in 99% (findings 12, 16). No single head carries it: seven heads look two back, zeroing any one leaves 96% of the code, and zeroing block 1's or block 2's heads leaves 55% or 70% (finding 14) | Done in feasibility. Still open: which heads write the code, by patching each head's output from a window with a different token two back | Supported (feasibility); the copy is spread over several heads |
| H2 | The model stores `b` upside down to keep from predicting it, since `b` is the most tempting wrong answer | The lens ranks `b` 3rd after block 0, last after blocks 1 and 2; each `b`-latent's smallest effect is `b` (Observations 2, 3). The whole code ranks `b` last for all 48 values (finding 10). The flip appears at step 6,000, at the checkpoint where `b` appears at `⇒` (finding 11). Induction's next-token latents, whose token is the answer, write it upward (finding 10). `b`'s set is upside down after block 1 too; induction's current-token set ranks its own token low, at 46 of 64, and in these windows the next token is never the current one (finding 17) | Rank `b` in the final logits against the window's other entities; steer along `b`'s set at `⇒` and check whether `b`'s final logit falls. Rerun both at steps 5,000 and 6,000 | Open, stronger |
| H3 | A latent adds up its entity's nearby occurrences | Self-loop facts give three of the top contexts (Observation 4). Over the held-out windows, a self-loop's mean code is 1.32, against 0.75 elsewhere (finding 16) | Done in feasibility | Supported (feasibility) |
| H4 | Something at position 16 keeps these latents quiet | 0 of 40 subject contexts at position 16 (Observation 4). Over the held-out windows, position 16 holds 3.3% of the mass, as positions 12 and 20 hold 3.4% (finding 16) | Done in feasibility: chance in a top-8 list | Refuted (feasibility) |
| H5 | The five shared latents (48, 50, 64, 90, 984) carry the format, not a variable | Active in all 8 windows whatever `b` is, and at 4% to 36% of all positions; largest effects `⇒`, `;`, `→`. The same five are active at `⇒` in at least 90% of 2,048 held-out windows, and leaving them out doesn't change the sign (finding 10) | Patch only those latents between windows and check the answer doesn't move | Open |
| H6 | Before step 6,000, two-hop answers by an earlier route that never holds `b` at `⇒` | 74% accuracy at step 5,000 with no probe, patch or lens sign for `b` at `⇒`, while `c` is already readable after block 2 (0.36), which fades as the composing route appears (finding 11). A second run, not bit-identical, learned in the same order (finding 11). Zeroing block 1's heads at step 20,000 leaves 24.6% accuracy, well above chance (finding 14), so the earlier route may survive as a backup. On a cloud stand-in, block 3's write at `⇒` barely matters through step 5,000: mean-ablating it moves the prediction by 0.35 nats while the model answers 69.6%, against 4.58 at step 10,000 (finding 30) | At step 5,000: patch each fact position from G4.3's aligned sources and see which moves the answer; train a dictionary there. At step 20,000: compare what block 1's heads zeroed leaves right with what step 5,000 got right | Open |
| H7 | Head 1.0, which looks one back from fact tokens, carries `b` to `⇒` | Zeroing it alone leaves 51% of `b`'s set mass at `⇒` and 71.6% accuracy, while no two-back head matters alone (finding 14) | Its attention from `⇒`; patch its output at `⇒` from aligned sources and check that the answer swaps | Open |
| H8 | A set chosen over every position misses what a position class carries | The token two back's set, chosen over every position, moves the entity's set in 0.6% of swaps, while changing the token moves it in 99.8%, and open-this-up ranks that set first (findings 12, 13). Chosen at object positions, it moves it in 6.8% (finding 19) | Done in feasibility: the edge fails for another reason, H9 | Refuted (feasibility) |
| H9 | The entity two back is copied twice: by block 1's heads from after block 0, and by block 2's heads from after block 1 | Swapping the token's set after block 1 moves the entity's set in at most 6.8%, while removing it leaves half its mass (finding 19); zeroing block 1's or block 2's heads leaves 55% or 70% of the code (finding 14) | Swap the token's set two back after block 0, which reaches both copies; G6.3 records that edge | Open |
| H10 | Phase 6's names miss modular addition's sum because the sum is held as phases of a few frequencies, which value-by-value descriptions can't name | The sum is named as planted for 23.9% of values, and steering runs at its null's rate, while five frequencies crystallize the write at KL 0.019 (findings 28, 31) | Declare, for each key frequency, the sum's phase in a few bins as a description, and point name-this-direction at it | Open |

## Ideas to explore

Things worth building or trying, roughly in order of payoff.

| Idea | Why | Where it fits |
| --- | --- | --- |
| Find two-hop's earlier route | The model answered 74% of windows at step 5,000 by a route that never held `b` at `⇒` (H6, finding 11). Only a replayable run can show a capability learned twice | Phase 6's generalize-this reads the checkpoints; the route itself is a study of its own |
| Which batches taught the composing route | Everything changed between steps 5,000 and 6,000. Leave-a-batch-out replay over those 1,000 steps could find what taught it | Phase 8, the ledger (D35) |
| Dictionaries at earlier checkpoints | The final dictionaries read `b` at F1 0.506 at step 6,000, where a probe reads 0.981, and can't see the earlier route at all (finding 11) | Phase 6 reads checkpoints through the final dictionaries, with a probe beside them; dictionaries per checkpoint stay later (D44) |
| A blind prediction on the reference run | State a latent or edge the views suggest for a TinyStories behavior (a repeated name, he/she agreement) before any intervention, then confirm it by patch. That is the step from calibration to explanation | After Phase 6's gestures |
| Descriptions for TinyStories | Token offsets name only 8.6% to 30.1% of the reference run's latents (finding 22). Declared descriptions such as inside a quote, the sentence's subject, or he/she agreement would be names a meter can check | After Phase 6, with the blind prediction |
| Name latents from both ends of the effect | Top contexts and largest effects misname the `b`-latents (D46). Read the effect from the set, weighted by its codes: it ranks `b` last for every value, where the F1-best latent alone does for 25% (finding 10) | Phase 6, name-this-direction |
| Contexts by position class | Global top contexts never showed the role at `⇒`; split them into at `⇒`, at fact subjects and at objects. A `b`-latent spends 3.6% of its mass at `⇒` (finding 16) | Phase 6 viewer |
| Both ends in the active-latents table | Its largest-effect column shows e42, e17, e47 for the e40 latent | Phase 6 viewer |
| Feature sets, not single latents | Done in feasibility: the top 4 swap 91.0% and 96.5%, against 5.9% and 7.8% for one (finding 9) | Phase 6: a name is a set |
| Model seed 1 | Does the dictionary follow `b` to block 1 in model seed 1, where patching found it (D36)? That separates the model from the tool (D34). Dictionary seeds are done: names transfer, ids don't (finding 15) | Any time; cheap |
| Wider reference dictionaries | Is the reference run's 41% edge density the model, or its dictionaries' 6% to 22% unexplained variance? | Later |
| Names for distributed codes | Modular addition's sum is held in five frequencies, which no value's set of four latents carries (finding 31) | After Phase 7 (H10) |
| Crystallize through a dictionary | Write a name's sets at their mean codes instead of a table, so a crystallization speaks in the gestures' terms | After Phase 7 |
| Crystallize the reference run's block 0 by the current token | Block 0's latents are named `token(p)` for 30.1% of those sampled (finding 22); a table by the current token would say how much of block 0 is a lookup | After Phase 7, recorded only (D44) |

## What these results don't show yet

So far this is calibration on models whose answers are known, not discovery in a real network.

- **Toy scale:** both zoo models have 4 layers, and two-hop has 51 tokens. Passing here is necessary before trusting the tools elsewhere, not sufficient.
- **Nothing judged on real text:** the reference run's dictionaries leave 6% to 22% of the variance unexplained, and none of its results can be judged (D44). Its meters travel with every view for that reason.
- **Feasibility isn't a gate:** findings 9 to 22 come from experiments outside the gates, on one dictionary seed per site. Phase 6's gates measure them again.
- **Few declared names on real text:** token offsets name 8.6% to 30.1% of the reference run's latents, most of them as the current token (finding 22).
- **No one-to-one concepts:** a single latent moves 9% to 23% of what its variable's whole code moves. A set of four moves nearly all of it, but a `b`-latent spends most of its activity on other roles (findings 9, 16).
- **Known methods:** dictionaries, lenses and patching are established. What is distinctive is the discipline around them: twin controls, determinism, replayable training, and a meter on every view.

## Lessons about the tools

The biggest lesson: a latent's identity showed up in its effect, not in its contexts.

- **Window variables undercount:** G5.6's `query_latents` labels each context by its window's `b`, whatever the position. That found the `b`-latent in 6 of 8 windows; the effect panel found it in all 8. Check identity by the effect and the context strip, not by window variables alone.
- **Contexts hide roles:** a latent's top contexts are its largest codes over every position, so a role it plays only at `⇒` never shows there.
- **The effect panel warns itself:** its label already says "It is not a name for the latent."
- **The viewer's bare address loads Phase 4:** without a query string it shows the Phase 4 build and badges. Use G5.6's printed addresses, adding `&seq=` and `&latent=` to land on a latent.
- **Feasibility drifts:** the gates re-measured several feasibility numbers (Observation 8). Cite reports, not feasibility rounds.
- **The most selective latent isn't the strongest:** a value's F1-best latent is its largest-code latent for only 8.3% of values, and its effect gives `b` mixed signs. Read a value's effect from its set or its whole code (finding 10).
- **Order parts by absolute attribution:** ordered by signed attribution, `b`'s set never got within 0.1 of its pre-activation, because positive parts overshoot (finding 13).
- **Check a null's own role:** the head picked as a null for H1, the least two-back head, was a previous-token head, and the one head whose zeroing hurt (finding 14). Whole blocks zeroed made the comparison.
- **A carry looks like an edge:** the residual stream passes a swapped set on, even on random weights (finding 18). Freeze the blocks between before calling it an edge.
- **A copy made twice hides from a single swap:** swapping one copy's source leaves the other copy in place (finding 19). A failed swap isn't a missing edge until every route is covered.
- **Match across dictionary seeds by name, never by id:** names come back at F1 0.75 to 0.83, in latents whose decoders have a cosine of only about 0.4 with seed 0's (finding 15).
- **Judge a crystallization by the model's prediction, not the answer:** a table keyed by the answer can beat a model that hasn't learned (finding 24).
- **Part the function from the answer:** where the two agree on every ordinary window, only a counterfactual says which the component computes (finding 26).
- **Draw the component where the function holds:** modular addition's MLP alone isn't a function of the sum; the block's write is (finding 27).

## Where the evidence lives

Everything above can be rebuilt from the repo; the views live only on the Mac.

- **Rulings:** `docs/decisions.md`, especially D39 (the lens finding) and D41 to D46.
- **Reports:** `reports/phase4/` to `reports/phase6/`, each phase's `summary.md` its overview.
- **Features payloads:** `runs/phase5/views/<run>/features-<dictionary>-site-<n>.json`. `runs/` isn't in git; G5.6 rebuilds it byte-identically.
- **Phase 6 feasibility:** `runs/feasibility/gestures-round1-20261019-230631/`, `runs/feasibility/gestures-round2-20261019-232218/` and `runs/feasibility/gestures-round3-20261020-171131/` on the Mac, each with a `SUMMARY.txt` and one JSON per model. The scripts are in `~/nf-feasibility/`, not in the repo.
- **Phase 7 feasibility:** the cloud workspace, on 2026-10-23. The scripts ship beside the Phase 7 spec delta for `~/nf-feasibility/phase7/`, not the repo: `modadd.py`, `train_modadd.py`, `analyze_modadd.py`, `analyze_induction.py`, `analyze_counterfactual.py`, `names_modadd.py`, `hop_two_hop.py` and `hop_envelope.py`, with round 1's `round1_findings.md`.

To open a latent in the viewer, serve the repo root with `python3 -m http.server`, then change `seq` (window) and `latent` in this address:

```
http://localhost:8000/viewer/?views=../runs/phase5/views/two_hop/&trace=../runs/phase4/two_hop/store/&interventions=../runs/phase4/two_hop/interventions/&badges=../runs/phase5/badges.json#step=20000&seq=3&pos=25&features=7e25eeb11d31c87de0da81e50a02edb490e1a22cc0979d47ade90b4d5526ab3d-0&latent=879
```

To print latents' contexts and effects from the block-2 TopK payload, run this from the repo root, listing the latents after the dash:

```sh
python3 - 879 867 <<'EOF'
import json, sys
p = json.load(open("runs/phase5/views/two_hop/features-7e25eeb11d31c87de0da81e50a02edb490e1a22cc0979d47ade90b4d5526ab3d-site-0.json"))
h, v = p["held_out"], p["variables"]
name = lambda t: f"e{t}" if t < 48 else "→;⇒"[t - 48]
for lat in map(int, sys.argv[1:]):
    print(f"latent {lat}: effect top {[name(t) for t, _ in p['effect']['top'][lat][:3]]}, bottom {[name(t) for t, _ in p['effect']['bottom'][lat][:3]]}")
    for w, pos, code in p["contexts"][lat]:
        print(f"  window {w} position {pos}: token {name(h[w][pos])}, code {code:.2f}; a e{v['a'][w]}, b e{v['b'][w]}, c e{v['c'][w]}")
EOF
```

## Log

Newest first.

| Date | Change |
| --- | --- |
| 2026-10-23 | Phase 7 feasibility, rounds 1 and 2: findings 23 to 32; H10 added, H6 evidence added; ideas, lessons and where the evidence lives; "Where things stand" brought to H6 |
| 2026-10-20 | Phase 6 feasibility, round 3: findings 17 to 22; H8 refuted, H9 added, H2 evidence added; lessons and the reference run's names |
| 2026-10-20 | Phase 6 feasibility, rounds 1 and 2: findings 9 to 16; H1 and H3 supported, H4 refuted, H2 stronger; H6 to H8 added; ideas and lessons updated |
| 2026-10-18 | Started after H5 (D46), from the Phase 5 reports and apolloh0's viewer session on windows 0 and 3 |
