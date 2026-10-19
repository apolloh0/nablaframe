# Nablaframe research notebook

Status: working notes, started by apolloh0 at H5 on 2026-10-18 and updated as phases close. This notebook rules nothing: rulings live in [decisions.md](decisions.md), and the current phase's work in its spec. Charter: [charter.md](charter.md).

## Where things stand

Phase 5 is closed: H5 passed (D46, commit `8bef5da`), and Phase 6, the gestures, is next. This notebook keeps what the zoo has shown, what it hasn't, and what to try next.

The zoo's two-hop model has 4 layers and 51 tokens: entities `e0` to `e47`, then `→`, `;` and `⇒`. A window lists six facts `s → o ;` at positions 0 to 23, then `a ⇒ c`, with `⇒` at position 25. To answer, the model finds `b` in `a → b`, then `c` in `b → c`.

Each entry carries one of four tags:

- **Judged**: a gate held it to a bar.
- **Recorded**: measured and in a report, but not judged.
- **Hypothesis**: a reading of recorded results that no intervention has tested.
- **Idea**: something to build or try.

Numbers come from `reports/phase4/`, `reports/phase5/` and `docs/decisions.md`, unless marked as read off the viewer. To update: change a hypothesis's status in place, add new observations under the phase they came from, cite a report or decision for every number, and add a line to the log at the end.

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

Each of these latents ranks its own `b` lowest of the 51 tokens. For 867 and 879 the gap is wide: -4.33 and -4.28, with no other token below -0.96. Both lenses also rank `b` last at `⇒` after blocks 1 and 2, after ranking it near the top after block 0 (logit lens median rank 3, counting from 0; D39).

### 3. Why the lens misses `b`

In windows 0 and 3, the logit lens's top token at `⇒` after block 2 is the `b`-latent's largest effect: e42 (41.3%) for 867, e14 (27.1%) for 879. The lens reads the latent's positive side, while `b` sits at its negative end. Read off the viewer, in two windows only.

### 4. The same latents fire two tokens after their entity

None of the six latents' 48 top held-out contexts is at `⇒`. All sit on fact tokens, at positions 2, 4, 8, 12 and 20, with codes of 4.60 to 5.41, against 1.44 to 3.32 at `⇒`.

- **867 and 879:** in all 16 contexts the viewer shows, the entity is the object of the fact just ended, two tokens back. The latent fires on the next fact's subject (`e4 → e40 ;` then `e15`).
- **559 (e37):** all 8 contexts are at position 2. In the two windows whose variables pin that fact down, it is `e37 → c`, so the entity is again two tokens back, as the subject.
- **No position 16:** the subject positions 4, 8, 12 and 20 hold 18, 8, 3 and 11 of the 40 subject contexts; position 16, fact 5's subject, holds none.
- **Self-loops score high:** distractor facts whose object is their own subject (`e12 → e12`, `e40 → e40`) give 879's first and third contexts and 867's second.

### 5. One latent carries only part of a variable

Moving only `b`'s latents after block 2 swaps the answer in 23.0% (TopK) and 9.4% (Staircase) of windows, against 90.2% and 86.3% for the whole code. Induction's transcoder: its current-token latents are followed 12.5% to 16.8% of the time, against 77.3% to 80.9% (G5.2, G5.3).

### 6. Two-hop leans on attention after block 0

Ablating two-hop's block-0 MLP costs 4.209 nats; the MLPs in blocks 1 to 3 cost 0.005, 0.004 and 0.012. Attribution credits block 3's attention with +11.13 at the answer, against +2.61 for its MLP (G4.4, G5.4).

### 7. A real language model is far denser

The zoo's judged MLPs need 0.4% to 1.8% of their scored edges; the reference run's two MLPs need 41.1% and 40.6% (G5.5). Its dictionaries leave 6% to 22% of the variance unexplained, against at most 3.2% on the zoo.

### 8. Departures from feasibility

- **Two-hop's blocks 0 to 1:** the TopK pair takes 1,203 edges against its twin's 22,138 (feasibility: 6,825 against 21,824). Induction's still matches its twin, 20,819 against 20,035.
- **Single latents:** the Staircase SAE's swap (9.4%) is below D41's 14.6% to 33%; the transcoder's following (12.5% to 16.8%) is below 24% to 100%.
- **Training time:** reference dictionaries took 112 to 372 seconds, not D44's 22 to 25 minutes.
- **Jacobian SAE:** at `λ` 4 its share of entries above 0.01 falls from 0.74–0.92 to 0.02–0.24, but 90% of the mass still takes 86 to 143 of 256 entries, and its output FVU rises up to sixfold (D43).

## Hypotheses to test

Five readings of the observations above; no intervention has tested any of them yet. Phase 6's gestures can test most of them. Status is one of Open, Testing, Supported or Refuted.

| # | Hypothesis | Evidence so far | Test | Status |
| --- | --- | --- | --- | --- |
| H1 | Entity latents copy the entity two tokens back, probably through an attention head that looks two positions back | 16 of 16 contexts for 867 and 879; both pinned contexts for 559 (Observation 4) | Find a block-0 or block-1 head that attends two back at fact tokens; ablate it and check the latents vanish there. Swap the previous fact's object and check the latent follows | Open |
| H2 | The model stores `b` upside down to keep from predicting it, since `b` is the most tempting wrong answer | The lens ranks `b` 3rd after block 0, last after blocks 1 and 2; each `b`-latent's smallest effect is `b` (Observations 2, 3) | Rank `b` in the final logits against the window's other entities; steer along a `b`-latent's decoder row at `⇒` and check whether `b`'s final logit falls | Open |
| H3 | A latent adds up its entity's nearby occurrences | Self-loop facts give three of the top contexts (Observation 4) | Compare codes at matched positions in windows where the entity appears once or twice within reach | Open |
| H4 | Something at position 16 keeps these latents quiet | 0 of 40 subject contexts at position 16 (Observation 4) | Each latent's mean code by position over the held-out windows; it may be chance in a top-8 list | Open |
| H5 | The five shared latents (48, 50, 64, 90, 984) carry the format, not a variable | Active in all 8 windows whatever `b` is, and at 4% to 36% of all positions; largest effects `⇒`, `;`, `→` | Patch only those latents between windows and check the answer doesn't move | Open |

## Ideas to explore

Eight things worth building or trying, roughly in order of payoff.

| Idea | Why | Where it fits |
| --- | --- | --- |
| Dictionaries at earlier checkpoints | Watch the `b`-latent form as the capability does. Only this project can replay its training; two-hop's answer loss falls most steeply between about 4,000 and 6,000 steps (read off the viewer's timeline) | Phase 6 scope, or a phase of its own (D44 deferred it) |
| A blind prediction on the reference run | State a latent or edge the views suggest for a TinyStories behavior (a repeated name, he/she agreement) before any intervention, then confirm it by patch. That is the step from calibration to explanation | After Phase 6's gestures |
| Name latents from both ends of the effect | Top contexts and largest effects misname the `b`-latents; their identity is the smallest effect (D46) | Phase 6, name-this-direction |
| Contexts by position class | Global top contexts never showed the role at `⇒`; split them into at `⇒`, at fact subjects and at objects | Phase 6 viewer |
| Both ends in the active-latents table | Its largest-effect column shows e42, e17, e47 for the e40 latent | Phase 6 viewer |
| Feature sets, not single latents | One latent moves 9% to 23% of the answer; find the smallest set whose patch reaches 90% of the code's rate | Phase 6, assert-this-edge |
| Seeds | Do dictionary seeds 0 to 2 find the same `b`-latents? Does the dictionary follow `b` to block 1 in model seed 1, where patching found it (D36)? That separates the model from the tool (D34) | Any time; cheap |
| Wider reference dictionaries | Is the reference run's 41% edge density the model, or its dictionaries' 6% to 22% unexplained variance? | Later |

## What these results don't show yet

So far this is calibration on models whose answers are known, not discovery in a real network.

- **Toy scale:** both zoo models have 4 layers, and two-hop has 51 tokens. Passing here is necessary before trusting the tools elsewhere, not sufficient.
- **Nothing judged on real text:** the reference run's dictionaries leave 6% to 22% of the variance unexplained, and none of its results can be judged (D44). Its meters travel with every view for that reason.
- **Thin evidence for the hypotheses:** H1 to H5 rest on a few dozen contexts from six latents, read off one dictionary.
- **No one-to-one concepts:** a single latent moves 9% to 23% of what its variable's whole code moves.
- **Known methods:** dictionaries, lenses and patching are established. What is distinctive is the discipline around them: twin controls, determinism, replayable training, and a meter on every view.

## Lessons about the tools

The biggest lesson: a latent's identity showed up in its effect, not in its contexts.

- **Window variables undercount:** G5.6's `query_latents` labels each context by its window's `b`, whatever the position. That found the `b`-latent in 6 of 8 windows; the effect panel found it in all 8. Check identity by the effect and the context strip, not by window variables alone.
- **Contexts hide roles:** a latent's top contexts are its largest codes over every position, so a role it plays only at `⇒` never shows there.
- **The effect panel warns itself:** its label already says "It is not a name for the latent."
- **The viewer's bare address loads Phase 4:** without a query string it shows the Phase 4 build and badges. Use G5.6's printed addresses, adding `&seq=` and `&latent=` to land on a latent.
- **Feasibility drifts:** the gates re-measured several feasibility numbers (Observation 8). Cite reports, not feasibility rounds.

## Where the evidence lives

Everything above can be rebuilt from the repo; the views live only on the Mac.

- **Rulings:** `docs/decisions.md`, especially D39 (the lens finding) and D41 to D46.
- **Reports:** `reports/phase4/` and `reports/phase5/`, with `reports/phase5/summary.md` as the overview.
- **Features payloads:** `runs/phase5/views/<run>/features-<dictionary>-site-<n>.json`. `runs/` isn't in git; G5.6 rebuilds it byte-identically.

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
| 2026-10-18 | Started after H5 (D46), from the Phase 5 reports and apolloh0's viewer session on windows 0 and 3 |
