# Phase 6, delta 2: name-this-direction and assert-this-edge

Date: 2026-10-21. Spec: [../specs/phase-6.md](../specs/phase-6.md). Rulings: D49, D50, and the spec's "What each run declares".

This delta ships:

- **Stubs for three new modules in `gestures/`.** Each docstring is its contract.
  - `sets.py`: what both gestures share. It reads a dictionary site over held-out windows, builds each class's F1 table, draws the null, and steers through `dsteer`.
  - `name.py`: name-this-direction (D49).
  - `edge.py`: assert-this-edge (D50).
- **A complete test helper, `tests/phase6_fixtures.py`.** It builds toy models whose weights are set by hand: in one, a token's direction is carried through the blocks; in the other, the last MLP computes a map `G` of it. It also builds dictionaries of token directions and toy declarations.
- **14 tests in two new files,** the delta 2 tests the spec lists: `test_gestures_name.py` (9) and `test_gestures_edge.py` (5).
- **One edit to the spec:** work package 3 adds `gestures/sets.py` to its files.

The gates, the command line and the viewer come in later deltas.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase6-delta2.zip
python3 apply_phase6_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `14 failed, 293 passed`:

- name-this-direction and the shared module: 9 failed
- assert-this-edge: 5 failed

## Settled in the contracts

D49, D50 and the spec left these open. Each follows the spec's intent, feasibility round 3 (whose definitions set the bars) or the nearest existing rule.

- **What both gestures share lives in `sets.py`.** That covers reading a site, the class tables, the null and steering. `edge.py` imports it, not `name.py`, so the edge thread doesn't depend on the name module.
- **A site is read once, then measured many times.** `read_site` encodes the site's input tap over the held-out windows. It keeps latents below the site's active width, so `dsteer` never refuses a latent the table offers.
  - The descriptions come in a fixed order: the token descriptions, then the run's variables. That order breaks ties between candidates.
  - The gates can loop over a name's values on one reading. G6.2 needs this to aggregate follows over every value.
- **F1 counts every row of the class, including rows where the description says nothing (-1).** A latent firing there tracks no value, so those rows count against it, as they did in round 3.
- **Candidate ties go to the earlier description, then the lower value.** A latent that fires on no row of the class has no candidates and no name. Its record says so and stops.
- **Coverage is over every held-out position, not only the class's,** as D49 says.
- **The set's effect is weighted by its mean codes where the name holds,** as D49 says. The pointed latent's own effect is unweighted, since a positive weight doesn't change its ranks.
- **The other value `w` (one rule, no task code).** `w` is drawn from the description's values at the class, less those the window rules out:
  - for a token description, every token the window holds
  - for a variable, the values the variable takes in the window

  Where the run predicts for some of the remaining values, only those are drawn. For token descriptions this is the spec's rule. For two-hop's `b` it gives the spec's exception from the declaration alone (D29): the window takes only its own `b`, and `predict` keeps the distractor subjects.
- **Steering runs only where the run predicts.** A name is steered only at instances where the run predicts for some `w`. So induction's `token(p)` at `every` is steered at sole targets only, as the spec says.
  - Where the run predicts at no instance, the record gives the KL and each pass's top tokens, and says no causal test applies.
  - The null's KL sits beside it, since every measurement sits beside its control (D47).
- **The null adds from latents active somewhere at the class,** as in round 3, never from shared latents.
- **One call per gesture.** A gesture's passes are stacked in one batch, one call to `run_intervened`: four for a name, seven for an edge. That is one compile per record instead of one per pass, which matters for G6.2's hundreds of records.
  - The pass through the residual alone runs once per position. `patch` selects its positions for the whole batch.
- **An edge instance needs the downstream description to be `g(v)` where the declaration pairs it.** Every zoo pair meets that. A wrong value map gives no instances rather than wrong ones.
- **Through the residual alone:** the frozen blocks are the ones after the upstream place's block, up to and including the downstream place's.
  - The label follows D50 literally. With nothing following at all, it reads carried, because no computation was shown.
  - A reading's block is the `l` of its output tap, `blocks.<l>.…`, and must be its place's.
- **Kept is a median of ratios to the unintervened mass, from the same batch.** Masses are summed in float64.
- **A seed fixes everything.** Instances, `w`s and nulls are drawn as the docstrings say, step by step, so the same request rebuilds the same record (D47).

## Threads

The edge thread builds on `sets.py`, so the two threads run in two waves.

**Wave 1.** Paste into the Project:

```
Start one thread now, without proposing it first.

Thread "[P6] Names", on Opus at high effort:
Implement src/nablaframe/gestures/sets.py and src/nablaframe/gestures/name.py against their docstrings and docs/specs/phase-6.md (D49, "What each run declares"). Change only those two files. Every forward pass goes through sets.run_steered, so every edit is a dsteer or patch edit through run_intervened, and a gesture's passes share one call, as the docstrings say. Nothing specific to a task goes into either file (D29): what a run declares comes from its Declaration. Draw from the rng exactly as the docstrings say, so that a seed always gives the same instances and null. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_name.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** after "[P6] Names" has merged:

```
Start one thread now, without proposing it first.

Thread "[P6] Edges", on Opus at high effort:
Implement src/nablaframe/gestures/edge.py against its docstring and docs/specs/phase-6.md (D50, "What each run declares"), on gestures/sets.py as merged. Change only that file. Every forward pass goes through sets.run_steered, and the seven passes share one call, as the docstring says; only the pass through the residual alone runs once per position. Nothing specific to a task goes into it (D29). Draw from the rng exactly as the docstring says. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_edge.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** on clones of my mirror of `main` at `a41062b`, with this delta applied:

- **On the stubs:** each of the 14 new tests fails on its own, and all 293 existing tests pass.
- **On a private reference implementation, not shipped:** all 307 pass with no skips. The two new files take about 70 seconds on a cloud CPU, mostly compiling.
- **Planted bugs:** all 71 are caught by the tests aimed at them. They include:
  - **the class table:**
    - F1 computed as recall, or blind to rows where the description says nothing
    - values counted from 6, or at every position
    - activity or sharing counted at every position, or sharing strictly above 0.9
    - candidate ties broken the wrong way, an inactive latent given candidates, or only four candidates
    - ranks tied to the higher latent, or a set of three
    - a name's F1 taken as its set's mean
    - mean codes over every row
    - a bool taken as a value
    - variables before the token descriptions
    - all of a Staircase site's latents read
  - **the other value and the null:**
    - `w` drawn from the description's own values, or a value the window holds
    - the null removing from `v`'s set or shared latents, or adding latents already active, from `v`'s top 8, or never active at the class
    - drawing with replacement, or one draw too many
    - the instances kept in drawn order
  - **steering and the name's record:**
    - amounts in reverse, freezing only attention, freezing over several positions, or argmax ties to the higher token
    - effects not centred, or without the final layer norm's gain
    - ends with ties the wrong way, or four tokens at each end
    - coverage by count, or over part of the window
    - every candidate steered, predicted or not
    - the floor itself weak; the pointed latent forced into its set, ranked from 1, or in the set at rank 4; the name's F1 given for the set's
    - the set's effect unweighted, or the latent's effect taken for it
    - `w`'s set at `v`'s mean codes; the null at its own mean codes
    - an ablation that removes nothing
    - unchanged measured against the prediction, or the output read at the edit's position
    - the KL taken the other way or at the first position, or seven examples
  - **edges:**
    - an inactive downstream set taken as an instance
    - `w` with the same `g(w)`, or a token the window holds, or predictions ignored when drawing it
    - the downstream value unchecked, or the output predicted from the upstream description
    - follows counted on a tie, or single kept as a mean
    - freezing from the upstream block, or short of the downstream one
    - an edge across positions labelled carried
    - the null kept after the set's ablation
    - pairs counted after the checks; blocks or windows unchecked
    - the single swap adding the whole set; the instances unsorted

  One more change turned out to be equivalent, not a bug, and was dropped. Without `read_site`'s own check of the windows' width, `describe` still refuses such windows.

  **Not covered:** the label's threshold at exactly half. Each toy model either carries an edge entirely or not at all, so a test can't tell "at least half" from any other cut between 0 and 1.

**Not verified:**

- anything on the Mac
- the gestures on the zoo models, which G6.2 and G6.3 measure in delta 4
- their running time and memory at the zoo's sizes, and on the reference run, where a site read over many windows of 256 positions holds every code in memory
