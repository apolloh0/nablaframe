# Phase 6, delta 3: open-this-up and generalize-this

Date: 2026-10-21. Spec: [../specs/phase-6.md](../specs/phase-6.md). Rulings: D51, D52.

This delta ships:

- **Stubs for two new modules in `gestures/`.** Each docstring is its contract.
  - `open.py`: open-this-up (D51). It covers a target's pre-activation, the integrated-gradients attribution through the model, the three part orders, the error curve and its reading, and each top part's name.
  - `generalize.py`: generalize-this (D52). It reads a name across held-out slices, dictionary seeds and every checkpoint, and an edge across the same.
- **11 tests in two new files,** the delta 3 tests the spec lists: `test_gestures_open.py` (7) and `test_gestures_generalize.py` (4). They build on delta 2's toy models in `tests/phase6_fixtures.py`, which this delta doesn't change. The few extra pieces they need are built in the test files:
  - a dictionary with biases and a scale that change neither its codes nor what it writes
  - a seed dictionary that holds the same latents under other ids
  - checkpoints whose token directions are scrambled, or whose final layer norm reads them backwards

No existing file changes. The gates, the command line and the viewer come in later deltas.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase6-delta3.zip
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `11 failed, 307 passed`:

- open-this-up: 7 failed
- generalize-this: 4 failed

## Settled in the contracts

D51, D52 and the spec left these open. Each follows the spec's intent, feasibility round 3 or the nearest existing rule.

### open-this-up

- **A record opens one target in either of the request's two forms (D47): a latent, or a name's set given by its description and value.** Its instances are where that target fires.
  - G6.4's judged targets, such as "`b`'s set at `query` after block 2", take each instance's own value's set.
  - The gate builds those from `attribute`, `order` and `evaluate`, the parts `open_up` is made of. It checks the planted set's rank the same way.
  - The gesture knows nothing of what is planted (D29).
- **One compiled pass serves the attribution and every kept-parts reading.** The upstream output tap holds `decode(codes) + err`, and the codes, tokens and error are the pass's arguments (PR #38).
  - It isn't `dsteer`: integrated gradients need a gradient, and kept parts span every position up to the target's, where `dsteer`'s rows are one position per sequence.
  - The arithmetic is `dsteer`'s: removing codes keeps the error (D48).
- **Parts are the positions up to and including the target's, times the upstream site's latents.** A later position can't reach the target through a causal model.
- **"An earlier site" means the upstream output tap comes before the downstream input tap,** in the trace registry's tap order.
- **The random order shuffles only the active parts,** and puts the rest after them in part order, as round 3 did. Each instance's shuffle is drawn after the instances are cut, so a seed fixes everything.
- **A reading of "not within 128" counts as larger than every `k` in the median.** The median is None when it lands on one.
- **A part is named at the smallest class that holds its position,** ties going to the class declared first:
  - two-hop: position 25 at `query`, 22 at `objects`, 24 at `every`
  - induction: mechanism positions at `mechanism`

### generalize-this

- **Each slice, seed or checkpoint takes one steering pass over the description's values,** each instance with its own value (`steer_description`). One pass per value would mean thousands of compiles across a run's checkpoints.
  - The rules for drawing instances and `w` are `name.draw_instances`'s.
  - The spec's per-value numbers remain available from `name.steer`.
- **Slices split the held-out windows in window order.** Each slice's F1 comes from its own table, with the full table's sets. Its steering draws only from its windows, with the full table's amounts and null.
- **Seed dictionaries come from the caller as readings.** The name is derived again at each one, never matched by latent id. The cosine compares decoder rows between the same value's sets.
- **Checkpoints go through the final dictionaries (D44):**
  - The final table gives the sets and the null's top-8 exclusions.
  - The checkpoint gives the amounts, the shared latents and the null's pool. That was round 3's mix.
  - The probe is D41's ridge probe on the site's input tap.
  - Accuracy is next-token accuracy at the name's class: the answer at two-hop's `query`, the next token at induction's mechanism positions.
- **The residual patch the spec puts beside each checkpoint is task code** (G4.2's, G4.3's). The caller passes it through `extras(step, params)`, and the record stores what it returns. The gesture itself names no task (D29).
- **`holds`** requires both F1 and follows to be at least 0.5. Where the run predicts nothing for the description, F1 alone decides. With no steering instance, the name doesn't hold. The function is public so its edges can be tested; the toy checkpoints only ever give 0 or 1.
- **The envelope records its runs, the run that reaches the final checkpoint, the gap and the probe's first reading of at least 0.9.** G6.5's three checks read from these:
  - step 0 is outside the envelope
  - one unbroken run reaches the final checkpoint
  - that run starts within two checkpoints of the probe's first reading
- **An edge is generalized with `edge.edge` on readings of each slice, seed and checkpoint,** with sets derived at each. No gate judges it, and D52 asks only for follows, kept and the label.

## Threads

The two threads share no files and depend on nothing in each other. Paste into the Project, in one wave:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P6] Open", on Opus at high effort:
Implement src/nablaframe/gestures/open.py against its docstring and docs/specs/phase-6.md (D51). Change only that file. The attribution and every kept-parts reading go through one compiled pass whose codes, tokens and error term are arguments (PR #38), as the docstring says, with the error held at its unintervened value. Nothing specific to a task goes into it (D29). Draw from the rng exactly as the docstring says. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_open.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Generalize", on Opus at high effort:
Implement src/nablaframe/gestures/generalize.py against its docstring and docs/specs/phase-6.md (D52), on gestures/sets.py, name.py and edge.py as merged. Change only that file. Every steering pass goes through sets.run_steered, one call per slice, seed or checkpoint, as the docstring says. Nothing specific to a task goes into it (D29): task readings come from the caller through extras. Draw from the rng exactly as the docstring says. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_generalize.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** on clones of my mirror of `main` at `78d1bf0`, with this delta's files added:

- **On the stubs:** each of the 11 new tests fails on its own, and all 307 existing tests pass.
- **On a private reference implementation, not shipped:** all 318 pass with no skips. The two new files take about two minutes on a cloud CPU, mostly compiling.
- **Planted bugs:** all 59 are caught by the tests aimed at them. They include:
  - **pre-activations and the forward pass:**
    - `b_pre`, `scale_in`, the jsae unit or all but one bias left out of the pre-activation, or `b_pre` left out of the compiled one
    - the error term dropped, decoded without its bias or its scale, or taken from one window
    - the target read at position 0, or a set reduced to its first latent
    - a chunk given its first rows' operands
    - the taps' order unchecked
  - **attribution:**
    - gradients at the endpoints, summed rather than averaged, or not multiplied by the codes
  - **orders:**
    - absolute order without the absolute value, or with ties to the higher part
    - signed order smallest first
    - random order shuffling every part, or putting the inactive parts first
    - shuffles drawn before the cut, or instances kept in drawn order
  - **the curve:**
    - keeping every part but the first `k`
    - the error over the full pre-activation
    - "not within 128" counted as 128
    - no instance left out for want of a span
    - candidates counted after the cut
    - a name's instances ignoring its value
  - **top parts:**
    - taken in signed order
    - named at the run's first class, or with a weak name labelled named
    - the class ties going to the later class, or the largest class chosen
  - **generalize-this:**
    - F1 with the table's own sets or only one latent of them
    - slices taken every eighth window, or a slice steering over every window
    - checkpoint sets or F1 read again at the checkpoint
    - `holds` on F1 alone, without F1, without steering instances, strictly above 0.5, or never for a name the run can't steer
    - the final run taken as the last run, the gap counting holding rows, the probe's first reading taken from the first holding row, runs broken a row early, or the probe threshold strict
    - extras dropped, or accuracy read on the final model
    - seeds matched by latent id, or seed F1 with seed 0's sets
    - the cosine against the seed's own set, or not normalized
    - steps in any order, or seed readings unchecked
    - an edge's slices over every window

  Three more changes were set aside before the run, as equivalent on the toy models:
  - a probe read on the final model: every checkpoint's block-0 activations decode the token perfectly
  - a null whose shared latents come from the set table: no toy table's shared latents differ
  - a strict error threshold: no error lands exactly on 0.1

  **Not covered:** attribution through attention. The toy models move nothing between positions, so every earlier position's attribution is zero there. G6.4's induction target, read across positions, is the first test of it.

**Not verified:**

- anything on the Mac
- the gestures on the zoo, which G6.4 and G6.5 measure in delta 4
- their time and memory at the zoo's sizes and every checkpoint, and on the reference run
