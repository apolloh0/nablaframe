# Phase 4, delta 2: the two-hop collision rules, and Phase 3's gates kept to their six operations

Date: 2026-10-09. Spec: [../specs/phase-4.md](../specs/phase-4.md). Rulings: D29 to D35.

This delta ships:

- **4 tests,** in two new files:
  - `test_zoo_two_hop_rules.py`: the collision rules ruled on PR #44, and the patch source's matching rule
  - `test_phase3_gate_operations.py`: G3.2 and G3.5 keep to the six operations H3 ruled on, the break PR #45 found
- **Contract text** for `zoo/two_hop.py`'s `source_for`, and for the Phase 3 gates.
- **Spec edits:** the collision rules, the patch source's rule, and this delta's tests.

The Phase 4 gates come in delta 3. The two-hop feasibility check found that G4.3's judged swap assumes where the model holds its intermediate, which the task's construction doesn't guarantee (see "Feasibility" below). The gates follow once that's ruled.

## Apply

Every file in the zip is new. Apply it after PRs #44 (with its collision change) and #45 have merged. From the repo root:

```
unzip -n nablaframe-phase4-delta2.zip
python3 apply_phase4_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `11 failed, 167 passed`:

- the 8 stubs waiting on wave 2: ground truth 2, induction 3, two-hop 3
- this delta's 3 red tests: the two Phase 3 gate tests, and the patch-source rule

The collision-rules test passes already, as a pin on PR #44's change.

## Contract changes

**`src/nablaframe/zoo/two_hop.py`** (replaces `source_for`'s bullet)

- "`source_for(task, target, counterpart, rng)`: a window whose intermediate is `counterpart`, a subject of one of the target's distractor facts, and whose answer is neither the target's answer nor the counterpart's object in the target. Windows are drawn with `task.window(rng, intermediate=counterpart)` until one qualifies. So an unchanged, a swapped and a transplanted answer are always three different tokens."

**`gates/phase3/_common.py`, `g32_determinism.py` and `g35_viewer.py`**

- "`PHASE3_OPERATIONS = ("zero", "mean", "resample", "patch", "add", "swap")`, in `_common.py`: D22's six operations, as H3 ruled on them (D28). G3.2's `sample_specs` and G3.5's `examples` iterate it, never `spec.OPERATIONS`, so their committed reports stay reproducible however many operations later phases add. Later operations are tested by their own phase's gates."

## Threads

Once this delta is applied, start two threads now. They share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P3] Phase 3 gates keep their six operations", on Sonnet at medium effort:
Implement the "gates/phase3/_common.py, g32_determinism.py and g35_viewer.py" part of "Contract changes" in docs/deltas/phase-4-delta-2.md, writing its bullet into _common.py's module docstring verbatim. Change only those three files, and in g32 and g35 only the lines that use OPERATIONS. Never run a gate for results and never write under reports/. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase3_gate_operations.py tests/test_phase3_gates.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Two-hop measurements", on Sonnet at high effort:
Implement src/nablaframe/zoo/two_hop.py against its docstring and docs/specs/phase-4.md, with the "src/nablaframe/zoo/two_hop.py" part of "Contract changes" in docs/deltas/phase-4-delta-2.md written into source_for's bullet verbatim. Change only that file. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_two_hop.py tests/test_zoo_two_hop_rules.py` passes with no skips, and every test that passed before still does.
```

This two-hop message replaces wave 2's in `docs/deltas/phase-4-delta-1.md`. Wave 2's ground-truth and induction thread is unchanged.

## Verification record

**Verified** on a fresh clone of `main` (`bfd1b5e`) with PRs #44 and #45 applied, #44 carrying its collision change:

- **On the stubs:** this delta's red tests fail as intended. The collision pin passes.
- **On a private reference, not shipped:** every new test passes. The reference covers the frozen Phase 3 operations, the collision rules and `source_for`.
- **Planted bugs:** each of 6 is caught by the test aimed at it:
  - G3.2 or G3.5 still iterating `spec.OPERATIONS`
  - an answer that may equal the queried subject
  - a distractor that may share the answer as its object
  - patch sources drawn without their rule
  - patch sources avoiding only the target's answer

**PR #45, checked independently** at a 4-layer model:

- **All 28 new null edits are bit for bit:** 12 `jswap`s at no positions and 16 one-head zeros.
- **A real swap** exchanges the coordinates to within 1e-7, leaving the orthogonal part of the residual unchanged.

**PR #44's collisions, measured** over 20,000 windows of its generator:

- the answer equals the query 2.1% of the time
- it equals the intermediate 2.1% of the time
- some distractor shares it as its object 8.0% of the time

Hence the ruling, and the pin.

## Feasibility, measured before the gates are written

Both zoo models were trained here at the spec's sizes, through the real trainer and batch source.

**Induction** forms cleanly:

- mechanism positions 0.18 at step 250, 0.96 at step 500, and 1.000 from step 750 on
- control positions 0.020, against a bar of 0.031
- random-weights twin 0.015
- The loss bump falls between steps 250 and 500, inside the dense checkpoints.

**Two-hop** at the spec's config:

- **Behaviour:** after a plateau near 0.25, it climbs to 0.80 by step 11,500 and flattens. That's short of the 0.9 bar.
- **Patching the query positions** from a source window, at every block's output, only ever transplants the source's answer, from 0.19 at block 0 to 0.71 at block 3. The swap rate is near 0.05 at every block, against G4.3's bar of 0.8.
- **Aligned sources** differ only in the intermediate. Patching any single fact position, any whole fact, or every fact position after block 0 never moves the source's answer in. Patching every position reproduces it exactly, 0.79, which confirms the method.
- **The intermediate is present at `⇒` and causally used:** the logit lens ranks it 2nd to 3rd at every block. Swapping only its logit-lens coordinates for the counterpart's, at `⇒` after block 0, sends the answer to the counterpart's successor 30% of the time, against a 5–9% base rate, fading with depth.

**So where and how strongly a model holds its intermediate is a measurement, not something the task's construction guarantees.** G4.3's judged swap bar assumed the query positions hold it. The gates wait for a ruling on that, and on how two-hop reaches its 0.9 bar.

**Not verified:** anything on the Mac.
