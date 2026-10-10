# Phase 4, delta 3: the zoo revised by D36, and the Phase 4 gates

Date: 2026-10-10. Spec: [../specs/phase-4.md](../specs/phase-4.md). Rulings: D29 to D37.

This delta ships:

- **D36's contract changes** to three zoo modules:
  - the periodic induction task
  - attention to every induction target, with its uniform baseline
  - G4.3's aligned sources
- **Stubs for G4.1 to G4.5,** their summary and their helpers. The helpers fix every setting at the feasibility runs' values (D37).
- **17 tests** in five new files, plus explicit edits to two committed tests that asserted the old induction task.
- **Contract text** for `reports.py`.
- **Spec edits:** J's fit positions, and this delta's tests.

## Apply

Apply and commit `apply_d36.py` first; this delta's script refuses to run without D36. Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase4-delta3.zip
python3 apply_phase4_delta3.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `18 failed, 177 passed`:

- **the zoo's 12:**
  - the periodic task: 4, plus the edited `test_induction_windows`
  - every target: 3, plus the edited `test_induction_scores`
  - aligned sources: 3
- **the gates' 6:**
  - the helpers: 4
  - the report rules: 2

The settings test passes already, as a pin.

## Contract changes

**`src/nablaframe/zoo/tasks.py`** (replaces `InductionTask`'s bullet)

- "`InductionTask(vocab_size=64, window=64, min_period=8, max_period=32)`: vocab `t0` to `t63`. Each window repeats a block of `P` distinct tokens to fill `window` tokens, token `i` being `block[i % P]` (D36). `draw_window(rng, *, period=None)` draws one, in this order: `P = rng.integers(min_period, max_period + 1)`, skipped when `period` is given, then `block = rng.choice(vocab_size, size=P, replace=False)`. A `period` outside `min_period` to `max_period` raises ValueError. `batch_for_step` draws its windows in sequence from one generator. Mechanism positions `max_period` to `window - 2`; control positions `0` to `min_period - 2`; chance `1 / vocab_size`. ValueError unless `2 <= min_period <= max_period <= vocab_size` and `max_period <= window - 2`."

The method is `draw_window` because `window` is already the field holding the window's length.

**`src/nablaframe/zoo/induction.py`** (replaces the `induction_scores` and `uniform_baseline` bullets, and adds two; `induction_targets` is unchanged)

- "`induction_target_mask(windows, positions)`: bool (Batch, len(positions), Pos), true at every position `k + 1` with `k < p` and `windows[b, k] == windows[b, p]`. These are the positions following an earlier occurrence of the current token, so each holds the right next token (D36)."
- "`induction_scores(pattern, windows, positions)`: `pattern` is one layer's attention, (Batch, Head, QPos, KPos). Per head, the mean over (sequence, position) pairs with at least one target of the attention from that position, summed over all its targets, in float64. Shape (Head,). `windows` may be longer than the pattern, since no target lies after its position. ValueError when no pair has a target."
- "`uniform_baseline(windows, positions)`: what uniform causal attention would score on the same targets, the mean over the same pairs of the number of targets over `p + 1`. ValueError when no pair has a target."
- "`new_token_sources(windows, position, rng, vocab_size)`: a copy of `windows` as int32, with `position` replaced in each window, in order, by `rng.choice` of the ids below `vocab_size` that the window doesn't contain."

**`src/nablaframe/zoo/two_hop.py`** (adds a bullet)

- "`aligned_sources(task, targets, rng)`: `(sources, controls, counterparts)` for G4.3 (D36). For each target window, in order:
  - the counterpart `k` is the first subject in window order that is neither `a` nor `b`
  - `d′` is `rng.choice` of the entities the window doesn't contain
  - the source is the target with `a`'s object set to `k`, `k`'s object set to `d′` and the last token set to `d′`: it asks `a → k → d′` and differs in nothing else
  - the control is the target with only `k`'s object set to `d′`

  Sources and controls are int32, shaped like `targets`; `counterparts` is (Batch,). With them, `patch_outcomes` counts a swap as `k`'s object in the target and a transplant as `d′`. ValueError unless `targets` is (Batch, `task.window_length`). The targets are left unchanged."

**`src/nablaframe/reports.py`**

- "`PHASE4_GATE_IDS = ("G4.1", "G4.2", "G4.3", "G4.4", "G4.5")`. The validator accepts them, and a Phase 4 report must use schema version 2."
- Status rules:
  - "G4.1 pass requires:
    - `recompute_identical` true
    - `checkpoints_written == checkpoints_scheduled`
    - `events_recorded == events_expected`
    - `induction_mechanism_accuracy >= 0.9`
    - `induction_control_accuracy` and `induction_twin_accuracy` each `<= 2 * induction_chance`
    - `two_hop_accuracy >= 0.9`
    - `two_hop_broken_accuracy` and `two_hop_twin_accuracy` each `<= 2 * two_hop_chance`"
  - "G4.2 pass requires:
    - `best_head_attention >= 0.5`
    - `twin_best_attention <= 2 * uniform_baseline`
    - `ablation_fraction_removed >= 0.5`
    - `swap_fraction_removed >= 0.5`
    - `patch_after_block0_min >= 0.9`
    - `patch_after_block1_max <= 0.1`"
  - "G4.3 pass requires `control_unchanged_min >= 0.95`. Its swap and transplant profiles are recorded, not judged (D36)."
  - "G4.4's status is always `pass`: it records, not judges (D34)."
  - "G4.5 pass requires `builds_identical` true and `external_urls == 0`."
- Required measurements:
  - "G4.1: `recompute_identical`, `checkpoints_written`, `checkpoints_scheduled`, `events_recorded`, `events_expected`, `induction_mechanism_accuracy`, `induction_control_accuracy`, `induction_twin_accuracy`, `induction_chance`, `two_hop_accuracy`, `two_hop_broken_accuracy`, `two_hop_twin_accuracy`, `two_hop_chance`"
  - "G4.2: `mechanism_accuracy`, `chance`, `attention_by_head`, `best_head_attention`, `induction_heads`, `twin_best_attention`, `uniform_baseline`, `ablation_fraction_removed`, `fraction_removed_by_head`, `swap_step`, `swap_fraction_removed`, `patch_followed_by_tap`, `patch_after_block0_min`, `patch_after_block1_max`"
  - "G4.3: `unpatched_accuracy`, `swap_by_site`, `transplant_by_site`, `control_unchanged_by_site`, `control_unchanged_min`, `best_swap`"
  - "G4.4: `lens_ranks`, `jswap_by_layer`, `attribution`"
  - "G4.5: `formation_loss_step`, `formation_attention_step`, `builds_identical`, `external_urls`"

**`gates/phase4/`:** each stub's docstring is its contract.

## Spec edits

- **J's fit positions:** D17's valid positions for a sequence of 26 are 16 to 24, not 16 to 25. D17 leaves out the last position, which in the reference's setting has no target.
- **The tests list** gains this delta's.

## Threads

Paste into the Project, in two waves.

**Wave 1** has four threads, which share no files:

```
Start four threads now, without proposing them first. They share no files.

Thread "[P4] Periodic induction task", on Sonnet at high effort:
Implement the "src/nablaframe/zoo/tasks.py" part of "Contract changes" in docs/deltas/phase-4-delta-3.md, writing its bullet over InductionTask's in the module docstring verbatim. Change only src/nablaframe/zoo/tasks.py, and in it only InductionTask; TwoHopTask stays as it is. The draw order is pinned so G4.1 retraces the feasibility run's batches: don't reorder it. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_tasks.py tests/test_zoo_tasks_periodic.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Every induction target", on Sonnet at high effort:
Implement the "src/nablaframe/zoo/induction.py" part of "Contract changes" in docs/deltas/phase-4-delta-3.md, writing its bullets into the module docstring verbatim, in place of the two they replace. Change only src/nablaframe/zoo/induction.py; induction_targets stays as it is. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_induction.py tests/test_zoo_induction_targets.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Aligned two-hop sources", on Sonnet at high effort:
Implement the "src/nablaframe/zoo/two_hop.py" part of "Contract changes" in docs/deltas/phase-4-delta-3.md, adding its bullet to the module docstring verbatim. Change only src/nablaframe/zoo/two_hop.py, and leave source_for and patch_outcomes as they are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_two_hop.py tests/test_zoo_two_hop_rules.py tests/test_zoo_two_hop_aligned.py` passes with no skips, and every test that passed before still does.

Thread "[P4] Phase 4 report rules", on Sonnet at medium effort:
Implement the "src/nablaframe/reports.py" part of "Contract changes" in docs/deltas/phase-4-delta-3.md, written into the docstring's matching lists verbatim, following the Phase 3 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase4_reports.py tests/test_phase3_reports.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once all four have merged:

```
Start one thread now, without proposing it first.

Thread "[P4] Phase 4 gates", on Opus at high effort:
Implement every file in gates/phase4/ against its docstring, docs/specs/phase-4.md and D36. You may read and reuse gates/_common.py and the phase 1 to 3 _common.py modules, loaded by path as gates/phase3/_common.py loads its predecessors, but not change them. Change only files in gates/phase4/. Every intervened pass goes through run_intervened. Never run a gate for results and never write under reports/. Exercise each gate in a scratch folder with its settings shrunk (fewer steps, checkpoints and windows), and say in the PR what that covered. A full-size run here is fine to check the code, but its numbers aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

Once wave 2 has merged, on the Mac, in order, committing each report before the next gate:

```
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase4/g41_zoo_training.py
git add reports/phase4 && git commit -m "G4.1 report"
```

Then the same for `g42_induction.py`, `g43_two_hop.py`, `g44_lenses.py` and `g45_formation_viewer.py`, and finally `summarize.py`. G4.2 to G4.5 refuse to start until G4.1 passes.

## Feasibility, measured before the gates were written

All three rounds ran on the Mac, the reference backend, through the real trainer, interventions and batch sources.

**Induction, periodic task, round 3,** with mechanism accuracy, each induction head's attention, the heads zeroed together, the swap and the patch:

| Seed, batch | Mechanism | Heads' attention | Heads zeroed | Swap | Patch after block 0 / 1 |
| --- | --- | --- | --- | --- | --- |
| 0, 32 | 1.000 | 0.963 to 0.986, four heads | 1.001 | 0.929, from step 50 | 1.0 / 0.0 |
| 0, 128 | 1.000 | 0.979 to 0.985, three | 1.008 | 0.948, from step 0 | 0.996 to 1.0 / 0.0 |
| 2, 32 | 1.000 | 0.980 and 0.987, two | 1.011 | 0.901, from step 50 | 1.0 / 0.0 |
| 2, 128 | 1.000 | 0.977 to 0.986, three | 1.010 | 0.951, from step 0 | 1.0 / 0.0 |
| 1, 32 | 0.446 | none | not measured | 0.966 | 0.0 / up to 0.004 |
| 1, 128 | 0.522 | none | not measured | 0.966 | 0.0 / 0.0 |

- For every seed and batch: controls 0.003 to 0.016, twin 0.017, the twin's best head 0.048 against a uniform 0.046.
- For seeds 0 and 2, zeroing any one block-1 head removes at most 0.157.
- For seed 0 at batch 32, the loss drop and the attention rise both end at step 600.

**G4.1's induction run should retrace seed 0 at batch 32 exactly.** It uses the same batches, the evaluation and patch windows checked equal here, the same settings and the same Mac. So G4.2 should reproduce that row:

- attention by head `[0.985, 0.977, 0.963, 0.986]`
- the twin's best 0.0476 against a uniform 0.0458
- the heads together removing 1.001
- the swap from step 50 removing 0.929
- the patch `[1.0, 1.0, 1.0]` after block 0 and `[0.0, 0.0, 0.0]` after block 1

A mismatch means a gate computes something other than the feasibility run did.

**Two-hop, round 2,** at the spec's task and settings:

| Seed | Two-hop | Broken | Twin | Aligned controls, lowest | Best swap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.992 at step 7,000 | 0.0 | 0.008 | 0.984 | 0.855, query after block 2 |
| 1 | 0.984 at step 13,500 | 0.004 | 0.008 | 0.980 | 0.508, query after block 1 |

G4.1's two-hop run retraces seed 0's batches and learning-rate schedule through step 7,000, then trains on to 20,000. Expect these numbers only roughly, for two reasons:

- the gates draw their evaluation windows differently from that run
- G4.3 counts a control as unchanged when it matches the unpatched model's own answer, where that run compared it with `c`

## Verification record

**Verified** on a fresh clone of `main` (`20ce602`) with `apply_d36.py` applied:

- **On the stubs:** 18 fail and 177 pass. The settings test passes already, as a pin.
- **On a private reference implementation, not shipped:** all 195 pass. It covers the three zoo contracts, the report rules and the gate helpers.
- **The contract task and the gate helpers reproduce the feasibility run's training batches, evaluation windows and patch windows exactly.**
- **Planted bugs:** each of 35 is caught by the test aimed at it. They include:
  - the period drawn after the block, or never reaching 32
  - a fixed period that still draws one
  - attention to the most recent target only, or to the earlier occurrence itself
  - the old uniform baseline, or one over pairs without a target
  - the last distractor as the counterpart, or a `d′` the window already uses
  - a source that keeps the target's answer, or a control that moves `a`'s object too
  - G4.2 without its patch bar, or G4.3's control at 0.9
  - G4.4 allowed to fail
  - the swap checkpoint taken first rather than last, or strictly below twice chance

**Not verified:** anything on the Mac, and any gate.

**Known limit, for work package 7:** the views treat the last position as having no target. Two-hop's mechanism position, 25, is its last, so its views show no loss or rank there. The gates measure it from the full windows instead.
