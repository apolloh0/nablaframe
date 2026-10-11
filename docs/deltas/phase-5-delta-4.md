# Phase 5, delta 4: the gates G5.1 to G5.5

Date: 2026-10-10. Spec: [../specs/phase-5.md](../specs/phase-5.md). Rulings: D40 to D45.

This delta ships:

- **Stubs for G5.1 to G5.5 and their helpers,** in `gates/phase5/`. The helpers fix every setting at the spec's values, which are the feasibility runs'. Each stub's docstring is its contract.
- **11 tests** in two new files: the helpers, and the report rules.
- **Contract text for `reports.py`,** written into its docstring by the apply script.
- **Delta 4's tests,** listed in the spec's Tests section.

G5.6, the features view and the badges come in the next delta, with the view they build.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase5-delta4.zip
python3 apply_phase5_delta4.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect 10 of the new tests to fail: 8 for the helpers and 2 for the report rules. The settings test passes already, as a pin. The edge tests fail too until the edge-sparsity meter's PR merges.

## The gates

| Gate | Judged on | Pass |
| --- | --- | --- |
| G5.1 | Each kind trained twice from one key at short settings; 68 cache keys; `dpatch`'s two identities at every site of both zoo models | Byte-identical files; distinct keys; every identity bit for bit |
| G5.2 | Induction: meters, the faithful code (following and breaking), selective latents, against the twin | FVU at most 0.1; splices recover 90%; the code patch reaches 90% of the residual's rate; breaking keeps at most 0.05 more; mean best F1 at least 0.5 and twice the twin's |
| G5.3 | Two-hop: the same, the faithful code being G4.3's aligned patch at the query | As G5.2, with the control unchanged at least 95% of the time in place of breaking |
| G5.4 | The edge meter across every MLP, judged where the MLP's mean ablation costs 0.1 nats | Each judged trained pair needs at most a fifth of its twin's edges |
| G5.5 | The reference run's dictionaries, meters and the meter's curves | Recorded, not judged |

Every dictionary comes through one cache, keyed as D40 says, so no gate retrains what another trained. G5.1 checks that keys stay distinct before anything relies on them.

### Settled in the contracts

- **G5.1 trains at short settings,** 300 steps at batch 1,024. Determinism doesn't depend on length, and 300 steps cover the warmup, the hold and the decay. Its identity checks use these dictionaries too: D25's identities are a property of the operation, so any dictionary at the site tests them.
- **A TopK pair across an MLP is a `jsae` at `λ` 0.** Two TopK SAEs trained together on shared batches with no penalty: what feasibility trained as its `λ` 0 pairs, and the baseline the Jacobian SAE's `λ` 1 and 4 are recorded against.
- **The faithful code reuses Phase 4's patches.** G5.2 uses G4.2's windows and sources for following, and G5.3 uses G4.3's windows and aligned sources. So each residual rate equals the Phase 4 report's on the same Mac, a check anyone can make.
- **Breaking reads the unpatched pass's own prediction,** as G4.3 reads its control: the patch keeps the prediction when it doesn't change it.
- **G5.3 reads the best swap block from G4.3's report,** where the spec names block 2.
- **Two edge cases in the meter:** a twin whose curve never settles counts as needing every edge it scored, which can only overstate the ratio. A trained pair that never settles fails the bar.
- **The selectivity bar is a margin:** a trained mean best F1 minus twice the twin's, at least 0. A ratio would divide by a twin's F1 that can be 0.

## Contract changes

**`src/nablaframe/reports.py`** (written into the docstring by the apply script)

- "`PHASE5_GATE_IDS = ("G5.1", "G5.2", "G5.3", "G5.4", "G5.5")`. The validator accepts them, and a Phase 5 report must use schema version 2."
- "fail is allowed for any gate except G0.6, G2.5, G3.3, G3.4, G4.4 and G5.5"
- Status rules:
  - "G5.1 pass requires `files_identical` true, `cache_keys_distinct` true, `identities_checked >= 1` and `identities_exact == identities_checked`."
  - "G5.2 pass requires `fvu_max <= 0.1`, `recovered_min >= 0.9`, `faithful_ratio_min >= 0.9`, `break_excess_max <= 0.05`, `selective_f1_min >= 0.5` and `selective_margin_min >= 0`."
  - "G5.3 pass requires `fvu_max <= 0.1`, `recovered_min >= 0.9`, `faithful_ratio_min >= 0.9`, `control_unchanged_min >= 0.95`, `selective_f1_min >= 0.5` and `selective_margin_min >= 0`."
  - "G5.4 pass requires `edge_ratio_max <= 0.2`."
  - "G5.5's status is always `pass`: it records, not judges (D44)."
- Required measurements:
  - "G5.1: `files_identical`, `identical_by_kind`, `kinds_checked`, `cache_keys_checked`, `cache_keys_distinct`, `identities_checked`, `identities_exact`, `identities_by_site`"
  - "G5.2: `meters`, `fvu_max`, `recovered_min`, `faithful`, `faithful_ratio_min`, `break_excess_max`, `selectivity`, `selective_f1_min`, `selective_margin_min`, `recovery`, `robustness`, `train_seconds`"
  - "G5.3: `meters`, `fvu_max`, `recovered_min`, `faithful`, `faithful_ratio_min`, `control_unchanged_min`, `selectivity`, `selective_f1_min`, `selective_margin_min`, `best_swap_block`, `recovery`, `robustness`, `train_seconds`"
  - "G5.4: `edge_pairs`, `judged_mlps`, `edge_ratio_max`, `block_pairs`, `jsae`"
  - "G5.5: `meters`, `edge_pairs`, `train_seconds`"

A null where a bar applies fails it, as every earlier phase's rules do.

**`gates/phase5/`:** each stub's docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave 1** has two threads, which share no files:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P5] Phase 5 report rules", on Sonnet at medium effort:
Implement the Phase 5 parts of src/nablaframe/reports.py's docstring, which the delta's apply script wrote: PHASE5_GATE_IDS, Phase 5's status rules and its required measurements, following the Phase 4 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase5_reports.py tests/test_phase4_reports.py tests/test_reports.py` passes with no skips, and every test that passed before still does.

Thread "[P5] Phase 5 gate helpers", on Sonnet at high effort:
Implement the contract in gates/phase5/_common.py's docstring, against docs/specs/phase-5.md. Change only that file. It loads gates/phase4/_common.py by path, as that file loads its predecessors, and never changes it. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase5_gates.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once both have merged and so has "[P5] Edge-sparsity meter":

```
Start one thread now, without proposing it first.

Thread "[P5] Phase 5 gates", on Opus at high effort:
Implement gates/phase5/g51_determinism.py to g55_reference.py against their docstrings, docs/specs/phase-5.md and D40 to D45. You may add to gates/phase5/_common.py what several gates share, below its contract, without changing what the contract says; read the earlier phases' _common.py modules by path, but don't change them. Change only files in gates/phase5/. Every intervened pass goes through run_intervened, except the splice and the edge meter's passes, which dictionaries/measure.py and dictionaries/edges.py run. Every long step prints progress at least every 30 seconds. Never run a gate for results and never write under reports/. Exercise each gate in a scratch folder: make its prerequisites there with the Phase 1 and Phase 4 gates at shrunk settings, shrink its own settings (steps, windows, widths), and say in the PR what that covered and how long each gate would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

Once wave 2 has merged, on the Mac, in order, committing each report before the next:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase5/g51_determinism.py
git add reports/phase5 && git commit -m "G5.1 report"
```

Then the same for `g52_induction.py`, `g53_two_hop.py`, `g54_edges.py` and `g55_reference.py`. G5.2 to G5.5 refuse to start until G5.1 passes.

**Expect hours, not minutes.** Round 2's feasibility runs trained the same dictionaries at the same sizes: about 20 minutes for two-hop's four TopK SAEs together, 25 for its Staircase SAE, and 10 to 20 for each pair across an MLP. G5.3 trains its dictionaries on both models and two more seeds, and G5.4 trains every pair at three values of `λ`. Run them overnight, one at a time.

## Verification record

**Verified** from a clone of my mirror of `main` at delta 3:

- **On the stubs:** 8 helper tests and both report-rule tests fail, the settings pin passes, the 10 edge tests fail as they did before the edge thread merges, and the other 242 pass.
- **On a private reference implementation of the helpers and the report rules, not shipped:** all 11 pass, and so does every other test but the edge tests, which wait on their thread.
- **Planted bugs:** all 26 are caught by the test aimed at them. They include:
  - short settings ignored; a transcoder's taps swapped; a Staircase given a layer
  - robustness seeds sharing the training windows; held-out windows from the training seed; the reference run's held-out windows not cut to 64
  - `prev` and `next` swapped, or the mechanism's values taken position-major
  - breaking at `j` instead of `j - 1`, or drawn from following's generator
  - a cache key with the wrong code identity; training that drops the last short chunk
  - a cache that loads a dictionary under another key or saves one made under another
  - a heartbeat that keeps running after an exception
  - a cost or a rate that must exceed its bar rather than reach it; a margin against the twin once instead of twice; a twin that never settles read as None
  - G5.1 passing with no identity checked, G5.2 without its breaking bar, G5.3's control at 0.9, G5.4 at a quarter, G5.5 allowed to fail, G5.3 without `best_swap_block`, and Phase 5 reports allowed on schema 1

  Two more I planted change nothing, so no test can catch them: `jac_batch` left at its default, which is already 1,024, and a file count the other cache checks already imply.

**Not verified:** the gates themselves, which the wave 2 thread implements and exercises at shrunk sizes, and anything on the Mac.
