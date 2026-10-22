# Phase 6, delta 4: the command line and the gates G6.1 to G6.6

Date: 2026-10-21. Spec: [../specs/phase-6.md](../specs/phase-6.md). Rulings: D47 to D52.

This delta ships work package 5:

- **A stub for `gestures/cli.py`:** the command line, and the one path by which every gesture record is built. A request names what to measure, a `Context` resolves what it names, and `build` runs the gesture. The gates, the command line and G6.7's builds all go through it, so a record never depends on who built it.
- **Stubs for G6.1 to G6.6 and their helpers,** in `gates/phase6/`. The helpers fix every setting at the spec's values, which are feasibility round 3's, and hold the measurements several gates share.
- **Contract text** for `reports.py`, and two wordings settled in the docstrings of `gestures/generalize.py` and `gestures/open.py`. The apply script writes them, and lists delta 4's tests in the spec.
- **19 tests in three new files:** the command line (7), the gate helpers (10) and the report rules (2).

Each docstring is its contract. G6.7, the gestures panel and the badges come in delta 5.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase6-delta4.zip
python3 apply_phase6_delta4.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `17 failed, 320 passed`:

- the command line: 7 failed
- the gate helpers: 8 failed. The settings test and `test_ticking` pass already, as pins: the settings are constants, and the heartbeat is Phase 5's.
- the report rules: 2 failed

## The gates

| Gate | Judged on | Pass |
| --- | --- | --- |
| G6.1 | `dsteer`'s three identities at every TopK site of both zoo models; 64 requests; one record of each kind built twice and once through the command line | Every identity bit for bit; hashes distinct and stable; every record byte-identical |
| G6.2 | Every planted name, trained and twin: each value pointed at its best latent, and each value's set steered | The bars of the spec's names table |
| G6.3 | Every relation, trained (as the run's edge records) and twin | The bars of the edges table, on the hop and the copy |
| G6.4 | The three judged targets, each opened along the relation that plants it, trained and twin | The bars of the open-this-up table |
| G6.5 | Both judged names across slices, seeds and every checkpoint (as the run's generalize records), with the task's patch beside each checkpoint | The bars of the generalize-this table |
| G6.6 | The reference run: 256 latents named per block, `token(p+1)` values steered, their sets opened | Recorded, not judged |

## Settled in the contracts

D47 to D52 and the spec left these open. Each follows the spec's intent, round 3 or the nearest existing rule.

### The command line

- **One path for every record.** `cli.build` and `cli.record` run each kind's gesture function as merged, on readings it makes itself, so the gates, the command line and G6.7 can't build the same request two ways.
- **Read and open chunks are fixed in `cli.py`** (256 and 32). Open-this-up's results depend on its chunk at the 1e-7 level (PR #79), and a record's key doesn't hold it.
- **A record's code identity** is the blob hash of `cli.py`, `declare.py`, `sets.py` and `zoo/declarations.py`, and of the gesture's own modules (D18). The declarations decide what a description holds and what a run predicts, so they are part of what computed a record.
- **A request names one dictionary per site; the context supplies its seeds.** Generalize-this reads dictionary seeds 1 and 2 (D52), which a request can't name in its fixed form (D47). The context maps each seed-0 dictionary to its counterparts, and the record's inputs list them, so they are part of its key.
- **Generalize-this records carry no task extras.** The residual patch beside each checkpoint is task code (D29). If it went into the record, a record built by the command line would differ from the gate's under the same key. G6.5 measures it beside the record, in its report.
- **The command line is explicit about its files,** as the interventions command line is: the run's name picks its declaration, and the config, checkpoints, windows, dictionary folder, request and seeds are paths. Every sidecar must name the request's run.

### The gates

- **No gate trains a dictionary.** Each is the Phase 5 TopK SAE (or its twin's, or a robustness seed's) under the key Phase 5 computed, and a missing one stops the gate. Every gesture reads the Phase 5 held-out windows.
- **Records:** G6.3 writes the run's edge records and G6.5 its generalize records, through the record cache, and judges their results. So each judged edge and generalize-this is the record the panel will show. G6.2, G6.4 and G6.6 measure across a description's values, or per instance, which no single request covers; G6.7 builds their records. A twin isn't a checkpoint of its run, so it gets no records.
- **512 instances per name means per value** (D49 names a description's value). G6.2 steers each value with up to 512, as each name record does, and pools them, weighted by instances: the share over every instance. Round 3 drew 512 across a description's values, so the pool estimates the same rate from more instances.
- **Each value steers with a fresh generator from the seed,** so G6.2's steering for a value equals that of a name record pointed at a latent named for it, with the same seed.
- **G6.4's judged targets are read along the relation that plants them:** the carry, the hop and the copy. The target is the downstream place's set for each instance's own value, and the planted set is the upstream place's. `open_targets` measures them with `open.py`'s public functions, as delta 3 said the gate would, and leaves out the top parts' names, which open records carry.
- **One bar per target:** the reading's median over its bar, at most 1, so G6.4's three different bars are judged by one rule.
- **A judged reading that wasn't made fails its bar.** `least` and `greatest` give None where any value is None, and the report rules fail None, as every earlier phase's do.
- **G6.5's offset counts checkpoints,** not steps: two-hop's checkpoints are 1,000 steps apart and induction's 50 early on.
- **G6.6 follows the gestures' own definitions,** not round 3's where they differ: `name.steer` draws `w` from every value, not only named ones. Its readings needn't match round 3's.
- **G6.6 steers only the `token(p+1)` names,** as the spec says. That also settles the cost noted at PR #76: `predict` is asked of every option only where no prediction applies, so names without a prediction are never steered on the reference run.
- **G6.1's identities** are D48's three: nothing moved; at every position, the lowest latent whose code is zero removed; at every position, the latent with the largest code added at amount 0.

### Carried from the delta 3 review

- **`"values"` in a generalize record is a count.** The docstring now says so.
- **`evaluate` and `attribute` refuse what `open_up` refuses.** The docstring now says so; PR #79 already did.
- These are docstring edits, but they change both modules' blob hashes, so they come now, before any record exists (D18).
- **No guard against a zero decoder row.** Trained dictionaries renormalize every decoder row (D40), so a cosine can't be NaN on the zoo, and if one were, the record would refuse it.
- **PR #76's int64 counts stay.** The largest table the gates build, the reference run's 16,384 rows at width 2,048, takes about 270 MB per description while its F1 is computed.

## Contract changes

**`src/nablaframe/reports.py`** (written into the docstring by the apply script)

- "`PHASE6_GATE_IDS = ("G6.1", "G6.2", "G6.3", "G6.4", "G6.5", "G6.6")`. The validator accepts them, and a Phase 6 report must use schema version 2."
- "fail is allowed for any gate except G0.6, G2.5, G3.3, G3.4, G4.4, G5.5 and G6.6"
- Status rules:
  - "G6.1 pass requires `identities_checked >= 1`, `identities_exact == identities_checked`, `request_hashes_distinct` true, `request_hashes_stable` true and `records_identical` true."
  - "G6.2 pass requires `named_as_planted_min >= 0.7`, `follows_min >= 0.8`, `null_follows_max <= 0.05`, `twin_named_max <= 0.1` and `twin_follows_max <= 0.05`."
  - "G6.3 pass requires `follows_min >= 0.8`, `null_follows_max <= 0.1`, `output_follows_min >= 0.8`, `kept_max <= 0.4`, `null_kept_min >= 0.8`, `residual_alone_max <= 0.1`, `labels_computed` true and `twin_excess_max <= 0.1`."
  - "G6.4 pass requires `reading_ratio_max <= 1`, `within_min >= 0.8`, `planted_top_min >= 0.6`, `random_within_max <= 0.1` and `twin_planted_top_max <= 0.1`."
  - "G6.5 pass requires `slice_f1_min >= 0.5`, `slice_follows_min >= 0.8`, `seed_f1_min >= 0.5`, `seed_follows_min >= 0.8`, `step0_outside` true, `one_final_run` true and `start_offset_max <= 2`."
  - "G6.6's status is always `pass`: it records, not judges (D44)."
- Required measurements:
  - "G6.1: `identities_checked`, `identities_exact`, `identities_by_site`, `request_hashes_checked`, `request_hashes_distinct`, `request_hashes_stable`, `records_identical`, `records_by_kind`"
  - "G6.2: `names`, `named_as_planted_min`, `follows_min`, `null_follows_max`, `twin_named_max`, `twin_follows_max`"
  - "G6.3: `edges`, `follows_min`, `null_follows_max`, `output_follows_min`, `kept_max`, `null_kept_min`, `residual_alone_max`, `labels_computed`, `twin_excess_max`"
  - "G6.4: `targets`, `reading_ratio_max`, `within_min`, `planted_top_min`, `random_within_max`, `twin_planted_top_max`"
  - "G6.5: `names`, `slice_f1_min`, `slice_follows_min`, `seed_f1_min`, `seed_follows_min`, `step0_outside`, `one_final_run`, `start_offset_max`"
  - "G6.6: `blocks`, `seconds`"

A null where a bar applies fails it, as every earlier phase's rules do.

**`src/nablaframe/gestures/generalize.py`** (its docstring): `"values"` is "how many values `d` takes at `T`".

**`src/nablaframe/gestures/open.py`** (its docstring): `evaluate` raises ValueError, before anything runs, "for readings `open_up` refuses, no targets, a target whose window, position or latents fall outside the windows, the sequence or `down`'s active width, codes of another shape, or a `chunk` that isn't a positive int"; `attribute` "as `evaluate` raises it, codes aside".

**`src/nablaframe/gestures/cli.py`** and **`gates/phase6/`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave 1** has three threads, which share no files:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P6] Phase 6 report rules", on Sonnet at medium effort:
Implement the Phase 6 parts of src/nablaframe/reports.py's docstring, which the delta's apply script wrote: PHASE6_GATE_IDS, Phase 6's status rules and its required measurements, following the Phase 5 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase6_reports.py tests/test_phase5_reports.py tests/test_reports.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Gesture command line", on Opus at high effort:
Implement src/nablaframe/gestures/cli.py against its docstring and docs/specs/phase-6.md (D47). Change only that file. Each gesture runs through its own module as merged, every site is read with sets.read_site at READ_CHUNK, and nothing specific to a task goes into it beyond declaration_for's choice of declaration (D29). Every refusal happens before anything runs, and a command that fails writes nothing. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_cli.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Phase 6 gate helpers", on Opus at high effort:
Implement the contract in gates/phase6/_common.py's docstring, against docs/specs/phase-6.md. Change only that file. It loads gates/phase5/_common.py by path, as that file loads its predecessors, and never changes it. Open-this-up's passes go through gestures/open.py's public functions, and every other intervened pass through run_intervened. Draw from every rng exactly as the docstring says. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase6_gates.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once all three have merged:

```
Start one thread now, without proposing it first.

Thread "[P6] Phase 6 gates", on Opus at high effort:
Implement gates/phase6/g61_determinism.py to g66_reference.py against their docstrings, docs/specs/phase-6.md and D47 to D52. You may add to gates/phase6/_common.py what several gates share, below its contract, without changing what the contract says; read the earlier phases' _common.py modules by path, but don't change them. Change only files in gates/phase6/. No gate trains a dictionary: each comes from the Phase 5 cache, and a missing one stops the gate. Every record goes through gestures/cli.py, and every intervened pass through run_intervened, except open-this-up's, which gestures/open.py runs. Every long step prints progress at least every 30 seconds. Never run a gate for results and never write under reports/. Exercise each gate in a scratch folder: make its prerequisites there with the Phase 1, 4 and 5 gates at shrunk settings, shrink its own settings (windows, instances, checkpoints), and say in the PR what that covered and how long each gate would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

Once wave 2 has merged, on the Mac, in order, committing each report before the next:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g61_determinism.py
git add reports/phase6/g61_determinism.json
git commit -m "G6.1 report"
```

Then the same for `g62_names.py`, `g63_edges.py`, `g64_open.py`, `g65_generalize.py` and `g66_reference.py`. G6.2 to G6.6 refuse to start until G6.1 passes. G6.3 and G6.5 also write the zoo runs' edge and generalize records under `runs/phase6/gestures/`, outside git.

Their time at full size is unmeasured. G6.5 reads every checkpoint of both zoo runs, 21 and 41, and is likely the longest. The wave 2 thread's PR estimates each.

## Verification record

**Verified** on clones of my mirror of `main` at `4659d91`, with this delta's files added and the apply script run:

- **On the stubs:** each of the 17 new tests fails on its own. The settings pin and `test_ticking` pass, and so do all 318 existing tests.
- **On a private reference implementation of the command line, the helpers and the report rules, not shipped:** all 337 pass with no skips. The three new files take about five minutes on a cloud CPU, most of it the command line's builds of generalize-this.
- **`open_targets` equals `open_up`,** reading for reading, wherever every instance opens the same target.
- **Planted bugs:** all 74 are caught by the tests aimed at them. They include:
  - **the command line:**
    - two-hop's and induction's declarations swapped, the reference read at 256 tokens, or a declaration's length unchecked
    - a module left out of a record's code, or only the gesture's own
    - generalize-this keyed by one state, or without its seeds' dictionaries; windows hashed as tokens
    - a name at another seed; an edge read backwards; open-this-up at the read chunk, or its name without its value
    - generalize-this without its seeds, a seed read with seed 0's dictionary, an edge's seeds swapped, or a checkpoint dropped
    - another kind's code; a record never answered from the cache
    - another run, a step that isn't a checkpoint, a missing dictionary or a seed without a counterpart accepted
    - dictionaries searched at the top level only, or one held twice accepted; the key not printed; `--seeds` ignored
  - **the gate helpers:**
    - `least` skipping a None, `greatest` giving the least, the ratio inverted, the twin's excess reversed
    - steering pooled unweighted, or with non-causal steerings in it
    - a planted rank by latent alone, or the top 8 taken as ranks up to 8
    - step 0 read as the first step, any run reaching the end taken as the envelope, the offset counted in steps
    - the largest code taken among shared latents, or ties going to the higher latent
    - names taken strictly above the floor, or as planted by description alone
    - one generator for every value's steering, or each value seeded apart
    - the effect's ends swapped
    - relation candidates whose sets never fire, planted at the downstream position, at the wrong block, or across readings that don't share a model
    - the cut unsorted, shuffles drawn before it, no span kept, the planted rank in signed order, a mixed planted set accepted
    - the reference sample unsorted or with dead latents, its values strictly above the floor
    - identity edits removing the highest zero latent, adding a nonzero amount, or at one position
    - requests nested in another order, the open name at another value, repeated hashes unnoticed
  - **the report rules:** G6.6 allowed to fail; Phase 6 reports on schema 1; G6.1 without stable hashes or `records_by_kind`; G6.2's null at 0.1; G6.3 without its label; G6.4's ratio strictly below 1, or without the twin; G6.5 three checkpoints apart, or without step 0

  Two more changes were set aside before the run, as equivalent on the toy models: one of the set's latents' coverage in place of the set's (a token's four latents fire together), and a reading at an error strictly below 0.1 (no error lands exactly on it).

**Not verified:**

- the gates themselves, which the wave 2 thread implements and exercises at shrunk sizes
- anything on the Mac
- the time and memory of each gate at full size
