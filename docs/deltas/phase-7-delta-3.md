# Phase 7, delta 3: crystallize-this's records, the command line and the gates G7.1 to G7.4

Date: 2026-10-24. Spec: [../specs/phase-7.md](../specs/phase-7.md). Rulings: D54 to D58.

This delta ships the rest of work package 3 and all of work package 4:

- **Contract text for three existing modules,** which the apply script writes into their docstrings:
  - `gestures/records.py`: the crystallize-this request, a new `hashes` argument type for the records it cites, and the twin's state among a record's inputs
  - `gestures/cli.py`: the modular-addition run, every run's functions, the context crystallize-this needs, and the command line's new arguments
  - `reports.py`: the Phase 7 report rules
- **Stubs for G7.1 to G7.4 and their helpers,** in `gates/phase7/`. The helpers fix every setting at the spec's values, which are D54's and feasibility's, and hold the measurements several gates share.
- **17 tests in four new files:** the records (2), the command line (6), the gate helpers (7) and the report rules (2).
- **Explicit edits to two existing tests,** which pinned the four gesture kinds and the three runs that the contracts now extend.

Each docstring is its contract. G7.5, the gestures panel's crystallize-this, the badges and modular addition's views come in delta 4.

## Apply

Claude applied this delta on a fresh clone of `main`. Every file it brings is new; they were copied in, and then the apply script made the edits listed under "Contract changes". The script refuses to run twice. From the repo root:

```
python3 apply_phase7_delta3.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `18 failed, 363 passed`:

- the records: the 2 new tests, and `test_requests_round_trip_and_hash`, which now expects the fifth kind
- the command line: the 6 new tests, and `test_declarations_for_runs`, which now expects the fourth run
- the gate helpers: 6 failed. The settings test passes already, as a pin: the settings are constants.
- the report rules: 2 failed

## The gates

| Gate | Judged on | Pass |
| --- | --- | --- |
| G7.1 | The specimen trained from step 0, traced at all 45 checkpoints on its probe, and retrained; its behaviour from logits, and its clock from its weights and logits | Every bar of the spec's specimen table |
| G7.2 | Its TopK SAE, trained and twin; the names at `answer`; a name record for each value's best latent | Recorded, not judged |
| G7.3 | `set`'s identities at every residual-shaped tap of the three zoo models; 64 requests; each form's record built twice and once through the command line | Every identity bit for bit; hashes distinct and stable; every record byte-identical |
| G7.4 | Every judged and recorded crystallization, as the run's records, with its twin, its counterfactual and its envelope | The bars of the spec's crystallizations table |

## Settled in the contracts

D54 to D58 and the spec left these open. Each follows the spec's intent, delta 2's contracts, or the nearest Phase 6 rule.

### Records

- **A request cites by key.** `cites` holds the keys of the name and edge records a crystallization cites, so the record's key covers them (D11), as delta 2 said it would. They are stored sorted, so citing the same records in another order is the same request.
- **The twin's state is an input.** A crystallize-this record reads the twin's params (D30), so its state hash joins the checkpoints' under `"twin"`. Every crystallize-this record holds one, and no other kind's may.
- **Every checkpoint is an input.** The envelope fits the form again at each one, so a record built over other checkpoints is another record. Generalize-this does the same.
- **No dictionaries; two window sets.** crystallize-this reads no site, so its `dictionaries` is empty; its cited records name theirs, under their own keys. Its windows are the fitting and the held-out windows.
- **`instances` is the held-out windows' count.** crystallize-this reads every held-out row, so a request saying anything else is refused.

### The command line

- **One path for every record,** as in Phase 6. `cli.build` and `cli.record` call `crystallize.crystallize` as merged, with the context's every checkpoint, its twin and the records it cites.
- **The context gains five fields, each None by default:** the fitting windows, the run's functions, the twin, its state and `cited`. Phase 6's contexts are unchanged, and a crystallize-this request is refused before anything runs without them.
- **The code identity** adds `gestures/crystallize.py` and `zoo/functions.py` to the four every record carries. A run's functions decide its keys, its wrong functions and its counterfactuals.
- **The command line resolves a zoo run's functions and twin itself:** `functions_for(run)`, the task at its defaults, and `init_params(config, seed=1)` (D30). The reference run declares none, so crystallize-this is refused there.
- **Its new arguments:**
  - `--fit-windows`: the fitting windows
  - `--records`, repeatable: the folders the cited records are read from, after the out folder
  - `--dictionaries` becomes optional, since crystallize-this reads none

### The gates

- **G7.1 trains as G4.1 trains,** with the task's own batches, and saves and traces the run the same way.
  - Its clock counts frequencies with no cut at 8. `crystallize.frequencies` stops at 8, so a bar of "at most 8" judged on it could never fail: the trap D58 closed for wrong functions.
  - The twin's count, 39 in feasibility, is recorded beside it.
- **G7.1's R² fits on the training pairs and is judged on the held-out pairs,** as a crystallization is.
  - Each pair's logits are centred over the numbers, since a constant shift changes no prediction.
  - The frequencies are the ones the fourier form keeps, `crystallize.frequencies`'s.
- **G7.2 writes the name records G7.4 cites:** each value's best latent by F1, for `token(p+1)`, `token(p−2)` and `token(p−1)` at `answer`. These are the ones the spec has G7.5 build.
  - Its TopK SAE has Phase 5's settings. `p5.settings` takes only Phase 5's runs, so G7.2 builds them itself.
  - As in Phase 6, the twin gets no records.
- **G7.3 compares `set`'s identities with every tap recorded,** as G3.1 does. On the d32 specimen, a pass that records fewer taps can differ in the last bits: XLA fuses the program differently (the PR #90 review).
- **G7.3's records read two checkpoints,** step 0 and the final one. It builds each form three times, which over every checkpoint would take hours. G7.4's records read every checkpoint.
- **G7.3's requests:** the four modular-addition crystallizations, each at the first and last step, citing nothing or two records, with two seeds and at two blocks.
  - That makes 64 requests.
  - Block 1 isn't the function's own block, but a request is only hashed here.
- **G7.4 builds every crystallization as a run's record,** through the record cache. So the record G7.4 judges is the one the panel will show.
- **G7.4 cites what Phase 6 recorded,** read by key from `runs/phase6/gestures/<run>/`:
  - the name records G6.7 built, and the edge records G6.3 built, chosen by `crystallize.cite`
  - for modular addition, G7.2's records
  - so G7.4 also needs G6.3's and G6.7's reports to pass
- **The judgements:**
  - **On the twin:** the greatest distance, either way, between a wrong function's KL and the right one's.
  - **The envelope:** it holds at the final checkpoint, and step 0 is read and outside it, as the spec's bar says.
  - **The counterfactual's agreements:** taken over the crystallizations that have one. Modular addition's write has none.

## For delta 4: Phase 6's record keys

Every gesture record's code identity includes `zoo/declarations.py` and `gestures/cli.py` (D18). Delta 1 changed the first, for modular addition's declaration, and this delta changes the second. So every Phase 6 record built today gets a new key.

The records already written keep their keys, and anything that reads them by key still works. G7.4's citations read them that way.

The conflict is the spec's "G6.7 runs again after the badge correction":

- G6.7 stops when a record folder holds a key the current code doesn't produce.
- So its rerun would stop on the folders it wrote itself.

This needs a ruling before delta 4. The two ways out:

- **G7.5 writes the corrected badges file,** with every phase's badges, as the spec already has it do, and G6.7 doesn't run again. Phase 6's records and report stay as they are. This is my recommendation.
- **Rebuild Phase 6's records** under the current code. G6.3 and G6.5 would have to run again first, since G6.7 checks their reports' record keys, and G7.4's citations would then point at the old records.

## Contract changes

**`src/nablaframe/gestures/records.py`** (its docstring):

- "``KINDS = ("name", "edge", "open", "generalize", "crystallize")``"
- The form: "``crystallize``: ``block`` (int), ``part`` (name), ``class`` (name), ``function`` (name), ``form`` (name) and ``cites`` (hashes), the keys of the records it cites (D56, D57)"
- The new type: "Hashes are a list or tuple of distinct hashes, possibly empty, stored as a sorted read-only list, so requests listing the same hashes in another order are equal."
- The inputs: "a crystallize-this record's ``states`` must also map ``"twin"`` to the state hash of its run's twin (D30), and no other kind's may"

**`src/nablaframe/gestures/cli.py`** (its docstring):

- `RUNS` gains `"modular_addition"`, whose declaration is `zoo.declarations.modular_addition(ModularAdditionTask())`.
- `functions_for(run)`: `zoo.functions.for_run` of a zoo run and the same task. It raises ValueError for `"reference"`, which declares no functions.
- `Context` gains `fit_windows`, `functions`, `twin`, `twin_state` and `cited(key)`, each None by default.
- `MODULES["crystallize"]` adds `nablaframe.gestures.crystallize` and `nablaframe.zoo.functions`.
- **Inputs:** every step and `"twin"` in `states`; no dictionaries; both window sets' batch hashes, sorted, with no repeats.
- **Results:** `crystallize.crystallize(...)` with the request's arguments, the context's fitting and held-out windows, its twin, every checkpoint, and `cited` of each key.
- **Refusals,** before anything runs:
  - a context without the five fields, or fitting windows of another width
  - a function the run doesn't declare
  - `instances` other than the held-out windows' count
  - a cited key the context doesn't hold, or another run's record
- **The command line:**
  - `--dictionaries` is optional
  - `--fit-windows` gives the fitting windows
  - `--records`, repeatable, gives the folders the cited records are read from, the out folder first
  - for a zoo run, it resolves the functions, the twin and the twin's state itself

**`src/nablaframe/reports.py`** (its docstring):

- "`PHASE7_GATE_IDS = ("G7.1", "G7.2", "G7.3", "G7.4")`. The validator accepts them, and a Phase 7 report must use schema version 2."
- "fail is allowed for any gate except G0.6, G2.5, G3.3, G3.4, G4.4, G5.5, G6.6 and G7.2"
- Status rules:
  - G7.1 pass requires:
    - `recompute_identical` true
    - `checkpoints_written == checkpoints_scheduled` and `events_recorded == events_expected`
    - `held_accuracy >= 0.99`
    - `control_accuracy` and `twin_accuracy` each `<= 2 * chance`
    - `frequencies <= 8` and `logit_r2 >= 0.8`
  - "G7.2's status is always `pass`: it records, not judges (C11)."
  - "G7.3 pass requires `identities_checked >= 1`, `identities_exact == identities_checked`, `request_hashes_distinct` true, `request_hashes_stable` true and `records_identical` true."
  - G7.4 pass requires:
    - `kl_max <= 0.25`, `ablation_min >= 1` and `wrong_min >= 1`
    - `reads_agreement_min >= 0.8` and `answer_agreement_max <= 0.1`
    - `twin_within_max <= 0.05`
    - `envelope_ok` true
- Required measurements:
  - "G7.1: `recompute_identical`, `checkpoints_written`, `checkpoints_scheduled`, `events_recorded`, `events_expected`, `held_accuracy`, `train_accuracy`, `control_accuracy`, `twin_accuracy`, `chance`, `frequencies`, `energy`, `twin_frequencies`, `logit_r2`"
  - "G7.2: `fvu`, `twin_fvu`, `names`, `records`, `seconds`"
  - "G7.3: `identities_checked`, `identities_exact`, `identities_by_run`, `request_hashes_checked`, `request_hashes_distinct`, `request_hashes_stable`, `records_identical`, `records_by_form`"
  - "G7.4: `crystallizations`, `kl_max`, `ablation_min`, `wrong_min`, `reads_agreement_min`, `answer_agreement_max`, `twin_within_max`, `envelope_ok`"

A null where a bar applies fails it, as every earlier phase's rules do.

**`docs/specs/phase-7.md`:** delta 3's tests are listed in full.

**`tests/test_gestures_records.py`:**

- `KINDS` gains crystallize, and the arguments every form takes gain its own
- the unknown kind it refuses becomes `"steer"`

**`tests/test_gestures_cli.py`:**

- `RUNS` gains modular addition
- the unknown run becomes `"addition"`, and the unknown kind `"steer"`

**`gates/phase7/`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave 1** has three threads, which share no files:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P7] Phase 7 report rules", on Sonnet at medium effort:
Implement the Phase 7 parts of src/nablaframe/reports.py's docstring, which the delta's apply script wrote: PHASE7_GATE_IDS, Phase 7's status rules and its required measurements, following the Phase 6 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase7_reports.py tests/test_phase6_reports.py tests/test_reports.py` passes with no skips, and every test that passed before still does.

Thread "[P7] crystallize-this records and command line", on Opus at high effort:
Implement the crystallize-this parts of the docstrings of src/nablaframe/gestures/records.py and src/nablaframe/gestures/cli.py, which the delta's apply script wrote, against docs/specs/phase-7.md (D47, D56). Change only those two files. crystallize-this runs through gestures/crystallize.py as merged, called as crystallize.crystallize. Every refusal happens before anything runs, and a command that fails writes nothing. Records of the other four kinds keep exactly the inputs and results they had. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_crystallize_records.py tests/test_crystallize_cli.py tests/test_gestures_records.py tests/test_gestures_cli.py` passes with no skips, and every test that passed before still does.

Thread "[P7] Phase 7 gate helpers", on Opus at high effort:
Implement the contract in gates/phase7/_common.py's docstring, against docs/specs/phase-7.md (D54 to D58). Change only that file. It loads gates/phase6/_common.py by path, as that file loads its predecessors, and never changes it. Every intervened pass goes through run_intervened, and set_identities records every canonical tap in both passes, as G3.1's do. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase7_gates.py` passes with no skips, and every test that passed before still does.
```

**Wave 2,** once all three have merged:

```
Start one thread now, without proposing it first.

Thread "[P7] Phase 7 gates", on Opus at high effort:
Implement gates/phase7/g71_specimen.py to g74_crystallizations.py against their docstrings, docs/specs/phase-7.md and D54 to D58. You may add to gates/phase7/_common.py what several gates share, below its contract, without changing what the contract says. Read the earlier phases' _common.py modules and gate scripts by path, but don't change them. Change only files in gates/phase7/. Every record goes through gestures/cli.py, and every intervened pass through run_intervened. Every long step prints progress at least every 30 seconds, and each crystallization prints a line per checkpoint: the context's params_at is called once for each. Never run a gate for results and never write under reports/. Exercise each gate in a scratch folder: make its prerequisites there at shrunk settings, shrink its own settings (steps, windows, checkpoints, instances), and say in the PR what that covered and how long each gate would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

Once wave 2 has merged, on the Mac, in order, committing each report before the next:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g71_specimen.py
git add reports/phase7/g71_specimen.json
git commit -m "G7.1 report"
```

Then the same for `g72_names.py`, `g73_determinism.py` and `g74_crystallizations.py`. Each refuses to start until the gates before it pass. G7.2 and G7.4 write records under `runs/phase7/gestures/`, outside git.

Their time at full size is unmeasured. G7.4 is likely the longest. With its twin, one modular-addition crystallization over 45 checkpoints took about 20 minutes on a cloud CPU, and G7.4 has four of them, beside the zoo's three over 41 and 21 checkpoints. The wave 2 thread's PR estimates each gate.

## Verification record

**Verified** on clones of my mirror of `main` at `169f2ad`, with this delta's files added and the apply script run:

- **On the stubs:** the 18 tests above fail, and all 363 others pass.
- **On a private reference implementation, not shipped:** all 381 pass with no skips, its docstrings checked equal to the delta's.
  - The new tests take about two and a half minutes on a cloud CPU.
  - Most of that is the command line's crystallizations, each with its twin and two checkpoints.
- **Planted bugs:** all 90 are caught by the tests aimed at them. They include:
  - **records:**
    - cites kept in the order given, writable, repeated, or of any hex case
    - the twin's state allowed on any kind, missing from a crystallization's, or standing alone
    - a request's JSON holding read-only lists
    - the key checking every record's inputs as a crystallization's
  - **the command line:**
    - no modular-addition run, its task at p = 11, or its declaration induction's
    - a record's code without `zoo/functions.py`, or without the four modules every record carries
    - inputs:
      - only the request's step
      - no twin, or a checkpoint's state for the twin
      - only the held-out windows, or the windows repeated
    - the crystallization run:
      - at the final checkpoint whatever the request's step
      - without its twin, its checkpoints or its citations, or citing nothing
      - fitted on the held-out windows
      - at seed 0, always at the write, or always as a table
    - refusals left out:
      - a context without its twin or the twin's state
      - fitting windows of another width
      - an undeclared function, or any instances
      - another run's cited record, or cites checked only once results run
    - the command line itself:
      - the twin at seed 0, or the twin's state the template's
      - `--records` or the out folder unsearched, or `--fit-windows` ignored
      - `--dictionaries` still required
      - a key read from the first folder with an index, whether or not it holds the key
  - **the report rules:**
    - G7.2 allowed to fail, and Phase 7 reports on schema 1
    - G7.1 without its recompute, its twin or `energy`; held-out accuracy at 0.9, its control at three times chance, nine frequencies, or an R² of 0.7
    - G7.3 with nothing checked, or without stable hashes
    - G7.4 with a KL of 0.3 or an answer agreement of 0.2, or without its wrong functions, its twin or its envelope
  - **the gate helpers:**
    - the checkpoints stopping at 2,900; the probe with replacement or sorted; the default weight decay
    - frequencies cut at 8, ranked smallest first, without their sines, or held to 0.8 instead of 0.75
    - logits not centred, read at every token, or with `a` and `b` taken from positions 1 and 2; α fitted on the held-out pairs
    - `set` tried at a head's tap; identities judged by the logits alone or the taps alone; passes recording no taps
    - requests nested in another order, at 8,937 instances, never citing, or with the fourier base first
    - the twin's distance signed; the greatest wrong function
    - the final label read off the first row; step 0 outside when unread, or taken as the first row
    - the agreements swapped, or taken over every crystallization; checks without checkpoints accepted
    - the envelope judged by any; the least KL; the twin's least distance; no checks accepted

  One more change was set aside as equivalent: accepting an empty `k` in `logit_r2`. The reference's design matrix can't be stacked from no frequencies, so the call still raises ValueError.

  **Not covered:**
  - whether a share of exactly 0.75 is enough: shares built to sum to 0.75 land a rounding either side of it.
  - that `set_identities` writes every position. A `set` of a tap's own values is exact at any positions, so on these models a set at fewer positions gives the same answer. The spy checks which taps are written and recorded, not where.

**Not verified:**

- anything on the Mac
- the gates themselves, which the wave 2 thread implements and exercises at shrunk sizes
- the time and memory of each gate at full size
