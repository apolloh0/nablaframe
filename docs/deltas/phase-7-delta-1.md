# Phase 7, delta 1: modular addition, what each run offers crystallize-this, and `set`

Date: 2026-10-23. Spec: [../specs/phase-7.md](../specs/phase-7.md). Rulings: D54, D55, D56 and D57, and the spec's "What each run declares".

This delta ships:

- **A stub for a new module,** `zoo/functions.py`: each run's keys, functions and counterfactuals, the declarations crystallize-this will read. Its docstring is its contract.
- **Contract text and stubs for `zoo/tasks.py` and `zoo/declarations.py`,** which gain `ModularAdditionTask` and `modular_addition`. The apply script writes both into the docstrings and adds each as a stub at the end of its file.
- **Contract text for `interventions/spec.py` and `apply.py`,** which gain `set`. The apply script writes it into both docstrings.
- **9 tests** in three new files, the delta 1 tests the spec lists.

These are the spec's work packages 1 and 2. crystallize-this itself, the gates and the viewer come in later deltas.

## Apply

Every file in the zip is new. The apply script makes the explicit edits below to four existing files, and refuses to run twice. From the repo root:

```
unzip -n nablaframe-phase7-delta1.zip
python3 apply_phase7_delta1.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `9 failed, 343 passed`:

- the task and its declaration: 3 failed
- the functions: 2 failed
- `set`: 4 failed

## Contract changes

**`src/nablaframe/zoo/tasks.py`** (added to the module docstring; a stub class at the end)

- "`ModularAdditionTask(p=113, train_fraction=0.3, split_seed=0)` (D54): vocab `"0"` to `str(p - 1)`, then `"="`, whose id is `p`. A window is `a b = c`, four tokens with `c = (a + b) mod p`, so `window` is 4 and `seq` 3. Mechanism position 2, where the `=` predicts `c`; control position 0, which predicts `b` from `a`; chance `1 / p`."
  - "`pairs`: int64 (p * p, 2), row `i` being `(i // p, i % p)`."
  - "`split`: `(train, held)`, int64 arrays of pair indices, each ascending. With `order = numpy.random.default_rng(split_seed).permutation(p * p)` and `n = round(train_fraction * p * p)`, `train` holds `order[:n]` and `held` the rest."
  - "`windows(index)`: int32 (len(index), 4), the pairs `index` names, in that order, each `[a, b, p, (a + b) % p]`. ValueError for an index outside `range(p * p)` or one that isn't integers. `training()` and `held_out()`: `windows(train)` and `windows(held)`."
  - "`batch_for_step(seed, step, batch)`: `windows(rng.choice(train, size=batch, replace=True))`, `rng` being `numpy.random.default_rng([seed, step])`."
  - "ValueError unless `p` is an int of at least 3, `train_fraction` is a float or int that leaves both parts non-empty, and `split_seed` is a non-negative int. A bool is not an int."

**`src/nablaframe/zoo/declarations.py`** (added to the module docstring; a stub function at the end)

- "`modular_addition(task)`, for a `ModularAdditionTask` (D54, D57), `seq` being `task.seq`:"
  - "`run` `"modular_addition"`. Classes: `"answer"` (2) and `"every"` (0 to 2). No variables."
  - "Planted: `token(p+1)` after block 0 at `answer`, recorded, never judged: Phase 6's names don't hold the sum (C11)."
  - "`predict`: `token(p+1)` at 2 moved to `w` gives `(2, w)`. Anything else gives None."
  - "No relations: `pairs` and `value_map` raise ValueError, naming the relation, as the reference run's do."
- Its closing sentence becomes: "Every declaration here checks its arguments as `gestures.declare` says every declaration does."

**`src/nablaframe/interventions/spec.py`** (added to the module docstring)

- "`OPERATIONS` gains `set` (D55), taking `values`, a content hash: the `array_hash` of the values it writes."

**`src/nablaframe/interventions/apply.py`** (added to the module docstring)

- "`set` (D55) applies at every editable tap whose values are shaped (batch, pos, d_model): every one but `attn.hook_z`, which raises ValueError before anything runs. `resolve` returns its `values`, checked with `array_hash`: a float32 array shaped (batch, the number of positions the edit selects, d_model), `d_model` being `params.wte`'s second dimension, with every value finite. Anything else raises ValueError before anything runs. At the `i`-th position the edit selects, in `position_index`'s order, each sequence's value at the tap becomes `values[:, i]`. Its operand enters the compiled program as an argument. Where the edit selects no positions it adds no operation at all, and values equal to the tap's own values on the unintervened pass reproduce that pass bit for bit (D25)."

**`src/nablaframe/zoo/functions.py`:** its docstring is its contract.

### Settled in the contracts

D54 to D57 and the spec left these open. Each follows the nearest existing rule:

- **A key reads only what the model has read.** The key at position `p` depends on tokens 0 to `p`, as a crystallized component can see no further. So:
  - **Induction's copied token** takes the period as far as `q` has read it: the first recurrence of `w[0]` at or before `q`, and -1 where there is none.
  - **Two-hop's hop** is read from the facts and the query's subject, `b`'s object, not from the window's last token, which the model never reads. On the task's own windows the two agree.
- **A function names its block.** What it reads depends on the block it replaces: two-hop's hop reads `b` after block 2 because it replaces block 3.
- **The counterfactuals' draws are pinned,** so a record rebuilds byte for byte (D47):
  - induction's broken position and token, window by window
  - two-hop's aligned sources, G4.3's function
  - modular addition's sources, redrawn round by round until no window shares its source's sum
- **Wrong functions are keys,** fitted alone in the same form, so a run declares them beside its functions.
- **`set`:**
  - **Its values hold only the selected positions,** in `position_index`'s order, like `dsteer`'s rows, so a crystallization at one position doesn't carry a whole sequence of values.
  - **It refuses `attn.hook_z`,** whose axes are per head, and the attention pattern isn't editable (D23).
  - **At no positions it adds no operation,** as `dpatch` and `dsteer` don't when their change is zero.
- **The modular-addition task:**
  - **`=` is token `p`,** after the numbers, so a number's id is its value.
  - **Position 0 is its control:** it predicts `b` from `a`, which nothing can.
  - **The split rounds with Python's `round`:** 3,831 of 12,769 pairs train.
- **`for_run` checks each function against its run's declaration:** its class, what it reads, and its relation.

## Threads

Paste into the Project, in one wave. Its two threads share no files and depend on nothing in each other:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P7] set", on Opus at high effort:
Implement the set bullets in the module docstrings of src/nablaframe/interventions/spec.py and src/nablaframe/interventions/apply.py, quoted under "Contract changes" in docs/deltas/phase-7-delta-1.md, against docs/specs/phase-7.md (D55). Change only those two files. The values enter the compiled program as an argument, as PR #38 established, and every refusal happens before anything runs. The identities are bit for bit; if one holds only approximately, stop and say so in the PR. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_set.py tests/test_dsteer.py tests/test_dpatch.py tests/test_interventions_apply.py tests/test_interventions_spec.py tests/test_jswap.py` passes with no skips, and every test that passed before still does.

Thread "[P7] Modular addition and functions", on Sonnet at high effort:
Implement ModularAdditionTask in src/nablaframe/zoo/tasks.py, modular_addition in src/nablaframe/zoo/declarations.py, and src/nablaframe/zoo/functions.py, against their docstrings and docs/specs/phase-7.md (D54, D56, D57, "What each run declares"). Replace the two stubs the apply script added; change only those three files. Every key reads only the tokens up to its own position, and every counterfactual draws exactly as its contract says, so records rebuild byte for byte. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_zoo_modadd.py tests/test_zoo_functions.py tests/test_zoo_tasks.py tests/test_gestures_declare.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** on clones of my mirror of `main` at `e08fe63`, with this delta's files added and the apply script run:

- **On the stubs:** the 9 new tests fail, and all 343 existing tests pass.
- **On a private reference implementation, not shipped:** all 352 pass with no skips, its docstrings checked equal to the delta's. The full lane takes about ten minutes on a cloud CPU.
- **The reference task is feasibility's.** Its split, held-out windows and batches of 512 equal those of the round 1 task, which trained every modular-addition run in the spec's tables.
- **Planted bugs:** all 70 are caught by the tests aimed at them. They include:
  - **the task:**
    - `=` left out of the vocabulary, put first, or given the id `p - 1`
    - `seq` 4, the control at position 1, chance over the vocabulary
    - pairs `b`-major; the split unsorted, floored, or training on the last `n`
    - the answer without its modulus, or `a - b`
    - windows sorted, or an index unchecked
    - batches without replacement, from every pair, or seeded by the step alone
    - `p` of 2 or a bool `p`, an empty held-out part or a string fraction accepted; `training()` giving the held-out windows
  - **the declaration:** the planted name judged, the answer at position 1, predictions at every position or for `token(p)` too, `pairs` returning nothing
  - **the keys and functions:**
    - induction: the copied token one place early, its period read from the whole window, current and previous swapped, keys defined everywhere, one wrong function
    - induction's broken windows: `j` drawn from the mechanism's start, a draw that may keep the token, the row at `j`, the answer the old token, the input changed in place
    - two-hop: the hop as `b` or read from the answer, `a` read at the arrow, `b` read across
    - two-hop's patch: after block 3, read from the answer, or with sources it draws itself
    - modular addition: the sum without its modulus, the difference backwards, the shift down, no fourier for the sum
    - its patch: from the window itself, every row redrawn, one round leaving a same sum, at the MLP's output
  - **`Function` and `RunFunctions`:** a table with two keys, a judged function with no relation, a relation with no `"same"` read, a wrong function that is its own key, a bool block, writable mappings, missing keys unchecked, a builder without its task check, windows unchecked by a key
  - **`set`:** taken at a head's tap; float64, NaN or an unchecked hash accepted; values shaped for every position; written in sorted order instead of the edit's; adding instead of replacing; an operation added at no positions; `values` declared an int

**Not verified:**

- anything on the Mac, including training the modular-addition run, which G7.1 does
- `set` on the zoo models, which G7.3 checks
- the functions as crystallize-this will fit them, which delta 2 brings
