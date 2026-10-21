# Phase 6, delta 1: `dsteer`, gesture records and what each run declares

Date: 2026-10-20. Spec: [../specs/phase-6.md](../specs/phase-6.md). Rulings: D47, D48, and the spec's "What each run declares".

This delta ships:

- **Stubs for a new package,** `gestures/`:
  - `records.py`: gesture requests and records, their hashes and keys, their files and index, and the cache
  - `declare.py`: what every run shares (the token descriptions, which descriptions count at a class, the declaration record) and the reference run's declaration

  Each docstring is its contract. `__init__.py` is complete.
- **A stub for `zoo/declarations.py`:** the two zoo tasks' declarations, with the induction task's periods and sole targets.
- **Contract text** for `interventions/spec.py` and `apply.py`, which gain `dsteer` and `array_hash`. The apply script writes it into both docstrings.
- **13 tests** in three new files, the delta 1 tests the spec lists.

The gestures themselves, the gates and the viewer come in later deltas.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase6-delta1.zip
python3 apply_phase6_delta1.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `13 failed, 280 passed`:

- `dsteer`: 5 failed
- gesture records: 4 failed
- declarations: 4 failed

## Contract changes

**`src/nablaframe/interventions/spec.py`** (added to the module docstring)

- "`OPERATIONS` gains `dsteer` (D48), taking `dictionary`, `remove`, `add` and `amounts`, each a content hash: the dictionary's `content_hash`, and each array's `array_hash`."
- "`array_hash(array)`: the SHA-256 of the canonical JSON of `{"dtype", "shape", "bytes"}`: the array's numpy dtype name (`"int32"`, `"float32"`), its shape as a list, and its `tensor_sha256`. Arrays equal in dtype, shape and values hash equal, whatever their byte order. Unlike the byte hash alone, it tells apart arrays whose bytes agree, such as an empty `remove` and an empty `amounts`. ValueError for an array that isn't of integers or floats."

**`src/nablaframe/interventions/apply.py`** (added to the module docstring)

- "`dsteer` (D48) applies at one of its dictionary's output taps, which picks the site, as `dpatch`'s does, and `resolve` returns its `dictionary` as for `dpatch`. `resolve` returns its three arrays too, each checked with `array_hash`: `remove` and `add`, int32 arrays shaped (n, 3) whose rows are `[sequence, position, latent]`, and `amounts`, float32 shaped (len(add),). Each of these raises ValueError before anything runs, beside `dpatch`'s refusals of the tap and the dictionary: an array of another dtype or shape; a row whose sequence isn't in `range(batch)`, whose position isn't one the edit selects, or whose latent isn't in `range(active_width(site))`; a row repeated within `remove` or within `add`; an amount that isn't finite. The codes `f` are `encode` of the target batch's values at the site's input tap, read as `dpatch` reads `f_target`, and a tap whose values aren't shaped (batch, pos, the dictionary's width there) raises ValueError. The change starts at zero for every sequence, position and latent; each `remove` row subtracts `f` there, and each `add` row adds its amount there, so a latent both removed and added ends at its amount. At the selected positions it adds `decode_delta` of the change, so the target's error term is kept. Where the change is zero at every selected position, as with empty `remove` and `add`, removing only latents whose code is zero, or adding only zero amounts, it adds no operation at all, so it reproduces the unintervened pass bit for bit (D25)."

**`src/nablaframe/gestures/records.py`, `declare.py`** and **`src/nablaframe/zoo/declarations.py`:** each docstring is its contract.

### Settled in the contracts

D47, D48 and the spec left these open. Each follows the nearest existing rule:

- **`dsteer`'s arrays are named by `array_hash`, not the byte hash.** `tensor_sha256` covers bytes alone, because a tap's spec pins its dtype and shape. `dsteer`'s arrays have no such spec, and an empty `remove` and an empty `amounts` have the same bytes, none, so a store keyed by the byte hash couldn't hold both.
- **`dsteer` reads its codes as `dpatch` does:** from the unintervened pass, after the spec's swaps, without its other edits.
- **A latent removed and added at one row ends at its amount:** removals apply first. That is how a gesture swaps a latent that is in both values' sets.
- **A request carries `instances` and `seed`.** They change what a gesture measures, so they are part of what was asked, and of the record's key through the request's hash.
- **A record's key leaves out its results,** as every cache key does (D11): it names what was asked of which inputs by which code, and rebuilding gives the same results. What catches a record changed after writing is the index's SHA-256 of each file, which the viewer can check as it checks the trace's tensors (D13).
- **The request forms:**
  - a name points at one latent
  - open-this-up opens a latent, or a name's set given by its description and value
  - generalize-this reads a description over all its values, as G6.5 does, or an edge
- **The zoo tasks' declarations live in `zoo/declarations.py`, built from each task** (D29). The reference run's lives in `gestures/declare.py`, because it declares nothing beyond what every run shares.
- **A description's "nothing" is -1,** as no token is negative.
- **The two-hop answer is a token description, not a variable:** `token(p+1)` at the query is `c`. The only variable two-hop adds is `b`.
- **A sole target is at least position 1:** the token at `s + P − 1` must have occurred before, at `s − 1`.

## Threads

Paste into the Project, in one wave. Its three threads share no files and depend on nothing in each other:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P6] dsteer", on Opus at high effort:
Implement the dsteer and array_hash bullets in the module docstrings of src/nablaframe/interventions/spec.py and src/nablaframe/interventions/apply.py, quoted under "Contract changes" in docs/deltas/phase-6-delta-1.md, against docs/specs/phase-6.md (D48). Change only those two files. Every operand enters the compiled program as an argument, as PR #38 established, and every refusal happens before anything runs. The three identities are bit for bit; if one holds only approximately, stop and say so in the PR. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dsteer.py tests/test_dpatch.py tests/test_interventions_apply.py tests/test_interventions_spec.py tests/test_jswap.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Gesture records", on Sonnet at high effort:
Implement src/nablaframe/gestures/records.py against its docstring and docs/specs/phase-6.md (D47). Change only that file. Files are written through a temporary file and a rename, and writing an identical record again must leave every byte as it was. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_records.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Declarations", on Sonnet at high effort:
Implement src/nablaframe/gestures/declare.py and src/nablaframe/zoo/declarations.py against their docstrings and docs/specs/phase-6.md ("What each run declares", D49, D50). Change only those two files. The gestures will carry no task-specific code, so everything about a task lives in zoo/declarations.py and is built from the task object. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_declare.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** from a clone of my mirror of `main` at `aee30b3`:

- **On the stubs:** the 13 new tests fail, and all 280 existing tests pass.
- **On a private reference implementation, not shipped:** all 293 pass with no skips.
- **Planted bugs:** all 66 are caught by the tests aimed at them. They include:
  - `array_hash` blind to dtype or shape, or taking booleans; `dsteer` without `amounts`, or its arrays checked by bytes alone
  - a missing refusal: an int64 `remove`, a sequence past the batch, a position the edit doesn't select, a Staircase site's full width, a repeated row, amounts of the wrong length or not finite, or rows checked only after running
  - a wrong change: removing adds the code, an add replaces instead of adding, rows indexed the wrong way round, decoding with the bias, codes read at the output tap, a zero change that still runs, the operand replacing the tap's value
  - requests: a hash ignoring the seed, bools taken as ints, numpy ints kept, names outside printable ASCII, a form matched by a subset of its keys, zero instances, writable arguments, extra JSON keys
  - records: a key covering the results or ignoring the code, unsorted inputs, no states, results kept by reference, a code identity without git's header, a stale key accepted, a key's meaning allowed to change, the index sorted by key alone, a file's bytes or its own key unchecked, a cache that always remakes, files without their newline
  - shared declarations: `token(p-k)` or `token(p+1)` one place off, -1 counted as a value, a strict count, one value counting, a variable named `token(p)`, an undeclared class, a negative block, repeated relations, variables unchecked, the reference predicting every description, the last position taken as in range
  - two-hop: `b` one place early, `b` moved to `a` or `b` predicting, `b`'s prediction `w` instead of its object, predictions at every position, the hop mapping `v` to `v`, two back paired backwards, the carry judged, subjects including the query's subject
  - induction: next-token predictions everywhere, the copy read one place late, sole targets one short, the period one short, the copy paired the wrong way

  Two more changes turned out equivalent, not bugs, and were dropped. Skipping the early exit for an empty `dsteer` changes nothing, because the zero-change check still adds no operation. Allowing position 0 as a sole target changes nothing either: a period is at most 32 and mechanism positions start at 32, so position 0 never qualifies.

**Not verified:** anything on the Mac; `dsteer` on the zoo models, which G6.1 checks; and the declarations as the gestures and gates will use them, which deltas 2 to 4 bring.
