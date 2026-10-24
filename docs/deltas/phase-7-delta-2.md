# Phase 7, delta 2: crystallize-this

Date: 2026-10-23. Spec: [../specs/phase-7.md](../specs/phase-7.md). Rulings: D56, D57 and D58, which this delta brings.

This delta ships:

- **A stub for `gestures/crystallize.py`,** the gesture itself: the three forms, the passes that put a part to a function's value, every measurement, the labels, the envelope and the citations. Its docstring is its contract.
- **D58,** which apolloh0 ruled while this delta was built. It drops one of modular addition's wrong functions, and makes crystallize-this refuse any like it. The apply script writes it into the decisions, the spec, the notebook, `zoo/functions.py`'s docstring and its test.
- **12 tests,** the delta 2 tests the spec lists, in `tests/test_gestures_crystallize.py`, with their fixtures in `tests/phase7_fixtures.py`.

This is the spec's work package 3 without its records and command line, which come with the gates in delta 3.

## Apply

Every file in the zip is new. The apply script makes the explicit edits below to five existing files, and refuses to run twice. From the repo root:

```
git pull
unzip -n nablaframe-phase7-delta2.zip
python3 apply_phase7_delta2.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `13 failed, 351 passed`:

- crystallize-this: the 12 new tests fail
- D58: `test_functions_keys_and_reads` now fails, until the thread drops the shifted key from `zoo/functions.py`'s code

## D58: a wrong function is never a relabelling

Building the reference, modular addition's `(a + b + 1) mod p` turned out unable to fail. D56 refits the form on each wrong key, and that key is the sum shifted by one. So its table is the sum's table, relabelled. On round 1's model it gives a KL of 0.02, against D57's bar of at least 1. Feasibility's 8.31 for it, and its 10.16 for `(a − b)`, came from reading the sum's own table at those keys, not from refitting.

apolloh0 chose to drop it. The entry:

> A wrong function is never a relabelling of the function's own keys. crystallize-this refuses a wrong key whose values and the function's keys' values determine each other on the fitting rows, since the form refitted on it is the function itself. Modular addition's wrong functions are `(a − b) mod p`, `a` and `b`, each refitted as D56 says; `(a + b + 1) mod p` is dropped from D57's, with its key.

Refitted on round 1's model, the three that stay give `(a − b)` 5.77, `a` 5.94 and `b` 5.90 as a table, and 4.44, 4.42 and 4.57 in the fourier form.

The edits:

- **`docs/decisions.md`:** D58, after D57.
- **`docs/specs/phase-7.md`:**
  - the status line names D58
  - modular addition's row lists `(a − b) mod p`, `a`, `b`
  - the wrong-function bar's measured value becomes "5.77 to 5.94 as a table, 4.42 to 4.57 in the fourier form"
- **`docs/notebook.md`:** finding 25 says how its two numbers were read and what refitting gives; a lesson, "A wrong function must cut the rows another way"; and a log line.
- **`src/nablaframe/zoo/functions.py`:** in its docstring, the keys become "``"sum"``, ``(a + b) % p``; ``"difference"``, ``(a - b) % p``; ``"a"``; ``"b"``. No key shifts the sum: refitted on one, a form would be the sum relabelled (D58)", and the sum's wrong functions ``("difference", "a", "b")``.
- **`tests/test_zoo_functions.py`:** the keys, their values and the sum's function, to match.

## The contract

`src/nablaframe/gestures/crystallize.py`'s docstring is the contract. In short:

- **`fit(form, keys, Y, n_values, *, k=None)`:** `table`, `additive` or `fourier`, each in float64, giving a `Fitted` with `value(keys)`.
- **`frequencies(embedding, p)`:** the fourier form's `K`, from the number tokens' embedding rows.
- **`part_taps` and `part_edits`:** a part's taps, and the edits that put it to given values. For `write`, `set` at the attention output and then `zero` at the MLP output.
- **`label` and `envelope`:** D56's labels, and the envelope across checkpoints.
- **`describes` and `cite`:** each key's description, and the records a crystallization cites.
- **`crystallize(...)`:** everything D56 measures, as a plain JSON mapping for a record's results:
  - the meter, loss, accuracy, R² and values seen
  - the fourier frequencies, the wrong functions and the counterfactual
  - the twin, the checkpoints and the envelope, and the citations

### Settled in the contracts

D56 to D58 and the spec left these open. Each follows the spec's intent, feasibility, or the nearest existing rule.

**Parts and functions**

- **A request names its part; a function doesn't.** D56 has the request name the run, block, part and class. `crystallize` checks the block and class against the function's own and takes any part. D57 decides which (function, part) pairs a gate judges: two-hop's `hop` at the write, and at the attention recorded. PR #89 read it this way.
- **Rows are the class's positions where every key is defined.** At the class's other positions the part keeps its own output. So a pass replaces only where the function says something.

**Passes**

- **Fixed chunks of 256 windows.** Results depend on the chunk at the 1e-7 level (PR #79). A record must rebuild byte for byte, and the record's key doesn't hold a chunk size.
- **The unintervened pass** is `model.instrumented`, recording the part's taps. Every pass that puts the part to values goes through `run_intervened`.

**Measurements**

- **The KL is `dictionaries.edges.kl`,** from the model's logits to the crystallized model's, averaged over rows. Agreement uses `sets.argmax`, as every gesture does.
- **Unseen keys are counted for every form.** A table gives them the overall mean (D56). In the additive form their least-squares coefficient is 0, so they add nothing.
- **The fourier form:**
  - It needs an odd `p`.
  - `K` is ranked by share, ties to the lower frequency.
  - It is cut at 8 even where 8 hold less than 0.75, and the record shows the share they hold.
  - The twin and every checkpoint keep the request's `K`, as the checkpoints do in D56.
- **Wrong functions are refitted** on the same rows (D56):
  - The additive form, which needs two keys, fits a wrong key alone as a table.
  - A wrong key undefined at a row is refused, and so is one that relabels the function's keys (D58).
- **The counterfactual** draws on the held-out windows with a fresh `default_rng(seed)`, for the trained model and again for the twin, so both read the same inputs. Its patch comes first in the spec. At the class's positions other than the case's rows, the keys are the case's windows'.

**Across training**

- **Each checkpoint fits the form again** (D56) and records the model's accuracy over the held-out rows.
- **"Before the envelope"** is the steps that hold before accuracy first reaches 0.9, or every step that holds where it never does.

**Citations**

- **A key's description** is the first description, in the declaration's order, that equals the key at every held-out row where the key is defined. That gives `token(p+1)` for every judged key, and for modular addition's MLP also `token(p-2)` and `token(p-1)`. Those are the names G7.5 builds.
- **`cite` chooses by the dictionary and site,** as a name request holds them. The gates name the dictionary at the part's output site, and an edge's downstream dictionary must be that one.
- **"Named as planted"** is a cited record naming the description and value, labelled named. The note appears below 0.7 of the key's values, G6.2's bar for names.
- **The cited records change a record's results,** so delta 3's request carries their keys, and the record's key covers them (D11).

## Thread

Paste into the Project:

```
Start this thread now, without proposing it first.

Thread "[P7] crystallize-this", on Opus at high effort:
Implement src/nablaframe/gestures/crystallize.py against its docstring and docs/specs/phase-7.md (D56 to D58), and in src/nablaframe/zoo/functions.py drop the shifted key and wrong function from modular_addition's code, as that module's docstring now says. Change only those two files. Every pass that puts a part to values goes through run_intervened, and every pass reads its windows in chunks of CHUNK, so a record rebuilds byte for byte. Fit in float64, as the docstring says. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_gestures_crystallize.py tests/test_zoo_functions.py tests/test_zoo_modadd.py tests/test_set.py` passes with no skips, and every test that passed before still does.
```

## Verification record

**Verified** on clones of my mirror of `main` at `9342183`, with this delta's files added and the apply script run:

- **On the stubs:** the 12 new tests fail, and so does `test_functions_keys_and_reads` until the shifted key goes. All 351 other tests pass.
- **On a private reference implementation, not shipped:** all 364 pass with no skips, its docstrings checked equal to the delta's. The new tests take about two minutes on a cloud CPU.
- **On the trained models feasibility left in the cloud workspace,** with the zoo's windows where the zoo has them, the reference reproduces feasibility's numbers. KL against the ablation, then the wrong functions:

| Crystallization | KL | Ablation | Wrong functions | Counterfactual agreement, by what it reads and by the answer |
| --- | --- | --- | --- | --- |
| Modular addition's write, table | 0.0207 | 4.14 | 5.77, 5.94, 5.90 | none |
| Its write, fourier (2, 36, 25, 44, 18) | 0.0187 | 4.14 | 4.44, 4.42, 4.57 | none |
| Its MLP, additive | 0.0191 | 7.50 | none | 1.000 and 0.006 |
| Its attention, additive | 0.0943 | 7.99 | none | none |
| Induction's block 1 attention, table | 0.0004 | 6.87 | 6.34, 6.91 | 1.000 and 0.000 |
| Two-hop's block 3 write, table | 0.126 | 4.54 | 4.50, 4.57 | 0.941 and 0.008 |
| Two-hop's block 3 attention, table | 0.161 | 4.87 | 4.82, 4.84 | 0.927 and 0.010 |

  - Modular addition's loss deltas are −0.033 as a table and −0.027 in the fourier form, as the spec records.
  - Every twin is inert, with its wrong functions within 0.003 of its right one.
  - The induction and two-hop models are feasibility's stand-ins of G4.1's runs. Feasibility read its own windows, so their numbers differ a little from the spec's.
- **Planted bugs:** all 67 are caught by the tests aimed at them. They include:
  - **parts and labels:** the write's taps the other way round, without its zero, or zeroing the attention; a bool block; inert at an ablation of exactly 1; holding only strictly below 0.25; inert judged by the KL
  - **forms and frequencies:**
    - unseen values reading zero, or a table of the overall mean
    - the additive design without its first key; the fourier design without its constant
    - a shifted cosine for the sine, which fits the same values but misstates each frequency's share
    - frequencies taken by a table; a key past its range; one key taken as additive
    - energy without the sines, nine frequencies, an even `p`, a frequency too many
  - **the envelope:** runs that skip a checkpoint, a final run that doesn't hold, accuracy strictly above 0.9, a gap that includes what holds, "before" reaching past accuracy
  - **passes:** zeros where the key is undefined; the write's own output without its MLP; the counterfactual's patch left out of the crystallized passes or one position late
  - **measurements:** the KL the wrong way round; the ablation by the held-out mean; "removed" as the ratio; agreement with the answer; the loss delta reversed or at the row's own token; the model's accuracy crystallized; R² about the fitted mean; unseen rows uncounted; values seen counted as rows
  - **the twin and the checkpoints:** the twin's own frequencies; checkpoints with their own frequencies, keeping the request's fit, or with the crystallized accuracy
  - **wrong functions:** read off the right table instead of refitted; no relabel guard; a key that merges values taken for a relabelling
  - **counterfactuals:** one generator for the trained model and the twin; reads and answer swapped; the case's rows keyed as the rest; the base unpatched; the held-out windows run instead of the case's
  - **citations:** descriptions compared where the key is undefined; a key defined nowhere described; names at any site or class; edges at any downstream site; records of two runs; weak names counted as planted; the note's threshold moved; cited records unchecked; edges without their label
  - **the request:** a block, class or form that isn't the function's; another run's declaration; steps without `params_at`; the row counts swapped
  - **D58:** the shifted key kept in `zoo/functions.py`

  One more change turned out equivalent and was dropped: citing edges with no relation asked. An edge record's relation is a name, so none matches.

  Six planted bugs escaped a first version of the tests, and the tests gained a case for each:
  - a frequency's coefficient share
  - a key undefined at some positions, in `describes` and in the passes, with a block after the replaced one in the induction fixture
  - a counterfactual that changes what the measured row reads
  - a form the function could take but doesn't admit
  - another run's declaration of the same shape

**Not verified:**

- anything on the Mac
- the records, the command line and the gates, which delta 3 brings
- time at gate size: with its twin, one crystallization takes from 36 seconds (induction) to about 2.5 minutes (modular addition) on a cloud CPU, before any checkpoints
