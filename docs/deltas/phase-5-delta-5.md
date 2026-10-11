# Phase 5, delta 5: the features view, dictionary badges and G5.6

Date: 2026-10-10. Spec: [../specs/phase-5.md](../specs/phase-5.md). Rulings: D38, D39, D44 and D45.

This is work package 6, the last of Phase 5. It ships:

- **Two new modules,** each with its docstring as its contract:
  - `views/features.py` builds a features payload: one site of a dictionary, its latents on the probe and on held-out windows.
  - `views/dictionary_badges.py` generates each shipped kind's badge, and `dpatch`'s, from the Phase 5 reports.
- **Two new gate scripts,** stubs whose docstrings are their contracts:
  - G5.6 builds every run's views again, with features payloads and D39's sentence, and writes one badges file for both phases.
  - The Phase 5 summary.
- **Contract changes** to `views/build.py`, `reports.py`, the viewer, and three of delta 4's gate stubs. The apply script writes all of them.
- **D39's sentence in code.** The apply script rewrites the three D14 badges' last sentence. This changes the bytes of every view built from now on, which is why G5.6 rebuilds them.
- **17 tests** in four new files, and an edit to one committed test.

## Apply

Apply this before starting delta 4's wave 2: it corrects two points in the contracts that wave implements (see "Two fixes to delta 4's contracts"). Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase5-delta5.zip
python3 apply_phase5_delta5.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `25 failed, 255 passed` before the edge-sparsity meter's PR merges, and `15 failed, 265 passed` after. The 15 are:

- the features payload: 7
- the build's features: 2
- the dictionary badges: 4
- G5.6's report rules: 1, plus the edited `test_phase5_gate_ids`

The constants' tests and the new wording's test pass already, as pins.

## What the viewer will show

For each run, a features panel lists its shipped dictionaries' sites. Choosing one shows three things:

- **Above it:** the dictionary's meters from its gate's report, the reconstruction's FVU on the probe, and its kind's badge.
- **At the page's position:** the probe's active latents.
- **For any latent:**
  - how often it fires
  - its top held-out contexts
  - its direct effect on the logits, labelled as that and not as a name

On the two-hop run, choose the TopK SAE at block 2 and position 25 (`⇒`) to see H5's question answered. Each context there also shows its window's `a`, `b` and `c`. A latent whose top contexts all share one `b` is that intermediate's latent, in the very place both lenses rank `b` last (D39).

| Run | Dictionaries | Features payloads | Meters from |
| --- | --- | --- | --- |
| Reference | TopK SAE at blocks 0 to 3, the Staircase SAE, transcoders at blocks 0 to 3 | 12 | G5.5 |
| Induction | TopK SAE at blocks 0 and 1, the Staircase SAE, transcoders at blocks 0 and 1 | 6 | G5.2 |
| Two-hop | TopK SAE at blocks 0 to 3, the Staircase SAE, transcoders at blocks 0 to 3 | 12 | G5.3 |

### The badges

| Badge | Judged by | Facts |
| --- | --- | --- |
| TopK SAE, Staircase SAE | G5.2, G5.3; the TopK SAE also G5.4 | Selective latents, trained against twin, for each judged variable. The code patch's swap at G4.3's best block, beside the residual patch's and the single-latent patch's. The gates' bounds on the faithful code. The TopK SAE adds each judged MLP's edge count, trained against twin |
| Transcoder | G5.2, G5.3 | Selective latents, and the induction model's bounds on the faithful code |
| `dpatch` | G5.1, G5.2, G5.3 | The identities, exact of those checked, and the faithful code's bounds |

## Settled in the contracts

- **A badge is about a kind,** wherever its dictionaries are trained. The reference run's TopK SAEs carry the TopK badge, beside their own recorded meters. TinyStories has nothing to judge (D44).
- **A badge never computes.** The reports hold the bars' bounds only across every kind a gate judges, so a badge cites those bounds and says so in a note. Its other facts are the kind's own entries.
- **One badges file holds both phases' badges.** G5.6 writes `runs/phase5/badges.json`, Phase 4's subjects followed by Phase 5's, so `?badges=` reads one file. `views/badges.py` is unchanged, so G4.6's badges still recompute byte-identically from its reports, as H4's evidence says.
- **The direct effect is the decoder's, through the final layer norm's gain.** The decoder row times the site's scale, centred, times `ln_f`'s gain, through the unembedding, with the norm's divisor held at 1. The payload carries that sentence as its label.
- **Contexts point into the held-out windows.** Each payload holds its windows' tokens once, and each context names a window and a position. Two-hop's payloads also carry each window's `a`, `b` and `c`.
- **The rebuilt views go to `runs/phase5/views/<run>/`,** not over the folders G2.6 and G4.5 built, which their reports describe.

### Two fixes to delta 4's contracts

- **G5.3's `faithful` used `"latents"` for two things at once:** the single-latent patch's rates and the count of latents it moves. The counts are now `"assigned"`, in G5.3 and, for consistency, in G5.2.
- **A dictionary's key tokens are now stated** in G5.2, G5.3 and G5.5: what the model reads of the training windows. G5.6 finds each shipped dictionary by recomputing its key, so the gates and G5.6 must make it the same way.

## Contract changes

All of these are written by the apply script.

**`src/nablaframe/views/build.py`**

- Code: the D14 badges' last sentence, "Planted structure at model level waits for Phase 4's zoo.", becomes "Its model-level result is in the badge below (D38).".
- Docstring: "`build_views` takes `features=None`: payloads from `views.features.features_payload`. Each is written with `export_view` as `features-<dictionary>-site-<site>`, after every step's views and before the index, and the index gains `"features"`: one entry per payload, in the given order, `{"dictionary", "kind", "site", "taps", "step", "file"}`, the first five copied from the payload and `file` its file name. Given as an empty list, the index gains `"features": []`. A payload that isn't a dict holding those five keys and a `"meter"`, two payloads with the same dictionary and site, or a payload whose step isn't one of the reader's raises ValueError before a file is written. Without it, the index and every file are byte-identical to a build without it."

**`src/nablaframe/views/jlens_view.py`**

- Code: the J-lens badge's last sentence changes the same way.

**`src/nablaframe/reports.py`** (docstring)

- "`PHASE5_GATE_IDS` gains G5.6."
- Status rule: "G5.6 pass requires `builds_identical` true, `external_urls == 0`, `dictionaries_missing == 0` and `subjects_missing == 0`."
- Required measurements: "G5.6: `badges`, `subjects_missing`, `dictionaries`, `dictionaries_missing`, `features`, `builds_identical`, `external_urls`"

**`gates/phase5/g52_induction.py`, `g53_two_hop.py`, `g55_reference.py`** (docstrings)

- G5.2 and G5.3: the counts of moved latents are named `"assigned"`.
- G5.2 and G5.5 state a dictionary's key tokens. G5.3 refers to G5.2's.

**`viewer/app.js`** (the header's contract, two new bullets)

- **Phase 5 badges:**
  - The badges file may also hold `topk`, `staircase`, `transcoder` and `dpatch`.
  - Their new units: a score and a ratio to three decimals, edges and a count as integers.
  - The compare panel shows `dpatch`'s badge, and reads a dpatch edit in words.
- **Features:**
  - When the index has `features`, a features panel lists them.
  - For the chosen one it shows: its meters and badge; the probe's active latents at the page's position; and, for one latent, its frequency, its top contexts with their windows' variables, and its effect's top and bottom tokens under the effect label.
  - It computes nothing: every number is the payload's.

**`gates/phase5/g56_features_badges.py`, `gates/phase5/summarize.py`, `views/features.py` and `views/dictionary_badges.py`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave A** has three threads, which share no files. They can start as soon as this delta is committed, alongside the edge meter's thread and delta 4's wave 2:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P5] Features view", on Sonnet at high effort:
Implement src/nablaframe/views/features.py against its docstring, and the `features` keyword that src/nablaframe/views/build.py's docstring now describes, against docs/specs/phase-5.md (D44, D45). Change only those two files. Without `features`, every file build_views writes must stay byte-identical to a build without it. Say in the PR how long one payload takes at the reference run's size: 2,048 latents, 64 held-out windows of 256 tokens, a vocabulary of 4,096. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_features.py tests/test_views_build_features.py tests/test_views_build.py tests/test_views_build_jlens.py tests/test_views_last_target.py` passes with no skips, and every test that passed before still does.

Thread "[P5] Dictionary badges", on Sonnet at high effort:
Implement src/nablaframe/views/dictionary_badges.py against its docstring, docs/specs/phase-5.md and D38. Change only that file. Every value on a badge comes from a report; nothing is typed in or computed. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_dictionary_badges.py tests/test_views_badges.py` passes with no skips, and every test that passed before still does.

Thread "[P5] G5.6 report rules", on Sonnet at medium effort:
Implement the G5.6 parts of src/nablaframe/reports.py's docstring, which the delta's apply script wrote: G5.6 in PHASE5_GATE_IDS, its status rule and its required measurements, following the Phase 5 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase5_reports_g56.py tests/test_phase5_reports.py tests/test_phase4_reports.py tests/test_reports.py` passes with no skips, and every test that passed before still does.
```

**Wave B** has two threads, which share no files. Start the viewer once "[P5] Features view" and "[P5] Dictionary badges" have merged. Start G5.6 once all of wave A and delta 4's "[P5] Phase 5 gates" have merged:

```
Start one thread now, without proposing it first.

Thread "[P5] Viewer: features panel and Phase 5 badges", on Opus at high effort:
Implement what viewer/app.js's header contract now says about Phase 5's badges and the features panel, in viewer/app.js and viewer/index.html, adding to the header's "How this page meets it" what your choices settle. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder on a tiny traced run whose views are built with features payloads from views/features.py (a TopK SAE, both sites of a Staircase SAE, and a transcoder, one payload with variables), with a badges file holding views/badges.py's and views/dictionary_badges.py's badges on fixture reports, and with a compare payload whose spec has a dpatch edit. Say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

```
Start one thread now, without proposing it first.

Thread "[P5] G5.6 and the Phase 5 summary", on Opus at high effort:
Implement gates/phase5/g56_features_badges.py and gates/phase5/summarize.py against their docstrings, docs/specs/phase-5.md, D38, D39, D44, D45 and docs/deltas/phase-5-delta-5.md. You may add to gates/phase5/_common.py what several gates share, below its contract, without changing what the contract says. Load the earlier phases' gates and _common.py modules by path, but don't change them. Change only files in gates/phase5/. Every long step prints progress at least every 30 seconds. Never run a gate for results and never write under reports/. Exercise G5.6 in a scratch folder: make its prerequisites there at shrunk settings (the runs, the cached dictionaries, and fixture reports where a gate would take hours), and exercise summarize.py on fixture reports for all six gates. Say in the PR what that covered and how long G5.6 would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

After G5.1 to G5.5 have run and both wave B threads have merged, on the Mac:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase5/g56_features_badges.py
git add reports/phase5 && git commit -m "G5.6 report"
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase5/summarize.py
git add reports/phase5 && git commit -m "Phase 5 summary"
```

G5.6 trains nothing: it loads every dictionary from the cache the earlier gates filled, and exits if one is missing. It rebuilds each run's views twice, as G2.6 and G4.5 did, so expect about as long as those two together, twice over. It prints the viewer's address for each run.

## Verification record

**Verified** on a clone of `main` (`254990b`), with delta 5 applied:

- **On the stubs:** `25 failed, 255 passed`, as listed under Apply.
- **On a private reference implementation, not shipped:** all 17 new tests pass, and so does the edited `test_phase5_gate_ids`.
  - The reference covers the features payload, the build's `features`, the badges and G5.6's rules.
  - The full lane fails only the 10 edge tests, which wait on their own thread.
- **The apply script** makes its 17 edits in nine files and stops on a second run.
- **One payload at the reference run's size** (2,048 latents, 64 windows of 256 tokens, a vocabulary of 4,096) takes the reference about 5 seconds here and writes 1.7 MB. So G5.6's 30 payloads add a few minutes per build.
- **Planted bugs:** all 49 are caught by the test aimed at them. They include:
  - the probe's latents in id order, smallest first, or with inactive ones listed
  - a frequency that counts zero codes, or divides by windows instead of positions
  - contexts tied to the higher window, one short, or numbered position-major
  - an effect not centred, without the layer norm's gain or the site's scale, tied to the higher token, or with the bottom taken from the top
  - a staircase site given the whole dictionary's width; a transcoder's codes read from its output tap, on the probe or on held-out windows
  - a meter of the output against itself; the caller's meters kept rather than copied
  - a Jacobian SAE, an out-of-vocabulary token, a negative step, another gate's meters or variables of the wrong length accepted
  - the dictionary's key written in place of its content hash
  - the index not written last, entries without their taps, duplicate or off-trace payloads accepted, an empty list leaving no `features` key, and the old wording
  - the badges' best block or last block fixed, the twin left out, judged MLPs sorted, the TopK SAE judged without G5.4 or `dpatch` without G5.1, the single-latent patch left out, two-hop's bounds on a transcoder, the edge note missing, G4.1 not required, a report filed under another gate's id accepted, and a ratio labelled a score
  - G5.6 passing without each of its four conditions, without `features`, or missing from `PHASE5_GATE_IDS`

  The harness first missed the misfiled report: the test's impostor lacked G5.2's paths, so it failed on a missing path whether or not the gate id was checked. The test now files a valid G5.3 report that holds every path G5.2's facts read.

**Not verified:** the viewer, G5.6 and the summary, which wave B builds, and anything on the Mac.
