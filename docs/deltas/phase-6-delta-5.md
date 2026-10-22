# Phase 6, delta 5: the records, the gestures panel, the badges and G6.7

Date: 2026-10-21. Spec: [../specs/phase-6.md](../specs/phase-6.md). Rulings: D38, D44 and D47 to D52.

This is work package 6, the last of Phase 6. It ships:

- **Two new modules,** each with its docstring as its contract:
  - `views/gestures.py` builds a run's gestures panel: every record in its records folder, each with its meter, and the name records by dictionary, site and latent, so the features view can link to them.
  - `views/gesture_badges.py` generates each gesture's badge, and `dsteer`'s, from the Phase 6 reports.
- **Two new gate scripts,** stubs whose docstrings are their contracts:
  - G6.7 builds every record twice, writes each run's panel, and writes one badges file for all three phases.
  - The Phase 6 summary.
- **Contract changes** to `reports.py`, the viewer's header, one committed test and the spec's G6.7 row. The apply script writes all of them.
- **6 tests** in three new files, and an edit to one committed test.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase6-delta5.zip
python3 apply_phase6_delta5.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `7 failed, 336 passed`. The 7 are:

- the gestures panel: 3
- the gesture badges: 2
- G6.7's report rules: 1, plus the edited `test_phase6_gate_ids`

Then commit, naming the files, and leave out the zip and the apply script:

```
rm nablaframe-phase6-delta5.zip apply_phase6_delta5.py
git add docs/deltas/phase-6-delta-5.md docs/specs/phase-6.md src/nablaframe/reports.py src/nablaframe/views/gestures.py src/nablaframe/views/gesture_badges.py gates/phase6/g67_records_panel.py gates/phase6/summarize.py tests/test_views_gestures.py tests/test_views_gesture_badges.py tests/test_phase6_reports_g67.py tests/test_phase6_reports.py viewer/app.js
git commit -m "Phase 6 delta 5: the records, the gestures panel, the badges and G6.7"
git push
```

## What the viewer will show

Each run's address gains `?gestures=`, its records folder, and its badges file becomes `runs/phase6/badges.json`. A gestures panel lists the run's records by kind, each with its meter first and its gesture's badge above it:

- **a name:** named or weak; follows beside the null; F1 and coverage; the set and its effect at both ends; the top candidates
- **an edge:** follows and kept beside their nulls, then the label, beside follows through the residual alone
- **open-this-up:** the error curve over parts in three orders, and each instance's top parts with their names
- **generalize-this:** the slices and the seeds, and every checkpoint on the timeline, with the envelope shaded and the probe and accuracy beside it

The features panel links each latent to its name records. So H6's second question can be answered in one session:

1. Open the two-hop run at `⇒` and choose the TopK SAE after block 2 in the features panel.
2. Choose an active latent and open its name record. It shows the name `b = v`, its set, the set's effect with `b` at the bottom, and steering beside its null.
3. Open `b`'s generalize-this record. On the timeline, its envelope runs from step 6,000, where the probe first reads `b`, to the final checkpoint, with the accuracy outside it shown beside it.

| Run | Name records | Edge records | Open-this-up records | Generalize-this |
| --- | --- | --- | --- | --- |
| Two-hop | every value of `b` after blocks 1 and 2 and of the answer after block 3, each pointed at its best latent; the latents active at `⇒` in the probe windows at those blocks, up to 8 per window | hop, carry, two back | the carry's and the hop's targets, for every value | `b` after block 2 |
| Induction | every value of `token(p)` after block 0 and `token(p+1)` after block 1; the latents active at the probe windows' last position at those blocks | copy | the copy's target, for every value | `token(p+1)` after block 1 |
| Reference | G6.6's 256 sampled latents per block, and each steered value's best latent | none | each steered value's set, at blocks 1 to 3 | none |

That is at most 436 records for two-hop and 322 for induction. The reference run gets 1,024 name records, plus a name record per steered value and an open-this-up record per steered value above block 0: G6.6 steered 38 values, 30 of them above block 0. Requests that coincide are built once, so each run gets fewer.

### The badges

| Badge | Judged by | Facts |
| --- | --- | --- |
| `name` | G6.1, G6.2 | Its records built twice, and through the command line, identical. G6.2's bounds. For each judged name: the values named as planted, steering's follows, and the values the set's effect ranks first and last |
| `edge` | G6.1, G6.3 | The records' identity. G6.3's bounds and whether every judged edge is computed. Each edge's follows, its null and its label, the carry and two back marked recorded |
| `open` | G6.1, G6.4 | The records' identity. G6.4's bounds. Each target's median reading beside its bar, its share within 128 parts and its planted set in the top 8; the induction twin's planted share |
| `generalize` | G6.1, G6.5 | The records' identity. G6.5's bounds. Each name's envelope and where the probe first reads it |
| `dsteer` | G6.1 | The identities, exact of those checked |

## Settled in the contracts

- **The panel sits beside the records.** `views/gestures.py` writes `panel.json` into each run's records folder (D47), next to the `index.json` that `records.py` keeps. The views G5.6 built don't change, so G6.7 rebuilds none, and the viewer finds the panel through `?gestures=`.
- **The viewer reads each record's own file.** The panel lists every record with its SHA-256 and its meter, so the list needs no fetch per record. When a record is chosen, the viewer fetches its file and checks it against that SHA-256, as it checks the trace's tensors.
- **A meter is copied, never computed.** Each kind's meter is a fixed set of entries of its results, in a fixed order, with None where a step didn't happen. A name with no steering instance, or no causal test, says which.
- **A name record per value is pointed at that value's best latent.** At G6.2's instances and seed, its steering is G6.2's for that value wherever the latent is named as planted. Requests that coincide are built once.
- **The readout's latents are the features view's.** At each planted name's block, the latents active at each probe window's last position, largest code first, up to 8: `⇒` for two-hop, the last mechanism position for induction.
- **Open-this-up gets a record per target and value.** G6.4's targets change from one instance to the next, which no single request covers. So each judged target gets one request per value of its downstream description.
- **The reference records repeat G6.6's draws,** with the same generator, and G6.7 checks that the values it steers are the ones G6.6's report lists.
- **G6.7 also checks the records it shows are the ones judged.** The edge and generalize records G6.3 and G6.5 name in their reports must be among G6.7's, key for key. The apply script adds this to the spec's G6.7 row, as the report rule `reports_match`.
- **No gesture module changes from here to H6.** A record's key holds the blob hash of `gestures/cli.py`, `declare.py`, `sets.py`, `name.py`, `edge.py`, `open.py`, `generalize.py` and `zoo/declarations.py`. An edit to any of them gives every record a new key, so G6.3's and G6.5's records would no longer be the ones shown, and `reports_match` fails.
- **An edge's label sits beside follows and its null,** in its meter and in the panel, and the edge badge says why: a label says little where follows is no higher than its null. The twin's hop is labelled carried at follows 0.115 against a null of 0.141.
- **G6.5's patch is labelled as its own.** The generalize badge notes that it moves position 25 only, so it isn't read as G4.3's query patch at positions 24 and 25.
- **The reference steering is explained, not changed.** `name.steer` draws `w` from every value the window doesn't hold, and on the reference run few of those have a set that names them, which is why G6.6's steering rarely follows. The name badge says so. Changing the draw would change every record's key.
- **One badges file holds all three phases' badges,** `runs/phase6/badges.json`, so `?badges=` reads one file. `views/badges.py` and `views/dictionary_badges.py` are unchanged.

## Contract changes

All of these are written by the apply script.

**`src/nablaframe/reports.py`** (docstring)

- "`PHASE6_GATE_IDS` gains G6.7."
- Status rule: "G6.7 pass requires `builds_identical` true, `external_urls == 0`, `subjects_missing == 0` and `reports_match` true."
- Required measurements: "G6.7: `badges`, `subjects_missing`, `records`, `builds_identical`, `external_urls`, `reports_match`"

**`viewer/app.js`** (the header's contract, three new bullets)

- **Phase 6 badges:** the badges file may also hold `name`, `edge`, `open`, `generalize` and `dsteer`, with four new units (parts, a span of steps, a flag and a label). A dsteer edit reads in words, beside its badge.
- **Gestures:** with `?gestures=`, a panel lists `panel.json`'s records by kind, meter first, and draws a chosen record's own file once its SHA-256, key and request match its entry. Each kind's sections are listed, and a generalize-this record's checkpoints join the timeline while it's chosen. It computes nothing.
- **Features link to names:** a latent with name records shows their names and labels, and opens them in the gestures panel.

**`tests/test_phase6_reports.py`:** `test_phase6_gate_ids` expects G6.7.

**`docs/specs/phase-6.md`:** G6.7's pass condition adds that the edge, generalize and reference records are the ones G6.3, G6.5 and G6.6 measured.

**`gates/phase6/g67_records_panel.py`, `gates/phase6/summarize.py`, `views/gestures.py` and `views/gesture_badges.py`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave A** has three threads, which share no files. They can start as soon as this delta is committed:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P6] Gestures panel", on Sonnet at high effort:
Implement src/nablaframe/views/gestures.py against its docstring, docs/specs/phase-6.md and D47. Change only that file. Every value in a meter is copied from a record's results, never computed. Import nothing that imports JAX. Say in the PR how long gesture_panel takes over a folder of 1,000 name records the size of the tests' toy ones. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_gestures.py tests/test_gestures_records.py` passes with no skips, and every test that passed before still does.

Thread "[P6] Gesture badges", on Sonnet at high effort:
Implement src/nablaframe/views/gesture_badges.py against its docstring, docs/specs/phase-6.md and D38. Change only that file. Every value on a badge comes from a report; nothing is typed in or computed. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_gesture_badges.py tests/test_views_dictionary_badges.py tests/test_views_badges.py` passes with no skips, and every test that passed before still does.

Thread "[P6] G6.7 report rules", on Sonnet at medium effort:
Implement the G6.7 parts of src/nablaframe/reports.py's docstring, which the delta's apply script wrote: G6.7 in PHASE6_GATE_IDS, its status rule and its required measurements, following the Phase 6 rules. Change only src/nablaframe/reports.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase6_reports_g67.py tests/test_phase6_reports.py tests/test_phase5_reports_g56.py tests/test_reports.py` passes with no skips, and every test that passed before still does.
```

**Wave B** has two threads, which share no files. Start both once all of wave A has merged:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P6] Viewer: gestures panel and Phase 6 badges", on Opus at high effort:
Implement what viewer/app.js's header contract now says about Phase 6's badges, the gestures panel and the features panel's links to names, in viewer/app.js and viewer/index.html, adding to the header's "How this page meets it" what your choices settle. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder: on a tiny traced run whose views are built with a features payload from views/features.py; with a records folder of that run holding a record of each kind and form, built with gestures/cli.py's build and written with records.write_record and views/gestures.write_panel, among them a name record of a latent the features payload shows active; with a badges file holding views/badges.py's, views/dictionary_badges.py's and views/gesture_badges.py's badges on fixture reports; and with a compare payload whose spec has a dsteer edit. Say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.

Thread "[P6] G6.7 and the Phase 6 summary", on Opus at high effort:
Implement gates/phase6/g67_records_panel.py and gates/phase6/summarize.py against their docstrings, docs/specs/phase-6.md, D38, D44, D47 to D52 and docs/deltas/phase-6-delta-5.md. You may add to gates/phase6/_common.py what several gates share, below its contract, without changing what the contract says. Load the earlier phases' gates and _common.py modules by path, but don't change them. Change only files in gates/phase6/; never change a gesture module, since a record's key holds their blob hashes. Every record goes through gestures/cli.py. Every long step prints progress at least every 30 seconds. Never run a gate for results and never write under reports/. Exercise G6.7 in a scratch folder: make its prerequisites there at shrunk settings, including G6.3, G6.5 and G6.6 run there so that reports_match is exercised for real, with fixture reports only where a gate would take hours; and exercise summarize.py on the scratch reports of all seven gates. Say in the PR what that covered, how many records each run gets, and how long G6.7 would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

After both wave B threads have merged, on the Mac:

```
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g67_records_panel.py
git add reports/phase6/g67_records_panel.json
git commit -m "G6.7 report"
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/summarize.py
git add reports/phase6/summary.md
git commit -m "Phase 6 summary"
git push
```

G6.7 trains nothing: it loads every dictionary from the Phase 5 cache and exits if one is missing. It builds every record twice, so its time is the wave B thread's estimate. It prints the viewer's address for each run. With the reports pushed, H6 is apolloh0's ruling, after a session in the viewer.

## Verification record

**Verified** on clones of my mirror of `main` at `03c8b86`, with this delta's files added and the apply script run:

- **On the stubs:** `7 failed, 336 passed`, the seven listed under Apply.
- **On a private reference implementation, not shipped:** `343 passed`, with no skips, in the full jax lane. The reference covers the panel, the badges and G6.7's rules, and its docstrings are the ones this delta ships.
- **The apply script** makes its 7 edits in four files and stops on a second run.
- **The tests' records are real.** `cli.build` makes them on the toy run, so each meter reads the results its gesture returns. Between them they reach every branch the meters name: a causal name, a name with no prediction, a latent that never fires, an edge through the residual alone and one across positions, open-this-up by latent and by name, and generalize-this of a name and of an edge. Edited copies add a name with no steering instance, open-this-up with no instance measured, and, for each kind, entries the toy leaves equal set apart.
- **Timing, here:** `records.write_record` rereads every record in its folder on each write, so a folder's build time grows with the square of its records. 400 toy name records took 43 seconds, which puts the reference run's 1,100 at about 5 minutes per build, on top of computing them. `gesture_panel` read the 400 in 0.6 seconds.
- **Planted bugs:** all 69 are caught by the test aimed at them. They include:
  - a name's coverage, set F1 or ranks read from the pointed latent, the ranks swapped, no instance counted as 0, no name giving an empty meter, a KL beside causal shares, the label left out
  - an edge's residual-alone figure read from the unfrozen swap, its output's null from its follows, its keys reordered
  - open-this-up's reading, curve or random share read from another order
  - generalize-this's final row from the first checkpoint, its slices' follows as their F1, an edge without its relation, every record read as a name
  - a payload hashed without the newline, its file named without its prefix, its request reduced to its arguments, read-only values or one meter shared between calls, a missing entry read as None
  - the panel in key order rather than the index's, an unlisted file, two runs or a malformed index let through, records read without their SHA-256, open records linked as names, places keyed with a colon, latents as ints, a latent's records in reverse, an empty index with an empty run
  - the panel file without its newline, started before the panel is made, or a missing folder made
  - badges with places unsorted, recorded names judged, first and last swapped, a name's null for its follows, the twin's figures for the trained, every edge marked recorded, a label read as a fraction, every twin judged, a reading from signed order, the envelope's first run for its final one, the wrong units for a flag, a bar, a ratio, a span, an offset or an F1, the identities reordered, dsteer given the records' flags, names judged without G6.1, dsteer judged by every gate, the patch note missing, a misfiled or invalid report accepted, a missing flag read as false, a label without its place
  - G6.7 missing from the Phase 6 gates, always passing, passing without `reports_match` or `records`, or with an external URL

  Eight first escaped, and the tests were strengthened. On the toy, a name's set and its pointed latent have the same effect ranks, an edge's frozen swap follows as often as the unfrozen one, and open-this-up's three orders and generalize-this's slices read alike, so the test now edits a copy of each record to set those apart. One planted bug, the edge's keys left in the order the code builds them, turned out to be that order already, and now sorts them. And a malformed index and a missing folder were refused only because the folder held unlisted records, so the test now tries both on folders holding none.

**Not verified:** the viewer, G6.7 and the summary, which wave B builds, and anything on the Mac.
