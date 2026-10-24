# Phase 7, delta 4: crystallize-this in the panel, the badges, modular addition's views and G7.5

Date: 2026-10-24. Spec: [../specs/phase-7.md](../specs/phase-7.md). Rulings: C11, D38, D47, D53 to D60.

This is work package 5, the last of Phase 7. It ships:

- **One new module,** with its docstring as its contract: `views/crystallize_badges.py` generates crystallize-this's badge and `set`'s from the Phase 7 reports.
- **Two new gate scripts,** stubs whose docstrings are their contracts:
  - G7.5 builds modular addition's views, builds every Phase 7 record twice, writes each run's panel, and writes one badges file for every phase.
  - The Phase 7 summary.
- **Contract text for six existing files,** which the apply script writes:
  - `views/gestures.py`: crystallize-this's meter in the panel
  - `views/gesture_badges.py`: D53's correction to the open-this-up badge's twin line
  - `views/features.py`: G7.2 may be cited for a features payload's meters
  - `gates/phase7/g72_names.py`: G7.2 records its `meters`
  - `reports.py`: G7.5's report rules
  - `viewer/app.js`: the header's contract for Phase 7's badges, several records folders and crystallize-this records
- **5 tests in three new files,** and explicit edits to three committed tests, which pinned the meter kinds, the features meter gates and the Phase 7 gate ids.

## Apply

Claude applied this delta on a fresh clone of `main`. Every file it brings is new; they were copied in, and then the apply script made the edits listed under "Contract changes". The script refuses to run twice. From the repo root:

```
python3 apply_phase7_delta4.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `9 failed, 378 passed`:

- the panel: `test_crystallize_payloads`, and `test_gesture_payloads_are_plain_json_and_repeatable`, which now expects the fifth meter
- the badges: the 3 new tests
- the report rules: `test_g75_status_rules`, and `test_phase7_gate_ids` and `test_phase7_status_rules`, which now expect G7.5
- the features view: `test_features_constants`, which now expects G7.2 among the meter gates

## What the viewer will show

Modular addition joins the viewer, with its views, its trace, its features payload and its gestures panel. Every run's badges file becomes `runs/phase7/badges.json`. The gestures panel gains crystallize-this records, each with its meter first and its badge above it. A chosen one shows:

- **the meter:** the label, the KL beside the mean ablation's, the share removed, agreement and the loss delta
- **the function:** its keys, what it reads, and the records it cites, each opening in the panel
- **the form:** the keys the fitting rows showed, or the fourier form's frequencies, each with its share of the embedding's energy and of the fitted coefficients
- **the controls:** the wrong functions, the counterfactual both ways, the twin
- **the timeline:** rows for its KL, the ablation's KL and the model's accuracy at every checkpoint, with the envelope shaded, the gap marked, and the checkpoints where it holds before the model is accurate marked apart

So H7's second question can be answered in one session:

1. Open modular addition at `=` and choose its write in the fourier form.
2. The meter says it holds: on the Mac, KL 0.018 against the ablation's 4.14, with the loss delta beside it.
3. The form lists frequencies 2, 36, 25, 44 and 18, each with its shares.
4. The function cites G7.2's 130 name records. They name the sum as planted for 16.8% of its values, so the record's note says why Phase 6's names fall short (C11).
5. On the timeline, the envelope runs from step 1,100 to 10,000, and the six checkpoints from 1,100 to 1,600, before held-out accuracy reaches 0.9 at 1,700, are marked apart.

| Run | Its address's records folders | Phase 7 records |
| --- | --- | --- |
| Modular addition | `runs/phase7/gestures/modular_addition/` | its 4 crystallizations; a name record for each value's best latent of `token(p+1)`, `token(p-2)` and `token(p-1)` at `answer`, which are G7.2's 130 on the Mac; and the latents active at `=` in the 64 probe windows, up to 8 per window |
| Two-hop | Phase 6's, then `runs/phase7/gestures/two_hop/` | its write and its attention |
| Induction | Phase 6's, then `runs/phase7/gestures/induction/` | its attention |

On the cloud specimen, the readout is 54 distinct latents, 13 of them already best for some value, so modular addition gets about 170 name records. The zoo runs' crystallizations cite Phase 6's records, which stay where G6.3 and G6.7 wrote them (D59). So a zoo run's address names both folders, and the viewer reads them together.

### The badges

| Badge | Judged by | Facts |
| --- | --- | --- |
| `crystallize` | G7.3, G7.4 | Each form's records built twice, and through the command line, identical. G7.4's bounds. For each crystallization, judged or recorded: its label, its KL beside the ablation's, its envelope's run to the final checkpoint and its twin's label. Modular addition's recorded names from G7.2: the FVU beside the twin's, and for each description the values named as planted and steering's follows beside its null's |
| `set` | G7.3 | The identities, exact of those checked |

Its notes say that a crystallization is judged by KL, never accuracy or loss; that it is demonstrated over an envelope; why modular addition's names don't hold its sum (C11); and which crystallizations are recorded (D57). The badges file carries every earlier phase's badges too, with the open-this-up badge's twin line now ending "most of any judged target" (D53, D59).

## Settled in the contracts

- **The meter is the record's own.** A crystallize-this record's meter in the panel is its results' `meter`, copied entry by entry: the label, KL, the ablation's KL, the share removed, agreement and the loss delta. Nothing is worked out from the losses or the twin.
- **Several records folders.** `?gestures=` may name several folders, separated by commas. A zoo run's crystallizations cite records in Phase 6's folders, which stay as they are (D59), so its address names Phase 6's folder and Phase 7's, and the viewer lists both. `panel.json` doesn't change shape.
- **`set` gets a badge.** D38 gives each operation a badge, as `dsteer` and `dpatch` have one. Its facts are G7.3's identities.
- **G7.2 records its meters.** A features payload cites its dictionary's meters at a path in a gate's report, and G7.2 records its FVU and the twin's as two separate measurements. So G7.2 gains `meters`, `{"fvu", "twin_fvu"}`, and `features.py` lets G7.2 be cited. It isn't a required measurement, so the report already committed stays valid. G7.2 runs again on the Mac, about 3 minutes, before G7.5, which refuses a G7.2 report without `meters`.
- **G7.5 takes each crystallization's request from G7.4's record.** Its citations were chosen from the records folder as G7.4 found it. G7.5 adds the readout's name records to modular addition's folder, which would change a citation chosen again. So G7.5 reads the request of each record G7.4's report names, and checks it is that crystallization's. The second build cites from its own folder, where the name records are built first.
- **The readout uses every probe window.** D60 made the probe 64 windows, so "up to 8 per window" could mean up to 512 records. But a latent active in many windows is one request, and the dictionary is 256 wide. On the cloud specimen the readout is 54 distinct latents.
- **G7.5 builds every record twice,** as the spec says. The second build computes the seven crystallizations again, which took G7.4 about 30 minutes on the Mac.
- **A Phase 7 summary,** as Phases 5 and 6 have, for H7.
- **Phase 6's records and panels stay as they are** (D59). G7.5 writes panels only into Phase 7's folders.

## Contract changes

All of these are written by the apply script.

**`src/nablaframe/views/gestures.py`** (docstring)

- `METER_KEYS` gains `"crystallize": ("label", "kl", "ablation", "removed", "agreement", "loss_delta")`.
- `gesture_meter`'s crystallize bullet: each is `r["meter"]`'s entry of that name.

**`src/nablaframe/views/gesture_badges.py`** (docstring): `twin_planted_top_max`'s label ends "most of any judged target", since G6.4 measures it over the judged target only (D53).

**`src/nablaframe/views/features.py`** (docstring): `METER_GATES = ("G5.2", "G5.3", "G5.5", "G7.2")`, G7.2's at `["meters"]`.

**`gates/phase7/g72_names.py`** (docstring): "`meters`: `{"fvu", "twin_fvu"}`, the two again, the mapping modular addition's features payload cites."

**`src/nablaframe/reports.py`** (docstring)

- "`PHASE7_GATE_IDS` gains G7.5."
- Status rule: "G7.5 pass requires `builds_identical` true, `external_urls == 0`, `subjects_missing == 0` and `reports_match` true."
- Required measurements: "G7.5: `badges`, `subjects_missing`, `records`, `builds_identical`, `external_urls`, `reports_match`"

**`viewer/app.js`** (the header's contract, three new bullets): Phase 7 badges and a set edit's words; several records folders in `?gestures=`; crystallize-this records, with their meter, function, form, controls and timeline rows.

**`docs/specs/phase-7.md`:** delta 4's tests.

**Tests:**

- `tests/test_views_gestures.py`: `test_gesture_payloads_are_plain_json_and_repeatable` expects the crystallize meter keys.
- `tests/test_views_features.py`: `test_features_constants` expects G7.2 among the meter gates.
- `tests/test_phase7_reports.py`: `test_phase7_gate_ids` expects G7.5, which its `PASSING` gains.

**`views/crystallize_badges.py`, `gates/phase7/g75_records_views.py` and `gates/phase7/summarize.py`:** each docstring is its contract.

## Threads

Paste into the Project, in two waves.

**Wave A** has three threads, which share no files. They can start as soon as this delta is committed:

```
Start three threads now, without proposing them first. They share no files.

Thread "[P7] Crystallize payloads", on Sonnet 5.5 at high effort:
Implement the crystallize-this parts of src/nablaframe/views/gestures.py's docstring, which the delta's apply script wrote: its meter keys and its meter. Change only that file. Every value in a meter is copied from a record's results, never computed. Import nothing that imports JAX. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_crystallize_payloads.py tests/test_views_gestures.py` passes with no skips, and every test that passed before still does.

Thread "[P7] Crystallize badges and D53's line", on Sonnet 5.5 at high effort:
Implement src/nablaframe/views/crystallize_badges.py against its docstring, docs/specs/phase-7.md and D38, in the form of views/gesture_badges.py's badges, and make the open-this-up line that src/nablaframe/views/gesture_badges.py's docstring now names end "most of any judged target" (D53). Change only those two files. Every value on a badge comes from a report; nothing is typed in or computed. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_views_crystallize_badges.py tests/test_views_gesture_badges.py tests/test_views_dictionary_badges.py tests/test_views_badges.py` passes with no skips, and every test that passed before still does.

Thread "[P7] G7.5 report rules and the features meter gates", on Sonnet 5.5 at medium effort:
Implement what the delta's apply script wrote into the docstrings of src/nablaframe/reports.py, G7.5 in PHASE7_GATE_IDS with its status rule and its required measurements, following G6.7's, and of src/nablaframe/views/features.py, G7.2 among METER_GATES. Change only those two files. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase7_reports_g75.py tests/test_phase7_reports.py tests/test_views_features.py tests/test_reports.py` passes with no skips, and every test that passed before still does.
```

**Wave B** has two threads, which share no files. Start both once all of wave A has merged:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P7] Viewer: crystallize-this, records folders and Phase 7 badges", on Opus 5.5 at high effort:
Implement what viewer/app.js's header contract now says about Phase 7's badges and a set edit, several records folders in ?gestures=, and crystallize-this records, in viewer/app.js and viewer/index.html, adding to the header's "How this page meets it" what your choices settle. Plain JavaScript, no build step, nothing outside the repo. Change only those two files. Exercise the page in a scratch folder: on a tiny modular-addition run, traced, with its views built by build_views and a features payload from views/features.py whose meters cite a fixture G7.2 report; with two records folders of that run, each with views/gestures.write_panel: one holding name records built with gestures/cli.py's build on a small dictionary, and the other crystallize-this records of each form, built the same way and citing the first folder's name records, one with a counterfactual and one in the fourier form; with a badges file holding views/badges.py's, views/dictionary_badges.py's, views/gesture_badges.py's and views/crystallize_badges.py's badges on fixture reports; and with a compare payload whose spec has a set edit. Say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.

Thread "[P7] G7.5, G7.2's meters and the Phase 7 summary", on Opus 5.5 at high effort:
Implement gates/phase7/g75_records_views.py and gates/phase7/summarize.py against their docstrings, docs/specs/phase-7.md, D38, D47, D53 to D60 and docs/deltas/phase-7-delta-4.md, and add to gates/phase7/g72_names.py the `meters` its docstring now names, changing nothing else it measures. You may add to gates/phase7/_common.py what several gates share, below its contract, without changing what the contract says. Load the earlier phases' gates and _common.py modules by path, but don't change them. Change only files in gates/phase7/; never change a gesture module, since a record's key holds their blob hashes. Every record goes through gestures/cli.py. Every long step prints progress at least every 30 seconds, and each crystallization a line per checkpoint. Never run a gate for results and never write under reports/. Exercise G7.5 in a scratch folder: make its prerequisites there at shrunk settings, with G7.1 to G7.4 run there so that reports_match is exercised for real, and fixture reports only where a gate would take hours; and exercise summarize.py on the scratch reports of all five gates. Say in the PR what that covered, how many records each run gets, and how long G7.5 would take at full size on this machine. Numbers from here aren't gate results: only the Mac's are. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest` passes in full with no skips.
```

## Running the gates

After both wave B threads have merged, on the Mac, in order:

```
git pull
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g72_names.py --overwrite
git add reports/phase7/g72_names.json
git commit -m "G7.2 report, with its meters"
caffeinate -i env UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g75_records_views.py
git add reports/phase7/g75_records_views.json
git commit -m "G7.5 report"
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/summarize.py
git add reports/phase7/summary.md
git commit -m "Phase 7 summary"
git push
```

G7.2 runs again only to record its meters: its dictionaries come from the cache and its records are found, so it measures what it measured before. G7.5 trains nothing. It builds every record twice, so expect about 35 minutes, most of it the seven crystallizations again. It prints the viewer's address for each run. With the reports pushed, H7 is apolloh0's ruling, after a session in the viewer.

## Verification record

**Verified** on clones of my mirror of `main` at `aad82a6`, with this delta's files added and the apply script run:

- **On the stubs:** the 9 tests above fail, and all 378 others pass, with no skips.
- **On a private reference implementation, not shipped:** all 387 pass with no skips, its docstrings checked equal to the delta's. It covers the panel's meter, both badges, D53's line, the features meter gates and G7.5's report rules.
  - `test_crystallize_payloads` takes about 40 seconds on a cloud CPU, most of it building its records.
  - The panel's records are real: `cli.build` makes a crystallize-this record of each form on test_crystallize_cli's toy, one with a counterfactual, and two name records of the same run on a small dictionary.
- **On the Mac's own records and reports:**
  - The reference panel reads the Phase 7 records folders as G7.4 left them: modular addition's 130 name records and 4 crystallizations, two-hop's 2 and induction's 1, each checked against its SHA-256 and key, in 0.2 seconds. Each meter is its record's: for example the fourier form's KL 0.0185 against 4.14.
  - The reference badges read G7.2 to G7.4's committed reports: `crystallize` judged pass with 59 facts, and `set` judged pass, 51 identities exact of 51. Phase 6's reports give the open-this-up line "twin: the planted set's first part in the top 8 parts, most of any judged target".
- **The apply script** makes its 16 edits in ten files and stops on a second run.
- **Planted bugs:** all 48 are caught by the tests aimed at them. They include:
  - **the meter:** its keys left out or reordered, the results' meter passed through whole, the twin's meter, the share removed worked out from the KLs, the loss delta from the losses, agreement read as accuracy, a missing meter or entry read as None, a crystallization metered as generalize-this, crystallizations linked as names
  - **the badges:**
    - forms, places or names walked in the report's order
    - the build flags swapped, the two agreements swapped, the envelope flag left out
    - a KL, the twin's distance or the FVU in the wrong unit; a span as a step
    - every crystallization marked judged, or judged ones marked recorded; names not marked recorded
    - the trained label for the twin's, the envelope's first run for its final one, the twin's names for the trained, the null's follows for the follows
    - crystallize judged by G7.3 alone, `set` by G7.4 too; notes or identities reordered; sources unsorted
    - a missing, invalid or misfiled report accepted; other reports refused; a missing path read as None; values shared with the reports
    - D53's line left as it was
  - **the features view:** G7.2 not among the meter gates
  - **the report rules:** G7.5 missing from the Phase 7 gates, always passing, or passing without `reports_match`, with an external URL, with a subject missing or with builds that differ; G7.5 without its records

  Three first escaped, and the tests were strengthened. An invalid report and a misfiled one were refused only because a path was missing, so the test now tries a report whose status its measurements contradict, and a valid G7.4 report that also holds G7.3's measurements. A third, another open-this-up line rewritten, changes nothing the contract pins, so it was dropped.

**Not verified:** the viewer, G7.5, G7.2's meters and the summary, which wave B builds, and anything on the Mac.
