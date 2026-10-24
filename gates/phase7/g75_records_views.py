"""G7.5: modular addition's views, every Phase 7 record built twice, the gestures panels, and the badges (D38, D47, D53, D59).

Spec: docs/specs/phase-7.md, "Records, the viewer and badges" and "Gates";
docs/deltas/phase-7-delta-4.md; helpers in ``_common.py``. Report:
``reports/phase7/g75_records_views.json``. Run in nf-jax, on the Mac, after G7.1 to G7.4 pass and
G7.2 has run again with its ``meters``.

Refuses to start unless G4.1's, G5.6's, G6.7's and G7.1 to G7.4's reports pass, and G4.2 to G4.4's,
G5.1 to G5.4's and G6.1 to G6.5's exist and pass ``validate_report``, whatever their status: a
failed judged gate shows on its badge, never hidden by a missing one. G7.2's report must hold
``meters``; one without them is a setup error that says to run G7.2 again.

**Records.** Each run's requests, made in the order listed; one equal to an earlier one is built
once.

* ``modular_addition``, at G7.1's final checkpoint, its site read once with Phase 6's ``read`` on
  G7.2's TopK SAE (``_common.dictionary``, from the cache):

  1. **Names:** for each of ``NAMES`` at ``answer``, for each of its values ``v``, ascending, a name
     request pointed at ``v``'s best latent by F1, ``table.ranked(d, v)[0]``, at site 0, with Phase
     6's ``INSTANCES["name"]`` and ``SEEDS["name"]``: G7.2's records.
  2. **The readout:** on the probe windows (``probe(task)``, the tokens the model reads), the
     latents active at position 2, ``answer``, largest code first, ties to the lower latent, up to 8
     per window: what the features view lists there. The codes are ``core.encode`` of the site's
     input tap, from ``measure.activations`` at the final params with ``CHUNK``. A name request for
     each, at ``answer``, with the names' instances and seed.
  3. **Crystallizations:** for each of ``JUDGED`` and then ``RECORDED`` on the run, in order, the
     request of the record G7.4's report names, ``crystallizations[<name>]["key"]``, read from the
     run's records folder. A record missing there, or whose request isn't that crystallization's
     (its block, part, class, function and form, the run's final step, ``instances`` the held-out
     windows' count and ``SEED``), is a setup error.

* ``induction`` and ``two_hop``, at G4.1's final checkpoint: their crystallizations, as in step 3.

The name records' context holds the run's checkpoints, its held-out pairs and G7.2's dictionary, as
G7.2's did. Each crystallization's is the run's context as G7.4 built it (``_common.context`` with
every checkpoint and ``cited``), ``cited`` reading, for modular addition, the folder the build
writes into, whose name records are built before any crystallization, and for induction and two-hop,
Phase 6's records, ``runs/phase6/gestures/<run>/``. Every record goes through ``cli.record``.

**Two builds.** Before anything is built, every request's key is found, ``records.record_key`` of the
request, ``cli.inputs`` and ``cli.code``. A run's records folder, ``record_dir(run)``, holding a
record whose key isn't one of them stops the gate, naming it. The first build goes into that
folder, through the record cache, so the records G7.2 and G7.4 wrote are kept as they are; the
second into ``RUN_DIR / "scratch" / "gestures" / run``, which starts empty. Each folder then gets
``views.gestures.write_panel``. The two are compared file by file: every file in either folder, by
name, with its bytes. Phase 6's records and panels stay as they are (D59).

**Views.** Modular addition's views, built as G4.5 and G5.6 build a zoo run's: ``build_views`` over
G7.1's trace store, which must hold exactly G7.1's checkpoints, each with its logits, traced on the
probe; ``params_for`` its checkpoints; ``probe`` what the model reads of ``probe(task)``; ``vocab``
the task's; ``next_token`` the probe windows' last tokens; ``curves`` one curve, ``"answer
position's loss"`` in ``"nats"``: at each checkpoint, the mean over the probe of ``-log
softmax(logits[b, 2])[window[b, 3]]`` from the traced logits, in float64, null where it isn't
finite; and ``features``, one ``features_payload`` of G7.2's TopK SAE at site 0 with the final
params and step, that probe, ``held_out`` the tokens the model reads of the held-out pairs,
``meters`` ``{"gate": "G7.2", "path": ["meters"], "values": <that entry of G7.2's report>}``,
``variables`` each held-out pair's ``a``, ``b`` and ``c``, its tokens 0, 1 and 3, and ``chunk``
``CHUNK``. The first build goes to ``runs/phase7/views/modular_addition/`` and the second to
``runs/phase7/scratch/views/modular_addition/``; both start empty, and each computes its payloads
afresh. Compared file by file.

**Badges.** ``runs/phase7/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1 to
G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, then gesture_badges of G6.1 to G6.5, then
crystallize_badges of G7.2 to G7.4, as one mapping>, "reports": {<gate id>: {"git_commit",
"timestamp_utc", "sha256"}}}`` for each of those sixteen reports, the last being the report file's
SHA-256. Built a second time into ``runs/phase7/scratch/badges.json`` and compared byte for byte.
One file holds every phase's badges, with D53's correction to the open-this-up badge, so the viewer
reads it with ``?badges=`` for every run (D59).

An existing second records folder, view folder or badges file, or an existing first view folder,
is replaced only with ``--overwrite``; a first records folder is never cleared.

Measurements: REQUIRED_MEASUREMENTS["G7.5"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``,
  ``dictionary_badges.SUBJECTS``, ``gesture_badges.SUBJECTS`` and ``crystallize_badges.SUBJECTS``
  without a badge.
* ``records``: per run, ``{"name", "crystallize"}``, how many records of each kind its first folder
  holds.
* ``builds_identical``: the two badges files hold the same bytes, the two view folders the same
  files with the same bytes, and each run's two records folders the same files with the same bytes.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.
* ``reports_match``: each crystallization's record, as built here, has the key G7.4's report names,
  and each name record G7.2's report names (``record_keys``) is in modular addition's first folder.
  So the records the panel shows are the ones G7.4 measured and G7.2 recorded.

Recorded beside them: ``builds``, each part's detail (files compared and files differing);
``matches``, every key compared; ``run_ids``; ``steps``; ``requests``, per run and step, how many
requests were made before equal ones were merged; ``readout``, per probe window, the latents of step
2; ``clock``, modular addition's fourier crystallization from its first folder: its key, and its
results' ``frequencies``, ``envelope`` and ``cites``; ``features``, the features payload's file and
dictionary; and ``addresses``.

Pass needs ``builds_identical``, ``external_urls == 0``, ``subjects_missing == 0`` and
``reports_match``. Every long step prints a progress line at least every 30 seconds, saying how many
records of how many it has built, and each crystallization a line per checkpoint, as G7.4 does. At
the end it prints the viewer's address for each run:

* modular addition: ``http://localhost:8000/viewer/?views=../runs/phase7/views/modular_addition/
  &trace=../runs/phase7/modular_addition/store/&badges=../runs/phase7/badges.json
  &gestures=../runs/phase7/gestures/modular_addition/#step=<final>&seq=0&pos=2
  &features=<dictionary>-0``, as one line
* each of G6.7's runs: G6.7's address, from its report's ``addresses``, with ``badges=`` pointing at
  ``../runs/phase7/badges.json``, and for induction and two-hop ``,../runs/phase7/gestures/<run>/``
  added after its Phase 6 records folder, so the viewer reads both (``viewer/app.js``)
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
