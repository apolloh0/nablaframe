"""G6.7: every gesture record built twice, the gestures panel, and the badges (D38, D47).

Spec: docs/specs/phase-6.md, "Records, the viewer and badges" and "Gates";
docs/deltas/phase-6-delta-5.md; helpers in ``_common.py``. Report:
``reports/phase6/g67_records_panel.json``. Run in nf-jax, on the Mac, after G6.1 to G6.6 have run.

Refuses to start unless G4.1's, G5.2's, G5.3's, G5.5's, G5.6's and G6.1's reports pass, and G4.2 to
G4.4's, G5.1's, G5.4's and G6.2 to G6.6's exist and pass ``validate_report``, whatever their status:
a failed judged gate shows on its badge, never hidden by a missing one.

**Records.** On each run at its final checkpoint, with the Phase 5 TopK SAEs from the cache, every
request below goes through ``cli.record`` with the run's context. A zoo run's context is
``_common.context`` holding the TopK SAE of every block, with the seed 1 and 2 TopK SAEs of its
``GENERALIZE`` block as its seeds, as G6.5 held them. The reference run's holds its run id, every
checkpoint of G1.4's run with its params and state hash, its held-out windows and the TopK SAE of
every block, with no seeds. Requests are made in the order listed; one equal to an earlier one is
built once.

* ``two_hop`` and ``induction``, each place read once with ``_common.read``:

  1. **Names:** for each place G6.2 reads (a planted name's description, block and class; two-hop's
     ``b`` after block 1 included), for each value ``v`` of the description at the class,
     ascending, a name request pointed at ``v``'s best latent by F1, ``table.ranked(d, v)[0]``, on
     the block's TopK SAE at site 0, with ``INSTANCES["name"]`` and ``SEEDS["name"]``. Where that
     latent is named ``d = v``, its steering is G6.2's for ``v``.
  2. **The readout:** on the probe windows (``p4.probe`` of the task, the tokens the model reads),
     at each of those places' blocks, the latents active at each window's last position, largest
     code first, ties to the lower latent, up to 8 per window: what the features view lists there.
     The codes are ``core.encode`` of the site's input tap, from ``measure.activations`` at the
     final params. A name request for each, at the place's class, with ``INSTANCES["name"]`` and
     ``SEEDS["name"]``.
  3. **Edges:** for each relation the run declares, G6.3's request: ``{"relation", "up",
     "up_site": 0, "down", "down_site": 0}`` with the TopK SAEs at its places' blocks,
     ``INSTANCES["edge"]`` and ``SEEDS["edge"]``.
  4. **Open-this-up:** for each of the run's targets in ``OPEN``, read along its relation, its
     downstream place the target and its upstream block the one opened into: for each value ``u``
     of the downstream description at its class, ascending, ``{"dictionary", "site": 0, "class",
     "upstream", "upstream_site": 0, "description", "value": u}``, with ``INSTANCES["open"]`` and
     ``SEEDS["open"]``.
  5. **Generalize-this:** the run's name in ``GENERALIZE``, G6.5's request: ``{"dictionary",
     "site": 0, "class", "description"}``, ``INSTANCES["name"]`` and ``SEEDS["generalize"]``.

* ``reference``, at each block ``l``, with ``T`` its reading's table at ``every`` and ``rng =
  numpy.random.default_rng([SEEDS["reference"], l])``, drawing as G6.6 draws:

  1. **Names:** each latent of ``reference_sample(T, REFERENCE["latents"], rng)``, at ``every``,
     with ``REFERENCE["per_value"]`` instances and ``SEEDS["reference"]``.
  2. **Steered values:** each value ``v`` of ``reference_values(T, "token(p+1)",
     REFERENCE["values"], rng)``, the same ``rng`` drawing after the sample: a name request pointed
     at ``v``'s best latent by F1, at ``every``, with the names' instances and seed.
  3. **Open-this-up,** at blocks 1 and above: for each such ``v``, the set of ``"token(p+1)" = v``
     at ``every``, opened into block ``l - 1``'s TopK SAE, with ``REFERENCE["open"]`` instances and
     ``SEEDS["reference"]``.

**Two builds.** Before anything is built, every request's key is found, ``records.record_key`` of
the request, ``cli.inputs`` and ``cli.code``. A run's record folder, ``RECORD_DIR / run``, holding a
record whose key isn't one of them stops the gate, naming it. The first build goes into that folder,
through the record cache, so the records G6.3 and G6.5 wrote are kept as they are; the second into
``RUN_DIR / "scratch" / "gestures" / run``, which starts empty. An existing second folder, or an
existing badges file, is replaced only with ``--overwrite``. Each folder then gets
``views.gestures.write_panel``. The two are compared file by file: every file in either folder, by
name, with its bytes.

**Badges.** ``runs/phase6/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1
to G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, then gesture_badges of G6.1 to G6.5, as
one mapping>, "reports": {<gate id>: {"git_commit", "timestamp_utc", "sha256"}}}`` for each of those
thirteen reports, the last being the report file's SHA-256. Built a second time into
``runs/phase6/scratch/badges.json`` and compared byte for byte. One file holds every model-level
badge, so the viewer reads it with ``?badges=`` and finds every phase's subjects.

Measurements: REQUIRED_MEASUREMENTS["G6.7"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``,
  ``dictionary_badges.SUBJECTS`` and ``gesture_badges.SUBJECTS`` without a badge.
* ``records``: per run, ``{"name", "edge", "open", "generalize"}``, how many records of each kind
  its first folder holds.
* ``builds_identical``: the two badges files hold the same bytes, and each run's two record folders
  hold the same files with the same bytes.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.
* ``reports_match``: each edge record G6.3's report names and each generalize record G6.5's names,
  by key, is in its run's first folder, and each reference block's steered values are G6.6's
  (``blocks[l]["steering"]["values"]``). So the records the panel shows are the ones the gates
  judged, and the reference's are the ones G6.6 recorded.

Recorded beside them: ``builds``, each part's detail (files compared and files differing);
``matches``, every key and value list compared; ``run_ids``; ``steps``; ``requests``, per run and
kind, how many requests were made before equal ones were merged; ``intermediate``, two-hop's ``b``
after block 2: its generalize record's key and its results' ``envelope``, and for each value ``v``
ascending, ``{"value", "latent", "key", "meter"}``, the name record of step 1 and its meter from the
panel; and ``addresses``.

Pass needs ``builds_identical``, ``external_urls == 0``, ``subjects_missing == 0`` and
``reports_match``. Every long step prints a progress line at least every 30 seconds, saying how many
records of how many it has built. At the end it prints the viewer's address for each run: G5.6's,
from its report's ``addresses``, with ``badges=`` pointing at ``../runs/phase6/badges.json`` and
``&gestures=../runs/phase6/gestures/<run>/`` added before the fragment.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
