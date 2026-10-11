"""G5.6: the features view, the dictionary badges, and every run's views rebuilt (D38, D39, D44, D45).

Spec: docs/specs/phase-5.md, D44, D45 and "Gates"; docs/deltas/phase-5-delta-5.md; helpers in
``_common.py``. Report: ``reports/phase5/g56_features_badges.json``. Run in nf-jax, on the Mac,
after G5.1 to G5.5 have run.

Refuses to start unless G4.1's and G5.1's reports pass, and G4.2 to G4.4's and G5.2 to G5.5's exist
and pass ``validate_report``, whatever their status: a failed judged gate shows on its badge, never
hidden by a missing one. It reads G4.1's two zoo runs, as G4.6 does, and the reference run, G1.4's
checkpoints and G1.6's trace, as G2.6 does.

* **Badges:** ``runs/phase5/badges.json``, canonical JSON: ``{"badges": <model_level_badges of G4.1
  to G4.4, then dictionary_badges of G4.1 and G5.1 to G5.4, as one mapping>, "reports": {<gate id>:
  {"git_commit", "timestamp_utc", "sha256"}}}`` for each of those eight reports, the last being the
  report file's SHA-256. Built a second time into ``runs/phase5/scratch/badges.json`` and compared
  byte for byte. One file holds every model-level badge, so the viewer reads it with ``?badges=``
  and finds both the Phase 4 subjects and the Phase 5 ones.
* **Shipped dictionaries:** on each run, the dictionaries its gate trained on the trained model at
  ``DICT_SEED``:

  - ``reference`` (G5.5): ``topk`` at every block, the ``staircase``, a ``transcoder`` at every block
  - ``induction`` (G5.2): ``topk`` at blocks 0 and 1, the ``staircase``, a ``transcoder`` at blocks 0
    and 1
  - ``two_hop`` (G5.3): ``topk`` at every block, the ``staircase``, a ``transcoder`` at every block

  Each comes through ``cached(DICTIONARY_DIR, key, make)`` under the key its gate trained it with:
  ``dictionary_key`` of the run's final state hash, ``sites``, ``settings(run, kind, d_model)`` and
  the tokens the model reads of the training windows, ``p4.tokens(task, zoo_windows(task, name,
  "train"))`` for a zoo run and ``reference_windows(train, validation, seq, "train")[:, :seq]`` for
  the reference run. ``make`` never trains: if it is called, the dictionary isn't in the cache, and
  the gate exits 1 naming the gate to run, with no report.
* **Features:** for each shipped dictionary, in the order above, and each of its sites in order,
  ``features_payload`` with the run's final parameters and step; ``probe``, the probe the run's
  views use; ``held_out``, the tokens the model reads of the run's held-out windows
  (``p4.tokens(task, zoo_windows(task, name, "held_out"))``, or ``reference_windows(...,
  "held_out")[:, :seq]``); ``meters``, ``{"gate": <the run's gate>, "path": ["meters", "<kind>
  <output tap>"], "values": <that entry of its report>}``; and for two-hop, ``variables``: each
  held-out window's ``a``, ``b`` and ``c`` from ``task.parse``. A missing meters entry is a setup
  error.
* **Views:** each run's views built with ``build_views`` as its gate built them, plus ``features``,
  the run's payloads: the reference run's as G2.6 builds them (its vocabulary, G1.4's
  validation-loss curve and, when G2.9's report passes, the J-lens), and each zoo run's as G4.5
  does (the task's vocabulary, ``next_token`` and ``curves``), using those gates' own functions,
  loaded by path. Their D14 badges now end with D39's sentence. The first build goes to
  ``runs/phase5/views/<run>/`` and the second to ``runs/phase5/scratch/views/<run>/``; both start
  empty, and each computes its payloads afresh. Compared file by file.

Measurements: REQUIRED_MEASUREMENTS["G5.6"].

* ``badges``: the badges mapping. ``subjects_missing``: the subjects of ``views.badges.SUBJECTS``
  and ``dictionary_badges.SUBJECTS`` without a badge.
* ``dictionaries``: per run, a list in the order above of ``{"kind", "sites", "key",
  "content_hash", "meters_gate", "files"}``, ``files`` naming its features payloads.
  ``dictionaries_missing``: the shipped dictionaries whose kind has no badge.
* ``features``: per run, the number of features payloads.
* ``builds_identical``: the two badges files hold the same bytes, and each run's two view folders
  hold the same files, by name, with the same bytes. ``builds`` gives each part's detail: files
  compared and files differing.
* ``external_urls``: external references in the viewer, counted as G4.5 counts them.

Recorded beside them: ``views_dirs``, ``run_ids`` and ``steps`` (each run's final step), and
``query_latents``: from the two-hop payload for the ``topk`` at G5.3's ``best_swap_block``, for
each probe window, its ``b`` from ``task.parse`` and, at position 25, each active latent with its
code and the ``b`` of each of its top contexts' windows, read from the payload. So the report shows
whether the latents active at ``⇒`` are the intermediate's.

Pass needs ``builds_identical``, ``external_urls == 0``, ``dictionaries_missing == 0`` and
``subjects_missing == 0``. At the end it prints the viewer's address for each run, with
``?views=``, ``?trace=`` and ``?badges=`` set, and for each zoo run ``?interventions=`` G4.6's
folder.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g56_features_badges.py``. Takes --overwrite and --allow-dirty. --overwrite also
#   lets it replace both badges files and every build folder.
# * Start checks, before anything is written: the reports, as above; the runs and stores, through
#   the checks G2.6 and G4.6 make; ``model_level_badges`` and ``dictionary_badges`` of the reports,
#   whose ValueError is a setup error; G5.3's ``best_swap_block`` a block of the two-hop model;
#   every shipped dictionary in the cache, and every meters entry in its report. Each fails with
#   exit 1 and no report.
# * Every long step prints a progress line at least every 30 seconds.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GATE_ID = "G5.6"
REPORT_NAME = "g56_features_badges.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
