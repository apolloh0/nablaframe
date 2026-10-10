"""G4.6: model-level badges and the zoo's demos for the viewer (D34, D38).

Spec: docs/specs/phase-4.md, "Gates"; docs/deltas/phase-4-delta-4.md. Report:
``reports/phase4/g46_badges_demos.json``. Run in nf-jax, on the Mac, after G4.1 to G4.4 have run.

Refuses to start unless G4.1's report passes and G4.2 to G4.4's reports exist and pass
``validate_report``, whatever their status: a failed judged gate is shown on its badge, never
hidden by a missing one.

* **Badges:** ``model_level_badges`` of the G4.1 to G4.4 reports, written as canonical JSON to
  ``runs/phase4/badges.json``: ``{"badges": <the badges>, "reports": {<gate id>:
  {"git_commit", "timestamp_utc", "sha256"}}}``, the last being each report file's SHA-256.
  Built a second time into ``runs/phase4/scratch/badges.json`` and compared byte for byte.
* **Demos:** compare payloads written by ``interventions.cli.main`` for each zoo model's final
  checkpoint, on what the model reads of ``probe(task)``, with ``--taps`` every block's
  ``hook_resid_post``, into ``runs/phase4/<name>/interventions/``, which starts empty:

  - induction: G4.2's ``induction_heads`` zeroed together through ``blocks.1.attn.hook_z``, one
    ``zero`` edit with ``head`` per head, in one spec
  - two-hop: two patches at G4.3's ``best_swap`` site and block, of
    ``blocks.<block>.hook_resid_post``: one from the aligned sources and one from their
    controls. Both come from ``aligned_sources(task, eval_windows(task),
    numpy.random.default_rng([SOURCE_SEED, 0]))``, rows 0 to 7, since the probe is the first
    eight evaluation windows, so they are G4.3's own sources for those windows.

  Every object a spec names is written to the command line's ``--objects`` folder. All of it
  is built a second time into ``runs/phase4/scratch/interventions/<name>/`` and compared file by
  file.

Measurements: REQUIRED_MEASUREMENTS["G4.6"]. ``badges`` is the badges mapping;
``subjects_missing`` counts ``SUBJECTS`` without a badge; ``builds_identical`` holds when the
badges and every compare folder match their second builds; ``payloads`` counts each model's
compare payloads. Recorded beside them, ``demo``: for the two-hop source patch on the probe's
eight windows, each window's ``c``, ``d`` and ``d′`` and its top-1 answer at position 25 before
and after the patch, read from the payload's ``top_tokens``. Pass needs ``builds_identical``
and ``subjects_missing == 0``.

At the end it prints the viewer's address for each zoo run, with ``?views=``, ``?trace=``,
``?interventions=`` and ``?badges=`` set.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
