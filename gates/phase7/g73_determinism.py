"""G7.3: ``set`` keeps its identities on every zoo model, crystallize-this requests hash apart, and each form's record rebuilds byte for byte (D47, D55, D56).

Spec: docs/specs/phase-7.md, D55, D56 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase7/g73_determinism.json``. Run in nf-jax, on the Mac, after G4.1, G5.2, G5.3, G7.1
and G7.2 pass.

* **Identities:** for the induction, two-hop and modular-addition models at their final
  checkpoints, ``set_identities`` on the tokens the model reads of the first
  ``IDENTITY_WINDOWS`` held-out windows: Phase 5's for the zoo runs, the held-out pairs for
  modular addition.
  ``identities_by_run``: ``{run: {tap: bool}}``; ``identities_checked`` and ``identities_exact``,
  counts over all three.
* **Request hashes:** ``request_hashes(requests(run id, its checkpoint steps, two record keys))``,
  the keys being any two of G7.2's name records: ``request_hashes_checked``,
  ``request_hashes_distinct`` and ``request_hashes_stable``.
* **Records:** on the modular-addition run's ``cli.Context`` with two checkpoints, step 0 and
  the final one: its training pairs as fitting windows, its held-out pairs, its twin, its
  functions, and ``cited`` reading ``runs/phase7/gestures/modular_addition/``. Two checkpoints
  keep nine builds affordable; G7.4 reads every one. One request per form, at the final
  checkpoint, with ``SEED``: the write as a table by the sum and the MLP additive in the sum and
  operands, each citing G7.2's records as G7.4 cites them; and the write in the fourier form,
  citing nothing. Each is built twice with
  ``cli.build``, never through a cache, and once more by the command line's code: ``cli.main``,
  called in this process, with ``--run modular_addition`` and files written under
  ``runs/phase7/g73_determinism/``: the config, both windows' ``.npy`` files, the request, a
  ``checkpoints`` folder holding copies of those two checkpoints with their sidecars, and a fresh
  ``--out``; ``--records`` is the run's record folder. ``records_by_form``: ``{form: [the two
  builds' canonical JSON is identical, the command line's file is that and a newline]}``.
  ``records_identical``: all three forms checked, every value true.

Measurements: REQUIRED_MEASUREMENTS["G7.3"]. Pass needs at least one identity checked and every
one exact, the request hashes distinct and stable, and ``records_identical``. The working folder,
``runs/phase7/g73_determinism/``, is started fresh, and ``--overwrite`` lets it replace one.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
