"""G6.1: ``dsteer`` keeps D48's identities at every TopK site, requests hash apart, and each kind of record rebuilds byte for byte (D47, D48).

Spec: docs/specs/phase-6.md, D47, D48 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase6/g61_determinism.json``. Run in nf-jax, on the Mac, after G4.1, G5.2 and G5.3 pass.

On G4.1's runs at their final checkpoints, with the Phase 5 TopK SAEs from the cache:

* **Identities:** for both zoo models, at every block's TopK SAE, ``dsteer_identities`` on the
  tokens of the first ``IDENTITY_WINDOWS`` held-out windows. ``identities_by_site``:
  ``{"<run> topk <output tap>": [one bool per identity, in IDENTITIES order]}``;
  ``identities_checked`` and ``identities_exact``, counts over both models.
* **Request hashes:** on the two-hop run, ``request_hashes(requests(run id, its checkpoint
  steps, (block 2's TopK content hash, block 3's)))``: ``request_hashes_checked``,
  ``request_hashes_distinct`` and ``request_hashes_stable``.
* **Records:** on the two-hop run's ``cli.Context``: its held-out windows, every checkpoint, the
  TopK SAE at every block, and ``ROBUST_SEEDS``'s TopK SAEs at every block as its seeds. One
  request of each kind, at the final checkpoint, with ``RECORD_INSTANCES`` instances and seed 0,
  ``v`` being the least value of ``b`` at block 2's TopK SAE at ``query``:

  - **name:** block 2's TopK SAE, site 0, class ``query``, the latent ``ranked("b", v)[0]``
  - **edge:** ``hop``, from block 2's TopK SAE to block 3's, sites 0
  - **open:** block 2's TopK SAE at class ``query``, the name ``b = v``, into block 1's
  - **generalize:** ``b`` at block 2's TopK SAE, class ``query``

  Each is built twice with ``cli.build``, never through a cache, and once more by the command
  line's code: ``cli.main``, called in this process, with ``--run two_hop``, files written under
  ``runs/phase6/g61_determinism/`` (the config, the held-out windows, the seeds' map, the
  request, and a fresh ``--out``), G4.1's checkpoints folder and the Phase 5 dictionary cache.
  ``records_by_kind``: ``{kind: [the two builds' canonical JSON is identical, the command line's
  file is that and a newline]}``. ``records_identical``: all four kinds checked, every value true.

Measurements: REQUIRED_MEASUREMENTS["G6.1"]. Pass needs at least one identity checked and every one
exact, the request hashes distinct and stable, and ``records_identical``. The working folder,
``runs/phase6/g61_determinism/``, is started fresh, and ``--overwrite`` lets it replace one.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
