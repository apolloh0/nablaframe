"""G5.1: dictionaries train bit for bit, their cache keys are distinct, and ``dpatch`` keeps D25's identities.

Spec: docs/specs/phase-5.md, D40, D45 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase5/g51_determinism.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's runs at their final checkpoints:

* **Determinism:** on the induction model, each of ``topk`` (block 0), ``staircase``,
  ``transcoder`` (block 0) and ``jsae`` (block 0, ``lam`` 1.0) is trained twice from one key with
  ``settings(..., short=True)`` on ``zoo_windows(task, "induction", "train")``, each time from
  scratch through ``train_dictionary``, never from the cache, and saved into a folder of its own.
  ``identical_by_kind``: whether both ``.safetensors`` files and both ``.json`` files are
  byte-identical; ``files_identical``, whether every kind's are; ``kinds_checked``, the kinds.
* **Cache keys:** ``dictionary_key`` for a ``topk`` at the induction model's block 0, over the 64
  combinations of two values each of width (``WIDTH_FACTOR`` and half of it, times d_model),
  ``k`` (16 and 8), ``steps`` (3,000 and 2,000), ``batch`` (4,096 and 2,048), ``lr`` (5e-4 and
  1e-3) and ``seed`` (0 and 1), then four more keys from the first, each with one other input
  changed: the state hash (the twin's), the sites (block 1), the training windows (the held-out
  ones) and the code (the blob hash of ``core.py`` instead of ``train.py``).
  ``cache_keys_checked`` (68) and ``cache_keys_distinct``.
* **Identities:** for both zoo models, one dictionary of each shipped kind at every site, trained
  with ``settings(..., short=True)`` through ``cached`` (``topk`` at every block, the
  ``staircase`` at each of its sites, a ``transcoder`` at every MLP). At each site's output tap, on
  the first 64 held-out windows, two ``dpatch`` edits at every position through
  ``run_intervened``: from a source equal to the target, and with ``latents=[]`` from the second 64
  held-out windows. Each must give logits bit for bit equal to the unintervened pass's.
  ``identities_by_site``: ``{"<model> <kind> <output tap>": [same source, no latents]}``, each
  true or false; ``identities_checked`` and ``identities_exact``, counts over both models.

Measurements: REQUIRED_MEASUREMENTS["G5.1"]. Pass needs ``files_identical``,
``cache_keys_distinct``, at least one identity checked and every one exact.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
