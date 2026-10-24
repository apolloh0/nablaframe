"""G7.2: the modular-addition specimen's dictionary and its names at ``answer``, recorded, not judged (C11, D57).

Spec: docs/specs/phase-7.md, "The names on modular addition"; helpers in ``_common.py``. Report:
``reports/phase7/g72_names.json``. Run in nf-jax, on the Mac, after G7.1 passes.

On G7.1's run at its final checkpoint, and on its twin:

* **The dictionary:** a TopK SAE at block 0's ``hook_resid_post`` with Phase 5's settings and
  ``DICTIONARY``'s steps and batch: ``core.Settings("topk", p5.WIDTH_FACTOR * 32, p5.K,
  DICTIONARY["steps"], DICTIONARY["batch"], lr=p5.LR, seed=p5.DICT_SEED,
  jac_batch=p5.JAC_BATCH)``, width 256 and ``k`` 16. ``p5.settings`` takes only Phase 5's runs,
  so the gate builds it. Trained on the activations of the training pairs, every position,
  through ``p5.train_dictionary`` and ``p5.cached`` into ``runs/phase7/dictionaries/``, under
  ``p5.dictionary_key`` of the run's final state, the site, the settings and the tokens the model
  reads of the training pairs. The twin's is trained the same way on the twin's
  activations, keyed by ``p5.twin_state``. ``fvu`` and ``twin_fvu``: each on the held-out pairs'
  activations, as Phase 5 measures it.
* **Names:** for each of ``NAMES`` at class ``answer`` on the held-out pairs, Phase 6's
  ``planted_name`` with ``INSTANCES["name"]`` and ``SEEDS["name"]``, trained and twin.
  ``names``: ``{description: {"trained": ..., "twin": ...}}``, each that mapping without its
  ``by_value``. This is what C11 rules on: the spec expects the sum named as planted for about a
  quarter of its values.
* **Records:** for each of ``NAMES`` and each of its values at ``answer``, a name record of that
  value's best latent by F1, ``ranked(d, v)[0]``, at the final checkpoint, with
  ``INSTANCES["name"]`` and ``SEEDS["name"]``, through ``cli.record`` into
  ``runs/phase7/gestures/modular_addition/``. The run's ``cli.Context`` holds G7.1's checkpoints,
  the held-out pairs and the trained dictionary. A latent best for two values is recorded once,
  and the twin, which isn't a checkpoint of its run, gets no records. G7.4 cites them, and G7.5 builds them again.
  ``records``: how many were written or found.

Measurements: REQUIRED_MEASUREMENTS["G7.2"]. Its status is always ``pass``: it records, not judges.
``seconds`` holds each stage's time.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
