"""G7.1: the modular-addition specimen trained, traced and retrained, holding its behaviour and its clock (D54).

Spec: docs/specs/phase-7.md, D54 and "The specimen"; helpers in ``_common.py``. Report:
``reports/phase7/g71_specimen.json``. Run in nf-jax, on the Mac.

``ModularAdditionTask()`` at its defaults, with ``configs()``:

* **Train:** from step 0 with ``run_until(..., batch_source=lambda k: task.batch_for_step(seed, k,
  batch))``, ``seed`` and ``batch`` being the ``TrainConfig``'s ``data_seed`` and ``batch``, as
  G4.1 trains, saving a checkpoint at every ``checkpoint_steps()`` step under
  ``runs/phase7/modular_addition/checkpoints/``, beside a run manifest and the task's tokenizer
  file, as G4.1 saves a zoo run: ``corpus_id`` ``"zoo:modular_addition"``, ``corpus_sha256`` the
  SHA-256 of the task's canonical JSON, its class name and fields. G4.1's training, manifest and
  checkpoint code may be loaded by path and reused; nothing in Phase 4 changes.
* **Trace:** every checkpoint, with every canonical tap and the logits, on what the model reads
  of ``probe(task)``, through ``TraceRun`` into ``runs/phase7/modular_addition/store/``, as G4.1
  traces the zoo runs.
* **Recompute:** trains again from step 0 into ``runs/phase7/scratch/`` and compares the final
  state hash with the first run's.
* **Behaviour,** from logits alone at the final checkpoint: ``held_accuracy`` and
  ``train_accuracy`` at position 2 on the held-out and training pairs; ``control_accuracy`` at
  position 0 on the held-out pairs, where nothing predicts ``b`` from ``a``; ``twin_accuracy`` at
  position 2 on the held-out pairs, the twin being ``init_params(config, seed=1)``; ``chance``,
  ``1 / p``.
* **The clock,** on the final checkpoint's weights and logits, never a view: ``frequencies``
  and ``energy``, ``fewest_frequencies(params.wte[:p], p)``'s count and share;
  ``twin_frequencies``, the twin's count; and ``logit_r2``, ``logit_r2`` of the training pairs'
  and the held-out pairs' logits at position 2, at the frequencies
  ``crystallize.frequencies(params.wte[:p], p)["k"]``: the five or so the fourier form keeps.

Measurements: REQUIRED_MEASUREMENTS["G7.1"]. ``checkpoints_scheduled`` is 45 and
``events_expected`` the canonical taps plus the logits, times 45. Also recorded: the run id, the
final state hash, training and held-out accuracy at every checkpoint, the first step each
reaches 0.99, and the seconds each stage took. Pass needs every bar. The run folder starts
fresh: an existing one is removed only with ``--overwrite``. If the trace engine's oracle refuses
a checkpoint, or the recompute isn't reproducible, the report fails and records the error, as
G4.1's does.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
