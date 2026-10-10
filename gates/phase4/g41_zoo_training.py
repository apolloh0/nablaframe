"""G4.1: both zoo models trained, traced and retrained to a seeded recompute, and held to their behaviour bars (D29, D30, D36).

Spec: docs/specs/phase-4.md, "The zoo" and "Gates"; helpers in ``_common.py``. Report:
``reports/phase4/g41_zoo_training.json``. Run in nf-jax, on the Mac.

For ``"induction"`` and then ``"two_hop"``, each with its task from ``zoo/tasks.py`` at its
defaults and its settings from ``configs(name)``:

* **Train:** from step 0 with ``run_until(..., batch_source=lambda k: task.batch_for_step(0, k,
  batch))``, saving a checkpoint at every ``checkpoint_steps(name)`` step under
  ``runs/phase4/<name>/checkpoints/``, beside a run manifest and the task's tokenizer file, so the
  trace engine and the views take the run as they take any (D29). The manifest's ``corpus_id`` is
  ``"zoo:<name>"`` and its ``corpus_sha256`` the SHA-256 of the task's canonical JSON, its class
  name and fields; its tokenizer fields name the tokenizer file and its SHA-256.
* **Trace:** every checkpoint, with every canonical tap and the logits, on what the model reads of
  ``probe(task)``, its first ``seq`` tokens, through ``TraceRun`` into
  ``runs/phase4/<name>/store/``, as G1.6 traces the reference run.
* **Recompute:** trains again from step 0 into a scratch folder under ``runs/phase4/scratch/`` and
  compares the final state hash with the first run's.
* **Behaviour,** from logits alone at the final checkpoint, with ``next_token_accuracy``:
  induction on ``eval_windows`` at its mechanism and its control positions, and its
  random-weights twin (``twin_params``) at the mechanism positions; two-hop on ``eval_windows``
  and on ``broken_windows`` at position 25, and its twin on ``eval_windows``.

Measurements: REQUIRED_MEASUREMENTS["G4.1"]. ``recompute_identical`` is true when both final
hashes match; ``checkpoints_scheduled`` is 62; ``events_expected`` is each model's canonical taps
plus the logits, times its checkpoints, 1,576. Also recorded: each model's final state hash,
mechanism accuracy at every checkpoint on the evaluation windows, and the seconds each stage
took. Pass needs every bar.

Run folders start fresh: an existing one is removed only with ``--overwrite``. ``--only
induction`` or ``--only two_hop`` runs one model and writes no report, so one model can be
exercised alone. If the trace engine's oracle refuses a checkpoint, or a recompute isn't
reproducible, the report fails and records the error, as G1.6's does.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
