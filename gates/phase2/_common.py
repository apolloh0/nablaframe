"""Shared by the Phase 2 gates. Read with docs/specs/phase-2.md, "Gates".

May import ``gates/phase1/_common.py`` (load it by path; both files are called
``_common``) but never change it. Reports go to ``reports/phase2/``, working files to
``runs/phase2/``. The reference run is G1.4's, at ``runs/phase1/g14_reference/``, and its
trace store is G1.6's, at ``runs/phase1/g16_trace/store/``. Every gate runs on the Mac
only, refuses a dirty tree unless ``--allow-dirty``, prints a progress line at least every
30 seconds during anything longer than a minute, and writes one schema-2 report.

* ``logits_bytes_expected(n_checkpoints, batch, seq, vocab)``: float32 logits for every
  checkpoint, ``4 * n_checkpoints * batch * seq * vocab``.
* ``random_accuracy_bound(vocab)``: ``2 / vocab``, twice chance (G2.3).
* ``chance_count(vocab, positions)``: correct guesses expected by chance, ``positions / vocab``.
* ``EXACT_PREFIX = 32`` and ``FIT_SIZES = (2_048, 8_192, 32_768)`` (G2.5).
* ``exact_jacobian_bytes(prefix, d_model)``: the float32 Jacobian of a ``prefix``-position
  sequence, ``4 * (prefix * d_model) ** 2``.
"""

from __future__ import annotations

EXACT_PREFIX = 32
FIT_SIZES = (2_048, 8_192, 32_768)


def logits_bytes_expected(n_checkpoints: int, batch: int, seq: int, vocab: int) -> int:
    raise NotImplementedError


def random_accuracy_bound(vocab: int) -> float:
    raise NotImplementedError


def chance_count(vocab: int, positions: int) -> float:
    raise NotImplementedError


def exact_jacobian_bytes(prefix: int, d_model: int) -> int:
    raise NotImplementedError
