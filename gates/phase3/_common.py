"""Shared by the Phase 3 gates. Read with docs/specs/phase-3.md, "Gates".

May import ``gates/phase1/_common.py`` and ``gates/phase2/_common.py`` (load them by path)
but never change them. Reports go to ``reports/phase3/``, working files to ``runs/phase3/``.
Every gate runs on the Mac only, refuses a dirty tree unless ``--allow-dirty``, prints a
progress line at least every 30 seconds during anything longer than a minute, writes one
schema-2 report, and runs every pass through ``run_intervened``.

* ``REFERENCE_SEED = 2`` and ``REFERENCE_WINDOWS = 64``: D22's reference batch, drawn from
  the training stream like D17's fit sample.
* ``EXPERIMENT_POSITIONS = 64``, ``LATER_WINDOW = 16``, ``FIT_FIRST = 16``.
* ``later_positions(p, seq_len)``: positions ``p + 1`` up to ``p + LATER_WINDOW``, cut at the
  window's end.
* ``fit_range(seq_len)``: positions ``FIT_FIRST`` to ``seq_len - 2``, D17's fit range.
"""

from __future__ import annotations

import numpy as np

REFERENCE_SEED = 2
REFERENCE_WINDOWS = 64
EXPERIMENT_POSITIONS = 64
LATER_WINDOW = 16
FIT_FIRST = 16


def later_positions(p: int, seq_len: int) -> np.ndarray:
    raise NotImplementedError


def fit_range(seq_len: int) -> np.ndarray:
    raise NotImplementedError
