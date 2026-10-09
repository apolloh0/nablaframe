"""The two-hop patch's sources and outcomes, and lens ranks.

Spec: docs/specs/phase-4.md, "Patch, on the two-hop model".

* ``QUERY_POSITIONS = (24, 25)``: the queried subject and the ``⇒``.
* ``source_for(task, target, counterpart, rng)``: a fresh window whose intermediate is
  ``counterpart``, a subject of one of the target's distractor facts.
* ``patch_outcomes(answers, targets, sources, counterparts)``: ``{"swap_rate",
  "transplant_rate"}`` as plain floats: the fraction of answers equal to the counterpart's
  object in the target's own facts, and the fraction equal to the source's own answer. Each is
  counted on its own, so one answer can count for both.
* ``token_ranks(readings, tokens)``: ``readings`` is (Layer, Batch, Vocab) at one position and
  ``tokens`` (Batch,). Each token's 0-based rank per layer and sequence: how many tokens score
  higher, plus how many score the same with a lower id. Shape (Layer, Batch).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

QUERY_POSITIONS = (24, 25)


def source_for(task: Any, target: np.ndarray, counterpart: int, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def patch_outcomes(answers: np.ndarray, targets: np.ndarray, sources: np.ndarray,
                   counterparts: Sequence[int]) -> dict[str, float]:
    raise NotImplementedError


def token_ranks(readings: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    raise NotImplementedError
