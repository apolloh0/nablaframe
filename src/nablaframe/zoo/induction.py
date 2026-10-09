"""Induction: targets, scores, head ablation, and the formation steps.

Spec: docs/specs/phase-4.md, "Discrimination tests at model level" and "The induction formation demo".

* ``induction_targets(windows, positions)``: for each sequence and position p, one past the
  most recent earlier occurrence of ``windows[b, p]``, or -1 if there is none. Shape
  (Batch, len(positions)).
* ``induction_scores(pattern, windows, positions)``: ``pattern`` is one layer's attention,
  (Batch, Head, QPos, KPos). Per head, the mean over (sequence, position) pairs with a
  target of the attention from that position to its target, in float64. Shape (Head,).
* ``uniform_baseline(positions)``: what uniform causal attention would give, the mean of
  ``1 / (p + 1)``.
* ``steepest_drop(steps, values)`` and ``steepest_rise(steps, values)``: the step ending the
  largest fall, or rise, between consecutive checkpoints. Ties go to the earlier step.
* ``head_zero(layer, head)``: the spec zeroing that head's output at every position, through
  ``blocks.<layer>.attn.hook_z``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def induction_targets(windows: np.ndarray, positions: Any) -> np.ndarray:
    raise NotImplementedError


def induction_scores(pattern: np.ndarray, windows: np.ndarray, positions: Any) -> np.ndarray:
    raise NotImplementedError


def uniform_baseline(positions: Any) -> float:
    raise NotImplementedError


def steepest_drop(steps: Sequence[int], values: Sequence[float]) -> int:
    raise NotImplementedError


def steepest_rise(steps: Sequence[int], values: Sequence[float]) -> int:
    raise NotImplementedError


def head_zero(layer: int, head: int) -> Any:
    raise NotImplementedError
