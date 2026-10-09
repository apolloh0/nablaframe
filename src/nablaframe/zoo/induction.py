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

from ..interventions.spec import InterventionSpec, edit


def _positions(positions: Any) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos.size == 0:
        raise ValueError("no positions")
    if pos.min() < 0:
        raise ValueError("positions must be non-negative")
    return pos


def induction_targets(windows: np.ndarray, positions: Any) -> np.ndarray:
    windows = np.asarray(windows)
    if windows.ndim != 2:
        raise ValueError("windows must be (Batch, Pos)")
    pos = _positions(positions)
    if pos.max() >= windows.shape[1]:
        raise ValueError(f"positions must lie in [0, {windows.shape[1]})")
    targets = np.full((windows.shape[0], pos.size), -1, dtype=np.int64)
    for b in range(windows.shape[0]):
        for i, p in enumerate(pos):
            earlier = np.flatnonzero(windows[b, :p] == windows[b, p])
            if earlier.size:
                targets[b, i] = earlier[-1] + 1
    return targets


def induction_scores(pattern: np.ndarray, windows: np.ndarray, positions: Any) -> np.ndarray:
    pattern = np.asarray(pattern)
    if pattern.ndim != 4:
        raise ValueError("pattern must be (Batch, Head, QPos, KPos)")
    windows = np.asarray(windows)
    if windows.ndim != 2 or windows.shape[0] != pattern.shape[0]:
        raise ValueError("windows must be (Batch, Pos) with the pattern's batch")
    pos = _positions(positions)
    if pos.max() >= pattern.shape[2]:
        raise ValueError(f"positions must lie in [0, {pattern.shape[2]})")
    targets = induction_targets(windows, pos)
    b, i = np.nonzero(targets >= 0)
    if b.size == 0:
        raise ValueError("no (sequence, position) pair has an induction target")
    attention = pattern[b, :, pos[i], targets[b, i]].astype(np.float64)  # (pairs, Head)
    return attention.mean(axis=0)


def uniform_baseline(positions: Any) -> float:
    pos = _positions(positions)
    return float(np.mean(1.0 / (pos.astype(np.float64) + 1.0)))


def _steepest(steps: Sequence[int], values: Sequence[float], sign: float) -> int:
    steps, values = list(steps), np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(steps) != values.size or values.size < 2:
        raise ValueError("steps and values must be equal-length sequences of at least two checkpoints")
    change = sign * np.diff(values)
    return int(steps[int(np.argmax(change)) + 1])  # argmax returns the first maximum: ties to the earlier step


def steepest_drop(steps: Sequence[int], values: Sequence[float]) -> int:
    return _steepest(steps, values, -1.0)


def steepest_rise(steps: Sequence[int], values: Sequence[float]) -> int:
    return _steepest(steps, values, 1.0)


def head_zero(layer: int, head: int) -> Any:
    return InterventionSpec((edit(f"blocks.{layer}.attn.hook_z", "all", "zero", head=head),))
