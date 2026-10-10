"""Induction: targets, scores, head ablation, and the formation steps.

Spec: docs/specs/phase-4.md, "Discrimination tests at model level" and "The induction formation demo".

* ``induction_targets(windows, positions)``: for each sequence and position p, one past the
  most recent earlier occurrence of ``windows[b, p]``, or -1 if there is none. Shape
  (Batch, len(positions)).
* ``induction_target_mask(windows, positions)``: bool (Batch, len(positions), Pos), true at every position ``k + 1`` with ``k < p`` and ``windows[b, k] == windows[b, p]``. These are the positions following an earlier occurrence of the current token, so each holds the right next token (D36).
* ``induction_scores(pattern, windows, positions)``: ``pattern`` is one layer's attention, (Batch, Head, QPos, KPos). Per head, the mean over (sequence, position) pairs with at least one target of the attention from that position, summed over all its targets, in float64. Shape (Head,). ``windows`` may be longer than the pattern, since no target lies after its position. ValueError when no pair has a target.
* ``uniform_baseline(windows, positions)``: what uniform causal attention would score on the same targets, the mean over the same pairs of the number of targets over ``p + 1``. ValueError when no pair has a target.
* ``new_token_sources(windows, position, rng, vocab_size)``: a copy of ``windows`` as int32, with ``position`` replaced in each window, in order, by ``rng.choice`` of the ids below ``vocab_size`` that the window doesn't contain.
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


def _windows(windows: Any) -> np.ndarray:
    windows = np.asarray(windows)
    if windows.ndim != 2:
        raise ValueError("windows must be (Batch, Pos)")
    return windows


def induction_target_mask(windows: np.ndarray, positions: Any) -> np.ndarray:
    windows = _windows(windows)
    pos = _positions(positions)
    if pos.max() >= windows.shape[1]:
        raise ValueError(f"positions must lie in [0, {windows.shape[1]})")
    mask = np.zeros((windows.shape[0], pos.size, windows.shape[1]), dtype=bool)
    for b in range(windows.shape[0]):
        for i, p in enumerate(pos):
            mask[b, i, 1 : p + 1] = windows[b, :p] == windows[b, p]  # earlier k marks position k + 1
    return mask


def induction_scores(pattern: np.ndarray, windows: np.ndarray, positions: Any) -> np.ndarray:
    pattern = np.asarray(pattern)
    if pattern.ndim != 4:
        raise ValueError("pattern must be (Batch, Head, QPos, KPos)")
    windows = _windows(windows)
    if windows.shape[0] != pattern.shape[0]:
        raise ValueError("windows must be (Batch, Pos) with the pattern's batch")
    pos = _positions(positions)
    if pos.max() >= pattern.shape[2]:
        raise ValueError(f"positions must lie in [0, {pattern.shape[2]})")
    mask = induction_target_mask(windows, pos)[:, :, : pattern.shape[3]]  # no target lies after its position
    paired = mask.any(axis=2)  # (Batch, len(positions))
    if not paired.any():
        raise ValueError("no (sequence, position) pair has an induction target")
    attention = pattern[:, :, pos, :].astype(np.float64)  # (Batch, Head, len(positions), KPos)
    on_targets = np.einsum("bhpk,bpk->bhp", attention, mask.astype(np.float64))
    return on_targets.transpose(1, 0, 2)[:, paired].mean(axis=1)  # (Head, pairs) -> (Head,)


def uniform_baseline(windows: np.ndarray, positions: Any) -> float:
    windows = _windows(windows)
    pos = _positions(positions)
    mask = induction_target_mask(windows, pos)
    paired = mask.any(axis=2)
    if not paired.any():
        raise ValueError("no (sequence, position) pair has an induction target")
    uniform = mask.sum(axis=2) / (pos.astype(np.float64) + 1.0)  # (Batch, len(positions))
    return float(uniform[paired].mean())


def new_token_sources(windows: np.ndarray, position: int, rng: np.random.Generator, vocab_size: int) -> np.ndarray:
    windows = _windows(windows)
    if not 0 <= position < windows.shape[1]:
        raise ValueError(f"position must lie in [0, {windows.shape[1]})")
    sources = windows.astype(np.int32, copy=True)
    for b in range(windows.shape[0]):
        unused = np.setdiff1d(np.arange(vocab_size), windows[b])
        if unused.size == 0:
            raise ValueError("a window uses every id below vocab_size")
        sources[b, position] = rng.choice(unused)
    return sources


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
