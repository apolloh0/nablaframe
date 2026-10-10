"""The two-hop patch's sources and outcomes, and lens ranks.

Spec: docs/specs/phase-4.md, "Patch, on the two-hop model".

* ``QUERY_POSITIONS = (24, 25)``: the queried subject and the ``⇒``.
* ``source_for(task, target, counterpart, rng)``: a window whose intermediate is ``counterpart``, a subject of one of the target's distractor facts, and whose answer is neither the target's answer nor the counterpart's object in the target. Windows are drawn with ``task.window(rng, intermediate=counterpart)`` until one qualifies. So an unchanged, a swapped and a transplanted answer are always three different tokens.
* ``patch_outcomes(answers, targets, sources, counterparts)``: ``{"swap_rate",
  "transplant_rate"}`` as plain floats: the fraction of answers equal to the counterpart's
  object in the target's own facts, and the fraction equal to the source's own answer. Each is
  counted on its own, so one answer can count for both.
* ``token_ranks(readings, tokens)``: ``readings`` is (Layer, Batch, Vocab) at one position and
  ``tokens`` (Batch,). Each token's 0-based rank per layer and sequence: how many tokens score
  higher, plus how many score the same with a lower id. Shape (Layer, Batch).
* ``aligned_sources(task, targets, rng)``: ``(sources, controls, counterparts)`` for G4.3 (D36). For each target window, in order:

  * the counterpart ``k`` is the first subject in window order that is neither ``a`` nor ``b``
  * ``d′`` is ``rng.choice`` of the entities the window doesn't contain
  * the source is the target with ``a``'s object set to ``k``, ``k``'s object set to ``d′`` and the last token set to ``d′``: it asks ``a → k → d′`` and differs in nothing else
  * the control is the target with only ``k``'s object set to ``d′``

  Sources and controls are int32, shaped like ``targets``; ``counterparts`` is (Batch,). With them, ``patch_outcomes`` counts a swap as ``k``'s object in the target and a transplant as ``d′``. ValueError unless ``targets`` is (Batch, ``task.window_length``). The targets are left unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

QUERY_POSITIONS = (24, 25)


def _distractor_object(window: np.ndarray, counterpart: int) -> int:
    """The object of ``counterpart``'s fact in ``window``, refusing anything but a distractor subject."""
    w = np.asarray(window)
    end = len(w) - 3
    subjects, objects = w[0:end:4], w[2:end:4]
    a = int(w[end])
    b = int(objects[subjects == a][0])
    hits = np.flatnonzero(subjects == counterpart)
    if counterpart in (a, b) or hits.size != 1:
        raise ValueError(f"counterpart {counterpart} is not a subject of one of the window's distractor facts")
    return int(objects[hits[0]])


def source_for(task: Any, target: np.ndarray, counterpart: int, rng: np.random.Generator) -> np.ndarray:
    p = task.parse(target)
    d = _distractor_object(target, counterpart)
    while True:
        source = task.window(rng, intermediate=counterpart)
        if int(source[-1]) not in (p.c, d):
            return source


def aligned_sources(task: Any, targets: np.ndarray, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] != task.window_length:
        raise ValueError(f"targets must be (Batch, {task.window_length}), got {targets.shape}")
    sources, controls = targets.astype(np.int32), targets.astype(np.int32)  # astype copies, so targets stay as given
    counterparts = np.empty(len(targets), dtype=np.int64)
    end = task.window_length - 3
    for i, w in enumerate(targets):
        subjects = w[0:end:4]
        a = int(w[end])
        b = task.parse(w).b
        rest = [int(x) for x in subjects if x not in (a, b)]
        if not rest:
            raise ValueError(f"window {i} has no distractor subject to be the counterpart")
        k = rest[0]
        d_new = int(rng.choice(np.setdiff1d(np.arange(task.entities), w)))
        a_slot, k_slot = 4 * int(np.flatnonzero(subjects == a)[0]) + 2, 4 * int(np.flatnonzero(subjects == k)[0]) + 2
        sources[i, a_slot], sources[i, k_slot], sources[i, -1] = k, d_new, d_new
        controls[i, k_slot] = d_new
        counterparts[i] = k
    return sources, controls, counterparts


def patch_outcomes(answers: np.ndarray, targets: np.ndarray, sources: np.ndarray,
                   counterparts: Sequence[int]) -> dict[str, float]:
    answers, targets, sources = np.asarray(answers), np.asarray(targets), np.asarray(sources)
    n = len(answers)
    if n == 0 or not len(targets) == len(sources) == len(counterparts) == n:
        raise ValueError(f"need one target, source and counterpart per answer, got {n} answers, "
                         f"{len(targets)} targets, {len(sources)} sources, {len(counterparts)} counterparts")
    swapped = np.array([_distractor_object(w, int(c)) for w, c in zip(targets, counterparts)])
    transplanted = sources[:, -1]
    return {"swap_rate": float(np.mean(answers == swapped)),
            "transplant_rate": float(np.mean(answers == transplanted))}


def token_ranks(readings: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    readings, tokens = np.asarray(readings), np.asarray(tokens)
    if readings.ndim != 3 or tokens.shape != readings.shape[1:2]:
        raise ValueError(f"readings must be (Layer, Batch, Vocab) and tokens (Batch,), got "
                         f"{readings.shape} and {tokens.shape}")
    scores = np.take_along_axis(readings, tokens[None, :, None], axis=2)  # (L, B, 1)
    lower_id = np.arange(readings.shape[2])[None, None, :] < tokens[None, :, None]
    return (readings > scores).sum(axis=2) + ((readings == scores) & lower_id).sum(axis=2)
