"""Corpus cleaning, token streams, batch order, evaluation slice and probe set.

Spec: docs/specs/phase-1.md, "Data".

* ``clean_stories(raw)``: split on ``SEPARATOR``, strip each piece, ignore empty
  pieces, apply ``fold_to_ascii``, and drop any story still containing a non-ASCII
  character. Returns the kept stories, with case preserved, and how many were dropped.
* ``encode_split`` concatenates encoded stories into one ``uint16`` stream.
* ``batch_for_step(stream, seed, step, batch, seq)``: ``batch`` windows of
  ``seq + 1`` consecutive tokens, as ``int32``. Start offsets come from
  ``numpy.random.default_rng([seed, step]).integers(0, len(stream) - seq, size=batch)``
  and nothing else, so a resumed run sees exactly the batches an uninterrupted one
  would. The update from step k to step k + 1 uses batch k.
* ``eval_windows(validation, n_tokens, seq)``: consecutive windows of ``seq + 1``
  tokens starting at 0, seq, 2 * seq and so on, all inside ``validation[:n_tokens]``.
* ``probe_batch(validation, eval_tokens, seed, n, seq)``: ``n`` windows of ``seq``
  tokens as ``int32``, with starts drawn by ``default_rng(seed)`` from
  ``[eval_tokens, len(validation) - seq]``, so the probe never overlaps the
  evaluation slice.
* ``batch_hash``: SHA-256 of the batch as little-endian ``int32`` bytes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .tokenizer import Tokenizer

SEPARATOR = "<|endoftext|>"


def clean_stories(raw: str) -> tuple[list[str], int]:
    raise NotImplementedError


def encode_split(stories: Iterable[str], tokenizer: Tokenizer) -> np.ndarray:
    raise NotImplementedError


def batch_for_step(stream: np.ndarray, seed: int, step: int, batch: int, seq: int) -> np.ndarray:
    raise NotImplementedError


def eval_windows(validation: np.ndarray, n_tokens: int, seq: int) -> np.ndarray:
    raise NotImplementedError


def probe_batch(validation: np.ndarray, eval_tokens: int, seed: int, n: int, seq: int) -> np.ndarray:
    raise NotImplementedError


def batch_hash(batch: np.ndarray) -> str:
    raise NotImplementedError
