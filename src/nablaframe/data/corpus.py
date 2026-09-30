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
from hashlib import sha256

import numpy as np

from .tokenizer import Tokenizer, fold_to_ascii

SEPARATOR = "<|endoftext|>"


def clean_stories(raw: str) -> tuple[list[str], int]:
    stories: list[str] = []
    dropped = 0
    for piece in raw.split(SEPARATOR):
        piece = piece.strip()
        if not piece:
            continue
        piece = fold_to_ascii(piece)
        if piece.isascii():
            stories.append(piece)
        else:
            dropped += 1
    return stories, dropped


def encode_split(stories: Iterable[str], tokenizer: Tokenizer) -> np.ndarray:
    if len(tokenizer.vocab) > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocabulary of {len(tokenizer.vocab)} does not fit in uint16")
    chunks = [np.asarray(tokenizer.encode(story), dtype=np.uint16) for story in stories]
    if not chunks:
        return np.empty(0, dtype=np.uint16)
    return np.concatenate(chunks)


def batch_for_step(stream: np.ndarray, seed: int, step: int, batch: int, seq: int) -> np.ndarray:
    if len(stream) <= seq:
        raise ValueError(f"stream of {len(stream)} tokens is too short for windows of {seq + 1}")
    starts = np.random.default_rng([seed, step]).integers(0, len(stream) - seq, size=batch)
    return stream[starts[:, None] + np.arange(seq + 1)].astype(np.int32)


def eval_windows(validation: np.ndarray, n_tokens: int, seq: int) -> np.ndarray:
    available = min(n_tokens, len(validation))
    count = max(0, (available - 1) // seq)
    starts = np.arange(count) * seq
    return validation[starts[:, None] + np.arange(seq + 1)].astype(np.int32)


def probe_batch(validation: np.ndarray, eval_tokens: int, seed: int, n: int, seq: int) -> np.ndarray:
    last_start = len(validation) - seq
    if last_start < eval_tokens:
        raise ValueError(
            f"validation of {len(validation)} tokens has no room for a probe window of {seq} "
            f"after the first {eval_tokens} tokens")
    # The docstring's interval is closed, so the last window may end exactly at the stream's end.
    starts = np.random.default_rng(seed).integers(eval_tokens, last_start + 1, size=n)
    return validation[starts[:, None] + np.arange(seq)].astype(np.int32)


def batch_hash(batch: np.ndarray) -> str:
    return sha256(np.asarray(batch).astype("<i4").tobytes()).hexdigest()
