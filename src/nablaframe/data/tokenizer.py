"""Word-level tokenizer for TinyStories.

Spec: docs/specs/phase-1.md, "Data".

* ``fold_to_ascii``: NFKD, then drop combining marks, then map curly quotes and
  dashes to ASCII (``\u2018\u2019`` to ``'``, ``\u201c\u201d`` to ``"``,
  ``\u2013\u2014`` to ``-``). Standard library only.
* ``normalize``: ``fold_to_ascii`` then lowercase.
* ``pre_tokenize``: words match ``[a-z]+(?:'[a-z]+)*``; each digit and each other
  non-space character is its own token.
* The vocabulary is ``<unk>`` (id 0), ``<eos>`` (id 1), then the most frequent
  types in the training stories, ties broken alphabetically. ``size`` counts the
  two specials. Building from the same stories in any order gives identical JSON.
* ``encode`` appends ``<eos>``; unknown words become ``<unk>``. ``decode`` joins
  tokens with single spaces, so it is lossy by design.
* ``to_json`` is canonical JSON of ``{"vocab": [...]}``; ``sha256`` is its SHA-256.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

UNK, EOS = "<unk>", "<eos>"
UNK_ID, EOS_ID = 0, 1


def fold_to_ascii(text: str) -> str:
    raise NotImplementedError


def normalize(text: str) -> str:
    raise NotImplementedError


def pre_tokenize(text: str) -> list[str]:
    """Split already-normalized text into word, digit and symbol tokens."""
    raise NotImplementedError


@dataclass(frozen=True)
class Tokenizer:
    vocab: tuple[str, ...]

    @property
    def sha256(self) -> str:
        raise NotImplementedError

    def id(self, token: str) -> int:
        """The token's id, or ``UNK_ID`` if it is not in the vocabulary."""
        raise NotImplementedError

    def encode(self, story: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def to_json(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_json(cls, text: str) -> Tokenizer:
        raise NotImplementedError


def build_tokenizer(stories: Iterable[str], size: int) -> Tokenizer:
    raise NotImplementedError
