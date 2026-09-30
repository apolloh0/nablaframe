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

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256

UNK, EOS = "<unk>", "<eos>"
UNK_ID, EOS_ID = 0, 1

_PUNCT = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
})
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*|\S")


def fold_to_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.translate(_PUNCT)


def normalize(text: str) -> str:
    return fold_to_ascii(text).lower()


def pre_tokenize(text: str) -> list[str]:
    """Split already-normalized text into word, digit and symbol tokens."""
    return _TOKEN.findall(text)


@dataclass(frozen=True)
class Tokenizer:
    vocab: tuple[str, ...]

    @cached_property
    def _ids(self) -> dict[str, int]:
        # Reversed so the first occurrence wins, should a vocabulary ever repeat a token.
        return {token: i for i, token in reversed(list(enumerate(self.vocab)))}

    @property
    def sha256(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def id(self, token: str) -> int:
        """The token's id, or ``UNK_ID`` if it is not in the vocabulary."""
        return self._ids.get(token, UNK_ID)

    def encode(self, story: str) -> list[int]:
        ids = [self.id(token) for token in pre_tokenize(normalize(story))]
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"vocab": list(self.vocab)}, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> Tokenizer:
        return cls(vocab=tuple(json.loads(text)["vocab"]))


def build_tokenizer(stories: Iterable[str], size: int) -> Tokenizer:
    if size < 2:
        raise ValueError(f"size counts the two specials, so it must be at least 2, got {size}")
    counts: Counter[str] = Counter()
    for story in stories:
        counts.update(pre_tokenize(normalize(story)))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return Tokenizer(vocab=(UNK, EOS, *(token for token, _ in ranked[: size - 2])))
