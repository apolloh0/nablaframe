"""Zoo tasks: deterministic, seeded batch generators with the corpus's interface (D29, D32).

Spec: docs/specs/phase-4.md, "The zoo".

Every task has ``vocab`` (a tuple of strings), ``window`` (tokens per window), ``seq``
(``window - 1``, the model's sequence length), ``chance``, ``mechanism_positions`` and
``control_positions`` (int arrays of positions p, each predicting token p + 1),
``batch_for_step(seed, step, batch)`` (an int32 array (batch, window), drawn with
``numpy.random.default_rng([seed, step])``, so a step's batch never depends on another's),
and ``tokenizer_json()`` (``Tokenizer(vocab).to_json()``).

* ``InductionTask(vocab_size=64, window=64)``: vocab ``t0`` to ``t63``. Each window's first
  half is ``window // 2`` distinct tokens drawn without replacement, and its second half
  repeats the first. Mechanism positions ``window // 2`` to ``window - 2``; control positions
  ``0`` to ``window // 2 - 2``; chance ``1 / vocab_size``.
* ``TwoHopTask(entities=48, facts=6)``: vocab ``e0`` to ``e47``, then ``→``, ``;``, ``⇒``.
  A window is ``facts`` facts ``s → o ;`` in shuffled order, then ``a ⇒ c``: 27 tokens. Every
  fact's subject is distinct. The facts hold the chain ``a → b`` and ``b → c`` and distractors
  with random objects. The answer c is never a or b, and no distractor has c as its object, so
  c appears as an object only in b's fact. Mechanism position 25; no control positions (the
  controls are broken chains and the random-weights twin); chance ``1 / entities``.

  - ``window(rng, *, intermediate=None, broken=False)``: one window. ``intermediate`` fixes
    ``b``. ``broken`` replaces the ``b → c`` fact with another distractor and draws ``c`` from
    entities appearing nowhere in the facts.
  - ``parse(window)``: a ``TwoHop`` with ``a``, ``b``, ``c`` (ints) and ``facts`` (subject to
    object), where ``b = facts[a]`` and ``c`` is the window's last token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.tokenizer import Tokenizer


@dataclass(frozen=True)
class TwoHop:
    a: int
    b: int
    c: int
    facts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class InductionTask:
    vocab_size: int = 64
    window: int = 64

    def __post_init__(self) -> None:
        if self.window < 2 or self.window % 2:
            raise ValueError(f"window must be even and at least 2, got {self.window}")
        if self.window // 2 > self.vocab_size:
            raise ValueError(f"a first half of {self.window // 2} distinct tokens needs at least that many "
                             f"tokens in the vocabulary, got {self.vocab_size}")

    @property
    def vocab(self) -> tuple[str, ...]:
        return tuple(f"t{i}" for i in range(self.vocab_size))

    @property
    def seq(self) -> int:
        return self.window - 1

    @property
    def chance(self) -> float:
        return 1 / self.vocab_size

    @property
    def mechanism_positions(self) -> np.ndarray:
        return np.arange(self.window // 2, self.window - 1)

    @property
    def control_positions(self) -> np.ndarray:
        return np.arange(0, self.window // 2 - 1)

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        rng = np.random.default_rng([seed, step])
        half = self.window // 2
        first = np.array([rng.choice(self.vocab_size, size=half, replace=False) for _ in range(batch)],
                         dtype=np.int32).reshape(batch, half)
        return np.concatenate([first, first], axis=1)

    def tokenizer_json(self) -> str:
        return Tokenizer(self.vocab).to_json()


@dataclass(frozen=True)
class TwoHopTask:
    entities: int = 48
    facts: int = 6

    def __post_init__(self) -> None:
        if self.facts < 2:
            raise ValueError(f"the chain alone takes two facts, got facts={self.facts}")
        # A broken chain's facts mention at most 2 * facts entities, and its answer must be another.
        if self.entities <= 2 * self.facts:
            raise ValueError(f"{self.facts} facts need more than {2 * self.facts} entities, got {self.entities}")

    @property
    def vocab(self) -> tuple[str, ...]:
        return (*(f"e{i}" for i in range(self.entities)), "→", ";", "⇒")

    @property
    def window_length(self) -> int:
        return 4 * self.facts + 3

    @property
    def seq(self) -> int:
        return self.window_length - 1

    @property
    def chance(self) -> float:
        return 1 / self.entities

    @property
    def mechanism_positions(self) -> np.ndarray:
        return np.array([self.window_length - 2])

    @property
    def control_positions(self) -> np.ndarray:
        return np.arange(0)

    def window(self, rng: np.random.Generator, *, intermediate: int | None = None, broken: bool = False) -> np.ndarray:
        n, k = self.entities, self.facts
        if intermediate is not None and not 0 <= intermediate < n:
            raise ValueError(f"intermediate must be an entity id below {n}, got {intermediate}")
        b = int(rng.integers(n)) if intermediate is None else int(intermediate)
        others = rng.permutation(np.delete(np.arange(n), b))  # every entity but b, shuffled
        a = int(others[0])
        if broken:
            # a and k - 1 distractor subjects, none of them b, so the chain stops at b.
            subjects = others[:k]
            objects = np.concatenate([[b], rng.integers(0, n, size=k - 1)])
            unused = np.setdiff1d(np.arange(n), np.concatenate([subjects, objects]))
            c = int(rng.choice(unused))
        else:
            c = int(rng.choice(np.setdiff1d(np.arange(n), [a, b])))
            subjects = np.concatenate([[a, b], others[1:k - 1]])
            objects = np.concatenate([[b, c], rng.choice(np.delete(np.arange(n), c), size=k - 2)])
        order = rng.permutation(k)
        arrow, sep, query = n, n + 1, n + 2
        facts = np.stack([subjects[order], np.full(k, arrow), objects[order], np.full(k, sep)], axis=1)
        return np.concatenate([facts.reshape(-1), [a, query, c]]).astype(np.int32)

    def parse(self, window: np.ndarray) -> TwoHop:
        w = np.asarray(window)
        end = 4 * self.facts
        facts = {int(s): int(o) for s, o in zip(w[0:end:4], w[2:end:4])}
        a = int(w[end])
        return TwoHop(a=a, b=facts[a], c=int(w[end + 2]), facts=facts)

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        rng = np.random.default_rng([seed, step])
        return np.array([self.window(rng) for _ in range(batch)], dtype=np.int32).reshape(batch, self.window_length)

    def tokenizer_json(self) -> str:
        return Tokenizer(self.vocab).to_json()
