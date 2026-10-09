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
  with random objects. Mechanism position 25; no control positions (the controls are broken
  chains and the random-weights twin); chance ``1 / entities``.

  - ``window(rng, *, intermediate=None, broken=False)``: one window. ``intermediate`` fixes
    ``b``. ``broken`` replaces the ``b → c`` fact with another distractor and draws ``c`` from
    entities appearing nowhere in the facts.
  - ``parse(window)``: a ``TwoHop`` with ``a``, ``b``, ``c`` (ints) and ``facts`` (subject to
    object), where ``b = facts[a]`` and ``c`` is the window's last token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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

    @property
    def vocab(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def seq(self) -> int:
        raise NotImplementedError

    @property
    def chance(self) -> float:
        raise NotImplementedError

    @property
    def mechanism_positions(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def control_positions(self) -> np.ndarray:
        raise NotImplementedError

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        raise NotImplementedError

    def tokenizer_json(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class TwoHopTask:
    entities: int = 48
    facts: int = 6

    @property
    def vocab(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def window_length(self) -> int:
        raise NotImplementedError

    @property
    def seq(self) -> int:
        raise NotImplementedError

    @property
    def chance(self) -> float:
        raise NotImplementedError

    @property
    def mechanism_positions(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def control_positions(self) -> np.ndarray:
        raise NotImplementedError

    def window(self, rng: np.random.Generator, *, intermediate: int | None = None, broken: bool = False) -> np.ndarray:
        raise NotImplementedError

    def parse(self, window: np.ndarray) -> TwoHop:
        raise NotImplementedError

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        raise NotImplementedError

    def tokenizer_json(self) -> str:
        raise NotImplementedError
