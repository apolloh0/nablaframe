"""Zoo tasks: deterministic, seeded batch generators with the corpus's interface (D29, D32).

Spec: docs/specs/phase-4.md, "The zoo".

Every task has ``vocab`` (a tuple of strings), ``window`` (tokens per window), ``seq``
(``window - 1``, the model's sequence length), ``chance``, ``mechanism_positions`` and
``control_positions`` (int arrays of positions p, each predicting token p + 1),
``batch_for_step(seed, step, batch)`` (an int32 array (batch, window), drawn with
``numpy.random.default_rng([seed, step])``, so a step's batch never depends on another's),
and ``tokenizer_json()`` (``Tokenizer(vocab).to_json()``).

* ``InductionTask(vocab_size=64, window=64, min_period=8, max_period=32)``: vocab ``t0`` to ``t63``. Each window repeats a block of ``P`` distinct tokens to fill ``window`` tokens, token ``i`` being ``block[i % P]`` (D36). ``draw_window(rng, *, period=None)`` draws one, in this order: ``P = rng.integers(min_period, max_period + 1)``, skipped when ``period`` is given, then ``block = rng.choice(vocab_size, size=P, replace=False)``. A ``period`` outside ``min_period`` to ``max_period`` raises ValueError. ``batch_for_step`` draws its windows in sequence from one generator. Mechanism positions ``max_period`` to ``window - 2``; control positions ``0`` to ``min_period - 2``; chance ``1 / vocab_size``. ValueError unless ``2 <= min_period <= max_period <= vocab_size`` and ``max_period <= window - 2``.
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
* ``ModularAdditionTask(p=113, train_fraction=0.3, split_seed=0)`` (D54): vocab ``"0"`` to
  ``str(p - 1)``, then ``"="``, whose id is ``p``. A window is ``a b = c``, four tokens with
  ``c = (a + b) mod p``, so ``window`` is 4 and ``seq`` 3. Mechanism position 2, where the ``=``
  predicts ``c``; control position 0, which predicts ``b`` from ``a``; chance ``1 / p``.

  - ``pairs``: int64 (p * p, 2), row ``i`` being ``(i // p, i % p)``.
  - ``split``: ``(train, held)``, int64 arrays of pair indices, each ascending. With
    ``order = numpy.random.default_rng(split_seed).permutation(p * p)`` and
    ``n = round(train_fraction * p * p)``, ``train`` holds ``order[:n]`` and ``held`` the rest.
  - ``windows(index)``: int32 (len(index), 4), the pairs ``index`` names, in that order, each
    ``[a, b, p, (a + b) % p]``. ValueError for an index outside ``range(p * p)`` or one that isn't
    integers. ``training()`` and ``held_out()``: ``windows(train)`` and ``windows(held)``.
  - ``batch_for_step(seed, step, batch)``: ``windows(rng.choice(train, size=batch, replace=True))``,
    ``rng`` being ``numpy.random.default_rng([seed, step])``.
  - ValueError unless ``p`` is an int of at least 3, ``train_fraction`` is a float or int that
    leaves both parts non-empty, and ``split_seed`` is a non-negative int. A bool is not an int.
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
    min_period: int = 8
    max_period: int = 32

    def __post_init__(self) -> None:
        if not 2 <= self.min_period <= self.max_period <= self.vocab_size:
            raise ValueError(f"periods must satisfy 2 <= min_period <= max_period <= vocab_size, got "
                             f"{self.min_period}, {self.max_period}, {self.vocab_size}")
        if self.max_period > self.window - 2:
            raise ValueError(f"max_period must be at most window - 2 = {self.window - 2}, got {self.max_period}")

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
        return np.arange(self.max_period, self.window - 1)

    @property
    def control_positions(self) -> np.ndarray:
        return np.arange(0, self.min_period - 1)

    def draw_window(self, rng: np.random.Generator, *, period: int | None = None) -> np.ndarray:
        # The draw order is pinned (period, then block) so G4.1 retraces the feasibility run's batches.
        if period is None:
            period = int(rng.integers(self.min_period, self.max_period + 1))
        elif not self.min_period <= period <= self.max_period:
            raise ValueError(f"period must be from {self.min_period} to {self.max_period}, got {period}")
        block = rng.choice(self.vocab_size, size=period, replace=False)
        return block[np.arange(self.window) % period].astype(np.int32)

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        rng = np.random.default_rng([seed, step])
        return np.array([self.draw_window(rng) for _ in range(batch)], dtype=np.int32).reshape(batch, self.window)

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


def _is_int(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


@dataclass(frozen=True)
class ModularAdditionTask:
    p: int = 113
    train_fraction: float = 0.3
    split_seed: int = 0

    def __post_init__(self) -> None:
        if not _is_int(self.p) or self.p < 3:
            raise ValueError(f"p must be an int of at least 3, got {self.p!r}")
        if not _is_int(self.split_seed) or self.split_seed < 0:
            raise ValueError(f"split_seed must be a non-negative int, got {self.split_seed!r}")
        tf = self.train_fraction
        if isinstance(tf, (bool, np.bool_)) or not isinstance(tf, (int, float, np.integer, np.floating)) \
                or not np.isfinite(tf):
            raise ValueError(f"train_fraction must be a number, got {tf!r}")
        if not 1 <= self._n_train <= self.p * self.p - 1:
            raise ValueError(f"train_fraction {tf!r} leaves a part of the {self.p * self.p} pairs empty")

    @property
    def _n_train(self) -> int:
        return round(self.train_fraction * self.p * self.p)

    @property
    def vocab(self) -> tuple[str, ...]:
        return (*(str(i) for i in range(self.p)), "=")

    @property
    def window(self) -> int:
        return 4

    @property
    def seq(self) -> int:
        return self.window - 1

    @property
    def chance(self) -> float:
        return 1 / self.p

    @property
    def mechanism_positions(self) -> np.ndarray:
        return np.array([2])

    @property
    def control_positions(self) -> np.ndarray:
        return np.array([0])

    @property
    def pairs(self) -> np.ndarray:
        i = np.arange(self.p * self.p, dtype=np.int64)
        return np.stack([i // self.p, i % self.p], axis=1)

    @property
    def split(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.random.default_rng(int(self.split_seed)).permutation(self.p * self.p)
        n = self._n_train
        return np.sort(order[:n]).astype(np.int64), np.sort(order[n:]).astype(np.int64)

    def windows(self, index: np.ndarray) -> np.ndarray:
        index = np.asarray(index)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise ValueError(f"index must be a one-dimensional integer array, got {index.dtype} {index.shape}")
        if index.size and (index.min() < 0 or index.max() >= self.p * self.p):
            raise ValueError(f"index must lie in range({self.p * self.p})")
        a, b = index // self.p, index % self.p
        return np.stack([a, b, np.full_like(a, self.p), (a + b) % self.p], axis=1).astype(np.int32)

    def training(self) -> np.ndarray:
        return self.windows(self.split[0])

    def held_out(self) -> np.ndarray:
        return self.windows(self.split[1])

    def batch_for_step(self, seed: int, step: int, batch: int) -> np.ndarray:
        rng = np.random.default_rng([seed, step])
        return self.windows(rng.choice(self.split[0], size=batch, replace=True))

    def tokenizer_json(self) -> str:
        return Tokenizer(self.vocab).to_json()
