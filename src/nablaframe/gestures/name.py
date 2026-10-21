"""name-this-direction (D49): what a latent stands for, named only from declared descriptions and held with a set of four.

Spec: docs/specs/phase-6.md, D49 and "What each run declares". Built on ``sets.py``, with nothing
specific to a task (D29). Nothing here caches or judges: the gates ask for names, and a record holds
what ``name`` returns (D47).

* ``NAMED = "named"``, ``WEAK = "weak"``, ``EXAMPLES = 8``, ``TOP_TOKENS = 5`` and
  ``NO_PREDICTION = "No causal test applies: the run declares no prediction for this description
  at these instances."``
* ``latent_effects(dictionary, site, params, latents)``: float64 (len(latents), vocab), each
  latent's direct effect on the logits, as the features view computes it (D45):
  ``e = decode_delta(dictionary, site, one-hot)`` taken to float64, then
  ``((e - mean(e)) * ln_f.weight) @ wte.T``.
* ``ends(effect, value)``: ``{"top", "bottom", "rank_from_top", "rank_from_bottom"}`` for a float64
  ``effect`` over the vocabulary. ``top`` holds the ``TOP_TOKENS`` largest, largest first, and
  ``bottom`` the smallest, smallest first, each ``[token, effect]``, ties to the lower token.
  ``rank_from_top`` is ``value``'s place, from 0, with every token ordered largest first, ties to the
  lower token; ``rank_from_bottom`` the same with every token ordered smallest first. ValueError for
  a ``value`` outside the vocabulary.
* ``coverage(reading, latents, value)``: for each description in ``reading.described``, in order,
  the share of the latents' summed codes over every held-out position (every window, every position)
  that falls where the description is ``value``, summed in float64; every share None where the
  latents never fire.
* ``Instance``: a frozen dataclass ``(window, position, w, read, target)``: the held-out window's
  index, the position the edit is at, the other value ``w``, and the position the output is read at
  and the token the run predicts there when ``d`` is moved to ``w``, both None where no causal test
  applies.
* ``draw_instances(reading, cls, d, v, *, limit, rng)``: ``(causal, instances)``. Every
  (window, position) pair where ``d`` is ``v``, window by window and each window's positions in the
  class's order, is a candidate. Its options are ``table.values(d)`` less the values ``taken`` in its
  window, ascending; its predicting options are those for which ``declaration.predict(d, window,
  position, w)`` gives ``(read, target)``.

  - ``causal`` is True where some candidate has a predicting option: the eligible candidates are
    those with one, and each draws from its predicting options. Otherwise the eligible candidates
    are those with an option, and each draws from its options.
  - With more than ``limit`` eligible, ``sorted(rng.choice(len(eligible), size=limit,
    replace=False))`` keeps that many. Then each kept candidate, in order, draws
    ``w = choices[rng.integers(len(choices))]``.

  ValueError unless ``reading.table(cls).has(d, v)``, or for a ``limit`` that isn't a positive int.
* ``steer(reading, cls, d, v, *, instances, rng)``: D49's step 6 for the name ``d = v``, or None
  where ``draw_instances`` finds no instance. Its instances are drawn with ``limit=instances``, then
  the null with ``sets.null_sets(table, d, the codes at each instance's row, vs, ws, rng)``. Four
  passes run in one ``sets.run_steered`` call, whose sequences are every instance, each at its
  position, for each pass in turn:

  - **unintervened:** nothing removed or added
  - **swapped:** ``table.set_for(d, v)`` removed, and ``table.set_for(d, w)`` added at
    ``table.mean_codes(d, w)``'s values for those latents
  - **null:** the null's latents removed, and its added latents at the first amounts of ``w``'s set
  - **ablated:** ``v``'s set removed, nothing added

  Causal: ``{"causal": True, "instances", "follows", "unchanged", "ablated_changed",
  "null_follows", "null_unchanged"}``. A pass's output is ``sets.argmax`` of its logits at the
  instance's ``read``. ``follows`` is the share of instances where the swapped output is ``target``,
  ``unchanged`` where it is the unintervened output; ``ablated_changed`` the share where the ablated
  output isn't; ``null_follows`` and ``null_unchanged`` as for the swap.

  Otherwise: ``{"causal": False, "note": NO_PREDICTION, "instances", "kl", "null_kl", "examples"}``.
  ``kl`` is ``dictionaries.edges.kl`` of the unintervened and the swapped logits at each instance's
  position, a mean over instances; ``null_kl`` the same for the null. ``examples``: for the first
  ``EXAMPLES`` instances, ``{"window", "position", "w", "unsteered", "steered"}``, the last two the
  ``TOP_TOKENS`` tokens with the largest logits there, largest first, ties to the lower token.
* ``name(reading, cls, latent, *, instances, seed)``: D49's six steps for ``latent``, a plain JSON
  mapping that ``records.GestureRecord`` takes as its results. ``rng =
  numpy.random.default_rng(seed)``; ``table = reading.table(cls)``.

  - ``"class"``, ``"latent"``, ``"windows"`` (the held-out windows' count) and ``"rows"``
    (``table.rows``).
  - ``"candidates"``: ``[{"description", "value", "f1"}, ...]``, ``table.candidates(latent)``.
  - ``"name"``: None where there is no candidate, the latent firing on no row of the class, and then
    the mapping holds nothing more. Otherwise the top candidate, ``{"description", "value", "f1",
    "label"}``, labelled ``NAMED`` where its F1 is at least ``sets.FLOOR`` and ``WEAK`` below it.
    Every step below runs for both, with ``d`` and ``v`` the name's description and value.
  - ``"set"``: ``{"latents", "f1", "name_f1", "pointed_rank", "in_set"}``: ``table.set_for(d, v)``,
    each one's F1 for ``v``, the highest of those, ``latent``'s place in ``table.ranked(d, v)`` from
    0, and whether that place is below ``sets.SET_SIZE``.
  - ``"coverage"``: ``{"latent": coverage(reading, [latent], v), "set": coverage(reading, the set,
    v)}``.
  - ``"effect"``: ``{"set", "latent", "vocabulary"}``. ``set`` is ``ends`` of the set's effect, the
    sum over the set of ``table.mean_codes(d, v)[l]`` times ``latent_effects[l]``: weighted by its
    mean codes where the name holds. ``latent`` is ``{"latent": latent}`` with ``ends`` of the
    latent's own effect, labelled as that latent's, never as the name's. ``vocabulary`` is the
    vocabulary's size.
  - ``"steering"``: ``steer(reading, cls, d, v, instances=instances, rng=rng)``.

  ValueError for a latent outside ``range(reading.width)``, a class the run doesn't declare or at
  which no description counts, or ``instances`` or ``seed`` that aren't a positive int and a
  non-negative int.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NAMED = "named"
WEAK = "weak"
EXAMPLES = 8
TOP_TOKENS = 5
NO_PREDICTION = ("No causal test applies: the run declares no prediction for this description at these "
                 "instances.")


@dataclass(frozen=True)
class Instance:
    window: int
    position: int
    w: int
    read: int | None
    target: int | None


def latent_effects(dictionary: Any, site: int, params: Any, latents: Any) -> np.ndarray:
    raise NotImplementedError


def ends(effect: Any, value: Any) -> dict[str, Any]:
    raise NotImplementedError


def coverage(reading: Any, latents: Any, value: int) -> dict[str, float | None]:
    raise NotImplementedError


def draw_instances(reading: Any, cls: str, d: str, v: int, *, limit: int,
                   rng: np.random.Generator) -> tuple[bool, list[Instance]]:
    raise NotImplementedError


def steer(reading: Any, cls: str, d: str, v: int, *, instances: int, rng: np.random.Generator) -> dict[str, Any] | None:
    raise NotImplementedError


def name(reading: Any, cls: str, latent: int, *, instances: int, seed: int) -> dict[str, Any]:
    raise NotImplementedError
