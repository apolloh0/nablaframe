"""assert-this-edge (D50): does one name's set drive another's, and is the edge computed or only carried?

Spec: docs/specs/phase-6.md, D50 and "What each run declares". Built on ``sets.py``, with nothing
specific to a task (D29). Nothing here caches or judges: the gates ask for edges, and a record holds
what ``edge`` returns (D47).

* ``CARRIED = "carried"`` and ``COMPUTED = "computed"``.
* ``EdgeInstance``: a frozen dataclass ``(window, up, down, v, gv, w, gw, read, target)``: the
  held-out window's index; the upstream and downstream positions; the upstream value ``v`` and
  ``g(v)``; the other value ``w`` and ``g(w)``; and ``declaration.predict(dd, window, down, g(w))``'s
  read position and token, both None where it gives none.
* ``draw_edge_instances(up, down, relation, *, limit, rng)``: the instances, from two
  ``sets.SiteReading``s and a relation's name. With ``R = declaration.relation(relation)``, its
  places' descriptions ``du`` and ``dd``, ``Tu = up.table(R.up.cls)`` and
  ``Td = down.table(R.down.cls)``: window by window, each pair ``(pu, pd)`` of
  ``declaration.pairs(relation, window)``, in its order, is a candidate where

  - ``v``, ``du`` at ``pu``, has ``Tu.has(du, v)``
  - ``g(v) = declaration.value_map(relation, window, v)`` isn't None, ``Td.has(dd, g(v))``, and
    ``dd`` is ``g(v)`` at ``pd``
  - the downstream set for ``g(v)`` is active there: its codes in ``down.codes`` at ``pd`` sum
    above zero

  Its options are ``Tu.values(du)`` less the values ``taken`` in its window, ascending, each with a
  ``g(w)`` that isn't None, has ``Td.has(dd, g(w))`` and isn't ``g(v)``. Where
  ``declaration.predict(du, window, pu, w)`` gives a prediction for some option, the options are only
  those. The eligible candidates are those with an option. With more than ``limit``,
  ``sorted(rng.choice(len(eligible), size=limit, replace=False))`` keeps that many; then each kept
  candidate, in order, draws ``w = options[rng.integers(len(options))]``.

  ValueError unless both readings share their model, params and declaration, as the same objects,
  and have equal windows; for a relation the run doesn't declare; for a reading whose ``block`` isn't
  its place's; for a downstream block before the upstream one; for a description that doesn't count
  at its class; or for a ``limit`` that isn't a positive int.
* ``edge(up, down, relation, *, instances, seed)``: D50's measurements, a plain JSON mapping that
  ``records.GestureRecord`` takes as its results. ``rng = numpy.random.default_rng(seed)`` draws the
  instances with ``limit=instances``, then the null with ``sets.null_sets(Tu, du, the upstream codes
  at each instance's upstream position, vs, ws, rng)``. Passes run through ``sets.run_steered`` on
  ``up``, each instance at its upstream position, recording the downstream site's input tap; the
  downstream codes are ``core.encode`` of what that holds at each instance's downstream position,
  keeping latents below ``down.width``. With ``Sv`` and ``Sw`` the upstream sets of ``v`` and ``w``
  (``Tu.set_for``), and ``Dv`` and ``Dw`` the downstream sets of ``g(v)`` and ``g(w)``, a set's mass
  is its codes' sum, taken in float64. Seven passes run in one call, whose sequences are every
  instance for each pass in turn:

  - **unintervened:** nothing removed or added
  - **the swap:** ``Sv`` removed, and ``Sw`` added at ``Tu.mean_codes(du, w)``'s values for those
    latents
  - **the null's swap:** the null's latents removed, and its latents added at the first amounts of
    ``w``'s set
  - **the ablation:** ``Sv`` removed, nothing added; **the null's ablation:** the null's latents
    removed, nothing added
  - **the single swap:** ``Sv[0]`` removed, ``Sw[0]`` added at its amount; **the single ablation:**
    ``Sv[0]`` removed, nothing added

  The mapping:

  - ``"relation"``, ``"judged"`` (``R.judged``), ``"up"`` and ``"down"``, each ``{"description",
    "block", "class", "site", "tap"}`` with the site's output tap, ``"pairs"`` (every pair the
    declaration gave, over every window) and ``"instances"``.
  - ``"follows"``: the share of instances where, after the swap, ``Dw``'s mass exceeds ``Dv``'s.
    ``"null_follows"``: the same after the null's swap.
  - ``"kept"``: the median over instances of ``Dv``'s mass after the ablation over its mass in the
    unintervened pass. ``"null_kept"``: the same after the null's ablation.
  - ``"output"``: None where no instance has a ``read``; otherwise, over those instances,
    ``{"instances", "follows", "null_follows"}``: the shares where the output at ``read``
    (``sets.argmax`` of the logits there) is ``target`` after the swap and after the null's swap.
  - ``"residual_alone"``: None where no instance has ``up == down``; otherwise, over those
    instances, ``{"instances", "blocks", "follows", "follows_unfrozen", "output_follows"}``. The
    swap runs again, one call per position, ascending, over those instances at that position, with
    ``freeze`` the ``blocks``, those after ``R.up.block`` up to and including ``R.down.block``.
    ``follows`` is its share as above, ``follows_unfrozen`` the swap's over the same instances, and
    ``output_follows`` the frozen swap's output share, None where none of them has a ``read``.
  - ``"label"``: ``CARRIED`` where ``residual_alone`` isn't None and its ``follows`` is at least half
    its ``follows_unfrozen``; ``COMPUTED`` otherwise, as every edge across positions is.
  - ``"single"``: ``{"follows", "kept"}`` with each name's best latent in place of its set: after the
    single swap, the share where ``Dw[0]``'s code exceeds ``Dv[0]``'s; and after the single ablation,
    the median of ``Dv[0]``'s code over its unintervened code, over the instances where that is above
    zero, None where it is at none.

  With no instance, ``"instances"`` is 0 and every key from ``"follows"`` on is None. ValueError as
  for ``draw_edge_instances``, or for ``instances`` or ``seed`` that aren't a positive int and a
  non-negative int.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CARRIED = "carried"
COMPUTED = "computed"


@dataclass(frozen=True)
class EdgeInstance:
    window: int
    up: int
    down: int
    v: int
    gv: int
    w: int
    gw: int
    read: int | None
    target: int | None


def draw_edge_instances(up: Any, down: Any, relation: str, *, limit: int,
                        rng: np.random.Generator) -> list[EdgeInstance]:
    raise NotImplementedError


def edge(up: Any, down: Any, relation: str, *, instances: int, seed: int) -> dict[str, Any]:
    raise NotImplementedError
