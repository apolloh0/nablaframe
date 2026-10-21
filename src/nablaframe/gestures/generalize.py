"""generalize-this (D52): where a name or an edge holds across held-out slices, dictionary seeds and the training run.

Spec: docs/specs/phase-6.md, D52. Built on ``sets.py``, ``name.py`` and ``edge.py``, with nothing
specific to a task (D29): the task's own readings at each checkpoint, such as a residual patch, come
from the caller through ``extras``. Nothing here caches or judges: the gates ask for generalize-this,
and a record holds what ``generalize_name`` or ``generalize_edge`` returns (D47).

* ``SLICES = 8``, ``HOLDS = 0.5`` and ``PROBE_READS = 0.9``.
* ``description_f1(table, sets_from, d)``: a description's F1 read again with given sets: the mean,
  over ``d``'s values at ``table`` that ``sets_from`` also has, of
  ``table.name_f1(d, v, sets_from.set_for(d, v))``, a float; None where there are none, or where
  ``d`` doesn't count at ``table``.
* ``steer_description(reading, cls, d, *, instances, rng, windows=None, sets_from=None)``: D49's
  steering for a description over all its values, each instance with its own value, or None where no
  instance is eligible. ``table = reading.table(cls)`` gives the amounts (``mean_codes``) and the
  null's shared and active latents; ``sets_from``, a ``ClassTable`` (``table`` by default), gives
  the sets and the null's top-8 exclusions (``ranked``). A value is usable where both tables have
  it.

  - Candidates: every (window, position), over ``windows`` (held-out window indices, ascending;
    every window by default) and each window's positions in the class's order, where ``d``'s value
    ``v`` is usable. Options: the usable values less those ``sets.taken`` in the window, ascending;
    predicting options: those for which ``declaration.predict(d, window, position, w)`` gives a
    prediction.
  - As in ``name.draw_instances``: ``causal`` is True where some candidate has a predicting option,
    and then the eligible candidates are those with one and draw from them; otherwise those with an
    option, drawing from all their options. More than ``instances`` eligible are cut with
    ``sorted(rng.choice(len(eligible), size=instances, replace=False))``; then each kept candidate,
    in order, draws ``w = choices[rng.integers(len(choices))]``.
  - The null: ``sets.null_sets`` with a table whose ``ranked`` is ``sets_from``'s and whose
    ``shared`` and ``active`` are ``table``'s, the codes at each instance's row, its ``v`` and ``w``,
    and ``rng``.
  - Three passes in one ``sets.run_steered`` call, every instance at its position for each pass in
    turn: unintervened; swapped, ``sets_from.set_for(d, v)`` removed and ``sets_from.set_for(d, w)``
    added at ``table.mean_codes(d, w)``'s values for those latents; the null, its latents removed and
    its added latents at the first amounts of ``w``'s set.

  Causal: ``{"causal": True, "instances", "follows", "null_follows"}``, the shares of instances whose
  output at the predicted position (``sets.argmax``) is the predicted token after the swap and after
  the null. Otherwise ``{"causal": False, "instances", "kl", "null_kl"}``, the mean
  ``dictionaries.edges.kl`` of the unintervened logits and the swapped, and the null's, at each
  instance's position. ValueError for a class the run doesn't declare, ``d`` not counting at it, or
  ``instances`` not a positive int.
* ``holds(f1, causal, follows)``: whether a name holds at a checkpoint: ``f1`` at least ``HOLDS``,
  and, where its steering is causal, ``follows`` at least ``HOLDS`` too; where the run predicts
  nothing for it (``causal`` False), ``f1`` alone decides; with no steering instance (``causal``
  None), or no ``f1``, it doesn't hold.
* ``envelope(rows)``: for checkpoint rows in step order, each with ``"step"``, ``"holds"`` and
  ``"probe"``: ``{"steps", "runs", "final", "gap", "probe_reads_from"}``. ``steps``: the steps whose
  row holds; ``runs``: ``[first, last]`` of each maximal run of consecutive rows that hold, in order;
  ``final``: the run that ends at the last row, None where the last row doesn't hold; ``gap``: the
  steps whose probe is at least ``PROBE_READS`` but whose row doesn't hold; ``probe_reads_from``: the
  first step whose probe is at least ``PROBE_READS``, or None. A probe of None reads nothing.
* ``generalize_name(reading, cls, d, *, instances, seed, steps, params_at, seed_readings=None,
  extras=None)``: D52 for the description ``d`` read over all its values, as a plain JSON mapping
  that ``records.GestureRecord`` takes as its results. ``reading`` is a ``sets.SiteReading`` at the
  run's final checkpoint, ``T = reading.table(cls)``; ``rng = numpy.random.default_rng(seed)`` is used
  by every ``steer_description`` below, in the order given.

  - ``"description"``, ``"class"``, ``"site"``, ``"tap"`` (the site's output tap) and ``"values"``
    (``d``'s values at ``T``).
  - ``"slices"``: the held-out windows split by ``numpy.array_split(range(B), SLICES)``. For each
    slice, ``{"windows", "f1", "follows", "null_follows"}``: its window count;
    ``description_f1`` of the ``ClassTable`` of the slice's codes and descriptions at the class's
    positions, with ``T``'s sets; and ``follows`` and ``null_follows`` from ``steer_description`` on
    ``reading`` over the slice's windows, None where it isn't causal or finds no instance. ``"slices"`` is
    ``{"rows", "f1", "follows"}``, the last two ``[least, greatest]`` over the rows that have one, or
    None.
  - ``"seeds"``: for each seed in ``seed_readings`` (a mapping from a dictionary's seed to a
    ``SiteReading`` of it at the same site, model, params and windows), ascending, ``{"seed", "f1",
    "follows", "null_follows", "cosine"}``. The name is derived again at that dictionary, never matched
    by latent id: ``f1`` is ``description_f1`` of its own table with its own sets, and the steering is
    ``steer_description`` on it, its ``follows`` None where it isn't causal or finds no instance.
    ``cosine`` is the mean, over the values both tables have, of the mean over that seed's set of each
    latent's highest cosine between decoder rows (``W_dec[unit(site)]``) and ``reading``'s set's; None
    where there are none. Where ``d`` doesn't count at the seed's class, all three are None and no
    steering is drawn.
  - ``"checkpoints"``: for each step in ``steps``, ascending, with ``params = params_at(step)`` and
    the reading ``sets.read_site`` gives at those params with ``reading``'s dictionary, site and
    windows (the final dictionaries, D44): ``{"step", "f1", "causal", "follows", "null_follows",
    "probe", "accuracy", "extras", "holds"}``.

    - ``f1``: ``description_f1`` of that reading's table with ``T``'s sets
    - the steering: ``steer_description`` on that reading with ``sets_from=T``, its ``causal``,
      ``follows`` and ``null_follows``, the last two None where it isn't causal; all three None where
      it finds no instance, or where ``d`` doesn't count at that reading's class, and then none is
      drawn
    - ``probe``: ``measure.probe_accuracy`` of the site's input tap values at the class's rows where
      ``d`` isn't -1, those rows' values, and the largest value plus one; None for fewer than two rows
    - ``accuracy``: ``measure.readout_loss``'s accuracy over the windows at the class's positions,
      from the unintervened logits
    - ``extras``: ``extras(step, params)`` as given, a mapping of plain JSON values; None without
      ``extras``
    - ``holds``: ``holds(f1, causal, follows)``
  - ``"envelope"``: ``envelope`` of the checkpoint rows.

  ValueError for ``d`` not counting at the class; ``steps`` that aren't non-empty, strictly
  ascending non-negative ints; a seed reading that doesn't share ``reading``'s model, params and
  declaration as the same objects, its windows and its site's taps; or ``instances`` or ``seed`` that
  aren't a positive int and a non-negative int.
* ``generalize_edge(up, down, relation, *, instances, seed, steps, params_at, seed_readings=None)``:
  D52 for an edge, a plain JSON mapping. Each reading below is measured with
  ``edge.edge(..., instances=instances, seed=seed)``, and its row holds ``"instances"``,
  ``"follows"``, ``"null_follows"``, ``"kept"``, ``"null_kept"`` and ``"label"`` from it.

  - ``"relation"``.
  - ``"slices"``: for each slice as above, ``sets.read_site`` of both sites over the slice's windows,
    with ``up``'s model and params; the row adds ``"windows"``.
  - ``"seeds"``: for each seed in ``seed_readings`` (a mapping from a seed to its ``(up, down)``
    readings), ascending; the row adds ``"seed"``.
  - ``"checkpoints"``: for each step in ``steps``, ascending, both sites read at ``params_at(step)``
    with the final dictionaries; the row adds ``"step"``.

  ValueError as ``edge.edge`` raises it, for ``steps`` as above, or for a seed's readings that don't
  share ``up``'s model, params, declaration and windows and their sites' taps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

SLICES = 8
HOLDS = 0.5
PROBE_READS = 0.9


def description_f1(table: Any, sets_from: Any, d: str) -> float | None:
    raise NotImplementedError


def steer_description(reading: Any, cls: str, d: str, *, instances: int, rng: np.random.Generator,
                      windows: Sequence[int] | None = None, sets_from: Any = None) -> dict[str, Any] | None:
    raise NotImplementedError


def holds(f1: float | None, causal: bool | None, follows: float | None) -> bool:
    raise NotImplementedError


def envelope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError


def generalize_name(reading: Any, cls: str, d: str, *, instances: int, seed: int, steps: Sequence[int],
                    params_at: Callable[[int], Any], seed_readings: Mapping[int, Any] | None = None,
                    extras: Callable[[int, Any], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    raise NotImplementedError


def generalize_edge(up: Any, down: Any, relation: str, *, instances: int, seed: int, steps: Sequence[int],
                    params_at: Callable[[int], Any], seed_readings: Mapping[int, Any] | None = None) -> dict[str, Any]:
    raise NotImplementedError
