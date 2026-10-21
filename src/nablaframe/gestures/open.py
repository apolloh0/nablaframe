"""open-this-up (D51): which upstream parts make a target, ranked by absolute attribution, and how many reproduce it.

Spec: docs/specs/phase-6.md, D51. Built on ``sets.py``, with nothing specific to a task (D29). JAX is
imported inside the functions that need it, so this module imports without it. Nothing here caches
or judges: the gates ask for open-this-up, and a record holds what ``open_up`` returns (D47).

* ``PARTS = (1, 2, 4, 8, 16, 32, 64, 128)``, ``IG_STEPS = 5``, ``ERROR = 0.1``, ``NO_SPAN = 1e-6``,
  ``TOP_PARTS = 8`` and ``ORDERS = ("absolute", "signed", "random")``.
* ``Target``: a frozen dataclass ``(window, position, latents)``: a held-out window's index, the
  position read there, and the downstream latents whose pre-activations it sums, a tuple of ints.
* ``class_of(declaration, position)``: the class a position's parts are named at: the smallest class
  the run declares that holds the position, ties to the one declared first. ValueError where none
  does.
* ``pre_activation(dictionary, site, y, latents)``: float64 (n,), the summed pre-activations of
  ``latents`` for inputs ``y`` (n, d_in) at the site, before the top ``k`` are kept: with
  ``u = dictionary.unit(site)``, ``Σ_l ((y / scale_in[site] − b_pre[site]) · W_enc[u][:, l] +
  b_enc[site, l])``, in float64. ValueError for no latents, or a latent outside
  ``range(dictionary.active_width(site))``.
* ``evaluate(up, down, targets, codes, *, chunk=32)``: float64 (n,), each target's summed
  pre-activation, as ``pre_activation`` defines it at ``down``'s site for ``target.latents``, when
  ``up``'s output tap holds ``decode(codes) + err`` at every position. ``up`` and ``down`` are
  ``sets.SiteReading``s; ``codes`` is (n, seq, up.width), the upstream codes target ``k``'s pass
  uses, latents beyond ``up.width`` being zero; ``err`` is the upstream error term on the
  unintervened pass, the output tap's value less ``core.decode`` of ``up.codes``, held fixed as the
  edge meter holds it (D42). The downstream input tap is read at each target's position. The pass is
  ``model.instrumented`` with that one intervention, compiled with ``jax.jit`` with the codes, tokens
  and error as arguments and computed in float32, ``chunk`` targets at a time.
* ``attribute(up, down, targets, *, chunk=32)``: ``Attribution(codes, attributions, full, none)``:
  ``codes``, float32 (n, seq, up.width), ``up.codes`` of each target's window; ``attributions``,
  float64 (n, seq, up.width), integrated gradients from zero, ``codes`` times the mean over ``α`` in
  ``(s + 0.5) / IG_STEPS``, ``s`` in ``range(IG_STEPS)``, of the gradient of ``evaluate``'s
  pre-activation with respect to the codes at ``α · codes``, taken with ``jax.grad`` in float32 and
  averaged in float64, and zero at every position after the target's; ``full``, ``evaluate`` with
  ``codes``; ``none``, ``evaluate`` with every code zero.
* ``order(attributions, codes, kind, rng=None)``: one target's parts in order, an int64 array of
  ``position * width + latent``, for ``attributions`` and ``codes`` shaped (positions, width), the
  positions up to and including the target's: every part there, each once.

  - ``"absolute"``: by absolute attribution, largest first, ties to the lower position and then
    the lower latent
  - ``"signed"``: by attribution, largest first, with the same ties
  - ``"random"``: the parts whose code is above zero, ascending, in the order
    ``rng.permutation`` gives them; then every other part, ascending

  ValueError for another ``kind``, mismatched shapes, or ``"random"`` without ``rng``.
* ``open_up(up, down, cls, *, instances, seed, latent=None, description=None, value=None,
  chunk=32)``: D51 for one target, as a plain JSON mapping that ``records.GestureRecord`` takes as
  its results. The target is ``down``'s ``latent``, or the name ``description = value``'s set at the
  class, ``down.table(cls).set_for(description, value)``: exactly one of the two forms.

  - **Instances:** every (window, position), window by window and each window's positions in the
    class's order, where the target fires: the latent's code, or the set's summed codes, above zero
    in ``down.codes``, and for a name, ``description`` is ``value`` there. With
    ``rng = numpy.random.default_rng(seed)``, more than ``instances`` are cut to ``instances`` with
    ``sorted(rng.choice(len(candidates), size=instances, replace=False))``. Then each instance, in
    order, draws its random order's shuffle, ``rng.permutation`` of its parts whose code is above
    zero, ascending, as ``order(..., "random", rng)`` draws it.
  - ``attribute``, with ``chunk``, gives each instance's attributions, ``full`` and ``none``.
    Instances whose ``|full − none|`` is below ``NO_SPAN`` are left out of everything below, and
    counted.
  - **Each order, each instance:** its parts in that order (``order`` of its attributions and codes
    up to its position). Keeping the first ``k`` for each ``k`` in ``PARTS``, those parts' codes stay
    and every other upstream code, at every position, is zero; the error is
    ``|evaluate(kept) − full| / |full − none|``. Its reading is the fewest ``k`` whose error is at most
    ``ERROR``, None for "not within 128".

  The mapping:

  - ``"class"``; ``"target"``, ``{"latents", "latent", "description", "value"}``, the unused form's
    keys None; ``"up"`` and ``"down"``, each ``{"site", "tap", "block"}`` with its output tap
  - ``"candidates"`` (where the target fires, before the cut), ``"instances"`` (drawn), ``"no_span"``
    (left out) and ``"measured"``
  - ``"steps"`` (``IG_STEPS``) and ``"parts"`` (``PARTS`` as a list)
  - ``"windows"``: ``[window, position]`` for each measured instance
  - ``"orders"``: for each of ``ORDERS``, ``{"error_median", "readings", "reading_median",
    "within"}``: the median error over measured instances at each ``k``; each measured instance's
    reading; their median, a reading of None counted as larger than any ``k``, and None where the
    median is such a reading; and the share of measured instances with a reading. With no measured
    instance, each is None and ``readings`` empty.
  - ``"top_parts"``: for each measured instance, its first ``TOP_PARTS`` parts by absolute
    attribution, each ``{"position", "latent", "attribution", "name"}``. ``name`` is steps 1 and 2 of
    D49 for that latent at ``class_of(declaration, position)`` on ``up``: its top candidate,
    ``{"description", "value", "f1", "label"}``, labelled as ``name.NAMED`` or ``name.WEAK``; None
    where it has no candidate.

  ValueError unless both readings share their model, params and declaration, as the same objects,
  and have equal windows; unless ``up``'s output tap comes before ``down``'s input tap in
  ``trace.registry.tap_names`` order; for a class the run doesn't declare; for both target forms or
  neither, a latent outside ``range(down.width)``, or a name that doesn't count at the class; or for
  ``instances`` or ``seed`` that aren't a positive int and a non-negative int.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PARTS = (1, 2, 4, 8, 16, 32, 64, 128)
IG_STEPS = 5
ERROR = 0.1
NO_SPAN = 1e-6
TOP_PARTS = 8
ORDERS = ("absolute", "signed", "random")


@dataclass(frozen=True)
class Target:
    window: int
    position: int
    latents: tuple[int, ...]


@dataclass(frozen=True)
class Attribution:
    codes: np.ndarray
    attributions: np.ndarray
    full: np.ndarray
    none: np.ndarray


def class_of(declaration: Any, position: int) -> str:
    raise NotImplementedError


def pre_activation(dictionary: Any, site: int, y: Any, latents: Any) -> np.ndarray:
    raise NotImplementedError


def evaluate(up: Any, down: Any, targets: Sequence[Target], codes: Any, *, chunk: int = 32) -> np.ndarray:
    raise NotImplementedError


def attribute(up: Any, down: Any, targets: Sequence[Target], *, chunk: int = 32) -> Attribution:
    raise NotImplementedError


def order(attributions: Any, codes: Any, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
    raise NotImplementedError


def open_up(up: Any, down: Any, cls: str, *, instances: int, seed: int, latent: int | None = None,
            description: str | None = None, value: int | None = None, chunk: int = 32) -> dict[str, Any]:
    raise NotImplementedError
