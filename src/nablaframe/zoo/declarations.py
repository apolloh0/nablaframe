"""The zoo tasks' declarations to the gestures (D29, D49, D50): classes, planted names, predictions and relations.

Spec: docs/specs/phase-6.md, "What each run declares". Each declaration is a
``gestures.declare.Declaration`` built from its task, so the gestures carry no task-specific code.

* ``two_hop(task)``, for a ``TwoHopTask`` with ``facts`` facts, windows of ``task.window_length``
  tokens and ``seq = task.seq``, one less. With ``q = seq - 1``, the ``⇒`` (25 for six facts):

  - ``run`` ``"two_hop"``. Classes: ``"query"`` (``q``), ``"subjects"`` (0, 4, ...,
    ``4 * (facts - 1)``), ``"objects"`` (2, 6, ..., ``4 * (facts - 1) + 2``), ``"every"``
    (``range(seq)``).
  - Variables: ``"b"``, the window's intermediate, ``task.parse(window).b``, at positions
    ``q - 1`` and ``q``, -1 elsewhere.
  - Planted: ``b`` after block 2 at ``query``, judged; ``token(p+1)`` after block 3 at
    ``query``, judged; ``b`` after block 1 at ``query``, recorded.
  - ``predict``: ``b`` at ``q`` moved to ``w`` gives ``(q, the object of w's fact)`` when ``w``
    is a distractor subject, the subject of a fact other than ``a``'s and ``b``'s, and None for
    any other ``w``. ``token(p+1)`` at ``q`` gives ``(q, w)``. Anything else gives None.
  - Relations: ``"hop"``, ``b`` after block 2 at ``query`` to ``token(p+1)`` after block 3 at
    ``query``, judged; ``"carry"``, ``b`` after block 1 at ``query`` to ``b`` after block 2 at
    ``query``, recorded; ``"two back"``, ``token(p)`` after block 0 at ``objects`` to
    ``token(p-2)`` after block 2 at ``subjects``, recorded.
  - ``pairs``: ``((q, q),)`` for the hop and the carry; for two back, ``(p, p + 2)`` for each
    object position ``p`` with ``p + 2`` a subject position, in order.
  - ``value_map``: the hop maps ``v`` to the object of ``v``'s fact in the window, None when
    ``v`` is no fact's subject; the carry and two back map ``v`` to ``v``.

* ``induction(task)``, for an ``InductionTask``, ``seq`` being ``task.seq``:

  - ``run`` ``"induction"``. Classes: ``"mechanism"`` (``task.mechanism_positions``) and
    ``"every"`` (``range(seq)``). No variables.
  - Planted: ``token(p)`` after block 0 at ``every``, judged; ``token(p+1)`` after block 1 at
    ``mechanism``, judged.
  - ``predict``: ``token(p+1)`` at a mechanism position ``p`` gives ``(p, w)``; ``token(p)`` at
    a sole target ``s`` gives ``(s + P - 1, w)``, ``P`` the window's period. Anything else gives
    None.
  - Relations: ``"copy"``, ``token(p)`` after block 0 at ``every`` to ``token(p+1)`` after block
    1 at ``mechanism``, judged. ``pairs``: ``(s, s + P - 1)`` for each sole target ``s``, in
    order. ``value_map``: ``v`` to ``v``.

* ``period(window)``: the smallest ``i >= 1`` with ``window[i] == window[0]``. ValueError when
  the window doesn't repeat its first token.
* ``sole_targets(task, window)``: the sole targets of an induction window, a sorted int64 array:
  the positions ``s`` with ``1 <= s <= P`` and ``s + P - 1`` a mechanism position. There, the
  token at ``s + P - 1`` has occurred once before, at ``s - 1``, so the induction head there copies
  from ``s`` and nowhere else. ValueError for a window of the wrong length or without a period.

Both declarations check their arguments as ``gestures.declare`` says every declaration does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gestures.declare import Declaration


def period(window: Any) -> int:
    raise NotImplementedError


def sole_targets(task: Any, window: Any) -> np.ndarray:
    raise NotImplementedError


def two_hop(task: Any) -> Declaration:
    raise NotImplementedError


def induction(task: Any) -> Declaration:
    raise NotImplementedError
