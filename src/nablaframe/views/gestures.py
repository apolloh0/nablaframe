"""The gestures panel: a run's gesture records as the viewer lists them, each with its meter first (D47).

Spec: docs/specs/phase-6.md, D47 and "Records, the viewer and badges"; contract:
docs/deltas/phase-6-delta-5.md. The viewer stays a pure decoder (D13, D27): every value here is
copied from a record's results, never computed, and the viewer reads each record's own file,
checking it against the SHA-256 its entry holds, as it checks the trace's tensors. Nothing here
runs a model or imports JAX.

* ``PANEL = "panel.json"``: the file ``write_panel`` writes into a run's records folder, beside the
  ``index.json`` that ``records.write_record`` keeps there.
* ``METER_KEYS``: each kind's meter keys, in order:

  - ``"name"``: ``("label", "description", "value", "f1", "set_f1", "coverage", "causal",
    "instances", "follows", "null_follows", "unchanged", "null_unchanged", "ablated_changed",
    "kl", "null_kl", "rank_from_top", "rank_from_bottom")``
  - ``"edge"``: ``("relation", "judged", "instances", "follows", "null_follows", "kept",
    "null_kept", "label", "residual_alone", "follows_unfrozen", "output_follows",
    "output_null_follows")``
  - ``"open"``: ``("class", "target", "candidates", "instances", "measured", "no_span", "parts",
    "reading", "within", "signed_within", "random_within", "error_median")``
  - ``"generalize"``: ``("form", "relation", "description", "class", "values", "envelope",
    "slices_f1", "slices_follows", "seeds", "final")``

* ``gesture_meter(record)``: a ``records.GestureRecord``'s meter, ``{key: value}`` for its kind's
  ``METER_KEYS`` in order, each value a copy of what its results hold, as plain JSON. A value the
  results don't hold because the step behind it didn't happen is None. With ``r`` the results:

  - **name:** where ``r["name"]`` is None (the latent fires on no row of the class), every value
    is None. Otherwise, with ``d`` the name's description: ``label``, ``description``, ``value``
    and ``f1`` are ``r["name"]``'s; ``set_f1`` is ``r["set"]["name_f1"]``; ``coverage`` is
    ``r["coverage"]["set"][d]``, the share of the set's mass where ``d`` is the name's value;
    ``rank_from_top`` and ``rank_from_bottom`` are ``r["effect"]["set"]``'s. With ``s =
    r["steering"]``: where ``s`` is None (no instance), ``causal``, ``instances`` and the seven
    shares and KLs are None. Otherwise ``causal`` and ``instances`` are ``s``'s; where ``s`` is
    causal, ``follows``, ``null_follows``, ``unchanged``, ``null_unchanged`` and
    ``ablated_changed`` are ``s``'s and ``kl`` and ``null_kl`` None; where it isn't, ``kl`` and
    ``null_kl`` are ``s``'s and the five shares None.
  - **edge:** ``relation``, ``judged``, ``instances``, ``follows``, ``null_follows``, ``kept``,
    ``null_kept`` and ``label`` are ``r``'s; ``residual_alone`` and ``follows_unfrozen`` are
    ``r["residual_alone"]``'s ``follows`` and ``follows_unfrozen``, and ``output_follows`` and
    ``output_null_follows`` are ``r["output"]``'s ``follows`` and ``null_follows``, each None where
    that entry is None.
  - **open:** ``class``, ``target``, ``candidates``, ``instances``, ``measured``, ``no_span`` and
    ``parts`` are ``r``'s; ``reading``, ``within`` and ``error_median`` are
    ``r["orders"]["absolute"]``'s ``reading_median``, ``within`` and ``error_median``;
    ``signed_within`` and ``random_within`` are ``r["orders"]["signed"]["within"]`` and
    ``r["orders"]["random"]["within"]``.
  - **generalize:** ``form`` is ``"name"`` where the request's arguments hold ``description``,
    and ``"edge"`` otherwise. ``seeds`` is ``r["seeds"]`` and ``final`` the last of
    ``r["checkpoints"]``, the final checkpoint's row, both as they are. A name adds
    ``description``, ``class``, ``values`` and ``envelope``, all ``r``'s, and ``slices_f1`` and
    ``slices_follows``, ``r["slices"]``'s ``f1`` and ``follows``, with ``relation`` None. An edge
    adds ``relation``, ``r``'s, with the other five None.

  ValueError, naming the record's key and the missing entry, where the results don't hold an entry
  this reads.
* ``gesture_payload(record)``: the record as the panel lists it: ``{"file", "key", "sha256",
  "kind", "request", "meter"}``: ``gesture-<key>.json``; ``record.key``; the SHA-256 of the bytes
  ``records.write_record`` writes for the record, its canonical JSON and a newline; its kind;
  ``records.request_json`` of its request; and ``gesture_meter``. Plain JSON, sharing no object
  with the record or with another call's result; records equal as values give equal payloads.
* ``gesture_panel(folder)``: the panel of a run's records folder, as ``records.write_record``
  leaves it: ``{"run", "records", "names"}``.

  - ``"records"``: ``gesture_payload`` of each record the folder's ``index.json`` lists, in the
    index's order, each read with ``records.read_record``, which checks the file's SHA-256 and key.
  - ``"run"``: the run id every record's request names; None where the index lists no record.
  - ``"names"``: the name records by place, so the features view can link a latent to them:
    ``{"<dictionary>-<site>": {"<latent>": [key, ...]}}`` from each name record's request
    arguments, each list in ``"records"`` order. Every other kind is left out.

  ValueError, naming the problem, for a folder without an ``index.json`` in ``records.py``'s form,
  a ``gesture-*.json`` file the index doesn't list, an entry ``read_record`` refuses, or two
  records whose requests name different runs.
* ``write_panel(folder)``: writes ``<folder>/PANEL``, the canonical JSON of
  ``gesture_panel(folder)`` and a newline, ASCII, through a temporary file and a rename, and
  returns its path. It writes nothing else: the records and their index stay as they are. With the
  records unchanged, writing it again gives the same bytes. Raises as ``gesture_panel`` does,
  before anything is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PANEL = "panel.json"
METER_KEYS = {
    "name": ("label", "description", "value", "f1", "set_f1", "coverage", "causal", "instances", "follows",
             "null_follows", "unchanged", "null_unchanged", "ablated_changed", "kl", "null_kl", "rank_from_top",
             "rank_from_bottom"),
    "edge": ("relation", "judged", "instances", "follows", "null_follows", "kept", "null_kept", "label",
             "residual_alone", "follows_unfrozen", "output_follows", "output_null_follows"),
    "open": ("class", "target", "candidates", "instances", "measured", "no_span", "parts", "reading", "within",
             "signed_within", "random_within", "error_median"),
    "generalize": ("form", "relation", "description", "class", "values", "envelope", "slices_f1", "slices_follows",
                   "seeds", "final"),
}


def gesture_meter(record: Any) -> dict[str, Any]:
    raise NotImplementedError


def gesture_payload(record: Any) -> dict[str, Any]:
    raise NotImplementedError


def gesture_panel(folder: str | Path) -> dict[str, Any]:
    raise NotImplementedError


def write_panel(folder: str | Path) -> Path:
    raise NotImplementedError
