"""The command line, and the one path by which every gesture record is built (D47).

Spec: docs/specs/phase-6.md, D47. A request names what to measure; a ``Context`` resolves
what it names, a run's model, checkpoints, held-out windows and dictionaries; and ``build``
runs the gesture and returns its record. The gates, the command line and the record builds
of G6.7 all go through here, so a record never depends on who built it. JAX is imported inside
the functions that need it, so this module imports without it.

* ``READ_CHUNK = 256`` and ``OPEN_CHUNK = 32``: the windows per pass when a site is read, and
  the targets per pass in open-this-up. They are fixed here because open-this-up's results
  depend on its chunk at the 1e-7 level (PR #79), and a record's key doesn't hold them.
* ``RUNS = ("induction", "reference", "two_hop")``. ``declaration_for(run, config)``: the run's
  declaration. ``"two_hop"`` and ``"induction"`` are ``zoo.declarations.two_hop`` and
  ``induction`` of the task at its defaults (``TwoHopTask()``, ``InductionTask()``), as G4.1
  trained them; ``"reference"`` is ``declare.reference(config.seq)``. ValueError for another
  run, or where the declaration's ``seq`` isn't ``config.seq``.
* ``Context``: a frozen dataclass, everything a request names, resolved:

  - ``run``: the run id, 64 lowercase hex characters
  - ``declaration``, ``model`` and ``n_layers``
  - ``steps``: the run's checkpoints, a tuple of ints, ascending and distinct
  - ``params_at(step)``: the params at a checkpoint; ``state_at(step)``: its state hash
  - ``windows``: the held-out windows, an integer array (B, seq + 1)
  - ``dictionary(content_hash)``: the ``Dictionary`` with that content hash; it raises
    ValueError for one the context doesn't hold
  - ``seeds``: ``{seed: {content hash: content hash}}``, for each dictionary seed beside seed 0
    (D41's robustness seeds), the dictionary that seed trained in place of each seed-0 dictionary.
    Only generalize-this reads it; it may be empty.

* ``MODULES``: ``{kind: module names}``, the modules whose code identity a record of each kind
  carries (D18): ``nablaframe.gestures.cli``, ``declare``, ``sets`` and
  ``nablaframe.zoo.declarations`` for every kind, and the gesture's own: ``name`` for a name;
  ``edge`` for an edge; ``name`` and ``open`` for open-this-up; ``name``, ``edge`` and
  ``generalize`` for generalize-this. ``code(kind)``: ``records.code_identity`` of those
  modules.
* ``inputs(request, context)``: the record's inputs. ``states``: the state hash at
  ``request.step``, and for generalize-this at every step in ``context.steps``, each keyed by
  its step as a decimal string. ``dictionaries``: the content hashes the request names, and for
  generalize-this each seed's counterparts of them, sorted, with no repeats. ``windows``:
  ``[batch_hash(context.windows)]``.
* ``results(request, context)``: the gesture's results. Every site is read with
  ``sets.read_site(context.model, params, n_layers=context.n_layers,
  declaration=context.declaration, dictionary=..., site=..., windows=context.windows,
  chunk=READ_CHUNK)``, ``params`` being ``context.params_at(request.step)``, taken once, so that
  readings share it.

  - **name:** ``name.name(reading, class, latent, instances=, seed=)``, the reading at
    ``dictionary`` and ``site``
  - **edge:** ``edge.edge(up, down, relation, instances=, seed=)``, the readings at ``up`` and
    ``up_site``, and at ``down`` and ``down_site``
  - **open:** ``open.open_up(up, down, class, instances=, seed=, chunk=OPEN_CHUNK)``, with the
    request's ``latent``, or its ``description`` and ``value``; ``up`` read at ``upstream`` and
    ``upstream_site``, ``down`` at ``dictionary`` and ``site``
  - **generalize, a name:** ``generalize.generalize_name(reading, class, description,
    instances=, seed=, steps=context.steps, params_at=context.params_at, seed_readings=)``,
    ``seed_readings`` holding, for each seed in ``context.seeds`` ascending, the reading of that
    seed's counterpart of ``dictionary`` at ``site``. No ``extras``: a task's own readings stay
    in the gates' reports.
  - **generalize, an edge:** ``generalize.generalize_edge(up, down, relation, instances=,
    seed=, steps=context.steps, params_at=context.params_at, seed_readings=)``, each seed's
    ``(up, down)`` read at its counterparts of ``up`` and ``down``

* ``build(request, context)``: the ``records.GestureRecord`` of ``request``,
  ``inputs(request, context)``, ``code(request.kind)`` and ``results(request, context)``,
  written nowhere. ``record(folder, request, context)``: ``records.cached_record`` of the same,
  so the results are computed only when ``folder`` doesn't hold the record already.
* ``inputs``, ``results``, ``build`` and ``record`` raise ValueError, before anything runs, for
  a request whose ``run`` isn't ``context.run``, whose ``step`` isn't one of ``context.steps``,
  that names a dictionary the context doesn't hold, or, for generalize-this, where a seed in
  ``context.seeds`` has no counterpart of one of the request's dictionaries.
* ``main(argv)``: the command line. Arguments:

  - ``--run``: one of ``RUNS``, which picks the declaration
  - ``--config``: a JSON of ``GPT2Config``'s fields
  - ``--checkpoints``: a folder of ``step-<n>.safetensors`` checkpoints with their sidecars.
    ``steps`` is every one of them, and ``state_at`` reads each sidecar's ``state_sha256``.
    Every sidecar must name the request's run; ``run`` is that run id.
  - ``--windows``: an integer ``.npy``, the held-out windows (B, seq + 1)
  - ``--dictionaries``: a folder searched at every depth for ``<content hash>.safetensors``,
    loaded with ``core.load_dictionary`` when a request names it. More than one file of that
    name is refused.
  - ``--request``: the request's JSON, as ``records.request_json`` writes it
  - ``--out``: the folder of the run's records
  - ``--seeds``, optional: a JSON ``{"<seed>": {"<content hash>": "<content hash>"}}``, the
    context's ``seeds``

  It writes the record with ``record`` and prints its key. Returns 0. Any error prints one line
  starting ``error:`` and returns 1, having written nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import declare, records

READ_CHUNK = 256
OPEN_CHUNK = 32
RUNS = ("induction", "reference", "two_hop")

_COMMON = ("nablaframe.gestures.cli", "nablaframe.gestures.declare", "nablaframe.gestures.sets",
           "nablaframe.zoo.declarations")
MODULES = {
    "name": _COMMON + ("nablaframe.gestures.name",),
    "edge": _COMMON + ("nablaframe.gestures.edge",),
    "open": _COMMON + ("nablaframe.gestures.name", "nablaframe.gestures.open"),
    "generalize": _COMMON + ("nablaframe.gestures.name", "nablaframe.gestures.edge", "nablaframe.gestures.generalize"),
}


@dataclass(frozen=True)
class Context:
    run: str
    declaration: Any
    model: Any
    n_layers: int
    steps: tuple[int, ...]
    params_at: Callable[[int], Any]
    state_at: Callable[[int], str]
    windows: np.ndarray
    dictionary: Callable[[str], Any]
    seeds: Mapping[int, Mapping[str, str]]


def declaration_for(run: str, config: Any) -> declare.Declaration:
    raise NotImplementedError


def code(kind: str) -> dict[str, str]:
    raise NotImplementedError


def inputs(request: records.GestureRequest, context: Context) -> dict[str, Any]:
    raise NotImplementedError


def results(request: records.GestureRequest, context: Context) -> dict[str, Any]:
    raise NotImplementedError


def build(request: records.GestureRequest, context: Context) -> records.GestureRecord:
    raise NotImplementedError


def record(folder: str | Path, request: records.GestureRequest, context: Context) -> records.GestureRecord:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
