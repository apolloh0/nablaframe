"""The command line, and the one path by which every gesture record is built (D47).

Spec: docs/specs/phase-6.md, D47. A request names what to measure; a ``Context`` resolves
what it names, a run's model, checkpoints, held-out windows and dictionaries; and ``build``
runs the gesture and returns its record. The gates, the command line and the record builds
of G6.7 all go through here, so a record never depends on who built it. JAX is imported inside
the functions that need it, so this module imports without it.

* ``READ_CHUNK = 256`` and ``OPEN_CHUNK = 32``: the windows per pass when a site is read, and
  the targets per pass in open-this-up. They are fixed here because open-this-up's results
  depend on its chunk at the 1e-7 level (PR #79), and a record's key doesn't hold them.
* ``RUNS = ("induction", "modular_addition", "reference", "two_hop")``.
  ``declaration_for(run, config)``: the run's declaration. ``"two_hop"``, ``"induction"`` and
  ``"modular_addition"`` are ``zoo.declarations.two_hop``, ``induction`` and
  ``modular_addition`` of the task at its defaults (``TwoHopTask()``, ``InductionTask()``,
  ``ModularAdditionTask()``), as G4.1 and G7.1 trained them; ``"reference"`` is
  ``declare.reference(config.seq)``. ValueError for another run, or where the declaration's
  ``seq`` isn't ``config.seq``. ``functions_for(run)``: ``zoo.functions.for_run`` of a zoo run
  and the same task; ValueError for ``"reference"``, which declares no functions, or another
  run.
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
  - Only crystallize-this reads the rest, and each is None by default:

    - ``fit_windows``: the fitting windows, an integer array (B, seq + 1)
    - ``functions``: the run's ``RunFunctions``
    - ``twin``: its twin's params (D30); ``twin_state``: their state hash
    - ``cited(key)``: the run's record with that key; it raises ValueError for one the context
      doesn't hold

* ``MODULES``: ``{kind: module names}``, the modules whose code identity a record of each kind
  carries (D18): ``nablaframe.gestures.cli``, ``declare``, ``sets`` and
  ``nablaframe.zoo.declarations`` for every kind, and the gesture's own: ``name`` for a name;
  ``edge`` for an edge; ``name`` and ``open`` for open-this-up; ``name``, ``edge`` and
  ``generalize`` for generalize-this; ``nablaframe.gestures.crystallize`` and
  ``nablaframe.zoo.functions`` for crystallize-this. ``code(kind)``: ``records.code_identity`` of
  those modules.
* ``inputs(request, context)``: the record's inputs. ``states``: the state hash at
  ``request.step``, and for generalize-this at every step in ``context.steps``, each keyed by
  its step as a decimal string. ``dictionaries``: the content hashes the request names, and for
  generalize-this each seed's counterparts of them, sorted, with no repeats. ``windows``:
  ``[batch_hash(context.windows)]``. For crystallize-this, ``states`` holds every step in
  ``context.steps`` and ``"twin"``, ``context.twin_state``; ``dictionaries`` is empty; and
  ``windows`` holds the batch hashes of ``context.fit_windows`` and ``context.windows``, sorted,
  with no repeats.
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
  - **crystallize:** ``crystallize.crystallize(context.model, params, n_layers=, functions=
    context.functions, declaration=, block=, part=, cls=class, function=, form=,
    fit_windows=context.fit_windows, held_windows=context.windows, seed=, twin=context.twin,
    steps=context.steps, params_at=context.params_at, cited=)``, ``cited`` being
    ``context.cited`` of each of ``cites``, in order. It reads no site.

* ``build(request, context)``: the ``records.GestureRecord`` of ``request``,
  ``inputs(request, context)``, ``code(request.kind)`` and ``results(request, context)``,
  written nowhere. ``record(folder, request, context)``: ``records.cached_record`` of the same,
  so the results are computed only when ``folder`` doesn't hold the record already.
* ``inputs``, ``results``, ``build`` and ``record`` raise ValueError, before anything runs, for
  a request whose ``run`` isn't ``context.run``, whose ``step`` isn't one of ``context.steps``,
  that names a dictionary the context doesn't hold, or, for generalize-this, where a seed in
  ``context.seeds`` has no counterpart of one of the request's dictionaries. For
  crystallize-this they also raise it for a context without ``fit_windows``, ``functions``,
  ``twin``, ``twin_state`` or ``cited``, or fitting windows of another width; a function the
  run doesn't declare; ``instances`` other than the held-out windows' count; and a cited key
  the context doesn't hold, or whose record is another run's.
* ``main(argv)``: the command line. Arguments:

  - ``--run``: one of ``RUNS``, which picks the declaration
  - ``--config``: a JSON of ``GPT2Config``'s fields
  - ``--checkpoints``: a folder of ``step-<n>.safetensors`` checkpoints with their sidecars.
    ``steps`` is every one of them, and ``state_at`` reads each sidecar's ``state_sha256``.
    Every sidecar must name the request's run; ``run`` is that run id.
  - ``--windows``: an integer ``.npy``, the held-out windows (B, seq + 1)
  - ``--dictionaries``, optional: a folder searched at every depth for
    ``<content hash>.safetensors``, loaded with ``core.load_dictionary`` when a request names
    it. More than one file of that name is refused. Without it the context holds no dictionary.
  - ``--request``: the request's JSON, as ``records.request_json`` writes it
  - ``--out``: the folder of the run's records
  - ``--seeds``, optional: a JSON ``{"<seed>": {"<content hash>": "<content hash>"}}``, the
    context's ``seeds``
  - ``--fit-windows``, optional: an integer ``.npy``, the context's ``fit_windows``
  - ``--records``, optional and repeatable: a folder of records with its ``index.json``.
    ``cited(key)`` is ``records.read_record`` from the first folder whose index holds ``key``:
    the out folder first, then these, in order. A folder without an ``index.json`` holds no
    record.

  For a zoo run, the context's ``functions`` is ``functions_for(run)``, its ``twin``
  ``init_params(config, seed=1)`` (D30), and its ``twin_state`` that twin's
  ``train.checkpoint.state_hash``.

  It writes the record with ``record`` and prints its key. Returns 0. Any error prints one line
  starting ``error:`` and returns 1, having written nothing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..zoo import declarations
from . import declare, edge, generalize, name, records, sets
from . import open as opening

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

_HASH = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT = re.compile(r"step-(0|[1-9][0-9]*)\.safetensors")
_SEED = re.compile(r"[1-9][0-9]*")


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
    from ..zoo.tasks import InductionTask, TwoHopTask

    if run == "two_hop":
        declaration = declarations.two_hop(TwoHopTask())
    elif run == "induction":
        declaration = declarations.induction(InductionTask())
    elif run == "reference":
        declaration = declare.reference(config.seq)
    else:
        raise ValueError(f"run must be one of {RUNS}, got {run!r}")
    if declaration.seq != config.seq:
        raise ValueError(f"the {run} run reads {declaration.seq} tokens, but the config's seq is {config.seq}")
    return declaration


def code(kind: str) -> dict[str, str]:
    if kind not in MODULES:
        raise ValueError(f"kind must be one of {tuple(MODULES)}, got {kind!r}")
    return records.code_identity(*(importlib.import_module(module) for module in MODULES[kind]))


def _named(request: records.GestureRequest) -> list[tuple[str, int]]:
    """The (dictionary, site) pairs the request reads, in the order its gesture takes them."""
    args = request.args
    if "relation" in args:
        return [(args["up"], args["up_site"]), (args["down"], args["down_site"])]
    if request.kind == "open":
        return [(args["upstream"], args["upstream_site"]), (args["dictionary"], args["site"])]
    return [(args["dictionary"], args["site"])]


def _check(request: records.GestureRequest, context: Context) -> None:
    """Every refusal, before anything runs: ValueError for anything the request names that the context lacks."""
    if not isinstance(request, records.GestureRequest):
        raise ValueError(f"request must be a GestureRequest, got {type(request).__name__}")
    if request.run != context.run:
        raise ValueError(f"the request is for run {request.run}, but the context holds run {context.run}")
    if request.step not in context.steps:
        raise ValueError(f"step {request.step} isn't one of the run's checkpoints {list(context.steps)}")
    windows = np.asarray(context.windows)
    seq = context.declaration.seq
    if (windows.ndim != 2 or windows.shape[0] < 1 or windows.shape[1] != seq + 1
            or not np.issubdtype(windows.dtype, np.integer)):
        raise ValueError(f"the windows must be an integer array (B, {seq + 1}) with B at least 1, got "
                         f"{windows.dtype} {windows.shape}")
    args, declaration = request.args, context.declaration
    for digest, site in _named(request):
        dictionary = context.dictionary(digest)  # ValueError for one the context doesn't hold
        if not 0 <= site < len(dictionary.sites):
            raise ValueError(f"dictionary {digest} has no site {site}; it has {len(dictionary.sites)}")
    if "relation" in args:
        declaration.relation(args["relation"])  # ValueError for a relation the run doesn't declare
    elif args["class"] not in declaration.classes:
        raise ValueError(f"run {declaration.run!r} declares no class {args['class']!r}; it declares "
                         f"{list(declaration.classes)}")
    if "latent" in args:
        digest, site = _named(request)[-1]
        width = context.dictionary(digest).active_width(site)
        if args["latent"] >= width:
            raise ValueError(f"latent {args['latent']} is beyond site {site}'s active width {width}")
    if "description" in args:
        known = declare.DESCRIPTIONS + tuple(declaration.variables)
        if args["description"] not in known:
            raise ValueError(f"run {declaration.run!r} has no description {args['description']!r}; it has "
                             f"{list(known)}")
    if request.kind == "generalize":
        for s in sorted(context.seeds):
            for digest, _ in _named(request):
                if digest not in context.seeds[s]:
                    raise ValueError(f"dictionary seed {s} has no counterpart of dictionary {digest}")
                context.dictionary(context.seeds[s][digest])


def inputs(request: records.GestureRequest, context: Context) -> dict[str, Any]:
    _check(request, context)
    steps = context.steps if request.kind == "generalize" else (request.step,)
    named = {digest for digest, _ in _named(request)}
    if request.kind == "generalize":
        named |= {context.seeds[s][digest] for s in context.seeds for digest, _ in _named(request)}
    return {"states": {str(step): context.state_at(step) for step in steps}, "dictionaries": sorted(named),
            "windows": [batch_hash(context.windows)]}


def results(request: records.GestureRequest, context: Context) -> dict[str, Any]:
    _check(request, context)
    params = context.params_at(request.step)  # taken once, so that every reading shares it

    def read(digest: str, site: int) -> Any:
        return sets.read_site(context.model, params, n_layers=context.n_layers, declaration=context.declaration,
                              dictionary=context.dictionary(digest), site=site, windows=context.windows,
                              chunk=READ_CHUNK)

    args, counts = request.args, {"instances": request.instances, "seed": request.seed}
    places = _named(request)
    if request.kind == "name":
        return name.name(read(*places[0]), args["class"], args["latent"], **counts)
    if request.kind == "edge":
        return edge.edge(read(*places[0]), read(*places[1]), args["relation"], **counts)
    if request.kind == "open":
        up, down = read(*places[0]), read(*places[1])
        target = ({"latent": args["latent"]} if "latent" in args
                  else {"description": args["description"], "value": args["value"]})
        return opening.open_up(up, down, args["class"], **counts, **target, chunk=OPEN_CHUNK)
    over = {"steps": context.steps, "params_at": context.params_at}
    if "relation" in args:
        (up, up_site), (down, down_site) = places
        seed_readings = {s: (read(context.seeds[s][up], up_site), read(context.seeds[s][down], down_site))
                         for s in sorted(context.seeds)}
        return generalize.generalize_edge(read(up, up_site), read(down, down_site), args["relation"], **counts,
                                          **over, seed_readings=seed_readings)
    digest, site = places[0]
    seed_readings = {s: read(context.seeds[s][digest], site) for s in sorted(context.seeds)}
    return generalize.generalize_name(read(digest, site), args["class"], args["description"], **counts, **over,
                                      seed_readings=seed_readings)


def build(request: records.GestureRequest, context: Context) -> records.GestureRecord:
    held = inputs(request, context)
    return records.GestureRecord(request, held, code(request.kind), results(request, context))


def record(folder: str | Path, request: records.GestureRequest, context: Context) -> records.GestureRecord:
    held = inputs(request, context)
    return records.cached_record(folder, request, held, code(request.kind), lambda: results(request, context))


class _Parser(argparse.ArgumentParser):
    """Raises on a bad command line, so ``main`` reports it in one line and returns 1."""

    def error(self, message: str) -> Any:
        raise ValueError(message)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = _Parser(prog="python -m nablaframe.gestures.cli",
                     description="Run a gesture request on a run's checkpoint and write its record.")
    parser.add_argument("--run", required=True, choices=RUNS, help="the run, which picks its declaration")
    parser.add_argument("--config", type=Path, required=True, help="a JSON of GPT2Config's fields")
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="a folder of step-<n>.safetensors checkpoints with their sidecars")
    parser.add_argument("--windows", type=Path, required=True, help="an integer .npy, the held-out windows")
    parser.add_argument("--dictionaries", type=Path, required=True,
                        help="a folder searched at every depth for <content hash>.safetensors")
    parser.add_argument("--request", type=Path, required=True, help="the request's JSON")
    parser.add_argument("--out", type=Path, required=True, help="the folder of the run's records")
    parser.add_argument("--seeds", type=Path, help='a JSON {"<seed>": {"<content hash>": "<content hash>"}}')
    return parser.parse_args(argv)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecars(folder: Path, run: str) -> dict[int, str]:
    """Each checkpoint's step and state hash; ValueError unless every sidecar names ``run``."""
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    states = {}
    for path in sorted(folder.iterdir()):
        match = _CHECKPOINT.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        step, sidecar = int(match.group(1)), path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{path} has no sidecar {sidecar.name}")
        data = _json(sidecar)
        if not isinstance(data, dict) or data.get("step") != step:
            raise ValueError(f"{sidecar} doesn't name step {step}")
        if data.get("run_id") != run:
            raise ValueError(f"{sidecar} names run {data.get('run_id')!r}, not the request's run {run}")
        state = data.get("state_sha256")
        if not isinstance(state, str) or _HASH.fullmatch(state) is None:
            raise ValueError(f"{sidecar} holds no state_sha256")
        states[step] = state
    if not states:
        raise ValueError(f"{folder} holds no step-<n>.safetensors checkpoints")
    return dict(sorted(states.items()))


def _dictionaries(folder: Path) -> Callable[[str], Any]:
    """The context's ``dictionary``: each one loaded when first named, from the one file of its name."""
    from ..dictionaries.core import load_dictionary

    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    found: dict[str, list[Path]] = {}
    for path in sorted(folder.rglob("*.safetensors")):
        if _HASH.fullmatch(path.stem) and path.is_file():
            found.setdefault(path.stem, []).append(path)
    loaded: dict[str, Any] = {}

    def dictionary(digest: str) -> Any:
        if digest not in loaded:
            paths = found.get(digest, [])
            if not paths:
                raise ValueError(f"{folder} holds no dictionary {digest}.safetensors")
            if len(paths) > 1:
                raise ValueError(f"{folder} holds {len(paths)} files named {digest}.safetensors: "
                                 f"{', '.join(str(p) for p in paths)}")
            loaded[digest] = load_dictionary(paths[0])
        return loaded[digest]

    return dictionary


def _seeds(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    data = _json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object of seeds")
    seeds = {}
    for seed, counterparts in data.items():
        if _SEED.fullmatch(seed) is None:
            raise ValueError(f"{path}: a seed beside seed 0 is a positive decimal int, got {seed!r}")
        if not isinstance(counterparts, dict) or not all(
                isinstance(v, str) and _HASH.fullmatch(k) and _HASH.fullmatch(v) for k, v in counterparts.items()):
            raise ValueError(f"{path}: seed {seed} must map content hashes to content hashes")
        seeds[int(seed)] = dict(counterparts)
    return dict(sorted(seeds.items()))


def _run(args: argparse.Namespace) -> str:
    """Resolves the context, then writes the record; a failure before the write writes nothing."""
    from ..model.gpt2 import GPT2Config, GPT2Model, init_params
    from ..train.checkpoint import load_params

    config = GPT2Config(**_json(args.config))
    declaration = declaration_for(args.run, config)
    request = records.request_from_json(_json(args.request))
    states = _sidecars(args.checkpoints, request.run)
    windows = np.load(args.windows, allow_pickle=False)
    if args.out.exists() and not args.out.is_dir():
        raise ValueError(f"{args.out} is not a folder")
    template = init_params(config, seed=0)

    def params_at(step: int) -> Any:
        if step not in states:
            raise ValueError(f"step {step} isn't one of the run's checkpoints {list(states)}")
        return load_params(args.checkpoints / f"step-{step}.safetensors", like=template)

    def state_at(step: int) -> str:
        if step not in states:
            raise ValueError(f"step {step} isn't one of the run's checkpoints {list(states)}")
        return states[step]

    context = Context(run=request.run, declaration=declaration, model=GPT2Model(config), n_layers=config.n_layers,
                      steps=tuple(states), params_at=params_at, state_at=state_at, windows=windows,
                      dictionary=_dictionaries(args.dictionaries), seeds=_seeds(args.seeds))
    return record(args.out, request, context).key


def main(argv: list[str] | None = None) -> int:
    try:
        key = _run(_arguments(argv))
    except Exception as exc:  # noqa: BLE001 - every error is one line and exit code 1
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
