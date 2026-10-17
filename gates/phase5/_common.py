"""Shared by the Phase 5 gates. Read with docs/specs/phase-5.md, "The dictionaries", "Discrimination tests on the zoo", "The edge-sparsity meter", "The reference run" and "Gates".

May load ``gates/phase4/_common.py``, and through it the earlier phases' ``_common.py``, by path,
but never changes them. Reports go to ``reports/phase5/``, working files to ``runs/phase5/``, and
dictionaries to ``runs/phase5/dictionaries/``. Every gate runs on the Mac only, refuses a dirty
tree unless ``--allow-dirty``, refuses to overwrite its report unless ``--overwrite``, prints a
progress line at least every 30 seconds during anything longer than a minute, and writes one
schema-2 report. Every intervened pass goes through ``run_intervened``, except the splice and the
edge meter's passes, which ``dictionaries/measure.py`` and ``dictionaries/edges.py`` run. Each
gate refuses to start unless G4.1's report exists, passes ``validate_report`` and says ``pass``,
and G5.2 to G5.5 also unless G5.1's does; G5.5 also needs G1.4's run.

The settings are the spec's, which are the feasibility runs' (D37):

* ``WIDTH_FACTOR = 8``, ``K = 16``, ``LR = 5e-4``, ``DICT_SEED = 0``, ``TRAIN_SEED = 6``,
  ``EVAL_SEED = 7``, ``JAC_BATCH = 1024``, ``LAMBDAS = (1.0, 4.0)``, ``ROBUST_SEEDS = (1, 2)``.
* Per run, keyed ``"induction"``, ``"two_hop"`` and ``"reference"``:
  ``STEPS = {3000, 3000, 2000}``, ``BATCH = {4096, 4096, 2048}``,
  ``TRAIN_WINDOWS = {8192, 16384, 2048}``, ``EVAL_WINDOWS = {1024, 2048, 64}``,
  ``METER_WINDOWS = {8, 8, 2}`` (the first of the held-out windows) and ``CHUNK = {256, 256, 32}``
  (sequences per pass when collecting activations).
* ``SHORT = {"steps": 300, "batch": 1024}``: G5.1's settings for the dictionaries it trains twice
  and patches through. 300 steps cover the warmup, the hold and the decay.
* ``BARS = {"fvu": 0.1, "ablation_cost": 0.1, "recovered": 0.9, "faithful_applies": 0.5,
  "faithful_ratio": 0.9, "break_excess": 0.05, "control_unchanged": 0.95, "selective_f1": 0.5,
  "selective_twin_factor": 2.0, "edge_ratio": 0.2}``.
* ``READOUT = {"induction": (32, ..., 62), "two_hop": (25,)}``: the readout positions, as tuples;
  the reference run reads every position.
* ``settings(name, kind, d_model, *, seed=DICT_SEED, lam=0.0, short=False)``: the
  ``core.Settings`` the spec gives: width ``WIDTH_FACTOR · d_model``, ``k = K``, the run's
  ``STEPS`` and ``BATCH`` (``SHORT``'s with ``short``), ``lr = LR``, ``seed``, ``lam`` and
  ``jac_batch = JAC_BATCH``, every other field at its default. ValueError for an unknown run.
* ``sites(kind, n_layers, layer=None)``: a ``topk`` at ``layer``'s ``hook_resid_post``; a
  ``staircase`` over every block's; a ``transcoder`` from ``layer``'s ``ln2.hook_normalized`` to its
  ``hook_mlp_out``; a ``jsae`` at ``layer``'s two, the pair across its MLP. As lists of
  ``(input tap, output tap)``. ValueError for a missing or out-of-range ``layer``, a ``layer``
  given to a ``staircase``, or another kind.
* ``zoo_windows(task, name, which, *, dict_seed=DICT_SEED)``: int32 windows of
  ``task.window_length`` tokens: ``"train"`` is ``task.batch_for_step(TRAIN_SEED + 10 · dict_seed,
  0, TRAIN_WINDOWS[name])``, so each robustness seed trains on windows of its own; ``"held_out"``
  is ``task.batch_for_step(EVAL_SEED, 0, EVAL_WINDOWS[name])``. ValueError otherwise.
* ``reference_windows(train, validation, seq, which, *, dict_seed=DICT_SEED)``: int32 windows of
  ``seq + 1`` tokens from the reference run's token streams: ``"train"`` is
  ``data.corpus.batch_for_step(train, TRAIN_SEED + 10 · dict_seed, 0, TRAIN_WINDOWS["reference"],
  seq)``; ``"held_out"`` the first ``EVAL_WINDOWS["reference"]`` of
  ``data.corpus.eval_windows(validation, EVAL_TOKENS, seq)``, the run's own evaluation slice, with
  ``EVAL_TOKENS`` Phase 1's.
* ``variables(task, name, windows)``: ``(by_position, n_values)``, the planted variables each
  held-out window carries. Two-hop: ``{24: {"a", "b", "c"}, 25: {"a", "b", "c"}}``, each one value
  per window from ``task.parse``, with ``n_values = task.entities``. Induction: ``{"mechanism":
  {"cur", "prev", "next"}}``, the token at each mechanism position ``p``, at ``p - 1`` and at
  ``p + 1``, one value per (window, position) in window-major order, with ``n_values =
  task.vocab_size``. Every value is an int64 array.
* ``induction_sources(task, windows, j, test)``: ``"follow"``: G4.2's sources,
  ``new_token_sources(windows, j, numpy.random.default_rng(j), task.vocab_size)``; ``"break"``: a new
  token at ``j - 1``, ``new_token_sources(windows, j - 1, numpy.random.default_rng(100 + j),
  task.vocab_size)``. ValueError for another test.
* ``dictionary_key(state_sha256, sites, settings, tokens)``: ``core.dictionary_key`` with the
  batch hashes ``[batch_hash(tokens)]`` and the code ``train.code_identity()``.
* ``twin_state(params)``: the state hash of a twin's parameters,
  ``train.checkpoint.state_hash(params)``; a twin has no checkpoint.
* ``train_dictionary(model, params, tokens, settings, sites, *, key, chunk, label, mlp_weights=None)``:
  a ``Dictionary`` trained on the sites' activations over ``tokens`` (``measure.activations`` in
  chunks of ``chunk``, every position, each site's input and output taps), carrying ``key``.
  It prints ``label`` and each ``progress`` line ``train`` reports, inside ``ticking``.
* ``cached(cache_dir, key, make)``: ``(dictionary, seconds)``. When ``cache_dir/<key>/`` holds
  a dictionary, it is loaded with ``load_dictionary`` and ``seconds`` is None. Otherwise
  ``make()`` trains it, timed; it must carry ``key``, and is saved there with
  ``save_dictionary``. A loaded dictionary whose ``key`` isn't ``key``, a folder holding anything
  but one dictionary's two files, or a made one with another key raises ValueError. So a gate
  retrains nothing another gate trained under the same key.
* ``ticking(label, interval_s=30, *, write=print)``: a context manager. While it is open, a
  background thread writes ``"<label>: still working, <elapsed> in"`` every ``interval_s``
  seconds, so long training never looks hung. It stops on exit, also on an exception.
* ``loss_recovered(clean, spliced, ablated)``: ``{"cost", "recovered"}``. ``cost`` is
  ``ablated - clean``; ``recovered`` is ``(ablated - spliced) / cost`` where ``cost >=
  BARS["ablation_cost"]``, and None elsewhere, where the spec records the change instead.
* ``faithful(residual, code)``: ``{"applies", "ratio"}``. It applies where the residual patch's
  rate is at least ``BARS["faithful_applies"]``; ``ratio`` is ``code / residual`` there, and None
  elsewhere.
* ``selective_margin(trained, twin)``: ``trained - BARS["selective_twin_factor"] · twin``, the
  amount by which a mean best F1 clears twice the twin's; the bar is that it isn't negative.
* ``edge_ratio(trained, twin, twin_scored)``: the trained pair's meter over its twin's, as a plain
  float. A twin reading of None, a curve that never settles, counts as ``twin_scored``, every
  edge the twin scored, which only overstates the ratio. A trained reading of None gives None,
  which fails the bar.

Every judgement helper returns plain Python floats, bools and None, ready for a report.
"""

# Beyond the contract above, this module holds the paths and modules several Phase 5 gates share:
#
# * ``p4``: gates/phase4/_common.py, and through it ``p3``, ``p2`` and ``p1``, all loaded by path
#   under their own module names (every one of these files is called ``_common``). ``p1`` is the
#   module the others loaded, so pointing one set of globals at a scratch folder points every gate
#   at it.
# * ``REPORT_DIR``, ``RUN_DIR`` and ``DICTIONARY_DIR``: reports/phase5/, runs/phase5/ and
#   runs/phase5/dictionaries/.
#
# Paths and Phase 1's settings are read from the modules' globals when a function runs, never bound
# at import. Nothing here imports JAX at module level, so every gate script imports in any
# environment.

from __future__ import annotations

import importlib.util
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

WIDTH_FACTOR, K, LR, DICT_SEED, TRAIN_SEED, EVAL_SEED = 8, 16, 5e-4, 0, 6, 7
JAC_BATCH, LAMBDAS, ROBUST_SEEDS = 1024, (1.0, 4.0), (1, 2)
RUNS = ("induction", "two_hop", "reference")
STEPS = {"induction": 3000, "two_hop": 3000, "reference": 2000}
BATCH = {"induction": 4096, "two_hop": 4096, "reference": 2048}
TRAIN_WINDOWS = {"induction": 8192, "two_hop": 16384, "reference": 2048}
EVAL_WINDOWS = {"induction": 1024, "two_hop": 2048, "reference": 64}
METER_WINDOWS = {"induction": 8, "two_hop": 8, "reference": 2}
CHUNK = {"induction": 256, "two_hop": 256, "reference": 32}
SHORT = {"steps": 300, "batch": 1024}
BARS = {"fvu": 0.1, "ablation_cost": 0.1, "recovered": 0.9, "faithful_applies": 0.5, "faithful_ratio": 0.9,
        "break_excess": 0.05, "control_unchanged": 0.95, "selective_f1": 0.5, "selective_twin_factor": 2.0,
        "edge_ratio": 0.2}
READOUT = {"induction": tuple(range(32, 63)), "two_hop": (25,)}

ZOO = ("induction", "two_hop")
KINDS = ("topk", "staircase", "transcoder", "jsae")
WHICH = ("train", "held_out")
TESTS = ("follow", "break")
_KEY = re.compile(r"[0-9a-f]{64}")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase5"
RUN_DIR = REPO_ROOT / "runs" / "phase5"
DICTIONARY_DIR = RUN_DIR / "dictionaries"


def _load_phase4_common() -> Any:
    """gates/phase4/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase4_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase4"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p4 = _load_phase4_common()
p3 = p4.p3
p2 = p4.p2
p1 = p4.p1


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def settings(name: str, kind: str, d_model: int, *, seed: int = DICT_SEED, lam: float = 0.0, short: bool = False) -> Any:
    if name not in RUNS:
        raise ValueError(f"no run {name!r}; expected one of {RUNS}")
    from nablaframe.dictionaries.core import Settings

    steps, batch = (SHORT["steps"], SHORT["batch"]) if short else (STEPS[name], BATCH[name])
    return Settings(kind, WIDTH_FACTOR * d_model, K, steps, batch, lr=LR, seed=seed, lam=lam, jac_batch=JAC_BATCH)


def sites(kind: str, n_layers: int, layer: int | None = None) -> list[tuple[str, str]]:
    if kind not in KINDS:
        raise ValueError(f"no dictionary kind {kind!r}; expected one of {KINDS}")
    if not _is_int(n_layers) or n_layers < 1:
        raise ValueError(f"n_layers must be a positive int, got {n_layers!r}")
    if kind == "staircase":
        if layer is not None:
            raise ValueError(f"a staircase covers every block; it takes no layer, got {layer!r}")
        return [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(n_layers)]
    if layer is None:
        raise ValueError(f"a {kind} needs a layer")
    if not _is_int(layer) or not 0 <= layer < n_layers:
        raise ValueError(f"layer must be an int in range({n_layers}), got {layer!r}")
    layer = int(layer)
    if kind == "topk":
        return [(f"blocks.{layer}.hook_resid_post",) * 2]
    if kind == "transcoder":
        return [(f"blocks.{layer}.ln2.hook_normalized", f"blocks.{layer}.hook_mlp_out")]
    return [(f"blocks.{layer}.ln2.hook_normalized",) * 2, (f"blocks.{layer}.hook_mlp_out",) * 2]


def zoo_windows(task: Any, name: str, which: str, *, dict_seed: int = DICT_SEED) -> np.ndarray:
    if name not in ZOO:
        raise ValueError(f"no zoo run {name!r}; expected one of {ZOO}")
    if which == "train":
        windows = task.batch_for_step(TRAIN_SEED + 10 * dict_seed, 0, TRAIN_WINDOWS[name])
    elif which == "held_out":
        windows = task.batch_for_step(EVAL_SEED, 0, EVAL_WINDOWS[name])
    else:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    return np.ascontiguousarray(windows, dtype=np.int32)


def reference_windows(train: np.ndarray, validation: np.ndarray, seq: int, which: str, *,
                      dict_seed: int = DICT_SEED) -> np.ndarray:
    from nablaframe.data.corpus import batch_for_step, eval_windows

    if which == "train":
        windows = batch_for_step(np.asarray(train), TRAIN_SEED + 10 * dict_seed, 0, TRAIN_WINDOWS["reference"], seq)
    elif which == "held_out":
        windows = eval_windows(np.asarray(validation), p1.EVAL_TOKENS, seq)[:EVAL_WINDOWS["reference"]]
    else:
        raise ValueError(f"which must be one of {WHICH}, got {which!r}")
    return np.ascontiguousarray(windows, dtype=np.int32)


def variables(task: Any, name: str, windows: np.ndarray) -> tuple[dict[Any, dict[str, np.ndarray]], int]:
    windows = np.asarray(windows)
    if name == "two_hop":
        parsed = [task.parse(window) for window in windows]
        values = {v: np.array([getattr(p, v) for p in parsed], dtype=np.int64) for v in ("a", "b", "c")}
        subject = 4 * task.facts  # the query's subject a, then the query token ⇒
        return {position: {v: values[v].copy() for v in values} for position in (subject, subject + 1)}, int(task.entities)
    if name == "induction":
        positions = np.asarray(task.mechanism_positions)
        by_name = {"cur": windows[:, positions], "prev": windows[:, positions - 1], "next": windows[:, positions + 1]}
        return {"mechanism": {v: a.reshape(-1).astype(np.int64) for v, a in by_name.items()}}, int(task.vocab_size)
    raise ValueError(f"no zoo run {name!r}; expected one of {ZOO}")


def induction_sources(task: Any, windows: np.ndarray, j: int, test: str) -> np.ndarray:
    from nablaframe.zoo.induction import new_token_sources

    if test == "follow":
        return new_token_sources(windows, j, np.random.default_rng(j), task.vocab_size)
    if test == "break":
        return new_token_sources(windows, j - 1, np.random.default_rng(100 + j), task.vocab_size)
    raise ValueError(f"test must be one of {TESTS}, got {test!r}")


def dictionary_key(state_sha256: str, sites: Sequence[Sequence[str]], settings: Any, tokens: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash
    from nablaframe.dictionaries import core, train

    return core.dictionary_key(state_sha256=state_sha256, sites=sites, settings=settings,
                               batch_hashes=[batch_hash(tokens)], code=train.code_identity())


def twin_state(params: Any) -> str:
    from nablaframe.train.checkpoint import state_hash

    return state_hash(params)


def _say(line: str) -> None:
    print(line, flush=True)


def train_dictionary(model: Any, params: Any, tokens: np.ndarray, settings: Any, sites: Sequence[Sequence[str]], *,
                     key: str, chunk: int, label: str, mlp_weights: Any = None) -> Any:
    from nablaframe.dictionaries import core, measure, train

    checked = core.check_sites(settings.kind, sites)
    taps = [tap for site in checked for tap in site]

    def progress(step: int, steps: int, losses: list[float]) -> None:
        _say(f"{label}: step {step:,}/{steps:,}, reconstruction {', '.join(f'{loss:.4g}' for loss in losses)}")

    _say(label)
    with ticking(label, write=_say):
        acts = measure.activations(model, params, tokens, taps, chunk=chunk)
        rows = {tap: value.reshape(-1, value.shape[-1]) for tap, value in acts.items()}
        data = [(rows[reads], rows[writes]) for reads, writes in checked]
        return train.train(settings, checked, data, mlp_weights=mlp_weights, key=key, progress=progress)


def cached(cache_dir: str | Path, key: str, make: Callable[[], Any]) -> tuple[Any, float | None]:
    from nablaframe.dictionaries import core

    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        raise ValueError(f"a cache key is 64 lowercase hex characters, got {key!r}")
    folder = Path(cache_dir) / key
    if folder.exists():
        if not folder.is_dir():
            raise ValueError(f"{folder} is not a folder")
        entries = sorted(folder.iterdir())
        if entries:
            stems = {entry.stem for entry in entries}
            if (len(entries) != 2 or len(stems) != 1 or not all(entry.is_file() for entry in entries)
                    or {entry.suffix for entry in entries} != {".json", ".safetensors"}):
                raise ValueError(f"{folder} must hold one dictionary's two files, not {[e.name for e in entries]}")
            dictionary = core.load_dictionary(folder / f"{stems.pop()}.safetensors")
            if dictionary.key != key:
                raise ValueError(f"{folder} holds a dictionary trained under key {dictionary.key}, not {key}")
            return dictionary, None
    start = time.perf_counter()
    dictionary = make()
    seconds = time.perf_counter() - start
    if getattr(dictionary, "key", None) != key:
        raise ValueError(f"made a dictionary under key {getattr(dictionary, 'key', None)}, not {key}; not saved")
    core.save_dictionary(dictionary, folder)
    return dictionary, float(seconds)


@contextmanager
def ticking(label: str, interval_s: float = 30.0, *, write: Callable[[str], Any] = print) -> Iterator[None]:
    if not interval_s > 0:
        raise ValueError(f"interval_s must be positive, got {interval_s!r}")
    start, stop = time.monotonic(), threading.Event()

    def beat() -> None:
        while not stop.wait(interval_s):
            write(f"{label}: still working, {p1._duration(time.monotonic() - start)} in")
            if write is print:
                sys.stdout.flush()

    thread = threading.Thread(target=beat, name=f"ticking: {label}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def loss_recovered(clean: float, spliced: float, ablated: float) -> dict[str, float | None]:
    cost = float(ablated) - float(clean)
    recovered = (float(ablated) - float(spliced)) / cost if cost >= BARS["ablation_cost"] else None
    return {"cost": cost, "recovered": recovered}


def faithful(residual: float, code: float) -> dict[str, Any]:
    residual = float(residual)
    applies = residual >= BARS["faithful_applies"]
    return {"applies": applies, "ratio": float(code) / residual if applies else None}


def selective_margin(trained: float, twin: float) -> float:
    return float(trained) - BARS["selective_twin_factor"] * float(twin)


def edge_ratio(trained: int | None, twin: int | None, twin_scored: int) -> float | None:
    if trained is None:
        return None
    return float(trained) / float(twin_scored if twin is None else twin)


# ---------------------------------------------------------------------------
# Beyond the contract: what the Phase 5 gates share
# ---------------------------------------------------------------------------
#
# * The start checks on earlier reports (``require_g41``, ``require_g43``, ``require_g51``), the report
#   paths, and Phase 5's working folders (``fresh_dir``, under RUN_DIR).
# * ``zoo_model``: G4.1's run of one zoo model at its final checkpoint, with its twin, read through
#   Phase 4's ``zoo_run`` and its checks; ``reference_model``: G1.4's run at its final checkpoint,
#   read through Phase 1's and Phase 2's helpers.
# * ``dictionary``: one dictionary of a model, through ``cached`` under the key the gates and G5.6
#   compute alike (``dictionary_spec``), its training time kept in the model's ``train_seconds``.
# * The measurements several gates make the same way: each site's meters, a site's codes and
#   activations at given positions, and the edge meter's reading of a pair.

VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "optax", "safetensors", "numpy")
G41_REPORT, G43_REPORT, G51_REPORT = "g41_zoo_training.json", "g43_two_hop.json", "g51_determinism.json"


def report_relative(name: str) -> str:
    """``reports/phase5/<name>``, as Phase 1's ``start`` takes it."""
    return str((Path(REPORT_DIR) / name).relative_to(p1.REPO_ROOT))


def _require(gate_id: str, path: Path, *, why: str) -> dict[str, Any]:
    """The report at ``path``, once it exists, passes ``validate_report``, is ``gate_id``'s and says ``pass``."""
    import json

    from nablaframe.reports import validate_report

    rel = p1.rel(path)
    if not Path(path).exists():
        raise SystemExit(f"{rel} is missing. Run {gate_id} first. No report written.")
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{rel} can't be read ({exc}). No report written.") from exc
    problems = validate_report(report) if isinstance(report, dict) else ["not a JSON object"]
    if problems:
        raise SystemExit(f"{rel} is not a valid report: {'; '.join(problems)}. No report written.")
    if report["gate_id"] != gate_id or report["status"] != "pass":
        raise SystemExit(f"{rel} is {report['gate_id']} with status {report['status']}, not a passing {gate_id}:"
                         f" {why}. No report written.")
    return report


def require_g41() -> dict[str, Any]:
    """G4.1's report, passing: no dictionary is judged on a model that missed its behaviour bars."""
    return p4.require_g41()


def require_g43() -> dict[str, Any]:
    """G4.3's report, passing: G5.3 reads its best swap block."""
    return _require("G4.3", Path(p4.REPORT_DIR) / G43_REPORT, why="G5.3 reads its best swap block")


def require_g51() -> dict[str, Any]:
    """G5.1's report, passing: nothing is measured with a trainer or an operation that failed it."""
    return _require("G5.1", Path(REPORT_DIR) / G51_REPORT,
                    why="no dictionary is measured before training and dpatch pass their determinism checks")


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    import shutil

    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 5 working folders live under {RUN_DIR}")
    if Path(path).exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)
    return Path(path)


def site_name(kind: str, output_tap: str) -> str:
    """A site as the gates name it, ``"<kind> <output tap>"``."""
    return f"{kind} {output_tap}"


def site_names(kind: str, dictionary_sites: Sequence[Sequence[str]]) -> list[str]:
    return [site_name(kind, writes) for _, writes in dictionary_sites]


# ---------------------------------------------------------------------------
# The models, and their dictionaries through the cache
# ---------------------------------------------------------------------------


def zoo_model(name: str, g41: dict[str, Any]) -> Any:
    """G4.1's run of ``name`` at its final checkpoint, and its twin, as the Phase 5 gates read them.

    ``tokens(windows)`` is what the model reads of windows; ``train_tokens(seed)`` those of the
    training windows for dictionary seed ``seed``, every dictionary's key tokens and training data;
    ``held_out`` the held-out windows and ``held_tokens`` what the model reads of them.
    """
    from types import SimpleNamespace

    from nablaframe.zoo.ground_truth import TWIN_SEED, twin_params

    run = p4.zoo_run(name, g41)
    task = run.task
    params = run.params_for(run.final)
    twin = twin_params(run.config)
    held_out = zoo_windows(task, name, "held_out")
    trained = {}

    def train_tokens(seed: int = DICT_SEED) -> np.ndarray:
        if seed not in trained:
            trained[seed] = p4.tokens(task, zoo_windows(task, name, "train", dict_seed=seed))
        return trained[seed]

    return SimpleNamespace(
        name=name, run=run, task=task, model=run.model, config=run.config, n_layers=run.config.n_layers,
        d_model=run.config.d_model, step=run.final, run_id=run.run_id, params=params, twin=twin,
        state=run.state_sha256(run.final), twin_state=twin_state(twin), twin_seed=TWIN_SEED,
        tokens=lambda windows: p4.tokens(task, windows), train_tokens=train_tokens, held_out=held_out,
        held_tokens=p4.tokens(task, held_out), readout=list(READOUT[name]), chunk=CHUNK[name],
        train_seconds={})


def reference_model() -> Any:
    """G1.4's run at its final checkpoint, read through Phase 1's and Phase 2's helpers, with its streams."""
    import json
    from types import SimpleNamespace

    p1.require_data()
    ref = p2.reference()
    step = ref.steps[-1]
    sidecar = p1.trace_checkpoint(ref.manifest, step).with_suffix(".json")
    state = json.loads(sidecar.read_text(encoding="utf-8"))["state_sha256"]
    train, validation = p1.load_split("train"), p1.load_split("validation")
    seq = ref.config.seq
    held_out = reference_windows(train, validation, seq, "held_out")
    if len(held_out) < EVAL_WINDOWS["reference"]:
        raise SystemExit(f"the validation split holds {len(held_out)} evaluation windows of {seq} tokens, fewer than"
                         f" {EVAL_WINDOWS['reference']}. No report written.")
    trained = {}

    def train_tokens(seed: int = DICT_SEED) -> np.ndarray:
        if seed not in trained:
            windows = reference_windows(train, validation, seq, "train", dict_seed=seed)
            trained[seed] = np.ascontiguousarray(windows[:, :seq], dtype=np.int32)
        return trained[seed]

    return SimpleNamespace(
        name="reference", ref=ref, model=ref.model, config=ref.config, n_layers=ref.config.n_layers,
        d_model=ref.config.d_model, step=step, run_id=ref.manifest.run_id, params=ref.params_for(step),
        twin=None, state=state, twin_state=None, train_tokens=train_tokens, held_out=held_out,
        held_tokens=np.ascontiguousarray(held_out[:, :seq], dtype=np.int32), readout=list(range(seq)),
        chunk=CHUNK["reference"], train_seconds={})


def dictionary_label(kind: str, n_layers: int, layer: int | None = None, *, lam: float = 0.0) -> str:
    """A dictionary's name in a report: its kind, then its block's output tap, or ``lam`` for a pair."""
    if kind == "staircase":
        return "staircase"
    writes = sites(kind, n_layers, layer)[-1][1]
    return f"{kind} {writes}" + (f" lam {lam:g}" if kind == "jsae" else "")


def dictionary_spec(m: Any, kind: str, layer: int | None = None, *, twin: bool = False, seed: int = DICT_SEED,
                    lam: float = 0.0, short: bool = False) -> tuple[list[tuple[str, str]], Any, str]:
    """``(sites, settings, key)`` of one dictionary of model ``m`` (or its twin): what ``dictionary`` trains it
    with, and the key it is cached under, which G5.6 recomputes to find it."""
    if twin and m.twin is None:
        raise ValueError(f"the {m.name} run has no twin")
    state = m.twin_state if twin else m.state
    dictionary_sites = sites(kind, m.n_layers, layer)
    chosen = settings(m.name, kind, m.d_model, seed=seed, lam=lam, short=short)
    return dictionary_sites, chosen, dictionary_key(state, dictionary_sites, chosen, m.train_tokens(seed))


def dictionary(m: Any, kind: str, layer: int | None = None, *, twin: bool = False, seed: int = DICT_SEED,
               lam: float = 0.0, short: bool = False) -> Any:
    """One dictionary of model ``m`` (or its twin), through ``cached(DICTIONARY_DIR, ...)``.

    Its key is ``dictionary_key`` of the model's state hash (``twin_state`` for the twin), its
    sites, ``settings(...)`` and ``m.train_tokens(seed)``, which it trains on when the cache doesn't
    hold it. A ``jsae`` trains with ``train.mlp_weights`` of its block. The training time, None from
    the cache, goes into ``m.train_seconds`` under ``"<trained|twin|seed n> <label>"``.
    """
    from nablaframe.dictionaries.train import mlp_weights

    params = m.twin if twin else m.params
    dictionary_sites, chosen, key = dictionary_spec(m, kind, layer, twin=twin, seed=seed, lam=lam, short=short)
    tokens = m.train_tokens(seed)
    who = "twin" if twin else ("trained" if seed == DICT_SEED else f"seed {seed}")
    name = f"{who} {dictionary_label(kind, m.n_layers, layer, lam=lam)}" + (" short" if short else "")
    weights = mlp_weights(params, layer) if kind == "jsae" else None

    def make() -> Any:
        return train_dictionary(m.model, params, tokens, chosen, dictionary_sites, key=key, chunk=m.chunk,
                                label=f"{m.name}: training {name}, key {key[:12]}", mlp_weights=weights)

    made, seconds = cached(DICTIONARY_DIR, key, make)
    if seconds is None:
        _say(f"{m.name}: {name} from the cache, key {key[:12]}")
    else:
        _say(f"{m.name}: {name} trained in {p1._duration(seconds)}")
    m.train_seconds[name] = seconds
    return made


# ---------------------------------------------------------------------------
# Measurements several gates make alike
# ---------------------------------------------------------------------------


def activations(m: Any, params: Any, tokens: np.ndarray, taps: Sequence[str], label: str) -> dict[str, np.ndarray]:
    """``measure.activations`` of ``taps`` over ``tokens``, in the run's chunks, inside ``ticking``."""
    from nablaframe.dictionaries import measure

    with ticking(label, write=_say):
        return measure.activations(m.model, params, tokens, list(dict.fromkeys(taps)), chunk=m.chunk)


def rows(values: np.ndarray, positions: Sequence[int] | None = None) -> np.ndarray:
    """A tap's values (batch, pos, d) as rows (N, d): every position, or ``positions`` in window-major order."""
    values = np.asarray(values)
    if positions is not None:
        values = values[:, list(positions)]
    return values.reshape(-1, values.shape[-1])


def codes(dictionary: Any, site: int, acts: dict[str, np.ndarray], positions: Sequence[int] | None = None) -> np.ndarray:
    """The site's codes of its input tap's values in ``acts``, as rows: every position, or ``positions``."""
    from nablaframe.dictionaries.core import encode

    return encode(dictionary, site, rows(acts[dictionary.sites[site][0]], positions))


def readout(logits: np.ndarray, m: Any) -> float:
    """``readout_loss`` at the run's readout positions on its held-out windows, the loss alone."""
    from nablaframe.dictionaries import measure

    return measure.readout_loss(logits, m.held_out, m.readout)[0]


class Meters:
    """Each site's meters on one model's held-out windows (D41), the unintervened and ablated passes made once.

    ``site(dictionary, site, twin=None)`` gives ``{"fvu", "dead_share", ["twin_fvu",
    "twin_dead_share",] "clean_loss", "spliced_loss", "ablated_loss", "cost", "recovered"}``: the
    FVU and dead share over every position; the readout losses of the unintervened pass, of
    ``spliced_logits`` and of ``ablated_logits`` at the site's output tap, and ``loss_recovered``.
    ``twin`` is ``(twin dictionary, twin activations)`` when the twin's FVU and dead share go beside.
    """

    def __init__(self, m: Any, acts: dict[str, np.ndarray]) -> None:
        self.m, self.acts = m, acts
        self.clean = readout(p3.unintervened(m.model, m.params, m.held_tokens, n_layers=m.n_layers)[0], m)
        self.ablated: dict[str, float] = {}

    def ablated_loss(self, tap: str) -> float:
        from nablaframe.dictionaries import measure

        if tap not in self.ablated:
            m = self.m
            self.ablated[tap] = readout(measure.ablated_logits(m.model, m.params, m.held_tokens, tap,
                                                               n_layers=m.n_layers), m)
        return self.ablated[tap]

    def site(self, dictionary: Any, site: int, twin: tuple[Any, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
        from nablaframe.dictionaries import measure

        m = self.m
        reads, writes = dictionary.sites[site]
        x, y = rows(self.acts[reads]), rows(self.acts[writes])
        out: dict[str, Any] = {"fvu": measure.fvu(dictionary, site, x, y), "dead_share": measure.dead_share(dictionary, site, x)}
        if twin is not None:
            twin_dictionary, twin_acts = twin
            tx, ty = rows(twin_acts[reads]), rows(twin_acts[writes])
            out |= {"twin_fvu": measure.fvu(twin_dictionary, site, tx, ty),
                    "twin_dead_share": measure.dead_share(twin_dictionary, site, tx)}
        spliced = readout(measure.spliced_logits(m.model, m.params, m.held_tokens, dictionary, site), m)
        ablated = self.ablated_loss(writes)
        out |= {"clean_loss": self.clean, "spliced_loss": spliced, "ablated_loss": ablated}
        out |= loss_recovered(self.clean, spliced, ablated)
        return out


def meter_summary(meters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``fvu_max`` over the sites, and ``recovered_min`` over those where ``recovered`` isn't None (None when none is)."""
    recovered = [entry["recovered"] for entry in meters.values() if entry["recovered"] is not None]
    return {"fvu_max": max(entry["fvu"] for entry in meters.values()) if meters else None,
            "recovered_min": min(recovered) if recovered else None}


def mlp_pair(m: Any, params: Any, tokens: np.ndarray, pair: Any, layer: int, label: str) -> dict[str, Any]:
    """The edge meter across block ``layer``'s MLP for a pair's sites 0 and 1 (D42).

    ``{"meter", "share", "scored_edges", "pair_kl", "curve"}``: ``mlp_edge_scores``, ``mlp_curve`` and
    ``meter``, with ``share`` the meter over the scored edges, None with the meter.
    """
    from nablaframe.dictionaries import edges
    from nablaframe.dictionaries.train import mlp_weights

    weights = mlp_weights(params, layer)
    with ticking(label, write=_say):
        scores = edges.mlp_edge_scores(m.model, params, tokens, (pair, 0), (pair, 1), weights)
        curve = edges.mlp_curve(m.model, params, tokens, (pair, 0), (pair, 1), weights, scores)
    return _reading(curve, label)


def block_pair(m: Any, params: Any, tokens: np.ndarray, up: tuple[Any, int], down: tuple[Any, int],
               label: str) -> dict[str, Any]:
    """The edge meter across a block, from ``up`` to ``down``, each ``(dictionary, site)``, as ``mlp_pair``'s."""
    from nablaframe.dictionaries import edges

    with ticking(label, write=_say):
        scores = edges.block_edge_scores(m.model, params, tokens, up, down)
        curve = edges.block_curve(m.model, params, tokens, up, down, scores)
    return _reading(curve, label)


def _reading(curve: dict[str, Any], label: str) -> dict[str, Any]:
    from nablaframe.dictionaries import edges

    reading = edges.meter(curve["curve"], curve["pair_kl"])
    scored = int(curve["scored_edges"])
    share = None if reading is None else float(reading) / scored
    _say(f"{label}: meter {reading if reading is not None else 'never settles'} of {scored:,} scored edges,"
         f" spliced whole KL {curve['pair_kl']:.4g}")
    return {"meter": None if reading is None else int(reading), "share": share, "scored_edges": scored,
            "pair_kl": float(curve["pair_kl"]), "curve": [[int(n), float(kl)] for n, kl in curve["curve"]]}


def f1_tables(dictionary: Any, site: int, acts: dict[str, np.ndarray], positions: Sequence[int] | None,
              values: dict[str, np.ndarray], n_values: int) -> dict[str, np.ndarray]:
    """``f1_table`` of the site's codes at ``positions`` for each variable in ``values``, in its order."""
    from nablaframe.dictionaries import measure

    rows_ = codes(dictionary, site, acts, positions)
    return {name: measure.f1_table(rows_, value, n_values)[0] for name, value in values.items()}


def readability(dictionary: Any, site: int, acts: dict[str, np.ndarray], positions: Sequence[int] | None,
                values: dict[str, np.ndarray], n_values: int, tables: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per variable, ``{"mean_best_f1", "probe_activation", "probe_code"}``: the code's mean best F1, and how well a
    ridge probe reads the variable from the site's input activation and from its code."""
    from nablaframe.dictionaries import measure

    x = rows(acts[dictionary.sites[site][0]], positions)
    rows_ = codes(dictionary, site, acts, positions)
    return {name: {"mean_best_f1": measure.mean_best_f1(tables[name]),
                   "probe_activation": measure.probe_accuracy(x, value, n_values),
                   "probe_code": measure.probe_accuracy(rows_, value, n_values)}
            for name, value in values.items()}


def selectivity_entry(trained: np.ndarray, twin: np.ndarray) -> dict[str, float]:
    """``{"trained", "twin", "margin"}`` for one variable at one site: the mean best F1s and ``selective_margin``."""
    from nablaframe.dictionaries import measure

    a, b = measure.mean_best_f1(trained), measure.mean_best_f1(twin)
    return {"trained": a, "twin": b, "margin": selective_margin(a, b)}


def number(value: Any) -> bool:
    """A plain int or float, not a bool or None: what a bar can be held to."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
