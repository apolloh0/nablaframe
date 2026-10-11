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
