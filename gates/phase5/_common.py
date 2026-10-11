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

from __future__ import annotations

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


def settings(name: str, kind: str, d_model: int, *, seed: int = DICT_SEED, lam: float = 0.0, short: bool = False) -> Any:
    raise NotImplementedError


def sites(kind: str, n_layers: int, layer: int | None = None) -> list[tuple[str, str]]:
    raise NotImplementedError


def zoo_windows(task: Any, name: str, which: str, *, dict_seed: int = DICT_SEED) -> np.ndarray:
    raise NotImplementedError


def reference_windows(train: np.ndarray, validation: np.ndarray, seq: int, which: str, *,
                      dict_seed: int = DICT_SEED) -> np.ndarray:
    raise NotImplementedError


def variables(task: Any, name: str, windows: np.ndarray) -> tuple[dict[Any, dict[str, np.ndarray]], int]:
    raise NotImplementedError


def induction_sources(task: Any, windows: np.ndarray, j: int, test: str) -> np.ndarray:
    raise NotImplementedError


def dictionary_key(state_sha256: str, sites: Sequence[Sequence[str]], settings: Any, tokens: np.ndarray) -> str:
    raise NotImplementedError


def twin_state(params: Any) -> str:
    raise NotImplementedError


def train_dictionary(model: Any, params: Any, tokens: np.ndarray, settings: Any, sites: Sequence[Sequence[str]], *,
                     key: str, chunk: int, label: str, mlp_weights: Any = None) -> Any:
    raise NotImplementedError


def cached(cache_dir: str | Path, key: str, make: Callable[[], Any]) -> tuple[Any, float | None]:
    raise NotImplementedError


@contextmanager
def ticking(label: str, interval_s: float = 30.0, *, write: Callable[[str], Any] = print) -> Iterator[None]:
    raise NotImplementedError
    yield


def loss_recovered(clean: float, spliced: float, ablated: float) -> dict[str, float | None]:
    raise NotImplementedError


def faithful(residual: float, code: float) -> dict[str, Any]:
    raise NotImplementedError


def selective_margin(trained: float, twin: float) -> float:
    raise NotImplementedError


def edge_ratio(trained: int | None, twin: int | None, twin_scored: int) -> float | None:
    raise NotImplementedError
