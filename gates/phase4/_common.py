"""Shared by the Phase 4 gates. Read with docs/specs/phase-4.md, "The zoo", "Discrimination tests at model level" and "Gates", and D36.

May load ``gates/phase1/_common.py``, ``gates/phase2/_common.py`` and ``gates/phase3/_common.py``
by path, but never changes them. Reports go to ``reports/phase4/``, working files to
``runs/phase4/``: each zoo model's run under ``runs/phase4/<name>/``. Every gate runs on the Mac
only, refuses a dirty tree unless ``--allow-dirty``, refuses to overwrite its report unless
``--overwrite``, prints a progress line at least every 30 seconds during anything longer than a
minute, writes one schema-2 report, and runs every intervened pass through ``run_intervened``.
G4.2 to G4.5 refuse to start unless G4.1's report exists, passes ``validate_report`` and says
``pass``: no tool is judged on a model that missed its behaviour bars.

The settings are the feasibility runs' (D37), so G4.1 retraces them on the same Mac.

* ``MODELS`` and ``TRAINING``: each zoo model's ``GPT2Config`` and ``TrainConfig`` fields, as
  plain dicts keyed ``"induction"`` and ``"two_hop"``. Induction: vocab 64, d_model 64, 2
  blocks, 4 heads, d_mlp 256, seq 63; batch 32, lr 1e-3, warmup 100, 4,000 steps. Two-hop:
  vocab 51, d_model 128, 4 blocks, 4 heads, d_mlp 512, seq 26; batch 64, lr 1e-3, warmup 500,
  20,000 steps. Both train with data and parameter seed 0, and every other ``TrainConfig``
  field at its default. ``configs(name)`` builds the pair, importing JAX inside; any other
  name raises ValueError.
* Seeds and sizes: ``FIT_SEED = 1``, ``EVAL_SEED = 2``, ``BROKEN_SEED = 3``,
  ``PATCH_SEED = 4``, ``SOURCE_SEED = 5``; ``EVAL_WINDOWS = 256``, ``PROBE_WINDOWS = 8``,
  ``FIT_WINDOWS = 64``.
* ``eval_windows(task)``: ``task.batch_for_step(EVAL_SEED, 0, EVAL_WINDOWS)``.
  ``probe(task)``: ``task.batch_for_step(EVAL_SEED, 0, PROBE_WINDOWS)``, the first eight
  evaluation windows, which every checkpoint's trace records. ``fit_windows(task)``: the
  first ``seq`` tokens of ``task.batch_for_step(FIT_SEED, 0, FIT_WINDOWS)``.
  ``broken_windows(task)``: ``EVAL_WINDOWS`` windows ``task.window(rng, broken=True)``
  drawn in sequence from ``numpy.random.default_rng([BROKEN_SEED, 0])``.
  ``induction_patch_windows(task)``: ``EVAL_WINDOWS`` windows ``task.draw_window(rng,
  period=PATCH_PERIOD)`` drawn in sequence from ``numpy.random.default_rng([PATCH_SEED, 0])``.
* ``PATCH_PERIOD = 20``, ``PATCH_TARGETS = (13, 16, 19)``, ``PATCH_LAG = 19``: G4.2's patch
  puts a new token at copy target ``j`` and reads the prediction at ``j + PATCH_LAG``.
* ``TWO_HOP_SITES``: G4.3's patch sites, ``{"query": (24, 25), "fact positions": 0 to 23,
  "every position": 0 to 25}``, as tuples.
* ``BARS``: the spec's bars, as ``{"behaviour": 0.9, "control_ratio": 2.0,
  "induction_attention": 0.5, "twin_attention_ratio": 2.0, "ablation_fraction": 0.5,
  "swap_fraction": 0.5, "patch_followed": 0.9, "patch_after_read": 0.1,
  "control_unchanged": 0.95}``.
* ``checkpoint_steps(name)``: ``"induction"``: every 50 steps from 0 through 1,500, then every
  250 through 4,000, 41 in all; ``"two_hop"``: every 1,000 from 0 through 20,000. Any other name
  raises ValueError.
* ``MIN_ABOVE_CHANCE = 0.1``. ``fraction_removed(before, after, chance)``: how much of the
  accuracy above chance an edit removed, ``(before - after) / (before - chance)``, as a plain
  float. It can exceed 1 when ``after`` falls below chance. ValueError when ``before - chance <
  MIN_ABOVE_CHANCE``, since a fraction of less is noise; the gate records null.
* ``last_at_chance(steps, accuracies, chance)``: the last step whose accuracy is at most twice
  ``chance``, G4.2's swap checkpoint (D36). ValueError if no step qualifies, or if the two
  sequences differ in length.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

MODELS = {
    "induction": {"vocab": 64, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_mlp": 256, "seq": 63},
    "two_hop": {"vocab": 51, "d_model": 128, "n_layers": 4, "n_heads": 4, "d_mlp": 512, "seq": 26},
}
TRAINING = {
    "induction": {"batch": 32, "seq": 63, "lr": 1e-3, "warmup_steps": 100, "total_steps": 4000, "data_seed": 0, "param_seed": 0},
    "two_hop": {"batch": 64, "seq": 26, "lr": 1e-3, "warmup_steps": 500, "total_steps": 20000, "data_seed": 0, "param_seed": 0},
}
FIT_SEED, EVAL_SEED, BROKEN_SEED, PATCH_SEED, SOURCE_SEED = 1, 2, 3, 4, 5
EVAL_WINDOWS, PROBE_WINDOWS, FIT_WINDOWS = 256, 8, 64
PATCH_PERIOD, PATCH_TARGETS, PATCH_LAG = 20, (13, 16, 19), 19
TWO_HOP_SITES = {"query": (24, 25), "fact positions": tuple(range(24)), "every position": tuple(range(26))}
BARS = {"behaviour": 0.9, "control_ratio": 2.0, "induction_attention": 0.5, "twin_attention_ratio": 2.0,
        "ablation_fraction": 0.5, "swap_fraction": 0.5, "patch_followed": 0.9, "patch_after_read": 0.1,
        "control_unchanged": 0.95}
MIN_ABOVE_CHANCE = 0.1


def configs(name: str) -> tuple[Any, Any]:
    raise NotImplementedError


def eval_windows(task: Any) -> np.ndarray:
    raise NotImplementedError


def probe(task: Any) -> np.ndarray:
    raise NotImplementedError


def fit_windows(task: Any) -> np.ndarray:
    raise NotImplementedError


def broken_windows(task: Any) -> np.ndarray:
    raise NotImplementedError


def induction_patch_windows(task: Any) -> np.ndarray:
    raise NotImplementedError


def checkpoint_steps(name: str) -> list[int]:
    raise NotImplementedError


def fraction_removed(before: float, after: float, chance: float) -> float:
    raise NotImplementedError


def last_at_chance(steps: Sequence[int], accuracies: Sequence[float], chance: float) -> int:
    raise NotImplementedError
