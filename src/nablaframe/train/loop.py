"""The training loop: AdamW on the reference model, deterministic by construction.

Spec: docs/specs/phase-1.md, "Trainer and checkpoints".

* ``init_state`` builds params with ``init_params(model_config, train_config.param_seed)``
  and a fresh optimizer state, at step 0.
* The optimizer is optax: gradient clipping by global norm ``clip``, then AdamW
  with ``betas``, ``eps`` and ``weight_decay`` applied to matrices only (leaves
  with two or more dimensions), on a schedule of linear warmup over
  ``warmup_steps`` then cosine decay to ``lr * min_lr_ratio`` at ``total_steps``.
  No dropout.
* The loss is mean next-token cross-entropy over every position of the batch.
* ``make_train_step`` returns a ``jit``-compiled function taking ``(state, batch)``,
  where ``batch`` is ``batch_for_step``'s ``(batch, seq + 1)`` int32 array. It
  returns the next state, with ``step`` one higher, and the loss as a float.
* ``run_until`` applies updates until ``state.step == until_step``. The update from
  step k uses ``batch_for_step(stream, data_seed, k, batch, seq)``. When
  ``checkpoint_dir`` and ``save_every`` are given, it saves a resume checkpoint
  whenever the step after an update is a multiple of ``save_every``. It calls
  ``heartbeat.tick`` after every update when a heartbeat is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

from ..model.gpt2 import GPT2Config
from .heartbeat import Heartbeat


@dataclass(frozen=True)
class TrainConfig:
    batch: int = 32
    seq: int = 256
    lr: float = 1e-3
    warmup_steps: int = 500
    total_steps: int = 12_208
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    clip: float = 1.0
    data_seed: int = 0
    param_seed: int = 0


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    step: int


def init_state(model_config: GPT2Config, train_config: TrainConfig) -> TrainState:
    raise NotImplementedError


def make_train_step(model_config: GPT2Config, train_config: TrainConfig) -> Callable[[TrainState, np.ndarray], tuple[TrainState, float]]:
    raise NotImplementedError


def run_until(state: TrainState, stream: np.ndarray, model_config: GPT2Config, train_config: TrainConfig, *,
              until_step: int, checkpoint_dir: str | Path | None = None, save_every: int | None = None,
              heartbeat: Heartbeat | None = None) -> TrainState:
    raise NotImplementedError
