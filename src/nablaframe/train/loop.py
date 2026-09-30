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
from functools import cache
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.corpus import batch_for_step
from ..model.gpt2 import GPT2Config, GPT2Model, init_params
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
    params = init_params(model_config, train_config.param_seed)
    return TrainState(params=params, opt_state=_optimizer(train_config).init(params), step=0)


def _optimizer(train_config: TrainConfig) -> optax.GradientTransformation:
    """Clipping by global norm, then AdamW on the warmup-cosine schedule.

    The schedule is optax's: the learning rate rises linearly from 0 at update 0 to
    ``lr`` at update ``warmup_steps``, then follows a cosine to ``lr * min_lr_ratio``
    at ``total_steps``. Levanter's default schedule is the same.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=train_config.lr, warmup_steps=train_config.warmup_steps,
        decay_steps=train_config.total_steps, end_value=train_config.lr * train_config.min_lr_ratio)
    b1, b2 = train_config.betas
    return optax.chain(
        optax.clip_by_global_norm(train_config.clip),
        optax.adamw(schedule, b1=b1, b2=b2, eps=train_config.eps, weight_decay=train_config.weight_decay,
                    mask=_matrices),
    )


def _matrices(params: Any) -> Any:
    return jax.tree.map(lambda leaf: jnp.ndim(leaf) >= 2, params)


def make_train_step(model_config: GPT2Config, train_config: TrainConfig) -> Callable[[TrainState, np.ndarray], tuple[TrainState, float]]:
    update = _compiled_update(model_config, train_config)

    def train_step(state: TrainState, batch: np.ndarray) -> tuple[TrainState, float]:
        expected = (train_config.batch, train_config.seq + 1)
        if np.shape(batch) != expected:
            raise ValueError(f"batch must be {expected} token ids, got {np.shape(batch)}")
        params, opt_state, loss = update(state.params, state.opt_state, jnp.asarray(batch, dtype=jnp.int32))
        # The step stays a Python int outside the compiled update, so it never
        # changes the compiled program and names checkpoints directly.
        return TrainState(params=params, opt_state=opt_state, step=int(state.step) + 1), float(loss)

    return train_step


@cache
def _compiled_update(model_config: GPT2Config, train_config: TrainConfig) -> Callable[..., Any]:
    """One compiled update per pair of configs, shared by every caller."""
    model = GPT2Model(model_config)
    optimizer = _optimizer(train_config)

    def loss_fn(params: Any, batch: jax.Array) -> jax.Array:
        logits = model.fast(params, batch[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:]).mean()

    def update(params: Any, opt_state: Any, batch: jax.Array) -> tuple[Any, Any, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return jax.jit(update)


def run_until(state: TrainState, stream: np.ndarray, model_config: GPT2Config, train_config: TrainConfig, *,
              until_step: int, checkpoint_dir: str | Path | None = None, save_every: int | None = None,
              heartbeat: Heartbeat | None = None, run_id: str | None = None) -> TrainState:
    """``run_id`` goes into each resume checkpoint's sidecar, and is null there when omitted."""
    from .checkpoint import save_checkpoint  # checkpoint.py imports TrainState from here

    if until_step < state.step:
        raise ValueError(f"state is already at step {state.step}, past until_step {until_step}")
    if save_every is not None and save_every < 1:
        raise ValueError(f"save_every must be at least 1, got {save_every}")
    saving = checkpoint_dir is not None and save_every is not None
    train_step = make_train_step(model_config, train_config)
    tokens = train_config.batch * train_config.seq
    while state.step < until_step:
        batch = batch_for_step(stream, train_config.data_seed, state.step, train_config.batch, train_config.seq)
        state, loss = train_step(state, batch)
        if saving and state.step % save_every == 0:
            save_checkpoint(state, checkpoint_dir, run_id=run_id)
        if heartbeat is not None:
            heartbeat.tick(step=state.step, loss=loss, tokens=tokens)
    return state
