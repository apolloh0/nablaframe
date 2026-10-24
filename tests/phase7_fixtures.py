"""Shared Phase 7 test helpers: the zoo's three runs at small task settings, each with a small random model. Not a test module.

Each model's parameters are nudged hard, so its components move its prediction by nats, not by rounding: a
crystallization's KL and its ablation's are then numbers a test can tell apart.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params
from nablaframe.zoo import declarations, functions
from nablaframe.zoo.tasks import InductionTask, ModularAdditionTask, TwoHopTask

MODADD = ModularAdditionTask(p=11, train_fraction=0.5)
INDUCTION = InductionTask(vocab_size=12, window=12, min_period=3, max_period=5)
TWO_HOP = TwoHopTask(entities=10, facts=3)


def loud(params, seed: int):
    """Every parameter nudged by 0.5 standard normal noise."""
    import jax

    rng = np.random.default_rng(seed)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    return jax.tree_util.tree_unflatten(
        treedef, [leaf + 0.5 * rng.standard_normal(np.shape(leaf)).astype(np.float32) for leaf in leaves])


def config(run: str) -> GPT2Config:
    if run == "modular_addition":
        return GPT2Config(vocab=len(MODADD.vocab), d_model=16, n_layers=1, n_heads=2, d_mlp=32, seq=MODADD.seq)
    if run == "induction":
        # A block after block 1, so what block 1 writes at one position reaches the others.
        return GPT2Config(vocab=INDUCTION.vocab_size, d_model=16, n_layers=3, n_heads=2, d_mlp=32, seq=INDUCTION.seq)
    return GPT2Config(vocab=len(TWO_HOP.vocab), d_model=16, n_layers=4, n_heads=2, d_mlp=32, seq=TWO_HOP.seq)


def world(run: str) -> SimpleNamespace:
    """A run's task, model, trained and twin params, functions, declaration, and fitting and held-out windows."""
    cfg = config(run)
    if run == "modular_addition":
        task, fit, held = MODADD, MODADD.training(), MODADD.held_out()
        declaration = declarations.modular_addition(task)
    elif run == "induction":
        task = INDUCTION
        fit, held = task.batch_for_step(1, 0, 48), task.batch_for_step(2, 0, 24)
        declaration = declarations.induction(task)
    else:
        task = TWO_HOP
        fit, held = task.batch_for_step(1, 0, 96), task.batch_for_step(2, 0, 32)
        declaration = declarations.two_hop(task)
    return SimpleNamespace(run=run, task=task, config=cfg, model=GPT2Model(cfg), n_layers=cfg.n_layers,
                           params=loud(init_params(cfg, seed=0), 3), twin=loud(init_params(cfg, seed=1), 4),
                           functions=functions.for_run(run, task), declaration=declaration,
                           fit=np.asarray(fit, dtype=np.int32), held=np.asarray(held, dtype=np.int32))
