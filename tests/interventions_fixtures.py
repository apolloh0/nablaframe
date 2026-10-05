"""Shared Phase 3 test helpers: a tiny model, two checkpoints, and a resolver. Not a test module."""

from __future__ import annotations

import numpy as np

from nablaframe.data.corpus import batch_hash
from nablaframe.model.gpt2 import GPT2Model, init_params
from nablaframe.trace.schema import tensor_sha256
from phase2_fixtures import TINY, _perturbed, probe_tokens

MODEL = GPT2Model(TINY)
PARAMS = _perturbed(init_params(TINY, seed=0))
OTHER = _perturbed(PARAMS)  # a second checkpoint, at step 4
TOKENS = probe_tokens(7)
SOURCE = probe_tokens(8)
REFERENCE = np.random.default_rng(9).integers(0, TINY.vocab, size=(6, TINY.seq)).astype(np.int32)
RESID = np.random.default_rng(10).standard_normal(TINY.d_model).astype(np.float32)
HEADS = np.random.default_rng(11).standard_normal((TINY.n_heads, TINY.d_model // TINY.n_heads)).astype(np.float32)
STORE = {batch_hash(SOURCE): SOURCE, batch_hash(TOKENS): TOKENS, batch_hash(REFERENCE): REFERENCE,
         tensor_sha256(RESID): RESID, tensor_sha256(HEADS): HEADS}
PARAMS_AT = {0: PARAMS, 4: OTHER}


def base_logits(params=PARAMS, tokens=TOKENS):
    return np.asarray(MODEL.instrumented(params, tokens, [])[0])
