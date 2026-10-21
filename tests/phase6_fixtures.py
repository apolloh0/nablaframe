"""Shared Phase 6 test helpers: toy models whose blocks are set by hand, dictionaries of token directions, and the
declarations that go with them. Not a test module.

Every toy model is a GPT-2 whose token embeddings are orthonormal directions with zero mean, so the final layer norm,
whose gain is near 1, keeps a direction's token on top. Its attention writes nothing, and so do its MLPs, except in the mapping model, whose
last MLP replaces each token's direction with ``G``'s image of it. Every position also holds a tenth of token 0's
direction, which no window contains and no dictionary encodes: with a token's direction removed, the output is 0.

The dictionaries are TopK SAEs at ``hook_resid_post`` with four latents per token direction ``t`` from 1 to 11,
``4 (t - 1) + j`` for ``j`` from 0 to 3, firing at codes near 0.95, 0.85, 0.75 and 0.65, a little higher for higher
tokens (``code(t, j)``), and decoding back to the direction together; a latent that fires everywhere at 0.5 (``SHARED_LATENT``); a latent that fires at 0.55 on tokens 9, 10 and 11
(``GROUP_LATENT``); and one that never fires (``DEAD_LATENT``). The last three decode to nothing. Every latent's bias
is -0.05 below its gain, so float noise in other directions never fires a latent.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nablaframe.dictionaries import core
from nablaframe.gestures import declare
from nablaframe.model.gpt2 import GPT2Config, GPT2Model, init_params

VOCAB, D_MODEL, D_MLP, SEQ = 12, 24, 16, 6
GAINS = (1.0, 0.9, 0.8, 0.7)
BIAS = -0.05


def gain(token: int, j: int) -> float:
    return GAINS[j] * (1 + 0.03 * token)


def code(token: int, j: int) -> float:
    """A token latent's code on its own token, the direction at unit length."""
    return gain(token, j) + BIAS
SHARED_LATENT, GROUP_LATENT, DEAD_LATENT = 44, 45, 46
WIDTH, K = 47, 6
GROUP = (9, 10, 11)
# The mapping model's g: two tokens to each odd token, no token to itself.
G = {1: 3, 2: 3, 3: 5, 4: 5, 5: 7, 6: 7, 7: 9, 8: 9, 9: 11, 10: 11, 11: 1}
PARTNER = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9}  # the other token with the same image


def latent(token: int, j: int = 0) -> int:
    return 4 * (token - 1) + j


def directions() -> np.ndarray:
    """Sixteen orthonormal directions in the residual stream, each with zero mean, as rows."""
    M = np.random.default_rng(0).standard_normal((D_MODEL, 16))
    M -= M.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(M)
    return Q.T


def config(n_layers: int) -> GPT2Config:
    return GPT2Config(vocab=VOCAB, d_model=D_MODEL, n_layers=n_layers, n_heads=2, d_mlp=D_MLP, seq=SEQ)


def params(n_layers: int = 2, *, mapping: bool = False):
    """The toy model's parameters: every block writes nothing, except the last MLP when ``mapping``."""
    import jax.numpy as jnp

    E = directions()
    base = init_params(config(n_layers), seed=0)
    zero = lambda a: jnp.zeros_like(a)  # noqa: E731
    blocks = []
    for l, b in enumerate(base.blocks):
        attn = type(b.attn)(c_attn=type(b.attn.c_attn)(zero(b.attn.c_attn.weight), zero(b.attn.c_attn.bias)),
                            c_proj=type(b.attn.c_proj)(zero(b.attn.c_proj.weight), zero(b.attn.c_proj.bias)))
        fc_w, fc_b = np.zeros((D_MODEL, D_MLP)), np.full(D_MLP, -10.0)
        proj_w = np.zeros((D_MLP, D_MODEL))
        if mapping and l == n_layers - 1:
            for u, gu in G.items():  # unit u fires, at about 9.5, only on token u's direction
                fc_w[:, u] = 4.0 * E[u]
                proj_w[u] = (E[gu] - E[u]) / 9.5
        mlp = type(b.mlp)(c_fc=type(b.mlp.c_fc)(jnp.asarray(fc_w, jnp.float32), jnp.asarray(fc_b, jnp.float32)),
                          c_proj=type(b.mlp.c_proj)(jnp.asarray(proj_w, jnp.float32), zero(b.mlp.c_proj.bias)))
        blocks.append(type(b)(ln_1=b.ln_1, attn=attn, ln_2=b.ln_2, mlp=mlp))
    wpe = np.tile(0.1 * E[0], (SEQ, 1))
    ln_f_gain = 1 + 0.1 * np.random.default_rng(1).standard_normal(D_MODEL)
    ln_f = type(base.ln_f)(jnp.asarray(ln_f_gain, jnp.float32), zero(base.ln_f.bias))
    return type(base)(wte=jnp.asarray(E[:VOCAB], jnp.float32), wpe=jnp.asarray(wpe, jnp.float32),
                      blocks=tuple(blocks), ln_f=ln_f)


def dictionary(block: int, *, decoder: str = "token", silent: tuple[int, ...] = ()) -> core.Dictionary:
    """A TopK SAE at ``blocks.<block>.hook_resid_post``. ``decoder``: ``"token"`` decodes each token's latents back to
    its direction, ``"negative"`` to minus it, ``"hollow"`` to nothing. The latents of the tokens in ``silent`` never
    fire, and with any token silent neither does the shared latent, which would otherwise join a silent token's set."""
    E = directions()
    W_enc, W_dec = np.zeros((1, D_MODEL, WIDTH)), np.zeros((1, WIDTH, D_MODEL))
    b_enc = np.zeros((1, WIDTH))
    sign = {"token": 1.0, "negative": -1.0, "hollow": 0.0}[decoder]
    for t in range(1, VOCAB):
        for j in range(4):
            W_enc[0, :, latent(t, j)] = gain(t, j) * E[t]
            b_enc[0, latent(t, j)] = -10.0 if t in silent else BIAS
            W_dec[0, latent(t, j)] = sign * E[t] / (4 * code(t, j))
    b_enc[0, SHARED_LATENT] = -1.0 if silent else 0.5
    W_enc[0, :, GROUP_LATENT] = 0.6 * sum(E[t] for t in GROUP)
    b_enc[0, GROUP_LATENT] = BIAS
    b_enc[0, DEAD_LATENT] = -1.0
    arrays = {"W_enc": W_enc, "W_dec": W_dec, "b_enc": b_enc, "b_pre": np.zeros((1, D_MODEL)),
              "b_dec": np.zeros((1, D_MODEL)), "scale_in": np.ones(1), "scale_out": np.ones(1)}
    tap = f"blocks.{block}.hook_resid_post"
    return core.Dictionary(core.Settings("topk", WIDTH, K, steps=1, batch=1), [(tap, tap)],
                           {n: v.astype(np.float32) for n, v in arrays.items()})


def staircase() -> core.Dictionary:
    """A Staircase SAE at blocks 0 and 1 whose two halves are each ``dictionary(0)``'s latents: at site 0, only the
    first half is active."""
    one = dictionary(0)
    a = one.arrays
    arrays = {"W_enc": np.concatenate([a["W_enc"]] * 2, axis=2), "W_dec": np.concatenate([a["W_dec"]] * 2, axis=1),
              "b_enc": np.tile(a["b_enc"], (2, 2)), "b_pre": np.zeros((2, D_MODEL)), "b_dec": np.zeros((2, D_MODEL)),
              "scale_in": np.ones(2), "scale_out": np.ones(2)}
    sites = [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(2)]
    return core.Dictionary(core.Settings("staircase", 2 * WIDTH, K, steps=1, batch=1), sites,
                           {n: np.asarray(v, np.float32) for n, v in arrays.items()})


def windows(n_cycles: int = 9, seed: int = 0) -> np.ndarray:
    """Windows of SEQ + 1 tokens. Over each 11 windows, every token from 1 to 11 sits at the first SEQ positions
    exactly SEQ times, at shuffled positions, six distinct tokens to a window; the last token is drawn at random."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(11 * n_cycles):
        first = [(i * SEQ + j) % 11 + 1 for j in range(SEQ)]
        rng.shuffle(first)
        rows.append(first + [int(rng.integers(1, VOCAB))])
    return np.array(rows, dtype=np.int64)


PAIRED = (4, 5)  # the positions the toy relations pair, so freezing runs at two positions, not six


def declaration(kind: str = "identity", *, n_layers: int = 2, predict_when=None) -> declare.Declaration:
    """A toy run's declaration.

    - ``"identity"``: ``token(p)`` moved to ``w`` at ``p`` predicts ``w`` read at ``p``. Relations, each pairing
      positions ``PAIRED``: ``"carry"``, ``token(p)`` after block 0 to ``token(p)`` after the last block at the
      same position; ``"next"``, ``token(p)`` after block 0 at ``p - 1`` to ``token(p-1)`` after the last block
      at ``p``.
    - ``"mapping"``: the variable ``"g"`` is ``G`` of the token at every position. ``token(p)`` moved to ``w``
      predicts ``G[w]``, and ``"g"`` moved to ``w`` predicts ``w``, each read at ``p``. Relation ``"map"``:
      ``token(p)`` after block 0 to ``"g"`` after the last block, same position, ``v`` to ``G[v]``. Two broken
      relations beside it: ``"wrong map"``, ``token(p)`` after block 0 to ``token(p)`` after the last block, maps
      ``v`` to ``PARTNER[v]``, whose downstream set is ``v``'s own, since both share an image; ``"never"`` maps
      nothing.
    - ``"silent"``: the identity's descriptions, and no prediction for any of them.
    - ``"elsewhere"``: the identity's, except that ``token(p)`` moved at ``p`` predicts that the output at ``p + 1``
      stays the token there, since nothing moves between positions; none at the last position.

    ``predict_when(description, window, position, w)``, if given, must also hold for a prediction.
    """
    last = n_layers - 1
    seq = SEQ
    variables = ("g",) if kind == "mapping" else ()
    known = declare.DESCRIPTIONS + variables
    when = predict_when or (lambda *a: True)

    def describe_variables(w: np.ndarray) -> dict[str, np.ndarray]:
        if kind != "mapping":
            return {}
        toks = np.asarray(w)[:, :seq]
        return {"g": np.vectorize(lambda t: G.get(int(t), -1))(toks).astype(np.int64)}

    def predict(description, window, position, w):
        window = declare.check_prediction_args(description, known, window, position, w, seq)
        if kind == "silent" or not when(description, window, position, w):
            return None
        if kind == "identity" and description == "token(p)":
            return (int(position), int(w))
        if kind == "elsewhere" and description == "token(p)":
            return (int(position) + 1, int(window[position + 1])) if position + 1 < seq else None
        if kind == "mapping" and description == "token(p)":
            return (int(position), G[int(w)]) if int(w) in G else None
        if kind == "mapping" and description == "g":
            return (int(position), int(w))
        return None

    def pairs(relation, window):
        declare.check_window(window, seq)
        if relation in ("carry", "map", "wrong map", "never"):
            return tuple((p, p) for p in PAIRED)
        if relation == "next":
            return tuple((p - 1, p) for p in PAIRED)
        raise ValueError(f"no relation {relation!r}")

    def value_map(relation, window, v):
        declare.check_window(window, seq)
        if relation in ("carry", "next"):
            return int(v)
        if relation == "wrong map":
            return PARTNER.get(int(v))
        if relation == "map":
            return G.get(int(v))
        if relation == "never":
            return None
        raise ValueError(f"no relation {relation!r}")

    up = declare.Place("token(p)", 0, "every")
    if kind == "mapping":
        relations = (declare.Relation("map", up, declare.Place("g", last, "every"), True),
                     declare.Relation("wrong map", up, declare.Place("token(p)", last, "every"), False),
                     declare.Relation("never", up, declare.Place("g", last, "every"), False))
    else:
        relations = (declare.Relation("carry", up, declare.Place("token(p)", last, "every"), False),
                     declare.Relation("next", up, declare.Place("token(p-1)", last, "every"), False))
    return declare.Declaration(run=f"toy {kind}", seq=seq, classes={"every": range(seq), "late": (3, 4, 5)},
                               variables=variables, planted=(), relations=relations,
                               describe_variables=describe_variables, predict=predict, pairs=pairs,
                               value_map=value_map)


def world(kind: str = "identity", *, n_layers: int = 2, decoder: str = "token", model=None, predict_when=None,
          silent: dict[int, tuple[int, ...]] | None = None) -> SimpleNamespace:
    """A toy model with a dictionary after every block, its declaration and its windows. ``silent`` maps a block to
    the tokens whose latents never fire in its dictionary."""
    silent = silent or {}
    return SimpleNamespace(model=model or GPT2Model(config(n_layers)), params=params(n_layers, mapping=kind == "mapping"),
                           n_layers=n_layers, declaration=declaration(kind, n_layers=n_layers,
                                                                      predict_when=predict_when),
                           dictionaries={l: dictionary(l, decoder=decoder, silent=silent.get(l, ()))
                                         for l in range(n_layers)},
                           windows=windows(), cache={})


def reading(w: SimpleNamespace, block: int):
    """``read(w, block)``, read once per world. Tests call it inside their bodies, so that on a stub each test fails
    on its own rather than its fixture erroring."""
    if ("reading", block) not in w.cache:
        w.cache["reading", block] = read(w, block)
    return w.cache["reading", block]


def read(w: SimpleNamespace, block: int, **kwargs):
    from nablaframe.gestures.sets import read_site

    return read_site(w.model, w.params, n_layers=w.n_layers, declaration=w.declaration,
                     dictionary=w.dictionaries[block], site=0, windows=w.windows, **kwargs)
