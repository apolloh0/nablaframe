r"""A Hugging Face GPT-2 export of our parameters, for the J-lens cross-check (D17).

Our layout is GPT-2's: ``c_attn``, ``c_proj`` and ``c_fc`` are laid out (in, out), as
GPT-2's Conv1D weights are, so the export renames tensors and changes no values.

* ``export_hf(params, config, tokenizer_path, out_dir, *, eos_token, eos_id, unk_token)``
  writes four files to ``out_dir`` and returns it:

  - ``model.safetensors``: float32 tensors named ``transformer.wte.weight``,
    ``transformer.wpe.weight``, ``transformer.ln_f.{weight,bias}`` and, per block i,
    ``transformer.h.<i>.{ln_1,attn.c_attn,attn.c_proj,ln_2,mlp.c_fc,mlp.c_proj}.{weight,bias}``.
    The unembedding is tied to ``wte``, so no ``lm_head.weight`` is written.
  - ``config.json``: canonical JSON with ``model_type`` "gpt2", ``architectures``
    ["GPT2LMHeadModel"], ``vocab_size``, ``n_positions`` (``config.seq``), ``n_embd``,
    ``n_layer``, ``n_head``, ``n_inner`` (``config.d_mlp``), ``activation_function``
    "gelu_new", ``layer_norm_epsilon`` 1e-5, ``tie_word_embeddings`` true, ``bos_token_id``
    and ``eos_token_id`` both ``eos_id``, and ``resid_pdrop``, ``embd_pdrop``, ``attn_pdrop`` 0.0.
  - `tokenizer_path` is our tokenizer's file (`{"vocab": [...]}`). `tokenizer.json` is a Hugging
    Face tokenizer built from it. It has a WordLevel model over our vocabulary, with `unk_token` as
    its unknown token, and a normalizer of NFKD, accent stripping and lowercasing. Its pre-tokenizer
    splits on whitespace, then isolates our token pattern `[a-z]+(?:'[a-z]+)*|\S`. `unk_token` and
    `eos_token` are added as special tokens, matched before splitting. Encoding text our tokenizer
    decoded reproduces its ids exactly. For raw text, NFKD and accent stripping approximate our
    ASCII folding, which also rewrites some punctuation, so raw-text ids are not promised identical.
  - ``tokenizer_config.json``: canonical JSON, ``{"tokenizer_class":
    "PreTrainedTokenizerFast", "model_max_length": config.seq, "eos_token": eos_token,
    "unk_token": unk_token}``.

  Two exports of the same inputs are byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..trace.schema import canonical_json


def export_hf(params: Any, config: Any, tokenizer_path: str | Path, out_dir: str | Path, *, eos_token: str,
              eos_id: int, unk_token: str) -> Path:
    from safetensors.numpy import save_file

    if not isinstance(eos_id, int) or isinstance(eos_id, bool) or not 0 <= eos_id < config.vocab:
        raise ValueError(f"eos_id must be a token id in [0, {config.vocab}), got {eos_id!r}")
    hf_tokenizer = _hf_tokenizer(Path(tokenizer_path), eos_token=eos_token, eos_id=eos_id, unk_token=unk_token)

    d, m = config.d_model, config.d_mlp
    tensors = {"transformer.wte.weight": _tensor(params.wte, (config.vocab, d), "wte"),
               "transformer.wpe.weight": _tensor(params.wpe, (config.seq, d), "wpe"),
               "transformer.ln_f.weight": _tensor(params.ln_f.weight, (d,), "ln_f.weight"),
               "transformer.ln_f.bias": _tensor(params.ln_f.bias, (d,), "ln_f.bias")}
    if len(params.blocks) != config.n_layers:
        raise ValueError(f"params have {len(params.blocks)} blocks, config has {config.n_layers} layers")
    for i, block in enumerate(params.blocks):
        h = f"h.{i}."
        for name, value, shape in (
            ("ln_1.weight", block.ln_1.weight, (d,)), ("ln_1.bias", block.ln_1.bias, (d,)),
            ("attn.c_attn.weight", block.attn.c_attn.weight, (d, 3 * d)),
            ("attn.c_attn.bias", block.attn.c_attn.bias, (3 * d,)),
            ("attn.c_proj.weight", block.attn.c_proj.weight, (d, d)),
            ("attn.c_proj.bias", block.attn.c_proj.bias, (d,)),
            ("ln_2.weight", block.ln_2.weight, (d,)), ("ln_2.bias", block.ln_2.bias, (d,)),
            ("mlp.c_fc.weight", block.mlp.c_fc.weight, (d, m)), ("mlp.c_fc.bias", block.mlp.c_fc.bias, (m,)),
            ("mlp.c_proj.weight", block.mlp.c_proj.weight, (m, d)),
            ("mlp.c_proj.bias", block.mlp.c_proj.bias, (d,)),
        ):
            tensors["transformer." + h + name] = _tensor(value, shape, h + name)

    hf_config = {
        "model_type": "gpt2", "architectures": ["GPT2LMHeadModel"],
        "vocab_size": config.vocab, "n_positions": config.seq, "n_embd": d, "n_layer": config.n_layers,
        "n_head": config.n_heads, "n_inner": m, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": True, "bos_token_id": eos_id, "eos_token_id": eos_id,
        "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0,
    }
    tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": config.seq,
                        "eos_token": eos_token, "unk_token": unk_token}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"))
    (out / "config.json").write_bytes(canonical_json(hf_config).encode("ascii"))
    (out / "tokenizer.json").write_bytes(canonical_json(hf_tokenizer).encode("ascii"))
    (out / "tokenizer_config.json").write_bytes(canonical_json(tokenizer_config).encode("ascii"))
    return out


# The pre-tokenizer's second step: data/tokenizer.py's ``_TOKEN``, each match its own piece.
_TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)*|\S"


def _hf_tokenizer(tokenizer_path: Path, *, eos_token: str, eos_id: int, unk_token: str) -> dict[str, Any]:
    """The Hugging Face ``tokenizers`` serialization (format "1.0") of our tokenizer's file.

    Written as plain JSON rather than through the ``tokenizers`` library, so the export needs no
    package beyond the base lane's and its bytes don't depend on that library's version. Raises
    ValueError if the vocabulary repeats a token, lacks ``unk_token`` or ``eos_token``, or puts
    ``eos_token`` at an id other than ``eos_id``.
    """
    vocab = json.loads(tokenizer_path.read_text(encoding="utf-8"))["vocab"]
    ids = {token: i for i, token in enumerate(vocab)}
    if len(ids) != len(vocab):
        raise ValueError(f"{tokenizer_path} repeats a token, so it has no WordLevel vocabulary")
    for name, token in (("unk_token", unk_token), ("eos_token", eos_token)):
        if token not in ids:
            raise ValueError(f"{name} {token!r} is not in {tokenizer_path}'s vocabulary")
    if ids[eos_token] != eos_id:
        raise ValueError(f"eos_token {eos_token!r} has id {ids[eos_token]} in {tokenizer_path}, but eos_id is {eos_id}")

    def special(token: str) -> dict[str, Any]:
        return {"id": ids[token], "content": token, "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": True}

    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": sorted((special(unk_token), special(eos_token)), key=lambda t: t["id"]),
        "normalizer": {"type": "Sequence",
                       "normalizers": [{"type": "NFKD"}, {"type": "StripAccents"}, {"type": "Lowercase"}]},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Split", "pattern": {"Regex": _TOKEN_PATTERN}, "behavior": "Isolated", "invert": False}]},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": ids, "unk_token": unk_token},
    }


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """``value`` as a C-contiguous float32 array, unchanged: any other dtype or shape raises ValueError."""
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"{name} is {array.dtype}{list(array.shape)}, expected float32{list(shape)}")
    return np.ascontiguousarray(array)
