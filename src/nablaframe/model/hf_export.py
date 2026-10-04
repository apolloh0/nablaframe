"""A Hugging Face GPT-2 export of our parameters, for the J-lens cross-check (D17).

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
  - ``tokenizer.json``: a byte-for-byte copy of ``tokenizer_path``.
  - ``tokenizer_config.json``: canonical JSON, ``{"tokenizer_class":
    "PreTrainedTokenizerFast", "model_max_length": config.seq, "eos_token": eos_token,
    "unk_token": unk_token}``.

  Two exports of the same inputs are byte-identical.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..trace.schema import canonical_json


def export_hf(params: Any, config: Any, tokenizer_path: str | Path, out_dir: str | Path, *, eos_token: str,
              eos_id: int, unk_token: str) -> Path:
    from safetensors.numpy import save_file

    if not isinstance(eos_id, int) or isinstance(eos_id, bool) or not 0 <= eos_id < config.vocab:
        raise ValueError(f"eos_id must be a token id in [0, {config.vocab}), got {eos_id!r}")

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
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    (out / "tokenizer_config.json").write_bytes(canonical_json(tokenizer_config).encode("ascii"))
    return out


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """``value`` as a C-contiguous float32 array, unchanged: any other dtype or shape raises ValueError."""
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"{name} is {array.dtype}{list(array.shape)}, expected float32{list(shape)}")
    return np.ascontiguousarray(array)
