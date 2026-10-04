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

from pathlib import Path
from typing import Any


def export_hf(params: Any, config: Any, tokenizer_path: str | Path, out_dir: str | Path, *, eos_token: str,
              eos_id: int, unk_token: str) -> Path:
    raise NotImplementedError
