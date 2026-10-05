"""Phase 2 delta 3: a Hugging Face GPT-2 export of our parameters, for the cross-check (D17).

Our layout is GPT-2's (c_attn, c_proj, c_fc, laid out (in, out)), so the export is a renaming.
Its logits are checked against the trace in the torch lane, by G2.8. Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from safetensors.numpy import load_file  # noqa: E402

from nablaframe.data.tokenizer import Tokenizer  # noqa: E402
from nablaframe.model.gpt2 import init_params  # noqa: E402
from nablaframe.model.hf_export import export_hf  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

OURS = Tokenizer(vocab=("<unk>", "<eos>", "the", "cat", "sat", ".", "lily's", ",", "3"))
PER_BLOCK = ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
             "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
             "mlp.c_proj.weight", "mlp.c_proj.bias")


def _export(tmp_path, name="hf"):
    params = _perturbed(init_params(TINY, seed=0))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(OURS.to_json())
    return params, tokenizer, export_hf(params, TINY, tokenizer, tmp_path / name, eos_token="<eos>", eos_id=1,
                                        unk_token="<unk>")


def test_export_names_shapes_values(tmp_path):
    params, tokenizer, out = _export(tmp_path)
    tensors = load_file(out / "model.safetensors")
    expected = {"transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight", "transformer.ln_f.bias"}
    expected |= {f"transformer.h.{i}.{name}" for i in range(TINY.n_layers) for name in PER_BLOCK}
    assert set(tensors) == expected, "tied to wte, so no lm_head.weight is written"
    assert all(t.dtype == np.float32 for t in tensors.values())
    assert np.array_equal(tensors["transformer.wte.weight"], np.asarray(params.wte))
    assert np.array_equal(tensors["transformer.wpe.weight"], np.asarray(params.wpe))
    block = params.blocks[1]
    assert np.array_equal(tensors["transformer.h.1.attn.c_attn.weight"], np.asarray(block.attn.c_attn.weight))
    assert np.array_equal(tensors["transformer.h.1.attn.c_proj.bias"], np.asarray(block.attn.c_proj.bias))
    assert np.array_equal(tensors["transformer.h.1.mlp.c_fc.weight"], np.asarray(block.mlp.c_fc.weight))
    assert np.array_equal(tensors["transformer.h.1.ln_2.bias"], np.asarray(block.ln_2.bias))
    assert np.array_equal(tensors["transformer.ln_f.weight"], np.asarray(params.ln_f.weight))

    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "gpt2" and config["architectures"] == ["GPT2LMHeadModel"]
    assert (config["vocab_size"], config["n_positions"], config["n_embd"], config["n_layer"], config["n_head"],
            config["n_inner"]) == (TINY.vocab, TINY.seq, TINY.d_model, TINY.n_layers, TINY.n_heads, TINY.d_mlp)
    assert config["activation_function"] == "gelu_new", "the tanh approximation of GELU, as our model"
    assert config["layer_norm_epsilon"] == 1e-5 and config["tie_word_embeddings"] is True
    assert config["bos_token_id"] == 1 and config["eos_token_id"] == 1, "inside the vocabulary, unlike GPT-2's 50256"
    assert all(config[k] == 0.0 for k in ("resid_pdrop", "embd_pdrop", "attn_pdrop"))

    from tokenizers import Tokenizer as HFTokenizer
    hf = HFTokenizer.from_file(str(out / "tokenizer.json"))  # a Hugging Face file now, not a copy of ours
    assert hf.token_to_id("<unk>") == 0 and hf.token_to_id("<eos>") == 1
    ids = [2, 3, 4, 5, 1, 6, 7, 0, 8, 5]  # with <eos> and <unk> mid-sequence, as in the probe windows
    assert hf.encode(OURS.decode(ids), add_special_tokens=False).ids == ids, "our decoded text round-trips exactly"
    assert hf.encode("THE dog sat.", add_special_tokens=False).ids == [2, 0, 4, 5], "lowercased; unknown words are <unk>"
    tokenizer_config = json.loads((out / "tokenizer_config.json").read_text())
    assert tokenizer_config == {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": TINY.seq,
                                "eos_token": "<eos>", "unk_token": "<unk>"}


def test_export_is_deterministic(tmp_path):
    _, _, first = _export(tmp_path, "a")
    _, _, second = _export(tmp_path, "b")
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir()) == ["config.json", "model.safetensors", "tokenizer.json",
                                                                 "tokenizer_config.json"]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
