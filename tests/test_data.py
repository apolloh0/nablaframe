"""Work package 2: tokenizer, corpus cleaning, batch order and probe set.

Spec: docs/specs/phase-1.md, "Data".
"""

from __future__ import annotations

import hashlib

import numpy as np

from nablaframe.data.corpus import batch_for_step, batch_hash, clean_stories, eval_windows, probe_batch
from nablaframe.data.tokenizer import EOS_ID, UNK_ID, Tokenizer, build_tokenizer, normalize, pre_tokenize

STORIES = ["Once upon a time, Lily saw a cat.", "The cat was happy. Lily smiled!", "Tom didn't see the dog."]


def test_tokenizer_build_is_deterministic():
    a = build_tokenizer(STORIES, size=16)
    b = build_tokenizer(list(reversed(STORIES)), size=16)
    assert a.to_json() == b.to_json() and a.sha256 == b.sha256
    assert Tokenizer.from_json(a.to_json()) == a
    assert a.vocab[:2] == ("<unk>", "<eos>") and len(a.vocab) == 16
    assert pre_tokenize(normalize("Lily didn\u2019t SEE Caf\u00e9 \u201c3\u201d.")) == [
        "lily", "didn't", "see", "cafe", '"', "3", '"', "."]
    assert pre_tokenize(normalize("It cost 42 coins")) == ["it", "cost", "4", "2", "coins"]


def test_vocabulary_ties_break_alphabetically():
    assert build_tokenizer(["b a c", "c a b"], size=5).vocab == ("<unk>", "<eos>", "a", "b", "c")
    assert build_tokenizer(["z z z y y x"], size=4).vocab == ("<unk>", "<eos>", "z", "y")


def test_unknown_words_map_to_unk():
    tok = build_tokenizer(["a b a"], size=4)
    ids = tok.encode("A zebra b")
    assert ids == [tok.id("a"), UNK_ID, tok.id("b"), EOS_ID]
    assert tok.decode(ids) == "a <unk> b <eos>"


def test_batch_depends_only_on_seed_and_step():
    stream = np.arange(1000, dtype=np.uint16)
    b5 = batch_for_step(stream, seed=3, step=5, batch=4, seq=8)
    batch_for_step(stream, seed=3, step=6, batch=4, seq=8)
    assert np.array_equal(b5, batch_for_step(stream, seed=3, step=5, batch=4, seq=8))
    assert b5.shape == (4, 9) and b5.dtype == np.int32
    assert not np.array_equal(b5, batch_for_step(stream, seed=3, step=6, batch=4, seq=8))
    assert not np.array_equal(b5, batch_for_step(stream, seed=4, step=5, batch=4, seq=8))
    for row in b5:
        assert np.array_equal(row, stream[row[0]: row[0] + 9]), "each row is one contiguous window"


def test_eval_slice_and_probe_never_overlap():
    validation = np.arange(5000, dtype=np.uint16)
    evaluation = eval_windows(validation, n_tokens=1024, seq=8)
    probe = probe_batch(validation, eval_tokens=1024, seed=0, n=8, seq=8)
    assert evaluation.shape[1] == 9 and evaluation.max() < 1024 <= probe.min()
    assert probe.shape == (8, 8) and probe.dtype == np.int32
    assert np.array_equal(probe, probe_batch(validation, eval_tokens=1024, seed=0, n=8, seq=8))
    assert batch_hash(probe) == hashlib.sha256(probe.astype("<i4").tobytes()).hexdigest()


def test_cleaning_drops_broken_stories():
    raw = ("A good story.<|endoftext|>\n Another one.  <|endoftext|><|endoftext|>"
           "Bad \u00e2\u20ac\u0153 text.<|endoftext|>Caf\u00e9 time.")
    stories, dropped = clean_stories(raw)
    assert stories == ["A good story.", "Another one.", "Cafe time."]
    assert dropped == 1
