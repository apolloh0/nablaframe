"""Phase 5 gate helpers: the dictionaries' settings, windows, variables, cache and bars, fixed before any gate runs (D40, D41, D42).

Contract: gates/phase5/_common.py. The settings are the feasibility runs', so the gates retrace them.
The tests that train need the jax lane.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from nablaframe.zoo.tasks import InductionTask, TwoHopTask


def _common():
    gates = REPO_ROOT / "gates" / "phase5"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase5_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_settings_and_bars_are_fixed():
    c = _common()
    assert (c.WIDTH_FACTOR, c.K, c.LR, c.DICT_SEED, c.TRAIN_SEED, c.EVAL_SEED) == (8, 16, 5e-4, 0, 6, 7)
    assert (c.JAC_BATCH, c.LAMBDAS, c.ROBUST_SEEDS) == (1024, (1.0, 4.0), (1, 2))
    assert c.STEPS == {"induction": 3000, "two_hop": 3000, "reference": 2000}
    assert c.BATCH == {"induction": 4096, "two_hop": 4096, "reference": 2048}
    assert c.TRAIN_WINDOWS == {"induction": 8192, "two_hop": 16384, "reference": 2048}
    assert c.EVAL_WINDOWS == {"induction": 1024, "two_hop": 2048, "reference": 64}
    assert c.METER_WINDOWS == {"induction": 8, "two_hop": 8, "reference": 2}
    assert c.CHUNK == {"induction": 256, "two_hop": 256, "reference": 32}
    assert c.SHORT == {"steps": 300, "batch": 1024}
    assert c.BARS == {"fvu": 0.1, "ablation_cost": 0.1, "recovered": 0.9, "faithful_applies": 0.5, "faithful_ratio": 0.9,
                      "break_excess": 0.05, "control_unchanged": 0.95, "selective_f1": 0.5, "selective_twin_factor": 2.0,
                      "edge_ratio": 0.2}
    assert c.READOUT == {"induction": tuple(range(32, 63)), "two_hop": (25,)}


def test_settings_and_sites():
    c = _common()
    s = c.settings("two_hop", "topk", 128)
    assert (s.kind, s.width, s.k, s.steps, s.batch, s.lr, s.seed, s.lam, s.jac_batch) == ("topk", 1024, 16, 3000, 4096, 5e-4, 0, 0.0, 1024)
    assert (s.warmup, s.decay_fraction, s.aux_weight, s.dead_after) == (100, 0.3, 1 / 32, 100), "every other field at its default"
    s = c.settings("reference", "jsae", 256, seed=2, lam=4.0)
    assert (s.width, s.steps, s.batch, s.seed, s.lam) == (2048, 2000, 2048, 2, 4.0)
    s = c.settings("induction", "staircase", 64, short=True)
    assert (s.width, s.steps, s.batch) == (512, 300, 1024)
    with pytest.raises(ValueError):
        c.settings("modular_addition", "topk", 64)
    assert c.sites("topk", 4, 2) == [("blocks.2.hook_resid_post",) * 2]
    assert c.sites("staircase", 3) == [(f"blocks.{l}.hook_resid_post",) * 2 for l in range(3)]
    assert c.sites("transcoder", 2, 1) == [("blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out")]
    assert c.sites("jsae", 2, 0) == [("blocks.0.ln2.hook_normalized",) * 2, ("blocks.0.hook_mlp_out",) * 2]
    for bad in (("topk", 2, None), ("topk", 2, 2), ("transcoder", 2, -1), ("staircase", 2, 0), ("pca", 2, 0), ("topk", 2, True)):
        with pytest.raises(ValueError):
            c.sites(*bad)


def test_windows():
    c = _common()
    task = TwoHopTask()
    train = c.zoo_windows(task, "two_hop", "train")
    assert train.dtype == np.int32 and train.shape == (16384, task.window_length)
    assert np.array_equal(train, task.batch_for_step(6, 0, 16384)), "training windows from seed 6"
    assert np.array_equal(c.zoo_windows(task, "two_hop", "train", dict_seed=2), task.batch_for_step(26, 0, 16384)), \
        "a robustness seed trains on windows of its own"
    induction = InductionTask()
    held = c.zoo_windows(induction, "induction", "held_out")
    assert held.shape == (1024, 64) and np.array_equal(held, induction.batch_for_step(7, 0, 1024))
    for bad in ((task, "two_hop", "test"), (task, "reference", "train")):
        with pytest.raises(ValueError):
            c.zoo_windows(*bad)
    from nablaframe.data.corpus import batch_for_step, eval_windows

    rng = np.random.default_rng(0)
    stream, validation = rng.integers(0, 100, 50_000), rng.integers(0, 100, 30_000)
    got = c.reference_windows(stream, validation, 16, "train")
    assert got.dtype == np.int32 and np.array_equal(got, batch_for_step(stream, 6, 0, 2048, 16))
    assert np.array_equal(c.reference_windows(stream, validation, 16, "train", dict_seed=1), batch_for_step(stream, 16, 0, 2048, 16))
    held = c.reference_windows(stream, validation, 16, "held_out")
    assert held.shape[0] == 64 and np.array_equal(held, eval_windows(validation, 262_144, 16)[:64]), "the run's evaluation slice"
    with pytest.raises(ValueError):
        c.reference_windows(stream, validation, 16, "test")


def test_variables():
    c = _common()
    task = TwoHopTask()
    windows = task.batch_for_step(7, 0, 50)
    by_position, n = c.variables(task, "two_hop", windows)
    assert n == 48 and sorted(by_position) == [24, 25]
    for position in (24, 25):
        assert sorted(by_position[position]) == ["a", "b", "c"]
        for v in ("a", "b", "c"):
            want = [getattr(task.parse(w), v) for w in windows]
            assert by_position[position][v].dtype == np.int64 and by_position[position][v].tolist() == want
    assert by_position[25]["c"].tolist() == windows[:, -1].tolist(), "the answer is the window's last token"
    induction = InductionTask()
    windows = induction.batch_for_step(7, 0, 3)
    by_position, n = c.variables(induction, "induction", windows)
    assert n == 64 and list(by_position) == ["mechanism"] and sorted(by_position["mechanism"]) == ["cur", "next", "prev"]
    values = by_position["mechanism"]
    assert len(values["cur"]) == 3 * 31, "every mechanism position of every window"
    assert values["cur"][:2].tolist() == windows[0, 32:34].tolist() and values["cur"][31] == windows[1, 32], "window-major"
    assert values["prev"][0] == windows[0, 31] and values["next"][0] == windows[0, 33] and values["next"][-1] == windows[2, 63]
    with pytest.raises(ValueError):
        c.variables(induction, "reference", windows)


def test_induction_sources():
    c = _common()
    from nablaframe.zoo.induction import new_token_sources

    task = InductionTask()
    windows = np.stack([task.draw_window(np.random.default_rng(i), period=20) for i in range(16)]).astype(np.int32)
    follow = c.induction_sources(task, windows, 13, "follow")
    assert np.array_equal(follow, new_token_sources(windows, 13, np.random.default_rng(13), 64)), "G4.2's sources"
    broken = c.induction_sources(task, windows, 16, "break")
    assert np.array_equal(broken, new_token_sources(windows, 15, np.random.default_rng(116), 64))
    assert (broken[:, 15] != windows[:, 15]).all() and np.array_equal(np.delete(broken, 15, 1), np.delete(windows, 15, 1)), \
        "a new token at j - 1, nothing else changed"
    with pytest.raises(ValueError):
        c.induction_sources(task, windows, 13, "swap")


def test_judgements():
    c = _common()
    got = c.loss_recovered(0.01, 0.05, 2.01)
    assert got["cost"] == pytest.approx(2.0) and got["recovered"] == pytest.approx(0.98)
    assert c.loss_recovered(1.0, 1.05, 1.1) == {"cost": pytest.approx(0.1), "recovered": pytest.approx(0.5)}, \
        "a cost of exactly the bar is judged"
    assert c.loss_recovered(1.0, 1.001, 1.012)["recovered"] is None, "an ablation costing less is recorded, not judged"
    assert c.faithful(0.86, 0.80) == {"applies": True, "ratio": pytest.approx(0.80 / 0.86)}
    assert c.faithful(0.5, 0.45) == {"applies": True, "ratio": pytest.approx(0.9)}, "half the time is enough to apply"
    assert c.faithful(0.49, 0.0) == {"applies": False, "ratio": None}
    assert c.selective_margin(0.8, 0.25) == pytest.approx(0.3) and c.selective_margin(0.5, 0.3) == pytest.approx(-0.1)
    assert c.edge_ratio(606, 58391, 58391) == pytest.approx(606 / 58391)
    assert c.edge_ratio(120, None, 3000) == pytest.approx(0.04), "a twin that never settles counts as every scored edge"
    assert c.edge_ratio(None, 500, 900) is None
    for value in (c.loss_recovered(0, 0.5, 1)["cost"], c.faithful(np.float32(0.9), np.float64(0.9))["ratio"],
                  c.selective_margin(np.float32(0.6), 0.1), c.edge_ratio(np.int64(5), np.int64(50), 60)):
        assert type(value) is float, "plain floats, ready for a report"
    assert type(c.faithful(np.float32(0.9), 0.9)["applies"]) is bool


def test_ticking():
    c = _common()
    import time

    lines = []
    with c.ticking("training", interval_s=0.05, write=lines.append):
        time.sleep(0.3)
    assert len(lines) >= 3 and all(line.startswith("training: still working") for line in lines)
    count = len(lines)
    time.sleep(0.2)
    assert len(lines) == count, "it stops on exit"
    with pytest.raises(RuntimeError):
        with c.ticking("failing", interval_s=0.05, write=lines.append):
            raise RuntimeError("boom")
    count = len(lines)
    time.sleep(0.2)
    assert len(lines) == count, "and on an exception"


def _tiny_dictionary(key, seed=0):
    from nablaframe.dictionaries import core

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((1, 8, 4))
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    arrays = {"W_enc": W.transpose(0, 2, 1), "W_dec": W, "b_enc": np.zeros((1, 8)), "b_pre": np.zeros((1, 4)),
              "b_dec": np.zeros((1, 4)), "scale_in": np.ones(1), "scale_out": np.ones(1)}
    return core.Dictionary(core.Settings("topk", 8, 2, steps=10, batch=4), [("blocks.0.hook_resid_post",) * 2],
                           {n: v.astype(np.float32) for n, v in arrays.items()}, key)


def test_cached(tmp_path):
    c = _common()
    from nablaframe.dictionaries import core

    key, made = "a" * 64, []

    def make():
        made.append(1)
        return _tiny_dictionary(key)

    first, seconds = c.cached(tmp_path, key, make)
    assert made == [1] and isinstance(seconds, float) and first.key == key
    files = sorted(p.name for p in (tmp_path / key).iterdir())
    assert files == [f"{core.content_hash(first)}.json", f"{core.content_hash(first)}.safetensors"]
    second, seconds = c.cached(tmp_path, key, make)
    assert made == [1] and seconds is None and core.content_hash(second) == core.content_hash(first), "trained once"
    with pytest.raises(ValueError):
        c.cached(tmp_path, "b" * 64, lambda: _tiny_dictionary("c" * 64))  # made under another key
    other = tmp_path / ("d" * 64)
    core.save_dictionary(_tiny_dictionary("e" * 64), other)
    with pytest.raises(ValueError):
        c.cached(tmp_path, "d" * 64, make)  # holds a dictionary under another key
    (tmp_path / key / "notes.txt").write_text("x")
    with pytest.raises(ValueError):
        c.cached(tmp_path, key, make)  # not just one dictionary's files


def test_keys_and_training():
    pytest.importorskip("jax")
    c = _common()
    from nablaframe.data.corpus import batch_hash
    from nablaframe.dictionaries import core, measure, train
    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.train.checkpoint import state_hash
    from phase2_fixtures import TINY, _perturbed

    params = _perturbed(init_params(TINY, seed=0))
    tokens = np.random.default_rng(1).integers(0, TINY.vocab, (24, TINY.seq)).astype(np.int32)
    s = core.Settings("topk", 32, 4, steps=20, batch=16, lr=1e-3)
    sites = c.sites("topk", 2, 1)
    key = c.dictionary_key("f" * 64, sites, s, tokens)
    assert key == core.dictionary_key(state_sha256="f" * 64, sites=sites, settings=s, batch_hashes=[batch_hash(tokens)],
                                      code=train.code_identity())
    assert c.twin_state(params) == state_hash(params)
    d = c.train_dictionary(GPT2Model(TINY), params, tokens, s, sites, key=key, chunk=5, label="tiny")
    acts = measure.activations(GPT2Model(TINY), params, tokens, ["blocks.1.hook_resid_post"], chunk=5)
    x = acts["blocks.1.hook_resid_post"].reshape(-1, TINY.d_model)
    want = train.train(s, sites, [(x, x)], key=key)
    assert d.key == key and core.content_hash(d) == core.content_hash(want), "trained on every position's activations"
    w = train.mlp_weights(params, 0)
    j = c.train_dictionary(GPT2Model(TINY), params, tokens, core.Settings("jsae", 32, 4, steps=5, batch=16, lam=1.0),
                           c.sites("jsae", 2, 0), key=key, chunk=24, label="tiny jsae", mlp_weights=w)
    assert j.settings.kind == "jsae" and j.d_in == j.d_out == TINY.d_model
    tc = c.train_dictionary(GPT2Model(TINY), params, tokens, core.Settings("transcoder", 32, 4, steps=5, batch=16),
                            c.sites("transcoder", 2, 1), key=key, chunk=24, label="tiny transcoder")
    acts = measure.activations(GPT2Model(TINY), params, tokens, ["blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out"])
    x, y = (acts[t].reshape(-1, TINY.d_model) for t in ("blocks.1.ln2.hook_normalized", "blocks.1.hook_mlp_out"))
    want = train.train(core.Settings("transcoder", 32, 4, steps=5, batch=16), c.sites("transcoder", 2, 1), [(x, y)], key=key)
    assert core.content_hash(tc) == core.content_hash(want), "a transcoder reads one tap and reconstructs the other"
