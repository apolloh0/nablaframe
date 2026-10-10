"""Phase 4 delta 4: the probe's last position scored against the window's final token, and timeline curves (D38).

Contract: docs/deltas/phase-4-delta-4.md, "src/nablaframe/views/". Needs the jax lane.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.views import attribution as attribution_module  # noqa: E402
from nablaframe.views import build as build_module  # noqa: E402
from nablaframe.views import jlens_view, readouts  # noqa: E402
from nablaframe.views.lens import lens_readings  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402

INDEX_KEYS = {"run_id", "steps", "views", "files", "probe", "n_layers", "vocab", "val_loss_curve"}
CURVES = [{"name": "mechanism positions' loss", "unit": "nats", "points": [{"step": 0, "value": 2.5}]},
          {"name": "best head's attention", "unit": "fraction", "points": [{"step": 0, "value": None}]}]


def _next(run):
    return (np.arange(run.probe.shape[0]) * 7 + 3).astype(np.int32) % TINY.vocab


def _log_probs(reader):
    logits = reader.tensor(0, "logits").astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def test_the_last_position_is_scored_against_the_next_token(tmp_path):
    run = traced_run(tmp_path)
    reader, nt = TraceReader(run.root), _next(run)
    plain = readouts.per_token_loss(reader, 0, run.probe)
    scored = readouts.per_token_loss(reader, 0, run.probe, next_token=nt)
    assert np.isnan(plain[:, -1]).all() and scored.dtype == np.float32
    assert np.array_equal(scored[:, :-1], plain[:, :-1]), "every other position is unchanged"
    logp = _log_probs(reader)
    rows = np.arange(len(nt))[:, None]
    targets = np.concatenate([run.probe[:, 1:], nt[:, None]], axis=1)
    expected = -logp[rows, np.arange(TINY.seq)[None, :], targets]
    assert np.allclose(scored, expected.astype(np.float32), rtol=0, atol=1e-6), "each position against its next token"

    readings = lens_readings(reader, 0, run.params, TINY.n_layers)
    ranks0, probs0 = jlens_view.target_ranks(readings, run.probe)
    ranks1, probs1 = jlens_view.target_ranks(readings, run.probe, next_token=nt)
    p = np.exp(readings.astype(np.float64) - readings.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    for layer in range(len(readings)):
        for seq in range(len(nt)):
            assert ranks0[layer][seq][-1] is None and ranks1[layer][seq][:-1] == ranks0[layer][seq][:-1]
            last = p[layer, seq, -1]
            want = int((last > last[nt[seq]]).sum() + ((last == last[nt[seq]]) & (np.arange(TINY.vocab) < nt[seq])).sum())
            assert ranks1[layer][seq][-1] == want and probs1[layer][seq][-1] == pytest.approx(last[nt[seq]], rel=1e-12)

    plain_parts, plain_error = attribution_module.attribution_from_trace(reader, 0, run.params, run.probe, TINY.n_layers)
    parts, error = attribution_module.attribution_from_trace(reader, 0, run.params, run.probe, TINY.n_layers, next_token=nt)
    total = sum(parts.values())
    target_logits = reader.tensor(0, "logits").astype(np.float64)[np.arange(len(nt)), -1, nt]
    all_logits = reader.tensor(0, "logits").astype(np.float64)[rows, np.arange(TINY.seq)[None, :], targets]
    assert np.isfinite(total).all() and error == pytest.approx(float(np.abs(total - all_logits).max()), rel=1e-9, abs=0), \
        "the decomposition error covers the last position too"
    assert error <= 1e-3
    for name in plain_parts:
        assert np.allclose(parts[name][:, :-1], plain_parts[name][:, :-1], rtol=0, atol=1e-12), name

    tied = np.zeros((1, 1, 2, 6), np.float32)
    tied[0, 0, 1] = [1, 3, 3, 0, 3, 0]
    ranks, _ = jlens_view.target_ranks(tied, np.zeros((1, 2), np.int32), next_token=np.array([2]))
    assert ranks[0][0][-1] == 1, "an equally probable token ranks above the target only with a lower id"

    for bad in (nt[:-1], nt[None], nt.astype(np.float32) + 0.5):
        with pytest.raises(ValueError):
            readouts.per_token_loss(reader, 0, run.probe, next_token=bad)


def test_build_views_with_next_tokens_and_curves(tmp_path):
    run = traced_run(tmp_path)
    reader, nt = TraceReader(run.root), _next(run)
    common = {"params_for": lambda step: run.params, "probe": run.probe, "n_layers": TINY.n_layers,
              "run_id": run.manifest.run_id}
    J = {0: np.eye(TINY.d_model, dtype=np.float32)}
    build_module.build_views(reader, tmp_path / "plain", **common, jacobians_for=lambda step: J)
    plain = json.loads((tmp_path / "plain" / "index.json").read_text())
    assert set(plain) == INDEX_KEYS, "without them the index is as before, so earlier runs' builds don't change"

    build_module.build_views(reader, tmp_path / "a", **common, next_token=nt, curves=CURVES, jacobians_for=lambda step: J)
    build_module.build_views(reader, tmp_path / "b", **common, next_token=nt, curves=CURVES, jacobians_for=lambda step: J)
    index = json.loads((tmp_path / "a" / "index.json").read_text())
    assert set(index) == INDEX_KEYS | {"next_token", "curves"}
    assert index["next_token"] == nt.tolist() and index["curves"] == CURVES
    for name in sorted(p.name for p in (tmp_path / "a").iterdir()):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name

    def payload(folder, view):
        files = json.loads((tmp_path / folder / "index.json").read_text())["files"][view]
        return json.loads((tmp_path / folder / files["0"]).read_text())

    expected = readouts.per_token_loss(reader, 0, run.probe, next_token=nt)
    loss = payload("a", "per_token_loss")["loss"]
    assert [row[-1] for row in loss] == pytest.approx(expected[:, -1].tolist())
    assert all(row[-1] is None for row in payload("plain", "per_token_loss")["loss"])
    assert all(row[-1] is not None for layer in payload("a", "logit_lens")["target_rank"] for row in layer)
    assert all(row[-1] is not None for layer in payload("a", "jlens")["target_rank"] for row in layer)
    assert all(row[-1] is not None for part in payload("a", "attribution")["contributions"] for row in part)
    assert all(row[-1] is None for part in payload("plain", "attribution")["contributions"] for row in part)

    for bad in ([{"name": "x", "points": [{"step": 0, "value": 1.0}]}],
                [{"name": "x", "unit": "nats", "points": [{"step": 0, "value": float("nan")}]}],
                [{"name": "x", "unit": "nats", "points": [{"step": 0.5, "value": 1.0}]}],
                [{"name": "", "unit": "nats", "points": []}]):
        with pytest.raises(ValueError):
            build_module.build_views(reader, tmp_path / "bad", **common, curves=bad)
    with pytest.raises(ValueError):
        build_module.build_views(reader, tmp_path / "bad", **common, next_token=nt[:-1])
