"""Phase 2 package 9: the view builder and the payload contract the viewer reads.

Spec: docs/specs/phase-2.md, "The scrubber" and D11. Needs the jax lane.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import nablaframe.trace.registry as registry  # noqa: E402
from nablaframe.trace.engine import TraceRun  # noqa: E402
from nablaframe.trace.registry import tap_names  # noqa: E402
from nablaframe.trace.store import TraceStore  # noqa: E402
from nablaframe.train.checkpoint import save_params  # noqa: E402
from nablaframe.views.attribution import attribution_from_trace  # noqa: E402
from nablaframe.views.build import BADGES, TOP_K, VIEWS, build_views, view_payloads  # noqa: E402
from nablaframe.views.lens import lens_meter, lens_readings  # noqa: E402
from nablaframe.views.readouts import TraceReader, per_token_loss  # noqa: E402
from phase2_fixtures import TINY, _perturbed, traced_run  # noqa: E402


def _two_step_run(tmp_path):
    """The fixture's step 0, plus a step 1 with different weights."""
    run = traced_run(tmp_path)
    params1 = _perturbed(run.params)
    save_params(params1, run.checkpoints, step=1, run_id=run.manifest.run_id)
    with TraceRun.open(run.root, run.manifest, run.model, checkpoint_dir=run.checkpoints, probe=run.probe,
                       like=run.params) as reopened:
        reopened.record(step=1, taps=tap_names(TINY.n_layers) + (registry.LOGITS,))
    run.params_for = {0: run.params, 1: params1}.__getitem__
    return run


def _nulled(array):
    return [[None if math.isnan(v) else v for v in row] for row in np.asarray(array, np.float64).tolist()]


def _build(run, out_dir, progress=None):
    return build_views(TraceReader(run.root), out_dir, params_for=run.params_for, probe=run.probe,
                       n_layers=TINY.n_layers, run_id=run.manifest.run_id, vocab=[f"t{i}" for i in range(TINY.vocab)],
                       val_loss_curve=[{"step": 0, "loss": 3.5}, {"step": 1, "loss": 3.4}], progress=progress)


def test_build_writes_every_view_with_a_meter(tmp_path):
    run = _two_step_run(tmp_path)
    paths = _build(run, tmp_path / "views")
    index = json.loads((tmp_path / "views" / "index.json").read_text())
    assert paths[-1].name == "index.json", "the index is written last"
    assert index["steps"] == [0, 1] and index["views"] == list(VIEWS) and index["run_id"] == run.manifest.run_id
    assert index["probe"] == run.probe.tolist() and index["n_layers"] == TINY.n_layers
    assert index["vocab"][3] == "t3" and index["val_loss_curve"][1] == {"step": 1, "loss": 3.4}
    for view in VIEWS:
        for step in (0, 1):
            payload = json.loads((tmp_path / "views" / index["files"][view][str(step)]).read_text())
            assert payload["step"] == step and "meter" in payload, (view, step)
    for view in ("logit_lens", "attribution"):
        payload = json.loads((tmp_path / "views" / index["files"][view]["0"]).read_text())
        assert payload["badge"] == BADGES[view], "each explanatory view states its discrimination level (D14)"


def test_build_payloads_match_the_views(tmp_path):
    run = _two_step_run(tmp_path)
    reader = TraceReader(run.root)
    payloads = view_payloads(reader, 1, run.params_for(1), run.probe, TINY.n_layers)
    assert set(payloads) == set(VIEWS)

    loss = payloads["per_token_loss"]
    assert loss["loss"] == _nulled(per_token_loss(reader, 1, run.probe))
    [logits_event] = [e for e in TraceStore(run.root, mode="r").current() if e.tap == "logits" and e.step == 1]
    assert loss["meter"] == {"kind": "hash_check", "tensor_sha256": logits_event.sha256}

    readings = lens_readings(reader, 1, run.params_for(1), TINY.n_layers)
    lens = payloads["logit_lens"]
    assert lens["meter"] == lens_meter(readings, reader.tensor(1, "logits"))
    assert np.asarray(lens["top_tokens"]).shape == (TINY.n_layers + 1, 2, TINY.seq, TOP_K)
    assert (np.asarray(lens["top_tokens"])[..., 0] == readings.argmax(-1)).all()
    probs = np.asarray(lens["top_probs"])
    assert (np.diff(probs, axis=-1) <= 0).all() and (probs > 0).all() and (probs.sum(-1) <= 1 + 1e-9).all()

    contributions, error = attribution_from_trace(reader, 1, run.params_for(1), run.probe, TINY.n_layers)
    attr = payloads["attribution"]
    assert attr["parts"] == list(contributions)
    assert attr["contributions"] == [_nulled(contributions[p]) for p in attr["parts"]]
    assert attr["meter"] == {"kind": "decomposition_error", "max_abs": error}

    layers = payloads["layer_structure"]
    assert layers["lens_kl"] == [m["kl_mean"] for m in lens["meter"]]
    pattern = reader.tensor(1, "blocks.1.attn.hook_pattern").astype(np.float64)
    entropy = -(pattern * np.log(np.where(pattern > 0, pattern, 1.0))).sum(-1).mean(axis=(0, 2))
    assert np.allclose(layers["attention_entropy"][1], entropy, rtol=0, atol=1e-12)


def test_build_is_deterministic(tmp_path):
    run = _two_step_run(tmp_path)
    first = _build(run, tmp_path / "a")
    second = _build(run, tmp_path / "b")
    assert [p.name for p in first] == [p.name for p in second]
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes(), a.name


def test_build_reports_progress(tmp_path):
    run = _two_step_run(tmp_path)
    calls = []
    _build(run, tmp_path / "views", progress=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 2), (2, 2)], "one call per step, so a gate can print a progress line every 30 s"
