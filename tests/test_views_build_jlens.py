"""Phase 2 delta 3b: the builder carries the J-lens, and every lens reading gives the actual next token's rank.

Spec: docs/specs/phase-2.md, "The scrubber"; contract: docs/deltas/phase-2-delta-4.md. Needs the jax lane.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.trace.schema import canonical_json  # noqa: E402
from nablaframe.views.build import VIEWS, build_views, view_payloads  # noqa: E402
from nablaframe.views.jlens_view import jlens_payload  # noqa: E402
from nablaframe.views.lens import lens_readings  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402

J = {0: (np.eye(16) + 0.3 * np.random.default_rng(3).standard_normal((16, 16))).astype(np.float32)}


def _build(run, out, **kw):
    return build_views(TraceReader(run.root), out, params_for=lambda step: run.params, probe=run.probe,
                       n_layers=TINY.n_layers, run_id=run.manifest.run_id, **kw)


def test_build_includes_the_jlens_when_given_jacobians(tmp_path):
    run = traced_run(tmp_path)
    _build(run, tmp_path / "plain")
    assert json.loads((tmp_path / "plain" / "index.json").read_text())["views"] == list(VIEWS), "unchanged without J's"
    _build(run, tmp_path / "with", jacobians_for=lambda step: J)
    index = json.loads((tmp_path / "with" / "index.json").read_text())
    assert index["views"] == list(VIEWS) + ["jlens"]
    written = json.loads((tmp_path / "with" / index["files"]["jlens"]["0"]).read_text())
    expected = jlens_payload(TraceReader(run.root), 0, run.params, J, run.probe, TINY.n_layers)
    assert written == json.loads(canonical_json(expected))


def test_lens_payloads_give_the_target_rank(tmp_path):
    run = traced_run(tmp_path)
    reader = TraceReader(run.root)
    lens = view_payloads(reader, 0, run.params, run.probe, TINY.n_layers)["logit_lens"]
    readings = lens_readings(reader, 0, run.params, TINY.n_layers).astype(np.float64)
    p = np.exp(readings - readings.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    ids = np.arange(TINY.vocab)
    for layer in range(TINY.n_layers + 1):
        for b in range(2):
            assert lens["target_rank"][layer][b][-1] is None and lens["target_prob"][layer][b][-1] is None
            for t in range(TINY.seq - 1):
                target = run.probe[b, t + 1]
                row = p[layer, b, t]
                rank = int((row > row[target]).sum() + ((row == row[target]) & (ids < target)).sum())
                assert lens["target_rank"][layer][b][t] == rank, "0 is the top token; ties go to the lower id"
                assert math.isclose(lens["target_prob"][layer][b][t], row[target], rel_tol=1e-12)
