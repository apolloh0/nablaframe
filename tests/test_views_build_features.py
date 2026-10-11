"""Phase 5 delta 5: features payloads in the view build, and the D14 badges' new last sentence (D39, D45).

Contract: src/nablaframe/views/build.py, ``build_views``' ``features``; docs/deltas/phase-5-delta-5.md.
Needs the jax lane.
"""

from __future__ import annotations

import json

import pytest

jax = pytest.importorskip("jax")

from nablaframe.trace.schema import canonical_json  # noqa: E402
from nablaframe.views.build import BADGES, build_views  # noqa: E402
from nablaframe.views.jlens_view import JLENS_BADGE  # noqa: E402
from nablaframe.views.readouts import TraceReader  # noqa: E402
from phase2_fixtures import TINY, traced_run  # noqa: E402

D38_SENTENCE = "Its model-level result is in the badge below (D38)."
HASH = "ab" * 32


def _feature(site, *, dictionary=HASH, step=0, **extra):
    return {"step": step, "dictionary": dictionary, "kind": "staircase", "site": site,
            "taps": [f"blocks.{site}.hook_resid_post"] * 2, "width": 12 * (site + 1), "k": 4,
            "frequency": [0.5, 0.25], "meter": {"kind": "reconstruction", "fvu": 0.04, "positions": 16}, **extra}


def _build(run, out, **kw):
    return build_views(TraceReader(run.root), out, params_for=lambda step: run.params, probe=run.probe,
                       n_layers=TINY.n_layers, run_id=run.manifest.run_id, vocab=[f"t{i}" for i in range(TINY.vocab)],
                       **kw)


def test_d14_badges_end_with_d39s_sentence():
    assert BADGES["logit_lens"] == ("Discrimination tests: recovers planted structure at decoder level, and is empty "
                                    "on random weights, checked by G2.3. " + D38_SENTENCE)
    assert BADGES["attribution"] == ("Discrimination tests: recovers planted structure at decoder level, and is an "
                                     "exact decomposition whose error is its meter, checked by G2.4. " + D38_SENTENCE)
    assert JLENS_BADGE == ("Discrimination tests: recovers planted structure at decoder level (a J that carries "
                           "planted directions to tokens), and is empty on random weights, checked by G2.10. Its "
                           "definition matches anthropics/jacobian-lens to 1.1e-7 (G2.8). " + D38_SENTENCE)


def test_features_are_written_and_indexed(tmp_path):
    run = traced_run(tmp_path)
    plain = _build(run, tmp_path / "plain")
    other = _feature(0, dictionary="cd" * 32)
    features = [_feature(1), _feature(0), other]
    paths = _build(run, tmp_path / "views", features=features)
    index = json.loads((tmp_path / "views" / "index.json").read_text())
    assert paths[-1].name == "index.json", "the index is still written last"
    names = [f"features-{HASH}-site-1.json", f"features-{HASH}-site-0.json", f"features-{'cd' * 32}-site-0.json"]
    assert index["features"] == [{"dictionary": p["dictionary"], "kind": "staircase", "site": p["site"], "taps": p["taps"],
                                  "step": 0, "file": name} for p, name in zip(features, names)], "in the given order"
    for payload, name in zip(features, names):
        assert (tmp_path / "views" / name).read_text() == canonical_json(payload), "written with export_view"
    assert {p.name for p in paths} == {p.name for p in plain} | set(names)
    plain_index = json.loads((tmp_path / "plain" / "index.json").read_text())
    assert {k: v for k, v in index.items() if k != "features"} == plain_index and "features" not in plain_index
    for path in plain[:-1]:
        assert (tmp_path / "views" / path.name).read_bytes() == path.read_bytes(), f"{path.name} is unchanged"
    _build(run, tmp_path / "none", features=[])
    assert json.loads((tmp_path / "none" / "index.json").read_text())["features"] == []


def test_bad_features_are_refused_before_anything_is_written(tmp_path):
    run = traced_run(tmp_path)
    cases = {
        "the same dictionary and site twice": [_feature(0), _feature(0, frequency=[1.0])],
        "a payload without a meter": [{k: v for k, v in _feature(0).items() if k != "meter"}],
        "a payload without its taps": [{k: v for k, v in _feature(0).items() if k != "taps"}],
        "a payload without its dictionary": [{k: v for k, v in _feature(0).items() if k != "dictionary"}],
        "a step the trace doesn't have": [_feature(0, step=5)],
        "a payload that isn't a dict": ["features"],
    }
    for i, (label, features) in enumerate(cases.items()):
        out = tmp_path / f"out{i}"
        with pytest.raises(ValueError):
            _build(run, out, features=features)
            pytest.fail(label)
        assert not out.exists() or not any(out.iterdir()), f"{label}: nothing written"
