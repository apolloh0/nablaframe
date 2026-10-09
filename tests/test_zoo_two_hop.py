"""Phase 4 package 5: the two-hop patch's sources and outcomes, and lens ranks."""

from __future__ import annotations

import numpy as np

from nablaframe.zoo.tasks import TwoHopTask
from nablaframe.zoo.two_hop import QUERY_POSITIONS, patch_outcomes, source_for, token_ranks


def test_sources_carry_the_counterpart():
    t, rng = TwoHopTask(), np.random.default_rng(3)
    assert QUERY_POSITIONS == (24, 25)
    for w in t.batch_for_step(seed=0, step=0, batch=20):
        p = t.parse(w)
        counterpart = next(s for s in p.facts if s not in (p.a, p.b))
        src = t.parse(source_for(t, w, counterpart, rng))
        assert src.b == counterpart, "the source's intermediate is the target's counterpart subject"


def test_patch_outcomes():
    t, rng = TwoHopTask(), np.random.default_rng(4)
    targets = t.batch_for_step(seed=1, step=0, batch=3)
    counterparts = [next(s for s in t.parse(w).facts if s not in (t.parse(w).a, t.parse(w).b)) for w in targets]
    sources = np.stack([source_for(t, w, c, rng) for w, c in zip(targets, counterparts)])
    swapped = t.parse(targets[0]).facts[counterparts[0]]
    transplanted = t.parse(sources[1]).c
    neither = next(e for e in range(48) if e not in (t.parse(targets[2]).facts[counterparts[2]], t.parse(sources[2]).c))
    out = patch_outcomes(np.array([swapped, transplanted, neither]), targets, sources, counterparts)
    assert out["swap_rate"] >= 1 / 3 and out["transplant_rate"] >= 1 / 3
    assert out == patch_outcomes(np.array([swapped, transplanted, neither]), targets, sources, counterparts)
    exact = patch_outcomes(np.array([t.parse(w).facts[c] for w, c in zip(targets, counterparts)]), targets, sources, counterparts)
    assert exact["swap_rate"] == 1.0


def test_token_ranks():
    readings = np.array([[[0.1, 0.9, 0.5, 0.5, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]], np.float32)  # (L, B, V)
    assert token_ranks(readings, np.array([1])).tolist() == [[0], [1]]
    assert token_ranks(readings, np.array([3])).tolist() == [[2], [3]], "ties go to the lower id"
