"""Phase 4 delta 3: G4.3's aligned sources and their distractor control (D36). Needs no JAX.

Contract: docs/deltas/phase-4-delta-3.md, "src/nablaframe/zoo/two_hop.py".
"""

from __future__ import annotations

import numpy as np
import pytest

from nablaframe.zoo import two_hop  # looked up per test, so a missing function fails its test alone
from nablaframe.zoo.tasks import TwoHopTask


def _object_slot(window: np.ndarray, subject: int) -> int:
    return int(np.flatnonzero(window[0:24:4] == subject)[0]) * 4 + 2


def test_aligned_sources_differ_only_where_the_chain_moves():
    t = TwoHopTask()
    targets = t.batch_for_step(seed=2, step=0, batch=40)
    sources, controls, counterparts = two_hop.aligned_sources(t, targets, np.random.default_rng(5))
    assert sources.dtype == controls.dtype == np.int32 and sources.shape == controls.shape == targets.shape
    assert counterparts.shape == (40,)
    mirror = np.random.default_rng(5)
    for w, s, c, k in zip(targets, sources, controls, counterparts):
        p = t.parse(w)
        assert k == next(x for x in w[0:24:4].tolist() if x not in (p.a, p.b)), "the first distractor subject"
        fresh = int(mirror.choice(np.setdiff1d(np.arange(48), w)))
        assert int(s[-1]) == fresh and fresh not in w.tolist(), "d′ is drawn in window order from entities the window doesn't use"
        assert sorted(np.flatnonzero(s != w).tolist()) == sorted([_object_slot(w, p.a), _object_slot(w, k), 26])
        q = t.parse(s)
        assert (q.a, q.b, q.c, q.facts[k]) == (p.a, k, fresh, fresh), "the source asks a → b′ → d′"
        assert q.facts[p.b] == p.c, "b's fact stays"
        assert np.flatnonzero(c != w).tolist() == [_object_slot(w, k)], "the control replaces only b′'s object"
        assert t.parse(c).c == p.c and t.parse(c).facts[k] == fresh


def test_outcomes_on_aligned_sources_are_three_different_tokens():
    t = TwoHopTask()
    targets = t.batch_for_step(seed=2, step=0, batch=40)
    sources, _, counterparts = two_hop.aligned_sources(t, targets, np.random.default_rng(5))
    unchanged = np.array([t.parse(w).c for w in targets])
    swapped = np.array([t.parse(w).facts[int(k)] for w, k in zip(targets, counterparts)])
    transplanted = sources[:, -1]
    assert ((unchanged != swapped) & (swapped != transplanted) & (unchanged != transplanted)).all()
    assert two_hop.patch_outcomes(swapped, targets, sources, counterparts) == {"swap_rate": 1.0, "transplant_rate": 0.0}
    assert two_hop.patch_outcomes(transplanted, targets, sources, counterparts) == {"swap_rate": 0.0, "transplant_rate": 1.0}
    assert two_hop.patch_outcomes(unchanged, targets, sources, counterparts) == {"swap_rate": 0.0, "transplant_rate": 0.0}


def test_aligned_sources_check_their_windows():
    t = TwoHopTask()
    targets = t.batch_for_step(seed=2, step=0, batch=2)
    for bad in (targets[0], targets[:, :-1]):
        with pytest.raises(ValueError):
            two_hop.aligned_sources(t, bad, np.random.default_rng(0))
    before = targets.copy()
    two_hop.aligned_sources(t, targets, np.random.default_rng(0))
    assert np.array_equal(targets, before), "the targets are left alone"
