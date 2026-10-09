"""The two-hop collision rules (PR #44's ruling).

The answer is never the queried subject or the intermediate, and no distractor has the answer as
its object. A patch source's answer is neither the target's answer nor the counterpart's object.
So in every window, an unchanged answer, a swapped one and a transplanted one are three different
tokens, and the intermediate's and the answer's lens ranks are never the same token's.
"""

from __future__ import annotations

import numpy as np

from nablaframe.zoo.tasks import TwoHopTask
from nablaframe.zoo.two_hop import source_for


def _check(p):
    assert p.c not in (p.a, p.b), "the answer is neither the queried subject nor the intermediate"
    assert all(o != p.c for s, o in p.facts.items() if s != p.b), "no other fact has the answer as its object"


def test_two_hop_answers_never_collide():
    t, rng = TwoHopTask(), np.random.default_rng(11)
    for _ in range(2000):
        _check(t.parse(t.window(rng)))
    for b in (0, 5, 47):
        for _ in range(100):
            p = t.parse(t.window(rng, intermediate=b))
            assert p.b == b
            _check(p)
    for w in t.batch_for_step(seed=0, step=7, batch=256):
        _check(t.parse(w))


def test_patch_sources_keep_the_outcomes_distinct():
    t, rng = TwoHopTask(), np.random.default_rng(12)
    for w in t.batch_for_step(seed=0, step=1, batch=150):
        p = t.parse(w)
        for counterpart in (s for s in p.facts if s not in (p.a, p.b)):
            src = t.parse(source_for(t, w, counterpart, rng))
            assert src.b == counterpart
            assert src.c not in (p.c, p.facts[counterpart]), "the transplanted answer differs from the other two"
