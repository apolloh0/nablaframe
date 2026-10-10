"""Phase 4 delta 4: the compare payload names what each pass predicts (D38).

Contract: docs/deltas/phase-4-delta-4.md, "src/nablaframe/interventions/cli.py". Needs the jax lane.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.interventions.apply import run_intervened  # noqa: E402
from nablaframe.interventions.cli import compare_payload  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit  # noqa: E402
from interventions_fixtures import MODEL, PARAMS, PARAMS_AT, STORE, TOKENS, base_logits  # noqa: E402
from phase2_fixtures import TINY  # noqa: E402


def _top(logits, k=5):
    x = np.asarray(logits, dtype=np.float64)
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    top = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    return top.tolist(), np.take_along_axis(p, top, axis=-1)


def test_both_passes_top_tokens():
    spec = InterventionSpec((edit("blocks.1.hook_resid_post", [2, 5], "zero"),))
    payload = compare_payload(MODEL, PARAMS, TOKENS, spec, n_layers=TINY.n_layers, resolve=STORE.__getitem__,
                              params_at=PARAMS_AT.__getitem__, taps=[])
    intervened = run_intervened(MODEL, PARAMS, TOKENS, spec, n_layers=TINY.n_layers, resolve=STORE.__getitem__,
                                params_at=PARAMS_AT.__getitem__)[0]
    assert set(payload["top_tokens"]) == set(payload["top_probs"]) == {"unintervened", "intervened"}
    for label, logits in (("unintervened", base_logits()), ("intervened", intervened)):
        tokens, probs = _top(logits)
        assert payload["top_tokens"][label] == tokens, label
        assert np.asarray(payload["top_probs"][label]).shape == (*TOKENS.shape, 5)
        assert np.allclose(payload["top_probs"][label], probs, rtol=1e-12, atol=0), label
    assert payload["top_tokens"]["intervened"][0][0] == payload["top_tokens"]["unintervened"][0][0], \
        "an edit at later positions leaves position 0's prediction alone"
    assert payload["top_tokens"]["intervened"][0][2] != payload["top_tokens"]["unintervened"][0][2] or \
        not np.allclose(payload["top_probs"]["intervened"][0][2], payload["top_probs"]["unintervened"][0][2])


def test_ties_go_to_the_lower_id():
    logits = np.zeros((1, 1, TINY.vocab), np.float32)
    logits[0, 0, [9, 4, 20]] = 2.0
    from nablaframe.interventions import cli

    tokens, probs = cli.top_tokens(logits)
    assert tokens == [[[4, 9, 20, 0, 1]]] and np.allclose(probs[0][0][:3], probs[0][0][0])
