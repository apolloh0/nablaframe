"""Phase 2 gate helpers: the numbers the gates are judged against.

Spec: docs/specs/phase-2.md, "Gates". Needs no JAX.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from conftest import REPO_ROOT


def _common():
    gates = REPO_ROOT / "gates" / "phase2"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase2_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logits_bytes_formula():
    common = _common()
    assert common.logits_bytes_expected(n_checkpoints=28, batch=8, seq=256, vocab=4096) == 939_524_096
    assert common.logits_bytes_expected(n_checkpoints=1, batch=2, seq=8, vocab=32) == 2_048


def test_random_weights_bound():
    common = _common()
    assert common.random_accuracy_bound(4096) == pytest.approx(2 / 4096)
    assert common.chance_count(vocab=4096, positions=262_144) == pytest.approx(64.0), \
        "over the evaluation slice, chance is about 64 correct guesses: a real measurement"


def test_exact_reference_uses_causal_prefixes():
    common = _common()
    assert common.EXACT_PREFIX == 32 and common.FIT_SIZES == (2_048, 8_192, 32_768)
    assert common.exact_jacobian_bytes(prefix=32, d_model=256) == (32 * 256) ** 2 * 4, \
        "the full Jacobian of a 32-position prefix at size S: 268 MB, not 17 GB"
