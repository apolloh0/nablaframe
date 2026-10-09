"""Phase 3's gates measure the six operations H3 ruled on, however many the spec module gains later.

G3.2's sample and G3.5's examples iterated ``spec.OPERATIONS``. When ``jswap`` joined it (PR #45),
G3.2's seeded sample changed and then failed, and G3.5 raised KeyError, with no test noticing. Their
committed reports must stay reproducible, so both iterate ``gates/phase3/_common.py``'s
``PHASE3_OPERATIONS``. Needs the jax lane.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from conftest import REPO_ROOT  # noqa: E402
from nablaframe.model.gpt2 import GPT2Model, init_params  # noqa: E402
from phase2_fixtures import TINY, _perturbed  # noqa: E402

GATES = REPO_ROOT / "gates" / "phase3"
SIX = ("zero", "mean", "resample", "patch", "add", "swap")
SHARED = ("_common", "g31_identities")


@contextlib.contextmanager
def _phase3_gates():
    """Gate scripts import ``_common`` by name, which every phase's gates share: isolate it, then restore."""
    path, saved = list(sys.path), {name: sys.modules.pop(name) for name in SHARED if name in sys.modules}
    sys.path.insert(0, str(GATES))
    try:
        yield
    finally:
        sys.path[:] = path
        for name in SHARED:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"phase3_operations_{name}", GATES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batches(seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.integers(0, TINY.vocab, (n, TINY.seq)).astype(np.int32) for n in (2, 2, 6))


def test_g32_samples_the_six_operations():
    probe, source, reference = _batches(0)
    with _phase3_gates():
        g32 = _load("g32_determinism")
        common = sys.modules["_common"]
        assert common.PHASE3_OPERATIONS == SIX
        specs = g32.sample_specs(TINY, (0, 4, 8), common.Objects(), reference, source)
    assert sorted(spec.edits[0].op for spec in specs) == sorted(SIX * 2), "every operation twice over, only the six"


def test_g35_examples_are_the_six_operations():
    probe, source, reference = _batches(1)
    params = _perturbed(init_params(TINY, seed=0))
    with _phase3_gates():
        g35 = _load("g35_viewer")
        chosen = g35.examples(GPT2Model(TINY), params, TINY, probe, source, reference, [0], sys.modules["_common"].Objects())
    assert [spec.edits[0].op for spec, _ in chosen] == list(SIX)
