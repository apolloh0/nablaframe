"""Phase 3 gate helpers: the numbers the experiments are defined by. Needs no JAX."""

from __future__ import annotations

import importlib.util
import sys

from conftest import REPO_ROOT


def _common():
    gates = REPO_ROOT / "gates" / "phase3"
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location("phase3_common", gates / "_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_constants():
    c = _common()
    assert (c.REFERENCE_SEED, c.REFERENCE_WINDOWS) == (2, 64), "D22's reference batch"
    assert (c.EXPERIMENT_POSITIONS, c.LATER_WINDOW, c.FIT_FIRST) == (64, 16, 16)


def test_position_helpers():
    c = _common()
    assert c.later_positions(10, 256).tolist() == list(range(11, 27)), "the next 16 positions"
    assert c.later_positions(250, 256).tolist() == [251, 252, 253, 254, 255], "cut at the window's end"
    assert c.later_positions(255, 256).tolist() == []
    assert c.fit_range(256).tolist() == list(range(16, 255)), "D17's fit range, 16 to T - 2"
