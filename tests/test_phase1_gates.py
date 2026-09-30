"""Phase 1 gate scripts: what can be pinned without the Mac.

Spec: docs/specs/phase-1.md, "Gates"; contracts in gates/phase1/.
"""

from __future__ import annotations

import importlib.util
import re
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from nablaframe.reports import PHASE1_GATE_IDS

GATES = REPO_ROOT / "gates" / "phase1"
SCRIPTS = {"G1.1": "g11_tokenizer", "G1.2": "g12_parity", "G1.3": "g13_determinism",
           "G1.4": "g14_reference", "G1.5": "g15_oracle", "G1.6": "g16_trace"}


def _load(name: str):
    if str(GATES) not in sys.path:
        sys.path.insert(0, str(GATES))  # as when a script runs, so `import _common` works
    spec = importlib.util.spec_from_file_location(f"phase1_{name}", GATES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("gate_id,script", sorted(SCRIPTS.items()))
def test_gate_scripts_declare_their_gate(gate_id, script):
    module = _load(script)
    assert module.GATE_ID == gate_id and gate_id in PHASE1_GATE_IDS
    assert module.REPORT_PATH == f"reports/phase1/{script}.json"
    assert callable(module.main)


def test_trace_schedule_and_storage():
    common = _load("_common")
    assert common.trace_steps(12_208) == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 1024, 2000, 2048, 3000,
                                          4000, 4096, 5000, 6000, 7000, 8000, 8192, 9000, 10000, 11000, 12000, 12208)
    assert common.trace_steps(600) == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600)
    assert common.trace_steps(1000) == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)

    jax = pytest.importorskip("jax")  # noqa: F841  (the model config lives in the jax lane)
    from nablaframe.model.gpt2 import GPT2Config
    per = common.trace_bytes_per_checkpoint(GPT2Config.from_size("S", 4096), 8)
    assert per == 98_566_144
    assert per * len(common.trace_steps(12_208)) == 2_759_852_032


def test_envelope_carries_the_commit():
    common = _load("_common")
    commit, dirty = common.git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", commit) and isinstance(dirty, bool)
    env = common.envelope("G1.2")
    assert env["schema_version"] == 2 and env["gate_id"] == "G1.2"
    assert env["git_commit"] == commit and env["git_dirty"] == dirty
    assert {"env_lock_sha256", "xla_flags", "thread_count", "chip", "timestamp_utc", "env_name"} <= set(env)
    assert "status" not in env and "measurements" not in env


def test_plain_converts_numpy():
    common = _load("_common")
    value = common.plain({"a": np.float32(1.5), "b": np.int64(3), "c": np.bool_(True),
                         "d": np.array([1, 2], dtype=np.int32), "e": [np.float64(0.25), None]})
    assert value == {"a": 1.5, "b": 3, "c": True, "d": [1, 2], "e": [0.25, None]}
    assert type(value["a"]) is float and type(value["b"]) is int and type(value["c"]) is bool
