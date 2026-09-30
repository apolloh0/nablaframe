"""Shared pieces for the Phase 1 gates.

Spec: docs/specs/phase-1.md, "Gates". Phase 0's gates/_common.py may be reused for
the lockfile hash, chip name and thread count; don't modify it.

* ``git_state()``: ``(commit, dirty)``. ``commit`` is ``git rev-parse HEAD``;
  ``dirty`` is true when tracked files differ from HEAD (``git diff --quiet HEAD``
  fails). Untracked files don't count, so reports written by earlier gates never
  make a later gate dirty.
* ``require_clean_tree(allow_dirty)``: gates refuse to start on a dirty tree
  unless ``--allow-dirty`` is passed, because a report must be reproducible from
  its commit. The flag's value is recorded as ``git_dirty``.
* ``envelope(gate_id, env_name)``: every schema 2 field except ``status`` and
  ``measurements``: schema_version 2, gate_id, env_name, env_lock_sha256,
  xla_flags, thread_count, chip, timestamp_utc, git_commit, git_dirty.
* ``Progress(label, total, interval_s=30)``: for work that isn't a training loop
  (tokenizing, encoding, tracing). ``advance(n)`` prints about every ``interval_s``:
  ``"<label>: 1,234,567/2,700,000 (46%) | 41,000/s | about 35 s left"``. Training
  loops use ``nablaframe.train.heartbeat.Heartbeat`` instead.
* ``watch_hint(folder)``: prints ``"Logs: <folder>. To watch: python3 scripts/watch.py <folder>"``.
  Any gate that sends work to a log file prints it first.
* ``trace_steps(total_steps)``: the trace checkpoint schedule. 0, the powers of two
  up to 8,192, every multiple of 1,000, and ``total_steps``; only steps up to
  ``total_steps``, sorted, no repeats. 28 steps for 12,208.
* ``trace_bytes_per_checkpoint(config, probe_sequences)``:
  ``4 * P * seq * (d_model * (7 * n_layers + 3) + n_layers * n_heads * seq)``.
  98,566,144 at size S with P = 8.
* ``plain(value)``: converts numpy scalars and arrays to plain JSON types, as
  reports require.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase1"
RUN_DIR = REPO_ROOT / "runs" / "phase1"
DATA_DIR = REPO_ROOT / "data" / "tinystories"
EVAL_TOKENS = 262_144
PROBE_SEED, PROBE_SEQUENCES = 0, 8


def git_state() -> tuple[str, bool]:
    raise NotImplementedError


def require_clean_tree(allow_dirty: bool) -> None:
    raise NotImplementedError


def envelope(gate_id: str, env_name: str = "nf-jax") -> dict[str, Any]:
    raise NotImplementedError


class Progress:
    def __init__(self, label: str, total: int, interval_s: float = 30.0) -> None:
        raise NotImplementedError

    def advance(self, n: int = 1) -> None:
        raise NotImplementedError


def watch_hint(folder: Path) -> None:
    raise NotImplementedError


def trace_steps(total_steps: int) -> tuple[int, ...]:
    raise NotImplementedError


def trace_bytes_per_checkpoint(config: Any, probe_sequences: int) -> int:
    raise NotImplementedError


def plain(value: Any) -> Any:
    raise NotImplementedError
