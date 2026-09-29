"""Shared plumbing for the Phase 0 gate scripts. Not a gate.

Spec: docs/specs/phase-0.md, "Gates": every report carries the gate id, pass or
fail, the measurements, the env lockfile hash, ``XLA_FLAGS``, thread count, chip
model, and a UTC timestamp. This module fills in everything but the measurements.

Nothing here imports JAX, PyTorch or Levanter, so every gate script imports
cleanly in any environment, including the default one the unit tests use.

Paths:

* Reports go to ``reports/phase0/`` through ``nablaframe.reports.write_report``.
* Everything else a gate produces (checkpoints, the tokenized data cache, the
  Hugging Face export, logs) goes under ``runs/phase0/``, which git ignores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.reports import validate_report, write_report

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE = REPO_ROOT / "uv.lock"
RUNS_DIR = REPO_ROOT / "runs" / "phase0"
DATA_CACHE_DIR = RUNS_DIR / "data_cache"


def env_lock_sha256() -> str:
    """SHA-256 of ``uv.lock``, which pins all three environments."""
    return hashlib.sha256(LOCKFILE.read_bytes()).hexdigest()


def thread_count() -> int:
    """CPU threads available to this process: the affinity mask where the OS has one, else the logical core count."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def chip() -> str:
    """The CPU model, e.g. "Apple M3 Max". Falls back to the machine type, never empty."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                name = line.split(":", 1)[1].strip()
                if name:
                    return name
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def macos_version() -> str | None:
    """The macOS version, e.g. "26.5", or None off macOS."""
    return platform.mac_ver()[0] or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Installed version of each distribution, None where it is not installed.

    The spec asks the reports to record what actually resolved, not what the lockfile allows.
    """
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def plain(value: Any) -> Any:
    """``value`` with numpy scalars and arrays turned into Python int, float, bool and lists.

    Reports reject numpy scalars (np.float64 included), so every measurement passes through here.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def finite_or_none(value: Any) -> float | None:
    """A finite float, or None for anything that is not one (NaN, infinity, None)."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def build_report(
    gate_id: str,
    status: str,
    measurements: Mapping[str, Any],
    env_name: str,
    *,
    xla_flags: str | None = None,
) -> dict[str, Any]:
    """A complete report dict. ``xla_flags`` defaults to this process's ``XLA_FLAGS``."""
    return {
        "schema_version": 1,
        "gate_id": gate_id,
        "status": status,
        "measurements": plain(measurements),
        "env_name": env_name,
        "env_lock_sha256": env_lock_sha256(),
        "xla_flags": os.environ.get("XLA_FLAGS", "") if xla_flags is None else xla_flags,
        "thread_count": thread_count(),
        "chip": chip(),
        "timestamp_utc": utc_timestamp(),
    }


def report_path(relative: str) -> Path:
    """A gate's REPORT_PATH, resolved against the repository root so gates run from any directory."""
    return REPO_ROOT / relative


def emit_report(report: Mapping[str, Any], path: Path, *, overwrite: bool) -> int:
    """Validate and write ``report``; print where it went. Returns the process exit code.

    An invalid report is a bug in the gate script, never a gate result: it raises,
    and nothing is written.
    """
    write_report(report, path, overwrite=overwrite)
    print(f"{report['gate_id']}: {report['status']}. Report written to {path.relative_to(REPO_ROOT)}")
    return 0


def check_writable(path: Path, *, overwrite: bool) -> None:
    """Refuse to start a run whose report could not be written at the end."""
    if path.exists() and not overwrite:
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} exists. Pass --overwrite to replace it.")


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working directory under runs/phase0/. An existing one is removed only with ``overwrite``.

    A leftover checkpoint would make Levanter resume instead of starting from step 0,
    so gates never reuse a working directory silently.
    """
    if RUNS_DIR not in path.resolve().parents:
        raise ValueError(f"refusing to manage {path}: gate working directories live under {RUNS_DIR}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} exists. Pass --overwrite to start over.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def load_report(path: Path) -> dict[str, Any]:
    """A committed or freshly written gate report, validated. Raises SystemExit if missing or invalid."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing. Run that gate first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_report(report)
    if problems:
        raise SystemExit(f"{path.relative_to(REPO_ROOT)} is not a valid report: {problems}")
    return report


def require_macos() -> None:
    """Gates run on apolloh0's Mac only, never in the cloud: refuse to measure anywhere else."""
    if platform.system() != "Darwin":
        raise SystemExit(f"Gates run on apolloh0's Mac only; this is {platform.system()}. No report written.")


def arg_parser(doc: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(doc or "").split("\n\n")[0])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report and clear this gate's working directory under runs/phase0/",
    )
    return parser


def rel(path: Path) -> str:
    """A repo-relative path for reports, so they carry no home-directory names."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
