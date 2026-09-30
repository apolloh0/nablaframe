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

# Beyond the contract above, this module holds what several Phase 1 gates share:
#
# * The command line every gate takes (--overwrite, --allow-dirty), the checks a
#   gate makes before it starts measuring (``start``), and writing its report
#   (``write``).
# * Where G1.1's outputs and G1.4's run live, and loading them: the tokenizer, the
#   token streams, the probe batch, G1.4's manifest and its trace checkpoints.
#   G1.5 and G1.6 build the model from G1.4's manifest, so they check exactly the
#   run G1.4 recorded.
# * ``corpus_sha256``: the SHA-256 of the pair of token streams that goes into
#   ``RunManifest.corpus_sha256``, defined once here: the SHA-256 of
#   ``canonical_json({"train": <sha256 of train.bin>, "validation": <sha256 of
#   validation.bin>})``. Hashing the per-file hashes keeps the boundary between the
#   two files, which concatenated bytes would lose. G1.1 records it and G1.4 puts it
#   in the manifest.
#
# Paths are read from this module's globals when a function runs, never bound at
# import, so everything here can be pointed at a scratch folder. Nothing here
# imports JAX at module level, so every gate script imports in any environment.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.reports import write_report

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase1"
RUN_DIR = REPO_ROOT / "runs" / "phase1"
DATA_DIR = REPO_ROOT / "data" / "tinystories"
EVAL_TOKENS = 262_144
PROBE_SEED, PROBE_SEQUENCES = 0, 8

# The repository whose commit a report records. Fixed to this file's checkout, even
# when REPO_ROOT is pointed elsewhere.
_GIT_ROOT = Path(__file__).resolve().parents[2]
ENV_NAME = "nf-jax"
SIZE = "S"


def _load_phase0_common() -> Any:
    """gates/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase0_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _GIT_ROOT / "gates" / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


phase0 = _load_phase0_common()


def git_state() -> tuple[str, bool]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=_GIT_ROOT, capture_output=True, text=True, timeout=60)

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed in {_GIT_ROOT}: {head.stderr.strip()}")
    # A file that was only touched still fails `git diff --quiet` until the index's stat
    # information is refreshed, as `git status` does. Its exit code doesn't matter.
    git("update-index", "-q", "--refresh")
    diff = git("diff", "--quiet", "HEAD")
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"git diff --quiet HEAD failed in {_GIT_ROOT}: {diff.stderr.strip()}")
    return head.stdout.strip(), diff.returncode == 1


def require_clean_tree(allow_dirty: bool) -> None:
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise SystemExit(
            f"Tracked files differ from commit {commit[:12]}, so this report could not be reproduced from its"
            " commit. Commit or stash the changes, or pass --allow-dirty to run anyway and record git_dirty: true."
        )


def envelope(gate_id: str, env_name: str = "nf-jax") -> dict[str, Any]:
    commit, dirty = git_state()
    return {
        "schema_version": 2,
        "gate_id": gate_id,
        "env_name": env_name,
        "env_lock_sha256": phase0.env_lock_sha256(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "thread_count": phase0.thread_count(),
        "chip": phase0.chip(),
        "timestamp_utc": phase0.utc_timestamp(),
        "git_commit": commit,
        "git_dirty": dirty,
    }


class Progress:
    def __init__(self, label: str, total: int, interval_s: float = 30.0, *,
                 clock: Callable[[], float] = time.monotonic,
                 write: Callable[[str], object] = lambda line: print(line, flush=True)) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self.label = label
        self.total = max(int(total), 0)
        self.interval_s = interval_s
        self.done = 0
        self._clock = clock
        self._write = write
        self._last_time = clock()
        self._last_done = 0

    def advance(self, n: int = 1) -> None:
        self.done += n
        now = self._clock()
        elapsed = now - self._last_time
        if elapsed < self.interval_s:
            return
        rate = (self.done - self._last_done) / elapsed
        line = f"{self.label}: {self.done:,}/{self.total:,}"
        if self.total:
            line += f" ({100 * self.done / self.total:.0f}%)"
        line += f" | {rate:,.0f}/s"
        if rate > 0 and self.total >= self.done:
            line += f" | about {_duration((self.total - self.done) / rate)} left"
        self._write(line)
        self._last_time = now
        self._last_done = self.done


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def watch_hint(folder: Path) -> None:
    print(f"Logs: {rel(folder)}. To watch: python3 scripts/watch.py {rel(folder)}", flush=True)


def trace_steps(total_steps: int) -> tuple[int, ...]:
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 0:
        raise ValueError(f"total_steps must be a non-negative int, got {total_steps!r}")
    powers = {2**k for k in range(14)}  # 1, 2, 4, ..., 8,192
    thousands = set(range(1000, total_steps + 1, 1000))
    return tuple(sorted(step for step in {0, total_steps} | powers | thousands if step <= total_steps))


def trace_bytes_per_checkpoint(config: Any, probe_sequences: int) -> int:
    d, n_layers, n_heads, seq = config.d_model, config.n_layers, config.n_heads, config.seq
    return 4 * probe_sequences * seq * (d * (7 * n_layers + 3) + n_layers * n_heads * seq)


def plain(value: Any) -> Any:
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
    """A finite float, or None for NaN, infinity and None: reports allow only finite numbers."""
    return phase0.finite_or_none(value)


# ---------------------------------------------------------------------------
# Starting a gate and writing its report
# ---------------------------------------------------------------------------


def arg_parser(doc: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(doc or "").split("\n\n")[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing report, and start this gate's working folder over")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run with uncommitted changes to tracked files; the report records git_dirty: true")
    return parser


def report_file(relative: str) -> Path:
    """A gate's REPORT_PATH, resolved against REPO_ROOT so gates run from any directory."""
    return REPO_ROOT / relative


def start(args: argparse.Namespace, report_relative: str) -> Path:
    """The checks every gate makes before measuring. Returns the report's path.

    The Mac only, a clean tree unless --allow-dirty, and a report that can be
    written at the end, so a long run never ends on a refusal.
    """
    phase0.require_macos()
    require_clean_tree(args.allow_dirty)
    path = report_file(report_relative)
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{rel(path)} exists. Pass --overwrite to replace it.")
    return path


def write(gate_id: str, status: str, measurements: Mapping[str, Any], path: Path, *, overwrite: bool,
          xla_flags: str | None = None) -> int:
    """Build the schema 2 report, validate and write it, and say where it went. Returns exit code 0.

    An invalid report is a bug in the gate, never a gate result: write_report raises
    and nothing is written.
    """
    report = envelope(gate_id, ENV_NAME)
    if xla_flags is not None:
        report["xla_flags"] = xla_flags
    report["status"] = status
    report["measurements"] = plain(measurements)
    write_report(report, path, overwrite=overwrite)
    print(f"{gate_id}: {status}. Report written to {rel(path)}", flush=True)
    return 0


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    if Path(RUN_DIR).resolve() not in path.resolve().parents:
        raise ValueError(f"refusing to manage {path}: gate working folders live under {RUN_DIR}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def rel(path: Path) -> str:
    """A path relative to REPO_ROOT when it is inside it, so reports carry no home-folder names."""
    try:
        return str(Path(path).resolve().relative_to(Path(REPO_ROOT).resolve()))
    except ValueError:
        return str(path)


def scrub(text: str) -> str:
    """``text`` with the repository's and the home folder's absolute paths shortened, for error text in reports."""
    for folder, short in ((Path(REPO_ROOT).resolve(), "."), (_GIT_ROOT, "."), (Path.home(), "~")):
        text = text.replace(str(folder), short)
    return text


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Write under a temporary name, sync, then rename, so ``path`` never holds a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# G1.1's outputs
# ---------------------------------------------------------------------------


def tokenizer_path() -> Path:
    return DATA_DIR / "tokenizer.json"


def split_path(split: str) -> Path:
    if split not in ("train", "validation"):
        raise ValueError(f"split must be train or validation, got {split!r}")
    return DATA_DIR / f"{split}.bin"


def require_data() -> None:
    """G1.1's outputs must exist; later gates can't measure without them."""
    missing = [rel(p) for p in (tokenizer_path(), split_path("train"), split_path("validation")) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Run G1.1 first. No report written.")


def load_tokenizer() -> Any:
    from nablaframe.data.tokenizer import Tokenizer

    return Tokenizer.from_json(tokenizer_path().read_text(encoding="utf-8"))


def load_split(split: str) -> np.ndarray:
    """A token stream as written by G1.1: little-endian uint16, read-only and memory-mapped."""
    path = split_path(split)
    if path.stat().st_size == 0:
        return np.empty(0, dtype="<u2")
    return np.memmap(path, dtype="<u2", mode="r")


def corpus_sha256(train: Path | None = None, validation: Path | None = None) -> str:
    """SHA-256 of ``canonical_json({"train": <sha256 of train.bin>, "validation": <sha256 of validation.bin>})``.

    This is RunManifest.corpus_sha256. Moving tokens from one file to the other changes it.
    """
    from nablaframe.trace.schema import canonical_json, sha256_hex

    hashes = {"train": sha256_file(train or split_path("train")),
              "validation": sha256_file(validation or split_path("validation"))}
    return sha256_hex(canonical_json(hashes).encode("utf-8"))


def model_config(vocab: int) -> Any:
    """The reference model's config: size S at the selected vocabulary."""
    from nablaframe.model.gpt2 import GPT2Config

    return GPT2Config.from_size(SIZE, vocab)


def probe(validation: np.ndarray, seq: int) -> np.ndarray:
    """The fixed probe batch: 8 windows drawn with probe seed 0 after the evaluation slice."""
    from nablaframe.data.corpus import probe_batch

    return probe_batch(np.asarray(validation), EVAL_TOKENS, PROBE_SEED, PROBE_SEQUENCES, seq)


# ---------------------------------------------------------------------------
# G1.4's run, as G1.5 and G1.6 read it
# ---------------------------------------------------------------------------


def g14_dir() -> Path:
    return RUN_DIR / "g14_reference"


def g14_manifest_path() -> Path:
    return g14_dir() / "manifest.json"


def g14_trace_dir() -> Path:
    return g14_dir() / "trace"


def g14_resume_dir() -> Path:
    return g14_dir() / "resume"


def load_g14_manifest() -> Any:
    """G1.4's RunManifest. Raises SystemExit if G1.4 has not written one."""
    from nablaframe.trace.schema import RunManifest

    path = g14_manifest_path()
    if not path.exists():
        raise SystemExit(f"{rel(path)} is missing. Run G1.4 first. No report written.")
    return RunManifest.from_json(path.read_text(encoding="utf-8"))


def config_from_manifest(manifest: Any) -> Any:
    from nablaframe.model.gpt2 import GPT2Config

    return GPT2Config(**{key: manifest.model_config[key]
                         for key in ("vocab", "d_model", "n_layers", "n_heads", "d_mlp", "seq")})


def trace_checkpoint(manifest: Any, step: int) -> Path:
    """G1.4's trace checkpoint for ``step``, after checking that its sidecar names this run.

    The loader checks the tensors against the sidecar's hash; this checks the sidecar
    belongs to the manifest's run, so a checkpoint left from another run is never
    measured. Raises SystemExit, since a gate can't measure without it.
    """
    path = g14_trace_dir() / f"step-{step}.safetensors"
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise SystemExit(f"{rel(path)} or its sidecar is missing. G1.4 must finish first. No report written.")
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    if recorded.get("run_id") != manifest.run_id or recorded.get("step") != step:
        raise SystemExit(f"{rel(sidecar)} names run {recorded.get('run_id')} step {recorded.get('step')},"
                         f" not G1.4's run {manifest.run_id} step {step}. No report written.")
    return path


def latest_checkpoint(folder: Path, run_id: str | None = None) -> Path | None:
    """The complete checkpoint in ``folder`` with the highest step, or None.

    Complete means the tensor file and its sidecar are both in place under their
    final names. With ``run_id``, only checkpoints whose sidecar names that run count.
    """
    best: tuple[int, Path] | None = None
    for path in Path(folder).glob("step-*.safetensors"):
        step = path.stem.removeprefix("step-")
        sidecar = path.with_suffix(".json")
        if not step.isdigit() or not sidecar.exists():
            continue
        if run_id is not None and json.loads(sidecar.read_text(encoding="utf-8")).get("run_id") != run_id:
            continue
        if best is None or int(step) > best[0]:
            best = (int(step), path)
    return best[1] if best else None
