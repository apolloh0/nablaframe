"""Shared by the Phase 2 gates. Read with docs/specs/phase-2.md, "Gates".

May import ``gates/phase1/_common.py`` (load it by path; both files are called
``_common``) but never change it. Reports go to ``reports/phase2/``, working files to
``runs/phase2/``. The reference run is G1.4's, at ``runs/phase1/g14_reference/``, and its
trace store is G1.6's, at ``runs/phase1/g16_trace/store/``. Every gate runs on the Mac
only, refuses a dirty tree unless ``--allow-dirty``, prints a progress line at least every
30 seconds during anything longer than a minute, and writes one schema-2 report.

* ``logits_bytes_expected(n_checkpoints, batch, seq, vocab)``: float32 logits for every
  checkpoint, ``4 * n_checkpoints * batch * seq * vocab``.
* ``random_accuracy_bound(vocab)``: ``2 / vocab``, twice chance (G2.3).
* ``chance_count(vocab, positions)``: correct guesses expected by chance, ``positions / vocab``.
* ``EXACT_PREFIX = 32`` and ``FIT_SIZES = (2_048, 8_192, 32_768)`` (G2.5).
* ``exact_jacobian_bytes(prefix, d_model)``: the float32 Jacobian of a ``prefix``-position
  sequence, ``4 * (prefix * d_model) ** 2``.
"""

# Beyond the contract above, this module holds what several Phase 2 gates share:
#
# * ``p1``: gates/phase1/_common.py, loaded by path under its own module name. The
#   Phase 2 gates reuse its command line, start checks (Mac only, clean tree, report
#   writable), report writing, Progress, G1.1's data and G1.4's run.
# * ``fresh_dir``: Phase 1's, for working folders under runs/phase2/ instead.
# * Loading what every Phase 2 gate reads: G1.4's model and checkpoints
#   (``reference``), and G1.6's store with D12's logits in it (``open_reader``).
#
# Paths are read from this module's globals when a function runs, never bound at
# import, so everything here can be pointed at a scratch folder. Nothing here
# imports JAX at module level, so every gate script imports in any environment.

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXACT_PREFIX = 32
FIT_SIZES = (2_048, 8_192, 32_768)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase2"
RUN_DIR = REPO_ROOT / "runs" / "phase2"


def logits_bytes_expected(n_checkpoints: int, batch: int, seq: int, vocab: int) -> int:
    return 4 * n_checkpoints * batch * seq * vocab


def random_accuracy_bound(vocab: int) -> float:
    return 2 / vocab


def chance_count(vocab: int, positions: int) -> float:
    return positions / vocab


def exact_jacobian_bytes(prefix: int, d_model: int) -> int:
    return 4 * (prefix * d_model) ** 2


# ---------------------------------------------------------------------------
# Phase 1's shared module, and what the Phase 2 gates build on it
# ---------------------------------------------------------------------------


def _load_phase1_common() -> Any:
    """gates/phase1/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase1_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase1"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p1 = _load_phase1_common()


def report_relative(name: str) -> str:
    """``reports/phase2/<name>``, as the gates' ``start`` and ``write`` take it."""
    return str((REPORT_DIR / name).relative_to(REPO_ROOT))


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 2 working folders live under {RUN_DIR}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def g16_store() -> Path:
    """G1.6's trace store of the reference run, which G2.1 adds D12's logits to."""
    return p1.RUN_DIR / "g16_trace" / "store"


def reference() -> SimpleNamespace:
    """G1.4's run as the Phase 2 gates read it: manifest, config, model, steps, and a params loader.

    Checks every trace checkpoint's sidecar names G1.4's run before anything is measured,
    so a gate never measures a checkpoint left from another run. Raises SystemExit if
    G1.4's manifest or any checkpoint is missing.
    """
    from nablaframe.model.gpt2 import GPT2Model, init_params

    manifest = p1.load_g14_manifest()
    config = p1.config_from_manifest(manifest)
    steps = tuple(manifest.checkpoint_schedule)
    paths = {step: p1.trace_checkpoint(manifest, step) for step in steps}
    like = init_params(config, 0)

    def params_for(step: int) -> Any:
        from nablaframe.train.checkpoint import load_params

        return load_params(paths[step], like=like)

    return SimpleNamespace(manifest=manifest, config=config, model=GPT2Model(config), steps=steps,
                           checkpoint_dir=p1.g14_trace_dir(), like=like, params_for=params_for)


def open_reader(root: Path, steps: tuple[int, ...], probe: Any) -> Any:
    """A ``TraceReader`` on ``root`` after checking it holds what the views decode.

    Every step must have a ``logits`` event (G2.1 records them) and events recorded on
    ``probe``. Raises SystemExit otherwise: a setup error, so the gate writes no report.
    """
    from nablaframe.data.corpus import batch_hash
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.readouts import TraceReader

    if not (Path(root) / "events.jsonl").exists():
        raise SystemExit(f"{p1.rel(root)} holds no trace store. Run G1.6, then G2.1, first. No report written.")
    reader = TraceReader(root)
    missing = [step for step in steps if step not in reader.steps()]
    if missing:
        raise SystemExit(f"{p1.rel(root)} has no events for steps {missing}. Run G1.6 first. No report written.")
    no_logits = []
    for step in steps:
        try:
            reader.event(step, LOGITS)
        except KeyError:
            no_logits.append(step)
    if no_logits:
        raise SystemExit(f"{p1.rel(root)} has no logits for steps {no_logits}. Run G2.1 first. No report written.")
    expected = batch_hash(probe)
    for step in steps:
        if reader.batch_hash(step) != expected:
            raise SystemExit(f"step {step} in {p1.rel(root)} was traced on a different batch than the probe."
                             " No report written.")
    return reader


def file_bytes(paths: Any) -> int:
    return sum(Path(path).stat().st_size for path in paths)
