"""Shared by the Phase 3 gates. Read with docs/specs/phase-3.md, "Gates".

May import ``gates/phase1/_common.py`` and ``gates/phase2/_common.py`` (load them by path)
but never change them. Reports go to ``reports/phase3/``, working files to ``runs/phase3/``.
Every gate runs on the Mac only, refuses a dirty tree unless ``--allow-dirty``, prints a
progress line at least every 30 seconds during anything longer than a minute, writes one
schema-2 report, and runs every pass through ``run_intervened``.

* ``REFERENCE_SEED = 2`` and ``REFERENCE_WINDOWS = 64``: D22's reference batch, drawn from
  the training stream like D17's fit sample.
* ``EXPERIMENT_POSITIONS = 64``, ``LATER_WINDOW = 16``, ``FIT_FIRST = 16``.
* ``later_positions(p, seq_len)``: positions ``p + 1`` up to ``p + LATER_WINDOW``, cut at the
  window's end.
* ``fit_range(seq_len)``: positions ``FIT_FIRST`` to ``seq_len - 2``, D17's fit range.
"""

# Beyond the contract above, this module holds what several Phase 3 gates share:
#
# * ``p2``: gates/phase2/_common.py, and ``p1``, gates/phase1/_common.py, both loaded by
#   path under their own module names (all three files are called ``_common``). ``p1`` is
#   the module ``p2`` loaded, so pointing one set of globals at a scratch folder points
#   every gate at it. The Phase 3 gates reuse their command line, start checks, report
#   writing, Progress, G1.1's data, G1.4's run (``p2.reference``) and G1.6's store.
# * ``fresh_dir``: Phase 1's, for working folders under runs/phase3/ instead.
# * The batches the gates run on: the probe (G1.6's), D22's reference batch, and the patch
#   source, a second batch of 8 validation windows drawn as the probe is with seed 1.
# * ``Objects``: what a spec's hashes name, for ``run_intervened``'s ``resolve`` and, written
#   out as ``<hash>.npy`` files, for the command line's ``--objects``.
# * ``run``: ``run_intervened`` with its results as numpy arrays. Every pass in every
#   Phase 3 gate goes through it, the unintervened ones included (an empty spec).
# * G2.9's J's, for G3.3 and G3.4: ``fit_report_problem`` and ``jacobians``, which check
#   and load them exactly as G2.10 does. They are restated here, not imported from G2.10,
#   because a Phase 3 gate may reuse only the three ``_common`` modules.
# * ``run_cli``: the command line's ``main`` in this process, with its one error line
#   captured; ``cli_inputs`` writes the files it reads; ``compare_trees`` compares two of
#   its output folders file by file.
#
# Paths are read from the globals of this module, ``p1`` and ``p2`` when a function runs,
# never bound at import, so everything here can be pointed at a scratch folder. Nothing
# here imports JAX at module level, so every gate script imports in any environment.

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_SEED = 2
REFERENCE_WINDOWS = 64
EXPERIMENT_POSITIONS = 64
LATER_WINDOW = 16
FIT_FIRST = 16


def later_positions(p: int, seq_len: int) -> np.ndarray:
    return np.arange(p + 1, min(p + 1 + LATER_WINDOW, seq_len), dtype=np.int64)


def fit_range(seq_len: int) -> np.ndarray:
    return np.arange(FIT_FIRST, seq_len - 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# Phase 1's and Phase 2's shared modules, and where Phase 3's files go
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase3"
RUN_DIR = REPO_ROOT / "runs" / "phase3"

PATCH_SOURCE_SEED = 1  # the probe is drawn with seed 0
PATCH_SOURCE_WINDOWS = 8
FIT_REPORT = "g29_jlens_fit.json"
# D17's fit sample: batch_for_step(train, FIT_SEED, FIT_STEP, FIT_WINDOWS, seq), as G2.7 to G2.10 draw it.
FIT_SEED, FIT_STEP, FIT_WINDOWS = 1, 0, 128


def _load_phase2_common() -> Any:
    """gates/phase2/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase2_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase2"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p2 = _load_phase2_common()
p1 = p2.p1


def report_relative(name: str) -> str:
    """``reports/phase3/<name>``, as the gates' ``start`` and ``write`` take it."""
    return str((Path(REPORT_DIR) / name).relative_to(p1.REPO_ROOT))


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 3 working folders live under {RUN_DIR}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def reference() -> Any:
    """G1.4's run, as the Phase 2 gates read it (``p2.reference``): manifest, config, model, steps, params."""
    return p2.reference()


# ---------------------------------------------------------------------------
# The batches
# ---------------------------------------------------------------------------


def probe(validation: np.ndarray, seq: int) -> np.ndarray:
    """G1.6's probe: 8 validation windows drawn with seed 0 after the evaluation slice, int32."""
    return p1.probe(validation, seq)


def reference_batch(train: np.ndarray, seq: int) -> np.ndarray:
    """D22's reference batch: 64 training windows drawn with seed 2, each window's first ``seq`` tokens, int32."""
    from nablaframe.data.corpus import batch_for_step

    return batch_for_step(np.asarray(train), REFERENCE_SEED, 0, REFERENCE_WINDOWS, seq)[:, :seq].astype(np.int32)


def patch_source(validation: np.ndarray, seq: int) -> np.ndarray:
    """A second batch of 8 validation windows, drawn as the probe is but with seed 1, int32.

    Raises SystemExit if it equals the probe, since a patch from the probe itself proves nothing.
    """
    from nablaframe.data.corpus import probe_batch

    source = probe_batch(np.asarray(validation), p1.EVAL_TOKENS, PATCH_SOURCE_SEED, PATCH_SOURCE_WINDOWS, seq)
    if np.array_equal(source, probe(validation, seq)):
        raise SystemExit("the patch source batch equals the probe. No report written.")
    return source


def fit_windows(train: np.ndarray, seq: int) -> np.ndarray:
    """D17's fit sample: 128 training windows drawn with seed 1, each window's first ``seq`` tokens, int32."""
    from nablaframe.data.corpus import batch_for_step

    return batch_for_step(np.asarray(train), FIT_SEED, FIT_STEP, FIT_WINDOWS, seq)[:, :seq].astype(np.int32)


class Objects:
    """Arrays by the hash a spec names them with: ``batch_hash`` for token batches, ``tensor_sha256`` for vectors."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}

    def batch(self, tokens: np.ndarray) -> str:
        from nablaframe.data.corpus import batch_hash

        tokens = np.asarray(tokens, dtype=np.int32)
        digest = batch_hash(tokens)
        self.arrays[digest] = tokens
        return digest

    def vector(self, vector: np.ndarray) -> str:
        from nablaframe.trace.schema import tensor_sha256

        vector = np.ascontiguousarray(vector, dtype=np.float32)
        digest = tensor_sha256(vector)
        self.arrays[digest] = vector
        return digest

    def resolve(self, digest: str) -> np.ndarray:
        return self.arrays[digest]

    def write(self, folder: Path) -> Path:
        """Each array as ``<folder>/<hash>.npy``, the command line's ``--objects``."""
        folder.mkdir(parents=True, exist_ok=True)
        for digest, array in sorted(self.arrays.items()):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            p1.write_atomic(folder / f"{digest}.npy", buffer.getvalue())
        return folder


class Checkpoints:
    """G1.4's parameters by step, each loaded once: ``params_at`` for ``run_intervened``'s swaps."""

    def __init__(self, ref: Any) -> None:
        self.ref = ref
        self.loaded: dict[int, Any] = {}

    def __call__(self, step: int) -> Any:
        if step not in self.loaded:
            if step not in self.ref.steps:
                raise ValueError(f"step {step} is not one of G1.4's trace checkpoints")
            self.loaded[step] = self.ref.params_for(step)
        return self.loaded[step]


def run(model: Any, params: Any, tokens: np.ndarray, spec: Any, *, n_layers: int,
        resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None = None,
        taps: Sequence[str] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``run_intervened``'s logits and recorded taps, as numpy arrays."""
    from nablaframe.interventions.apply import run_intervened

    logits, recorded = run_intervened(model, params, tokens, spec, n_layers=n_layers, resolve=resolve,
                                      params_at=params_at, taps=taps)
    return _numpy(logits), {tap: _numpy(value) for tap, value in recorded.items()}


def unintervened(model: Any, params: Any, tokens: np.ndarray, *, n_layers: int,
                 taps: Sequence[str] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The unintervened pass: ``run_intervened`` with an empty spec."""
    from nablaframe.interventions.spec import InterventionSpec

    return run(model, params, tokens, InterventionSpec(()), n_layers=n_layers, resolve=_no_objects, taps=taps)


def _no_objects(digest: str) -> np.ndarray:
    raise KeyError(f"an empty spec names no object, but {digest} was asked for")


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "array", value))


# ---------------------------------------------------------------------------
# G2.9's J's, checked and loaded as G2.10 does
# ---------------------------------------------------------------------------


def fit_report_problem() -> str | None:
    """Why G2.9's report doesn't pass, as a sentence, or None when it exists, is valid and passes."""
    from nablaframe.reports import validate_report

    path = Path(p2.REPORT_DIR) / FIT_REPORT
    rel = p1.rel(path)
    if not path.exists():
        return f"{rel} is missing"
    try:
        fit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{rel} can't be read ({exc})"
    problems = validate_report(fit) if isinstance(fit, dict) else ["not a JSON object"]
    if problems:
        return f"{rel} is not a valid report: {'; '.join(problems)}"
    if fit["gate_id"] != "G2.9" or fit["status"] != "pass":
        return f"{rel} is {fit['gate_id']} with status {fit['status']}, not a passing G2.9"
    return None


def jlens_fit_dir() -> Path:
    """G2.9's cache of J's, ``runs/phase2/jlens/fit/``."""
    return Path(p2.RUN_DIR) / "jlens" / "fit"


def jacobians(ref: Any, train: np.ndarray, steps: Sequence[int]) -> Callable[[int], dict[int, np.ndarray]]:
    """G2.9's J's for ``steps``: a function from a step to ``{layer: J}`` for layers 0 to n - 2.

    Checks first that G2.9's report passes and that every J is in G2.9's cache under
    ``jacobian_key`` with the trace checkpoint sidecar's ``state_sha256``, ``batch_hash`` of
    D17's 128 fit windows and this checkout's ``views/jlens.py`` code identity. Raises
    SystemExit, a setup error, naming whatever is missing.
    """
    from nablaframe.data.corpus import batch_hash
    from nablaframe.views.jlens import code_identity, jacobian_key, load_jacobian

    problem = fit_report_problem()
    if problem is not None:
        raise SystemExit(f"{problem}. The J-lens needs G2.9's J's. No report written.")
    root = jlens_fit_dir()
    fit_sha, code = batch_hash(fit_windows(train, ref.config.seq)), code_identity()
    keys = {}
    for step in steps:
        sidecar = Path(ref.checkpoint_dir) / f"step-{step}.json"
        state = json.loads(sidecar.read_text(encoding="utf-8"))["state_sha256"]
        for layer in range(ref.config.n_layers - 1):
            keys[(step, layer)] = jacobian_key(checkpoint_sha256=state, fit_sha256=fit_sha, layer=layer, code=code)
    missing = [pair for pair, key in keys.items() if not (root / f"{key}.npy").exists()]
    if missing:
        shown = ", ".join(f"step {step} layer {layer}" for step, layer in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise SystemExit(f"{len(missing)} of {len(keys)} J's for run {ref.manifest.run_id} are missing from"
                         f" {p1.rel(root)} ({shown}{more}), for this checkout's views/jlens.py at blob {code}."
                         " Run G2.9 first. No report written.")

    def for_step(step: int) -> dict[int, np.ndarray]:
        out = {}
        for (key_step, layer), key in keys.items():
            if key_step == step:
                J = load_jacobian(root, key)
                if J is None:
                    raise SystemExit(f"G2.9's J for step {step}, layer {layer} is missing from {p1.rel(root)}."
                                     " Run G2.9 again. No report written.")
                out[layer] = J
        return out

    return for_step


# ---------------------------------------------------------------------------
# The command line, run in this process
# ---------------------------------------------------------------------------


def cli_inputs(folder: Path, config: Any, tokens: np.ndarray, objects: Objects) -> dict[str, Path]:
    """The files the command line reads, written under ``folder``: config, tokens and objects."""
    import dataclasses

    folder.mkdir(parents=True, exist_ok=True)
    config_path, tokens_path = folder / "config.json", folder / "tokens.npy"
    p1.write_atomic(config_path, (json.dumps(dataclasses.asdict(config), sort_keys=True) + "\n").encode("utf-8"))
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(tokens, dtype=np.int32), allow_pickle=False)
    p1.write_atomic(tokens_path, buffer.getvalue())
    return {"config": config_path, "tokens": tokens_path, "objects": objects.write(folder / "objects")}


def write_spec(path: Path, spec: Any) -> Path:
    from nablaframe.interventions.spec import spec_json

    p1.write_atomic(path, (json.dumps(spec_json(spec), sort_keys=True) + "\n").encode("utf-8"))
    return path


def run_cli(inputs: Mapping[str, Path], checkpoints: Path, step: int, spec_path: Path, out: Path,
            taps: Sequence[str] = ()) -> tuple[int, str]:
    """``nablaframe.interventions.cli.main`` on these files: ``(exit code, its error line or "")``."""
    from nablaframe.interventions import cli

    argv = ["--config", str(inputs["config"]), "--checkpoints", str(checkpoints), "--step", str(step),
            "--tokens", str(inputs["tokens"]), "--spec", str(spec_path), "--objects", str(inputs["objects"]),
            "--out", str(out)]
    if taps:
        argv += ["--taps", ",".join(taps)]
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        code = cli.main(argv)
    return code, p1.scrub(errors.getvalue().strip())


def tree_files(folder: Path) -> dict[str, bytes]:
    """Every file under ``folder``, by its path relative to it, with its bytes."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def compare_trees(first: Path, second: Path) -> dict[str, Any]:
    """Whether two folders hold the same files, by relative path, with the same bytes."""
    a, b = tree_files(first), tree_files(second)
    differing = sorted(set(a) ^ set(b)) + sorted(name for name in set(a) & set(b) if a[name] != b[name])
    return {"identical": not differing and bool(a), "files_compared": len(set(a) | set(b)),
            "files_differing": differing}
