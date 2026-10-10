"""Shared by the Phase 4 gates. Read with docs/specs/phase-4.md, "The zoo", "Discrimination tests at model level" and "Gates", and D36.

May load ``gates/phase1/_common.py``, ``gates/phase2/_common.py`` and ``gates/phase3/_common.py``
by path, but never changes them. Reports go to ``reports/phase4/``, working files to
``runs/phase4/``: each zoo model's run under ``runs/phase4/<name>/``. Every gate runs on the Mac
only, refuses a dirty tree unless ``--allow-dirty``, refuses to overwrite its report unless
``--overwrite``, prints a progress line at least every 30 seconds during anything longer than a
minute, writes one schema-2 report, and runs every intervened pass through ``run_intervened``.
G4.2 to G4.5 refuse to start unless G4.1's report exists, passes ``validate_report`` and says
``pass``: no tool is judged on a model that missed its behaviour bars.

The settings are the feasibility runs' (D37), so G4.1 retraces them on the same Mac.

* ``MODELS`` and ``TRAINING``: each zoo model's ``GPT2Config`` and ``TrainConfig`` fields, as
  plain dicts keyed ``"induction"`` and ``"two_hop"``. Induction: vocab 64, d_model 64, 2
  blocks, 4 heads, d_mlp 256, seq 63; batch 32, lr 1e-3, warmup 100, 4,000 steps. Two-hop:
  vocab 51, d_model 128, 4 blocks, 4 heads, d_mlp 512, seq 26; batch 64, lr 1e-3, warmup 500,
  20,000 steps. Both train with data and parameter seed 0, and every other ``TrainConfig``
  field at its default. ``configs(name)`` builds the pair, importing JAX inside; any other
  name raises ValueError.
* Seeds and sizes: ``FIT_SEED = 1``, ``EVAL_SEED = 2``, ``BROKEN_SEED = 3``,
  ``PATCH_SEED = 4``, ``SOURCE_SEED = 5``; ``EVAL_WINDOWS = 256``, ``PROBE_WINDOWS = 8``,
  ``FIT_WINDOWS = 64``.
* ``eval_windows(task)``: ``task.batch_for_step(EVAL_SEED, 0, EVAL_WINDOWS)``.
  ``probe(task)``: ``task.batch_for_step(EVAL_SEED, 0, PROBE_WINDOWS)``, the first eight
  evaluation windows, which every checkpoint's trace records. ``fit_windows(task)``: the
  first ``seq`` tokens of ``task.batch_for_step(FIT_SEED, 0, FIT_WINDOWS)``.
  ``broken_windows(task)``: ``EVAL_WINDOWS`` windows ``task.window(rng, broken=True)``
  drawn in sequence from ``numpy.random.default_rng([BROKEN_SEED, 0])``.
  ``induction_patch_windows(task)``: ``EVAL_WINDOWS`` windows ``task.draw_window(rng,
  period=PATCH_PERIOD)`` drawn in sequence from ``numpy.random.default_rng([PATCH_SEED, 0])``.
* ``PATCH_PERIOD = 20``, ``PATCH_TARGETS = (13, 16, 19)``, ``PATCH_LAG = 19``: G4.2's patch
  puts a new token at copy target ``j`` and reads the prediction at ``j + PATCH_LAG``.
* ``TWO_HOP_SITES``: G4.3's patch sites, ``{"query": (24, 25), "fact positions": 0 to 23,
  "every position": 0 to 25}``, as tuples.
* ``BARS``: the spec's bars, as ``{"behaviour": 0.9, "control_ratio": 2.0,
  "induction_attention": 0.5, "twin_attention_ratio": 2.0, "ablation_fraction": 0.5,
  "swap_fraction": 0.5, "patch_followed": 0.9, "patch_after_read": 0.1,
  "control_unchanged": 0.95}``.
* ``checkpoint_steps(name)``: ``"induction"``: every 50 steps from 0 through 1,500, then every
  250 through 4,000, 41 in all; ``"two_hop"``: every 1,000 from 0 through 20,000. Any other name
  raises ValueError.
* ``MIN_ABOVE_CHANCE = 0.1``. ``fraction_removed(before, after, chance)``: how much of the
  accuracy above chance an edit removed, ``(before - after) / (before - chance)``, as a plain
  float. It can exceed 1 when ``after`` falls below chance. ValueError when ``before - chance <
  MIN_ABOVE_CHANCE``, since a fraction of less is noise; the gate records null.
* ``last_at_chance(steps, accuracies, chance)``: the last step whose accuracy is at most twice
  ``chance``, G4.2's swap checkpoint (D36). ValueError if no step qualifies, or if the two
  sequences differ in length.
"""

# Beyond the contract above, this module holds what several Phase 4 gates share:
#
# * ``p3``: gates/phase3/_common.py, and through it ``p2`` and ``p1``, all loaded by path under
#   their own module names (every one of these files is called ``_common``). ``p1`` is the
#   module ``p2`` and ``p3`` loaded, so pointing one set of globals at a scratch folder points
#   every gate at it. The Phase 4 gates reuse their command line, start checks, report writing,
#   ``Progress``, ``Objects``, ``run`` and ``unintervened`` (every pass through
#   ``run_intervened``, the unintervened ones with an empty spec) and ``compare_trees``.
# * ``fresh_dir``: Phase 1's, for working folders under runs/phase4/ instead.
# * Each zoo model's task (``task``) and its run folder: ``manifest.json``, ``tokenizer.json``,
#   ``checkpoints/`` and ``store/``, G4.4's ``jlens/`` for two-hop, and G4.5's ``views/``.
#   G4.1's recompute, and G4.5's second builds, go under ``runs/phase4/scratch/``.
# * ``tokens``: what a model reads of a batch of windows, each window's first ``seq`` tokens.
# * ``require_g41``: G4.2 to G4.5's start check on G4.1's report, and ``zoo_run``, G4.1's run as
#   they read it: manifest, config, model, checkpoint steps and a params loader, after checking
#   the manifest is the run G4.1's report names and every checkpoint's sidecar names that run.
#
# Paths are read from this module's globals, and those of ``p1``, ``p2`` and ``p3``, when a
# function runs, never bound at import, so everything here can be pointed at a scratch folder.
# Nothing here imports JAX at module level, so every gate script imports in any environment.

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MODELS = {
    "induction": {"vocab": 64, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_mlp": 256, "seq": 63},
    "two_hop": {"vocab": 51, "d_model": 128, "n_layers": 4, "n_heads": 4, "d_mlp": 512, "seq": 26},
}
TRAINING = {
    "induction": {"batch": 32, "seq": 63, "lr": 1e-3, "warmup_steps": 100, "total_steps": 4000, "data_seed": 0, "param_seed": 0},
    "two_hop": {"batch": 64, "seq": 26, "lr": 1e-3, "warmup_steps": 500, "total_steps": 20000, "data_seed": 0, "param_seed": 0},
}
FIT_SEED, EVAL_SEED, BROKEN_SEED, PATCH_SEED, SOURCE_SEED = 1, 2, 3, 4, 5
EVAL_WINDOWS, PROBE_WINDOWS, FIT_WINDOWS = 256, 8, 64
PATCH_PERIOD, PATCH_TARGETS, PATCH_LAG = 20, (13, 16, 19), 19
TWO_HOP_SITES = {"query": (24, 25), "fact positions": tuple(range(24)), "every position": tuple(range(26))}
BARS = {"behaviour": 0.9, "control_ratio": 2.0, "induction_attention": 0.5, "twin_attention_ratio": 2.0,
        "ablation_fraction": 0.5, "swap_fraction": 0.5, "patch_followed": 0.9, "patch_after_read": 0.1,
        "control_unchanged": 0.95}
MIN_ABOVE_CHANCE = 0.1

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase4"
RUN_DIR = REPO_ROOT / "runs" / "phase4"
NAMES = ("induction", "two_hop")
G41_REPORT = "g41_zoo_training.json"


def configs(name: str) -> tuple[Any, Any]:
    if name not in MODELS:
        raise ValueError(f"no zoo model {name!r}; expected one of {tuple(MODELS)}")
    from nablaframe.model.gpt2 import GPT2Config
    from nablaframe.train.loop import TrainConfig

    return GPT2Config(**MODELS[name]), TrainConfig(**TRAINING[name])


def eval_windows(task: Any) -> np.ndarray:
    return task.batch_for_step(EVAL_SEED, 0, EVAL_WINDOWS)


def probe(task: Any) -> np.ndarray:
    return task.batch_for_step(EVAL_SEED, 0, PROBE_WINDOWS)


def fit_windows(task: Any) -> np.ndarray:
    return np.ascontiguousarray(task.batch_for_step(FIT_SEED, 0, FIT_WINDOWS)[:, :task.seq])


def broken_windows(task: Any) -> np.ndarray:
    rng = np.random.default_rng([BROKEN_SEED, 0])
    return np.stack([task.window(rng, broken=True) for _ in range(EVAL_WINDOWS)]).astype(np.int32)


def induction_patch_windows(task: Any) -> np.ndarray:
    rng = np.random.default_rng([PATCH_SEED, 0])
    return np.stack([task.draw_window(rng, period=PATCH_PERIOD) for _ in range(EVAL_WINDOWS)]).astype(np.int32)


def checkpoint_steps(name: str) -> list[int]:
    if name == "induction":
        return list(range(0, 1_501, 50)) + list(range(1_750, 4_001, 250))
    if name == "two_hop":
        return list(range(0, 20_001, 1_000))
    raise ValueError(f"no zoo model {name!r}; expected one of {tuple(MODELS)}")


def fraction_removed(before: float, after: float, chance: float) -> float:
    above = float(before) - float(chance)
    if above < MIN_ABOVE_CHANCE:
        raise ValueError(f"accuracy {before} is {above:.4g} above chance {chance}, less than {MIN_ABOVE_CHANCE}:"
                         " a fraction of that is noise")
    return float((float(before) - float(after)) / above)


def last_at_chance(steps: Sequence[int], accuracies: Sequence[float], chance: float) -> int:
    steps, accuracies = list(steps), [float(a) for a in accuracies]
    if len(steps) != len(accuracies):
        raise ValueError(f"{len(steps)} steps but {len(accuracies)} accuracies")
    at_chance = [step for step, accuracy in zip(steps, accuracies) if accuracy <= 2 * chance]
    if not at_chance:
        raise ValueError(f"no checkpoint's accuracy is at most twice chance ({2 * chance:.4g})")
    return int(at_chance[-1])


# ---------------------------------------------------------------------------
# The earlier phases' shared modules, and where Phase 4's files go
# ---------------------------------------------------------------------------


def _load_phase3_common() -> Any:
    """gates/phase3/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase3_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase3"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p3 = _load_phase3_common()
p2 = p3.p2
p1 = p3.p1


def report_relative(name: str) -> str:
    """``reports/phase4/<name>``, as the gates' ``start`` and ``write`` take it."""
    return str((Path(REPORT_DIR) / name).relative_to(p1.REPO_ROOT))


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 4 working folders live under {RUN_DIR}")
    if path.exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def run_dir(name: str) -> Path:
    """``runs/phase4/<name>/``: one zoo model's run, as G4.1 writes it."""
    if name not in MODELS:
        raise ValueError(f"no zoo model {name!r}; expected one of {tuple(MODELS)}")
    return Path(RUN_DIR) / name


def scratch_dir() -> Path:
    """``runs/phase4/scratch/``: G4.1's recompute and G4.5's second builds."""
    return Path(RUN_DIR) / "scratch"


def task(name: str) -> Any:
    """The zoo model's task from ``zoo/tasks.py``, at its defaults."""
    from nablaframe.zoo.tasks import InductionTask, TwoHopTask

    if name == "induction":
        return InductionTask()
    if name == "two_hop":
        return TwoHopTask()
    raise ValueError(f"no zoo model {name!r}; expected one of {tuple(MODELS)}")


def tokens(task: Any, windows: np.ndarray) -> np.ndarray:
    """What the model reads of ``windows``: each window's first ``seq`` tokens, int32."""
    return np.ascontiguousarray(np.asarray(windows)[:, :task.seq], dtype=np.int32)


# ---------------------------------------------------------------------------
# G4.1's run, as G4.2 to G4.5 read it
# ---------------------------------------------------------------------------


def require_g41() -> dict[str, Any]:
    """G4.1's report, once it exists, passes ``validate_report`` and says ``pass``. Raises SystemExit otherwise."""
    from nablaframe.reports import validate_report

    path = Path(REPORT_DIR) / G41_REPORT
    rel = p1.rel(path)
    if not path.exists():
        raise SystemExit(f"{rel} is missing. Run G4.1 first. No report written.")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{rel} can't be read ({exc}). No report written.") from exc
    problems = validate_report(report) if isinstance(report, dict) else ["not a JSON object"]
    if problems:
        raise SystemExit(f"{rel} is not a valid report: {'; '.join(problems)}. No report written.")
    if report["gate_id"] != "G4.1" or report["status"] != "pass":
        raise SystemExit(f"{rel} is {report['gate_id']} with status {report['status']}, not a passing G4.1: no tool"
                         " is judged on a model that missed its behaviour bars. No report written.")
    return report


def checkpoint(folder: Path, run_id: str, step: int) -> Path:
    """The params checkpoint for ``step`` in ``folder``, after checking its sidecar names ``run_id`` and the step.

    Raises SystemExit if it or its sidecar is missing or names another run.
    """
    path = Path(folder) / f"step-{step}.safetensors"
    sidecar = path.with_suffix(".json")
    if not path.exists() or not sidecar.exists():
        raise SystemExit(f"{p1.rel(path)} or its sidecar is missing. G4.1 must finish first. No report written.")
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    if recorded.get("run_id") != run_id or recorded.get("step") != step:
        raise SystemExit(f"{p1.rel(sidecar)} names run {recorded.get('run_id')} step {recorded.get('step')},"
                         f" not run {run_id} step {step}. No report written.")
    return path


def zoo_run(name: str, g41: dict[str, Any]) -> SimpleNamespace:
    """G4.1's run of one zoo model: task, manifest, config, model, steps, and a params loader.

    Checks, before anything is measured, that the manifest is the run G4.1's report names, that
    its model and schedule are this module's, and that every checkpoint's sidecar names it.
    Raises SystemExit otherwise. ``params_for(step)`` loads each checkpoint once.
    """
    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.train.checkpoint import load_params
    from nablaframe.trace.schema import RunManifest

    folder = run_dir(name)
    path = folder / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{p1.rel(path)} is missing. Run G4.1 first. No report written.")
    manifest = RunManifest.from_json(path.read_text(encoding="utf-8"))
    named = (g41["measurements"].get("run_ids") or {}).get(name)
    if manifest.run_id != named:
        raise SystemExit(f"{p1.rel(path)} is run {manifest.run_id}, but G4.1's report names {named}. Run G4.1"
                         " again. No report written.")
    config, train_config = configs(name)
    if p1.config_from_manifest(manifest) != config or list(manifest.checkpoint_schedule) != checkpoint_steps(name):
        raise SystemExit(f"{p1.rel(path)} doesn't hold {name}'s model and checkpoint schedule. No report written.")
    steps = tuple(manifest.checkpoint_schedule)
    paths = {step: checkpoint(folder / "checkpoints", manifest.run_id, step) for step in steps}
    like = init_params(config, 0)
    loaded: dict[int, Any] = {}

    def params_for(step: int) -> Any:
        if step not in paths:
            raise ValueError(f"step {step} is not one of {name}'s checkpoints")
        if step not in loaded:
            loaded[step] = load_params(paths[step], like=like)
        return loaded[step]

    def state_sha256(step: int) -> str:
        return json.loads(paths[step].with_suffix(".json").read_text(encoding="utf-8"))["state_sha256"]

    return SimpleNamespace(name=name, task=task(name), manifest=manifest, run_id=manifest.run_id, config=config,
                           train_config=train_config, model=GPT2Model(config), steps=steps, final=steps[-1],
                           folder=folder, store=folder / "store", like=like, params_for=params_for,
                           state_sha256=state_sha256)


def logits(model: Any, params: Any, task: Any, windows: np.ndarray, *, n_layers: int,
           taps: Sequence[str] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The unintervened pass on what the model reads of ``windows``, through ``run_intervened``."""
    return p3.unintervened(model, params, tokens(task, windows), n_layers=n_layers, taps=taps)
