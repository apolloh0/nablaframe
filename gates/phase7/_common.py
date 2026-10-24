"""Shared by the Phase 7 gates. Read with docs/specs/phase-7.md, D54 to D58, "What each run declares", "Discrimination tests on the zoo" and "Gates".

May load ``gates/phase6/_common.py``, and through it the earlier phases' ``_common.py``, by path,
but never changes them. Reports go to ``reports/phase7/`` and working files to ``runs/phase7/``:
the modular-addition run under ``runs/phase7/modular_addition/``, its dictionaries under
``runs/phase7/dictionaries/``, and each run's Phase 7 records under ``runs/phase7/gestures/<run>/``.
Every gate runs on the Mac only, refuses a dirty tree unless ``--allow-dirty``, refuses to
overwrite its report unless ``--overwrite``, prints a progress line at least every 30 seconds
during anything longer than a minute, and writes one schema-2 report. Every record a gate writes
goes through ``cli.record``, and every intervened pass through ``run_intervened``.

Each gate refuses to start unless the reports it builds on exist, pass ``validate_report`` and say
``pass``: for G7.2 to G7.4, G7.1's; for G7.3 and G7.4, G4.1's, G5.2's and G5.3's, whose runs,
windows and dictionaries they read, and G7.2's, whose records they cite; and for G7.4, G6.3's and
G6.7's, whose records it cites (G6.3 wrote the zoo runs' edge records and G6.7 their name
records), and G7.3's.

The settings are the spec's (D54):

* ``MODEL = {"vocab": 114, "d_model": 32, "n_layers": 1, "n_heads": 4, "d_mlp": 128, "seq": 3}``
  and ``TRAINING = {"batch": 512, "seq": 3, "lr": 1e-3, "warmup_steps": 10, "total_steps":
  10000, "min_lr_ratio": 0.1, "weight_decay": 1.0, "data_seed": 0, "param_seed": 0}``, every
  other ``TrainConfig`` field at its default. ``configs()`` builds the ``GPT2Config`` and the
  ``TrainConfig``, importing JAX inside.
* ``checkpoint_steps()``: every 100 steps from 0 through 3,000, then every 500 through 10,000:
  45 steps, as a list of ints.
* ``EVAL_SEED = 2``, Phase 4's, and ``PROBE_WINDOWS = 64`` (D60: at 8, the trace's bitwise oracle refuses
  this model). ``probe(task)``: the held-out windows
  ``task.held_out()`` at ``numpy.random.default_rng(EVAL_SEED).choice(len(held), PROBE_WINDOWS,
  replace=False)``, in that order, which every checkpoint's trace records.
* ``DICTIONARY = {"steps": 3000, "batch": 4096}``: G7.2's TopK SAE at block 0's
  ``hook_resid_post``, with Phase 5's ``WIDTH_FACTOR``, ``K``, ``LR`` and ``DICT_SEED``.
* ``NAMES = ("token(p+1)", "token(p-2)", "token(p-1)")``: G7.2's descriptions at ``answer``.
* ``SEED = 50``: every crystallize-this request's seed, which draws its counterfactual.
* ``JUDGED`` and ``RECORDED``: D57's crystallizations, each ``(run, block, part, class,
  function, form)``. Judged: ``("induction", 1, "attention", "mechanism", "copied", "table")``,
  ``("two_hop", 3, "write", "query", "hop", "table")``, ``("modular_addition", 0, "write",
  "answer", "sum", "table")`` and ``("modular_addition", 0, "write", "answer", "sum",
  "fourier")``. Recorded: ``("modular_addition", 0, "mlp", "answer", "sum and operands",
  "additive")``, ``("modular_addition", 0, "attention", "answer", "operands", "additive")`` and
  ``("two_hop", 3, "attention", "query", "hop", "table")``.
* ``IDENTITY_WINDOWS = 64``: G7.3's windows per zoo model.
* ``BARS``: ``{"held_accuracy": 0.99, "control_ratio": 2.0, "frequencies": 8, "logit_r2": 0.8,
  "kl": 0.25, "ablation": 1.0, "wrong": 1.0, "reads_agreement": 0.8, "answer_agreement": 0.1,
  "twin_within": 0.05}``.

Measurements, each returning plain Python floats, ints, bools and None, ready for a report:

* ``fewest_frequencies(embedding, p)``: G7.1's clock. ``{"count", "held"}``: the fewest
  frequencies whose shares of the embedding's non-constant energy, computed and ranked as
  ``crystallize.frequencies`` computes and ranks them, sum to at least ``crystallize.ENERGY``,
  with no cut at ``MAX_FREQUENCIES``, and their summed share. ValueError as
  ``crystallize.frequencies`` raises it.
* ``logit_r2(fit_logits, fit_windows, held_logits, held_windows, p, k)``: G7.1's mechanism,
  D54's ``Σₖ αₖ cos(2πk(a + b − c)/p)``. Each logits array holds its pairs' logits at position 2,
  a float array (B, vocab), and each windows array the pairs themselves, (B, 4), ``a`` and ``b``
  being a window's tokens at 0 and 1. A pair's targets are its logits at the number tokens ``c``
  from 0 to ``p − 1``, in float64, centred over ``c``; its features, for each frequency in ``k``
  in order, ``cos(2πk((a + b − c) mod p)/p)``. ``α`` is ``numpy.linalg.lstsq(rcond=None)`` over
  every (pair, ``c``) of the fitting pairs, with no constant, and the result the R² over the
  held-out pairs', ``1 − Σ(y − ŷ)² / Σ(y − ȳ)²``, ``ȳ`` their mean; None where the denominator is
  0. ValueError for an empty ``k``, logits and windows of different counts, or logits with fewer
  than ``p`` columns.
* ``set_identities(model, params, tokens, *, n_layers)``: G7.3's identities, D55's on one model:
  for each editable tap but ``attn.hook_z``, in ``editable_taps`` order, whether ``set`` at
  every position to the tap's own values gives, through ``run_intervened``, logits and every
  canonical tap bit for bit equal to the pass with an empty spec. Both passes record every
  canonical tap, ``tap_names(n_layers)``, as G3.1's do: recording a tap changes how XLA fuses
  the program, so a pass that records fewer taps can differ in the last bits. ``{tap: bool}``.
* ``requests(run, steps, cites)``: G7.3's 64 crystallize-this requests on the modular-addition
  run, each with ``instances`` 8,938, its held-out pairs. Four bases, in order: the write as a
  table by the sum, the write in the fourier form, the MLP additive in the sum and operands, and
  the attention additive in the operands, each at block 0 and class ``answer``. Each base at
  ``steps[0]`` and ``steps[-1]``, then citing nothing and ``cites``, then with ``SEED`` and
  ``SEED + 1``, then at its own block and at block 1, nested in that order.
* ``request_hashes(requests)``: as Phase 6's.
* ``crystal_checks(results)``: G7.4's bars read off one crystallization's results:
  ``{"kl", "ablation", "wrong_min", "reads_agreement", "answer_agreement", "twin_within",
  "final_holds", "step0_outside"}``. ``kl`` and ``ablation``: the meter's. ``wrong_min``: the
  least wrong function's KL, None where it has none. ``reads_agreement`` and
  ``answer_agreement``: the counterfactual's, None where it has none. ``twin_within``: the
  greatest, over the twin's wrong functions, of the distance between that one's KL and the
  twin's own KL, None where it has none. ``final_holds``: the last checkpoint row is labelled
  ``crystallize.HOLDS``. ``step0_outside``: a checkpoint row is at step 0, and it isn't labelled
  ``HOLDS``. ValueError for results without checkpoints or a twin.
* ``judged(checks)``: G7.4's measurements over the judged crystallizations' ``crystal_checks``:
  ``{"kl_max", "ablation_min", "wrong_min", "reads_agreement_min", "answer_agreement_max",
  "twin_within_max", "envelope_ok"}``. Each is the greatest or least, as Phase 6's ``greatest``
  and ``least`` take them, over every crystallization, except the two agreements, which are
  over those with a counterfactual. ``envelope_ok``: every one's ``final_holds`` and
  ``step0_outside``. ValueError for no checks.
"""

# Beyond the contract above, the Phase 7 gates add here what several of them share: the start checks on
# earlier reports, the paths, each run's model, checkpoints, twin and windows, and each run's ``cli.Context``.
# Nothing here imports JAX at module level, so every gate script imports in any environment.

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nablaframe.gestures import crystallize, records
from nablaframe.interventions.apply import editable_taps, run_intervened
from nablaframe.interventions.spec import InterventionSpec, array_hash, edit
from nablaframe.trace.registry import tap_names
from nablaframe.zoo.tasks import ModularAdditionTask

MODEL = {"vocab": 114, "d_model": 32, "n_layers": 1, "n_heads": 4, "d_mlp": 128, "seq": 3}
TRAINING = {"batch": 512, "seq": 3, "lr": 1e-3, "warmup_steps": 10, "total_steps": 10000, "min_lr_ratio": 0.1,
            "weight_decay": 1.0, "data_seed": 0, "param_seed": 0}
EVAL_SEED, PROBE_WINDOWS = 2, 64
DICTIONARY = {"steps": 3000, "batch": 4096}
NAMES = ("token(p+1)", "token(p-2)", "token(p-1)")
SEED = 50
JUDGED = (("induction", 1, "attention", "mechanism", "copied", "table"),
          ("two_hop", 3, "write", "query", "hop", "table"),
          ("modular_addition", 0, "write", "answer", "sum", "table"),
          ("modular_addition", 0, "write", "answer", "sum", "fourier"))
RECORDED = (("modular_addition", 0, "mlp", "answer", "sum and operands", "additive"),
            ("modular_addition", 0, "attention", "answer", "operands", "additive"),
            ("two_hop", 3, "attention", "query", "hop", "table"))
IDENTITY_WINDOWS = 64
BARS = {"held_accuracy": 0.99, "control_ratio": 2.0, "frequencies": 8, "logit_r2": 0.8, "kl": 0.25, "ablation": 1.0,
        "wrong": 1.0, "reads_agreement": 0.8, "answer_agreement": 0.1, "twin_within": 0.05}

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "phase7"
RUN_DIR = REPO_ROOT / "runs" / "phase7"
RECORD_DIR = RUN_DIR / "gestures"


def _load_phase6_common() -> Any:
    """gates/phase6/_common.py, under its own module name: this file is also called ``_common``."""
    name = "nablaframe_gates_phase6_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "phase6"
                                                      / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


p6 = _load_phase6_common()
p5, p4, p3, p2, p1 = p6.p5, p6.p4, p6.p3, p6.p2, p6.p1
ticking = p6.ticking
least, greatest = p6.least, p6.greatest
request_hashes = p6.request_hashes


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def configs() -> tuple[Any, Any]:
    from nablaframe.model.gpt2 import GPT2Config
    from nablaframe.train.loop import TrainConfig

    return GPT2Config(**MODEL), TrainConfig(**TRAINING)


def checkpoint_steps() -> list[int]:
    return list(range(0, 3001, 100)) + list(range(3500, 10001, 500))


def probe(task: Any) -> np.ndarray:
    held = task.held_out()
    return held[np.random.default_rng(EVAL_SEED).choice(len(held), PROBE_WINDOWS, replace=False)]


# ---------------------------------------------------------------------------
# The clock (G7.1)
# ---------------------------------------------------------------------------


def fewest_frequencies(embedding: Any, p: int) -> dict[str, Any]:
    shares = crystallize._shares(embedding, p)  # the shares crystallize.frequencies ranks, with its refusals
    ranked = sorted(range(len(shares)), key=lambda i: (-shares[i], i))  # largest first, ties to the lower k
    count, held = 0, 0.0
    for i in ranked:  # no cut at MAX_FREQUENCIES
        count += 1
        held += float(shares[i])
        if held >= crystallize.ENERGY:
            break
    return {"count": count, "held": float(sum(shares[i] for i in ranked[:count]))}


def _clock(logits: Any, windows: Any, p: int, k: Sequence[int], what: str) -> tuple[np.ndarray, np.ndarray]:
    """One set of pairs' design over every (pair, c) and its targets, the logits centred over c."""
    logits, windows = np.asarray(logits), np.asarray(windows)
    if logits.ndim != 2 or logits.shape[1] < p:
        raise ValueError(f"the {what} logits must be shaped (B, vocab) with at least p = {p} columns, "
                         f"got {logits.shape}")
    if windows.ndim != 2 or windows.shape[1] < 2:
        raise ValueError(f"the {what} windows must be shaped (B, 4), got {windows.shape}")
    if len(logits) != len(windows):
        raise ValueError(f"the {what} logits hold {len(logits)} pairs but their windows {len(windows)}")
    y = logits[:, :p].astype(np.float64)
    y = y - y.mean(axis=1, keepdims=True)
    a, b = windows[:, 0].astype(np.int64), windows[:, 1].astype(np.int64)
    d = ((a + b)[:, None] - np.arange(p)[None, :]) % p
    X = np.stack([np.cos(2 * np.pi * int(f) * d / p) for f in k], axis=-1)
    return X.reshape(-1, len(k)), y.reshape(-1)


def logit_r2(fit_logits: Any, fit_windows: Any, held_logits: Any, held_windows: Any, p: int,
             k: Sequence[int]) -> float | None:
    k = list(k)
    if not k:
        raise ValueError("logit_r2 needs at least one frequency")
    X_fit, y_fit = _clock(fit_logits, fit_windows, p, k, "fitting")
    X_held, y_held = _clock(held_logits, held_windows, p, k, "held-out")
    alpha = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]  # no constant: the targets are centred over c
    total = float(np.sum((y_held - y_held.mean()) ** 2))
    if total == 0:
        return None
    return float(1.0 - np.sum((y_held - X_held @ alpha) ** 2) / total)


# ---------------------------------------------------------------------------
# set's identities and the requests (G7.3)
# ---------------------------------------------------------------------------


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "array", value))


def set_identities(model: Any, params: Any, tokens: Any, *, n_layers: int) -> dict[str, bool]:
    tokens = np.ascontiguousarray(tokens, dtype=np.int32)
    taps = list(tap_names(n_layers))

    def run(spec: Any, store: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        logits, recorded = run_intervened(model, params, tokens, spec, n_layers=n_layers,
                                          resolve=store.__getitem__, taps=taps)
        return {"logits": _numpy(logits), **{tap: _numpy(value) for tap, value in recorded.items()}}

    plain = run(InterventionSpec(()), {})
    exact = {}
    for tap in editable_taps(n_layers):
        if tap.endswith(".attn.hook_z"):
            continue
        values = np.ascontiguousarray(plain[tap], dtype=np.float32)
        digest = array_hash(values)
        got = run(InterventionSpec((edit(tap, "all", "set", values=digest),)), {digest: values})
        exact[tap] = bool(got.keys() == plain.keys() and all(
            got[name].dtype == want.dtype and got[name].shape == want.shape and np.array_equal(got[name], want)
            for name, want in plain.items()))
    return exact


def requests(run: str, steps: Sequence[int], cites: Sequence[str]) -> list[Any]:
    instances = len(ModularAdditionTask().held_out())  # its held-out pairs, 8,938
    bases = [{"part": "write", "class": "answer", "function": "sum", "form": "table"},
             {"part": "write", "class": "answer", "function": "sum", "form": "fourier"},
             {"part": "mlp", "class": "answer", "function": "sum and operands", "form": "additive"},
             {"part": "attention", "class": "answer", "function": "operands", "form": "additive"}]
    steps, cites = list(steps), list(cites)
    return [records.GestureRequest("crystallize", run, step, {**base, "block": block, "cites": cited}, instances,
                                   seed)
            for base in bases for step in (steps[0], steps[-1]) for cited in ([], cites)
            for seed in (SEED, SEED + 1) for block in (0, 1)]


# ---------------------------------------------------------------------------
# Judgements (G7.4)
# ---------------------------------------------------------------------------


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def crystal_checks(results: Mapping[str, Any]) -> dict[str, Any]:
    rows, twin = results.get("checkpoints"), results.get("twin")
    if not rows:
        raise ValueError("crystal_checks needs the crystallization's checkpoint rows")
    if twin is None:
        raise ValueError("crystal_checks needs the crystallization's twin")
    meter, counterfactual = results["meter"], results.get("counterfactual")
    twin_kl = float(twin["meter"]["kl"])
    at_zero = [row for row in rows if row["step"] == 0]
    return {
        "kl": _float(meter["kl"]),
        "ablation": _float(meter["ablation"]),
        "wrong_min": least([w["kl"] for w in results["wrong"]]),
        "reads_agreement": None if counterfactual is None else _float(counterfactual["reads"]["agreement"]),
        "answer_agreement": None if counterfactual is None else _float(counterfactual["answer"]["agreement"]),
        "twin_within": greatest([abs(float(w["kl"]) - twin_kl) for w in twin["wrong"]]),
        "final_holds": rows[-1]["label"] == crystallize.HOLDS,
        "step0_outside": bool(at_zero) and all(row["label"] != crystallize.HOLDS for row in at_zero),
    }


def judged(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = list(checks)
    if not checks:
        raise ValueError("judged needs at least one crystallization's checks")

    def over(name: str, *, counterfactual: bool = False) -> list[Any]:
        return [c[name] for c in checks if not counterfactual or c[name] is not None]

    return {
        "kl_max": greatest(over("kl")),
        "ablation_min": least(over("ablation")),
        "wrong_min": least(over("wrong_min")),
        "reads_agreement_min": least(over("reads_agreement", counterfactual=True)),
        "answer_agreement_max": greatest(over("answer_agreement", counterfactual=True)),
        "twin_within_max": greatest(over("twin_within")),
        "envelope_ok": all(c["final_holds"] is True and c["step0_outside"] is True for c in checks),
    }


# ---------------------------------------------------------------------------
# Beyond the contract: what the Phase 7 gates share
# ---------------------------------------------------------------------------
#
# * The start checks on earlier reports (``require``, ``start``), the report paths, and Phase 7's working
#   folders (``fresh_dir``, under RUN_DIR).
# * The modular-addition run, as G7.1 writes it (``modadd_run``) and as G7.2 to G7.4 read it
#   (``modadd_model``): its checkpoints, twin, windows, declaration and functions. ``zoo_model`` is Phase 6's
#   for induction and two-hop, plus their functions and fitting windows.
# * G7.2's TopK SAE through Phase 5's cache (``dictionary``), and the zoo runs' Phase 5 TopK SAEs through
#   Phase 6's ``topk``.
# * Each run's ``cli.Context`` (``context``), with ``params_at`` announcing every checkpoint it loads, and the
#   records a crystallization cites (``folder_records``, ``citations``).
#
# Paths are read from this module's globals, and those of ``p6`` and the earlier phases, when a function runs,
# never bound at import, so everything here can be pointed at a scratch folder.

VERSIONS = p6.VERSIONS
RUN_NAME = "modular_addition"
CHUNK = crystallize.CHUNK  # sequences per pass when collecting activations, as crystallize-this reads its windows
REPORTS = {
    "G6.3": ("phase6", "g63_edges.json", "G6.3 wrote the zoo runs' edge records, which the crystallizations cite"),
    "G6.7": ("phase6", "g67_records_panel.json",
             "G6.7 wrote the zoo runs' name records, which the crystallizations cite"),
    "G7.1": ("phase7", "g71_specimen.json", "nothing is measured on a modular-addition specimen that missed its bars"),
    "G7.2": ("phase7", "g72_names.json", "G7.2 trained the dictionary and wrote the name records the gates cite"),
    "G7.3": ("phase7", "g73_determinism.json",
             "no crystallization is judged before set, the requests and the records pass their determinism checks"),
}

say, number, show = p6.say, p6.number, p6.show


def report_relative(name: str) -> str:
    """``reports/phase7/<name>``, as Phase 1's ``start`` takes it."""
    return str((Path(REPORT_DIR) / name).relative_to(p1.REPO_ROOT))


def require(gate_id: str) -> dict[str, Any]:
    """The report a Phase 7 gate builds on, once it exists, passes ``validate_report`` and says ``pass``."""
    if gate_id not in REPORTS:
        return p6.require(gate_id)  # G4.1, G5.2 and G5.3
    phase, name, why = REPORTS[gate_id]
    folder = p6.REPORT_DIR if phase == "phase6" else REPORT_DIR
    return p5._require(gate_id, Path(folder) / name, why=why)


def start(args: Any, report_name: str, needs: Sequence[str]) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Phase 1's start checks, then every report in ``needs``: ``(the report's path, {gate id: report})``."""
    report = p1.start(args, report_relative(report_name))
    return report, {gate_id: require(gate_id) for gate_id in needs}


def fresh_dir(path: Path, *, overwrite: bool) -> Path:
    """An empty working folder under RUN_DIR. An existing one is removed only with ``overwrite``."""
    import shutil

    if Path(RUN_DIR).resolve() not in Path(path).resolve().parents:
        raise ValueError(f"refusing to manage {path}: Phase 7 working folders live under {RUN_DIR}")
    if Path(path).exists():
        if not overwrite:
            raise SystemExit(f"{p1.rel(path)} exists. Pass --overwrite to start it over.")
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)
    return Path(path)


def run_dir() -> Path:
    """``runs/phase7/modular_addition/``: the specimen's run, as G7.1 writes it."""
    return Path(RUN_DIR) / RUN_NAME


def dictionary_dir() -> Path:
    """``runs/phase7/dictionaries/``: G7.2's dictionaries, through Phase 5's cache."""
    return Path(RUN_DIR) / "dictionaries"


def record_dir(run: str) -> Path:
    """``runs/phase7/gestures/<run>/``: a run's Phase 7 records."""
    return Path(RECORD_DIR) / run


def task() -> Any:
    """The specimen's task at its defaults (D54)."""
    return ModularAdditionTask()


def tokens(windows: Any) -> np.ndarray:
    """What the specimen reads of windows: each one's first ``seq`` tokens, int32."""
    return p4.tokens(task(), windows)


# ---------------------------------------------------------------------------
# The runs, as G7.2 to G7.4 read them
# ---------------------------------------------------------------------------


def modadd_run(g71: Mapping[str, Any]) -> Any:
    """G7.1's run: manifest, config, model, steps and a params loader, as Phase 4's ``zoo_run`` reads a zoo run.

    Checks, before anything is measured, that the manifest is the run G7.1's report names, that its model and
    schedule are this module's, and that every checkpoint's sidecar names it. Raises SystemExit otherwise.
    ``params_for(step)`` loads each checkpoint once.
    """
    import json
    from types import SimpleNamespace

    from nablaframe.model.gpt2 import GPT2Model, init_params
    from nablaframe.train.checkpoint import load_params
    from nablaframe.trace.schema import RunManifest

    folder = run_dir()
    path = folder / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{p1.rel(path)} is missing. Run G7.1 first. No report written.")
    manifest = RunManifest.from_json(path.read_text(encoding="utf-8"))
    named = g71["measurements"].get("run_id")
    if manifest.run_id != named:
        raise SystemExit(f"{p1.rel(path)} is run {manifest.run_id}, but G7.1's report names {named}. Run G7.1 again."
                         " No report written.")
    config, train_config = configs()
    if p1.config_from_manifest(manifest) != config or list(manifest.checkpoint_schedule) != checkpoint_steps():
        raise SystemExit(f"{p1.rel(path)} doesn't hold the specimen's model and checkpoint schedule. No report"
                         " written.")
    steps = tuple(int(s) for s in manifest.checkpoint_schedule)
    paths = {step: p4.checkpoint(folder / "checkpoints", manifest.run_id, step) for step in steps}
    like = init_params(config, 0)
    loaded: dict[int, Any] = {}

    def params_for(step: int) -> Any:
        if step not in paths:
            raise ValueError(f"step {step} is not one of the specimen's checkpoints")
        if step not in loaded:
            loaded[step] = load_params(paths[step], like=like)
        return loaded[step]

    def state_sha256(step: int) -> str:
        return json.loads(paths[step].with_suffix(".json").read_text(encoding="utf-8"))["state_sha256"]

    return SimpleNamespace(name=RUN_NAME, task=task(), manifest=manifest, run_id=manifest.run_id, config=config,
                           train_config=train_config, model=GPT2Model(config), steps=steps, final=steps[-1],
                           folder=folder, store=folder / "store", like=like, params_for=params_for,
                           state_sha256=state_sha256)


def modadd_model(g71: Mapping[str, Any]) -> Any:
    """G7.1's run at its final checkpoint, with its twin, as the Phase 7 gates read it, shaped as Phase 5's
    ``zoo_model``: its training pairs are ``fit`` and the windows its dictionary trains on, its held-out pairs
    ``held_out``."""
    from types import SimpleNamespace

    from nablaframe.gestures import cli
    from nablaframe.zoo.ground_truth import TWIN_SEED, twin_params

    run = modadd_run(g71)
    params = run.params_for(run.final)
    twin = twin_params(run.config)
    fit, held_out = run.task.training(), run.task.held_out()
    return SimpleNamespace(
        name=RUN_NAME, run=run, task=run.task, model=run.model, config=run.config, n_layers=run.config.n_layers,
        d_model=run.config.d_model, step=run.final, run_id=run.run_id, params=params, twin=twin,
        state=run.state_sha256(run.final), twin_state=p5.twin_state(twin), twin_seed=TWIN_SEED,
        declaration=cli.declaration_for(RUN_NAME, run.config), functions=cli.functions_for(RUN_NAME),
        fit=fit, held_out=held_out, held_tokens=tokens(held_out), train_tokens=lambda seed=p5.DICT_SEED: tokens(fit),
        chunk=CHUNK, train_seconds={})


def zoo_model(name: str, g41: Mapping[str, Any]) -> Any:
    """Phase 6's ``zoo_model``, G4.1's run at its final checkpoint with its twin and declaration, plus its
    ``functions`` and ``fit``, Phase 5's training windows (``zoo_windows``), which crystallize-this fits on."""
    from nablaframe.gestures import cli

    m = p6.zoo_model(name, dict(g41))
    m.functions = cli.functions_for(name)
    m.fit = p5.zoo_windows(m.task, name, "train")
    return m


# ---------------------------------------------------------------------------
# G7.2's dictionary
# ---------------------------------------------------------------------------


def dictionary_spec(m: Any, *, twin: bool = False) -> tuple[list[tuple[str, str]], Any, str]:
    """``(sites, settings, key)`` of G7.2's TopK SAE at block 0's ``hook_resid_post`` of the specimen (or its twin):
    Phase 5's settings with ``DICTIONARY``'s steps and batch, keyed by ``p5.dictionary_key`` of the model's state,
    the site, the settings and the tokens the model reads of the training pairs."""
    from nablaframe.dictionaries.core import Settings

    sites = p5.sites("topk", m.n_layers, 0)
    settings = Settings("topk", p5.WIDTH_FACTOR * m.d_model, p5.K, DICTIONARY["steps"], DICTIONARY["batch"],
                        lr=p5.LR, seed=p5.DICT_SEED, jac_batch=p5.JAC_BATCH)
    state = m.twin_state if twin else m.state
    return sites, settings, p5.dictionary_key(state, sites, settings, m.train_tokens())


def dictionary(m: Any, *, twin: bool = False, train: bool = False) -> tuple[Any, float | None]:
    """G7.2's TopK SAE of the specimen (or its twin) from the cache under ``dictionary_dir()``, and the seconds it
    took to train, None from the cache. Only G7.2 trains it (``train``); for the others a missing one stops the
    gate."""
    from nablaframe.dictionaries import core

    sites, settings, key = dictionary_spec(m, twin=twin)
    label = f"{'twin' if twin else 'trained'} TopK SAE after block 0"
    params = m.twin if twin else m.params

    def make() -> Any:
        if not train:
            raise SystemExit(f"{m.name}: the cache holds no {label}, key {key}, in"
                             f" {p1.rel(dictionary_dir() / key)}. Run G7.2 first. No report written.")
        return p5.train_dictionary(m.model, params, m.train_tokens(), settings, sites, key=key, chunk=m.chunk,
                                   label=f"{m.name}: training the {label}, key {key[:12]}")

    made, seconds = p5.cached(dictionary_dir(), key, make)
    say(f"{m.name}: {label} {'from the cache' if seconds is None else 'trained in ' + p1._duration(seconds)}, key"
        f" {key[:12]}, content {core.content_hash(made)[:12]}")
    return made, seconds


# ---------------------------------------------------------------------------
# Contexts and citations
# ---------------------------------------------------------------------------


def announced(params_at: Any, label: str, steps: Sequence[int]) -> Any:
    """``params_at``, printing a line each time a checkpoint is loaded: the request's checkpoint first, as
    ``cli.results`` takes it, then one line per checkpoint as crystallize-this refits across training."""
    steps = [int(s) for s in steps]
    calls = [0]

    def load(step: int) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            say(f"{label}: step {step:,}, the request's checkpoint: measuring on it and on the twin")
        else:
            at = steps.index(step) + 1 if step in steps else "?"
            say(f"{label}: across training, step {step:,}, checkpoint {at} of {len(steps)}")
        return params_at(step)

    return load


def records_reader(folder: Path) -> Any:
    """A context's ``cited``: the record a records folder holds under a key, ValueError for one it doesn't."""
    def cited(key: str) -> Any:
        return records.read_record(folder, key)

    return cited


def context(m: Any, *, steps: Sequence[int] | None = None, dictionaries: Sequence[Any] = (),
            cited: Path | None = None) -> Any:
    """The run's ``cli.Context``: its held-out windows, ``steps`` (every checkpoint by default), ``dictionaries``,
    and, where ``cited`` names the records folder its crystallizations cite, everything crystallize-this needs:
    the fitting windows, the run's functions, its twin and its state.

    ``params_at`` is the run's own loader, so the final checkpoint's params are ``m.params``, the object every
    reading the gate makes shares.
    """
    from nablaframe.gestures import cli

    run = m.run
    steps = tuple(int(s) for s in (run.steps if steps is None else steps))
    extra = {}
    if cited is not None:
        extra = {"fit_windows": m.fit, "functions": m.functions, "twin": m.twin, "twin_state": m.twin_state,
                 "cited": records_reader(Path(cited))}
    return cli.Context(run=m.run_id, declaration=m.declaration, model=m.model, n_layers=m.n_layers, steps=steps,
                       params_at=run.params_for, state_at=run.state_sha256, windows=m.held_out,
                       dictionary=p6.held(dictionaries), seeds={}, **extra)


def folder_records(folder: Path) -> list[Any]:
    """Every record a records folder holds, by its ``index.json``. SystemExit for a folder without one: the
    records a gate cites must have been written before it runs."""
    folder = Path(folder)
    if not (folder / records._INDEX).is_file():
        raise SystemExit(f"{p1.rel(folder)} holds no {records._INDEX}, so it holds no records to cite. No report"
                         " written.")
    return [records.read_record(folder, entry["key"]) for entry in records._read_index(folder)]


def citations(m: Any, held: Sequence[Any], function: str, dictionary_hash: str) -> list[str]:
    """The keys a crystallization of ``function`` cites: ``crystallize.cite`` of ``held``, every record of the
    folder its context reads, with the function's class, ``describes`` of the run's held-out windows, the
    dictionary at its part's output site, site 0, and the function's relation; names, then edges."""
    f = m.functions.functions[function]
    chosen = crystallize.cite(held, cls=f.cls,
                              descriptions=crystallize.describes(m.declaration, m.functions, function, m.held_out),
                              dictionary=dictionary_hash, site=0, relation=f.relation)
    return sorted(chosen["names"] + chosen["edges"])


def crystallize_request(m: Any, entry: Sequence[Any], cites: Sequence[str], *, step: int | None = None) -> Any:
    """A crystallize-this request for one of ``JUDGED`` or ``RECORDED`` on run ``m``: at ``step`` (the final
    checkpoint by default), every held-out window, ``SEED``, citing ``cites``."""
    _, block, part, cls, function, form = entry
    args = {"block": int(block), "part": part, "class": cls, "function": function, "form": form,
            "cites": list(cites)}
    return records.GestureRequest("crystallize", m.run_id, int(m.step if step is None else step), args,
                                  len(m.held_out), SEED)


def crystallization_name(entry: Sequence[Any]) -> str:
    """A crystallization as the reports name it, ``"<run> <part> <function> <form>"``."""
    run, _, part, _, function, form = entry
    return f"{run} {part} {function} {form}"
