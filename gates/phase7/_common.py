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
* ``EVAL_SEED = 2``, Phase 4's, and ``PROBE_WINDOWS = 8``. ``probe(task)``: the held-out windows
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

MODEL = {"vocab": 114, "d_model": 32, "n_layers": 1, "n_heads": 4, "d_mlp": 128, "seq": 3}
TRAINING = {"batch": 512, "seq": 3, "lr": 1e-3, "warmup_steps": 10, "total_steps": 10000, "min_lr_ratio": 0.1,
            "weight_decay": 1.0, "data_seed": 0, "param_seed": 0}
EVAL_SEED, PROBE_WINDOWS = 2, 8
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


def configs() -> tuple[Any, Any]:
    raise NotImplementedError


def checkpoint_steps() -> list[int]:
    raise NotImplementedError


def probe(task: Any) -> Any:
    raise NotImplementedError


def fewest_frequencies(embedding: Any, p: int) -> dict[str, Any]:
    raise NotImplementedError


def logit_r2(fit_logits: Any, fit_windows: Any, held_logits: Any, held_windows: Any, p: int,
             k: Sequence[int]) -> float | None:
    raise NotImplementedError


def set_identities(model: Any, params: Any, tokens: Any, *, n_layers: int) -> dict[str, bool]:
    raise NotImplementedError


def requests(run: str, steps: Sequence[int], cites: Sequence[str]) -> list[Any]:
    raise NotImplementedError


def crystal_checks(results: Mapping[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def judged(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError
