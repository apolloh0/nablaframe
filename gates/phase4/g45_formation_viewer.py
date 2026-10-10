"""G4.5: the induction formation demo, and both zoo models' views.

Spec: docs/specs/phase-4.md, "The induction formation demo" and "Gates"; helpers in
``_common.py``. Report: ``reports/phase4/g45_formation_viewer.json``. Run in nf-jax, on the Mac,
after G4.1 passes.

* **Formation,** from G4.1's induction trace, what the scrubber shows: at every checkpoint, the
  mechanism positions' mean loss, from the traced logits against the full ``probe(task)``
  windows, and the best block-1 head's ``induction_scores`` of the traced
  ``blocks.1.attn.hook_pattern``. ``formation_loss_step`` is ``steepest_drop`` of the losses and
  ``formation_attention_step`` ``steepest_rise`` of the scores. Both curves are recorded.
* **Views:** each zoo model's views built with ``build_views``, with the task's vocabulary,
  ``next_token`` the probe windows' last tokens, and ``curves``, into
  ``runs/phase4/<name>/views/``, then again into a scratch folder, compared file by file:
  ``builds_identical``. For induction the curves are the formation curves, ``"mechanism
  positions' loss"`` in ``"nats"`` and ``"best block-1 head's attention to induction targets"``
  as a ``"fraction"``. For two-hop there is one curve, ``"answer position's loss"`` in
  ``"nats"``: at each checkpoint, the mean over the probe of ``-log
  softmax(logits[b, 25])[window[b, 26]]`` from the traced logits, in float64. ``external_urls``
  counts external references in the viewer as G2.6 does. Also recorded: ``curves``, each
  model's curve names, and ``answer_loss_by_step``.

Measurements: REQUIRED_MEASUREMENTS["G4.5"]. Pass needs ``builds_identical`` true and
``external_urls == 0``; the formation steps are recorded.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase4/g45_formation_viewer.py``. Takes --overwrite and --allow-dirty. It reads G4.1's
#   runs and their traces, after ``require_g41`` and ``zoo_run``'s checks; --overwrite also lets
#   it replace the build folders.
# * Each store must hold, at every checkpoint in G4.1's schedule, the logits and events recorded
#   on the probe's first ``seq`` tokens (``batch_hash``), or the gate exits 1 with no report.
# * Formation: at each checkpoint, the mean over the probe's windows and the mechanism positions
#   of ``-log softmax(logits[b, p])[probe[b, p + 1]]``, in float64, from the traced logits; and
#   the largest of block 1's heads' ``induction_scores`` at that checkpoint. Each head's own
#   curve is recorded beside it, as ``attention_by_head_by_step``.
# * Answer loss, for two-hop: at each checkpoint, the mean over the probe's windows of
#   ``-log softmax(logits[b, 25])[probe[b, 26]]``, in float64, from the traced logits.
# * Views: ``build_views`` of every traced step, with ``params_for`` loading G4.1's checkpoints,
#   the probe's first ``seq`` tokens (what the trace recorded), the run's ``n_layers`` and
#   ``run_id``, ``task.vocab``, ``next_token`` the probe windows' last tokens (the token after
#   the last position the trace recorded) and the model's curves, each point a checkpoint's
#   value (null if it isn't finite). No validation-loss curve and no J-lens: a zoo run has
#   neither at every step. The first build goes to ``runs/phase4/<name>/views/``, the second to
#   ``runs/phase4/scratch/views/<name>/``; both start empty. ``builds_identical`` holds when
#   both models' two builds hold the same files, by relative path, with the same bytes.
# * ``external_urls`` counts every match of ``EXTERNAL_URL``, copied from tests/test_viewer.py
#   as G2.6 copies it, in every file directly in viewer/; ``external_url_matches`` lists them.

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G4.5"
REPORT_NAME = "g45_formation_viewer.json"
PATTERN_TAP = "blocks.1.attn.hook_pattern"
ANSWER_POSITION = 25
# The timeline's curves (D38): each model's names and units, in the order the viewer draws them.
LOSS_CURVE = ("mechanism positions' loss", "nats")
ATTENTION_CURVE = ("best block-1 head's attention to induction targets", "fraction")
ANSWER_CURVE = ("answer position's loss", "nats")
# The offline rule's pattern, as tests/test_viewer.py states it.
EXTERNAL_URL = re.compile(r"https?://|src=[\"']//|import\s*\(\s*[\"']//")
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    runs = {name: _common.zoo_run(name, g41) for name in _common.NAMES}
    readers = {name: open_reader(run) for name, run in runs.items()}
    folders = {name: (run.folder / "views", _common.scratch_dir() / "views" / name) for name, run in runs.items()}
    for first, second in folders.values():
        for folder in (first, second):
            if folder.exists() and not args.overwrite:
                raise SystemExit(f"{p1.rel(folder)} exists. Pass --overwrite to start it over.")
    measurements = formation(runs["induction"], readers["induction"])
    measurements |= answer_loss(runs["two_hop"], readers["two_hop"])
    curves = {
        "induction": [curve(LOSS_CURVE, measurements["steps"], measurements["mechanism_loss_by_step"]),
                      curve(ATTENTION_CURVE, measurements["steps"], measurements["best_attention_by_step"])],
        "two_hop": [curve(ANSWER_CURVE, measurements["answer_steps"], measurements["answer_loss_by_step"])],
    }
    measurements["curves"] = {name: [c["name"] for c in model_curves] for name, model_curves in curves.items()}
    measurements |= views(runs, folders, curves, overwrite=args.overwrite)
    measurements |= offline(Path(p1.REPO_ROOT) / "viewer")
    measurements["run_ids"] = {name: run.run_id for name, run in runs.items()}
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G4.5 rule holds."""
    return "pass" if m["builds_identical"] is True and m["external_urls"] == 0 else "fail"


def probe(run: Any) -> np.ndarray:
    """The probe as the trace recorded it: the probe windows' first ``seq`` tokens."""
    return _common.tokens(run.task, _common.probe(run.task))


def next_token(run: Any) -> np.ndarray:
    """The probe windows' last tokens: what follows the last position the trace recorded."""
    return np.asarray(_common.probe(run.task)[:, -1], dtype=np.int64)


def open_reader(run: Any) -> Any:
    """A ``TraceReader`` on the run's store, once it holds every checkpoint's events and logits, on the probe."""
    from nablaframe.data.corpus import batch_hash
    from nablaframe.trace.registry import LOGITS
    from nablaframe.views.readouts import TraceReader

    root = run.store
    if not (root / "events.jsonl").exists():
        raise SystemExit(f"{p1.rel(root)} holds no trace store. Run G4.1 first. No report written.")
    reader = TraceReader(root)
    missing = [step for step in run.steps if step not in reader.steps()]
    if missing or tuple(reader.steps()) != run.steps:
        raise SystemExit(f"{p1.rel(root)} doesn't hold exactly G4.1's checkpoints (missing {missing}). Run G4.1"
                         " again. No report written.")
    expected = batch_hash(probe(run))
    for step in run.steps:
        try:
            reader.event(step, LOGITS)
        except KeyError:
            raise SystemExit(f"{p1.rel(root)} has no logits at step {step}. Run G4.1 again. No report written.")
        if reader.batch_hash(step) != expected:
            raise SystemExit(f"step {step} in {p1.rel(root)} was traced on a different batch than the probe."
                             " No report written.")
    return reader


def formation(run: Any, reader: Any) -> dict[str, Any]:
    """The scrubber's two curves: the mechanism positions' loss and the best block-1 head's attention."""
    from nablaframe.zoo.induction import induction_scores, steepest_drop, steepest_rise

    task = run.task
    windows = _common.probe(task)
    positions = task.mechanism_positions
    losses, best, by_head = [], [], []
    progress = p1.Progress("formation, checkpoints", len(run.steps))
    for step in run.steps:
        logits = reader.tensor(step, "logits").astype(np.float64)[:, positions]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        targets = windows[:, positions + 1]
        losses.append(float(-np.take_along_axis(log_probs, targets[..., None].astype(np.intp), axis=-1).mean()))
        scores = induction_scores(reader.tensor(step, PATTERN_TAP), windows, positions)
        by_head.append([float(s) for s in scores])
        best.append(float(scores.max()))
        progress.advance()
    loss_step, attention_step = steepest_drop(run.steps, losses), steepest_rise(run.steps, best)
    print(f"{GATE_ID}: the mechanism positions' loss drops most steeply at step {loss_step:,}; the best block-1 head's"
          f" attention to its targets rises most steeply at step {attention_step:,}", flush=True)
    return {
        "formation_loss_step": loss_step,
        "formation_attention_step": attention_step,
        "steps": list(run.steps),
        "mechanism_loss_by_step": losses,
        "best_attention_by_step": best,
        "attention_by_head_by_step": by_head,
        "probe_windows": int(len(windows)),
        "probe_sha256": _batch_hash(windows),
    }


def answer_loss(run: Any, reader: Any) -> dict[str, Any]:
    """Two-hop's timeline curve: the answer position's mean loss at every checkpoint, from the traced logits."""
    windows = _common.probe(run.task)
    targets = windows[:, ANSWER_POSITION + 1].astype(np.intp)
    losses = []
    progress = p1.Progress("answer loss, checkpoints", len(run.steps))
    for step in run.steps:
        logits = reader.tensor(step, "logits").astype(np.float64)[:, ANSWER_POSITION]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        losses.append(float(-np.take_along_axis(log_probs, targets[:, None], axis=-1).mean()))
        progress.advance()
    print(f"{GATE_ID}: two-hop's answer position's loss goes from {losses[0]:.4f} at step {run.steps[0]:,} to"
          f" {losses[-1]:.4f} at step {run.steps[-1]:,}", flush=True)
    return {"answer_steps": list(run.steps), "answer_loss_by_step": losses}


def curve(named: tuple[str, str], steps: list[int], values: list[float]) -> dict[str, Any]:
    """A timeline curve as ``build_views`` takes it: a value per checkpoint, null where it isn't finite."""
    name, unit = named
    return {"name": name, "unit": unit,
            "points": [{"step": int(step), "value": p1.finite_or_none(value)} for step, value in zip(steps, values)]}


def views(runs: dict[str, Any], folders: dict[str, tuple[Path, Path]], curves: dict[str, list[dict[str, Any]]], *,
          overwrite: bool) -> dict[str, Any]:
    """Each zoo model's views built twice, with its curves and next tokens, compared file by file."""
    by_model = {}
    for name, run in runs.items():
        first, second = folders[name]
        written = []
        for label, out in (("first build", first), ("second build", second)):
            _common.fresh_dir(out, overwrite=overwrite)
            written.append(build(run, out, label, curves[name]))
        comparison = _common.p3.compare_trees(first, second)
        print(f"{GATE_ID}: {name}: the two builds are {'byte-identical' if comparison['identical'] else 'different'}"
              f" over {comparison['files_compared']:,} files", flush=True)
        by_model[name] = {**comparison, "files_written": len(written[0]), "export_bytes": _bytes(written[0]),
                          "views_dir": p1.rel(first), "second_build_dir": p1.rel(second)}
    return {"builds_identical": all(m["identical"] for m in by_model.values()), "views": by_model}


def build(run: Any, out: Path, label: str, curves: list[dict[str, Any]]) -> list[Path]:
    """One ``build_views`` of a zoo run into ``out``, with a fresh reader and a progress line."""
    from nablaframe.views.build import build_views
    from nablaframe.views.readouts import TraceReader

    reader = TraceReader(run.store)
    progress = p1.Progress(f"views, {run.name}, {label}, steps", len(reader.steps()))
    paths = build_views(reader, out, params_for=run.params_for, probe=probe(run), n_layers=run.config.n_layers,
                        run_id=run.run_id, vocab=run.task.vocab, next_token=next_token(run), curves=curves,
                        progress=lambda done, total: progress.advance())
    print(f"{GATE_ID}: {run.name}, {label}: {len(paths):,} files written to {p1.rel(out)}", flush=True)
    return paths


def offline(viewer: Path) -> dict[str, Any]:
    """Matches of the offline rule's pattern in the viewer's files, as G2.6 counts them."""
    matches = []
    for path in sorted(Path(viewer).glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            matches += [{"file": path.name, "match": match.group(0)} for match in EXTERNAL_URL.finditer(text)]
    files = sorted(path.name for path in Path(viewer).glob("*") if path.is_file())
    print(f"{GATE_ID}: {len(matches)} external references in {', '.join(files) or 'no files'}", flush=True)
    return {"external_urls": len(matches), "external_url_matches": matches, "viewer_files": files}


def _bytes(paths: list[Path]) -> int:
    return sum(Path(path).stat().st_size for path in paths)


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
