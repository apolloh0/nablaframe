"""G6.6: the gestures on the reference run, at its final checkpoint (recorded; D44).

Spec: docs/specs/phase-6.md, D44 and "The reference run"; helpers in ``_common.py``. Report:
``reports/phase6/g66_reference.json``. Run in nf-jax, on the Mac, after G4.1, G5.5 and G6.1 pass,
with G1.4's run and the reference corpus in place.

On G1.4's run at its final checkpoint, with the Phase 5 TopK SAE at every block from the cache, as
G5.5 trained them, each read over the reference run's held-out windows with
``cli.declaration_for("reference", config)``, whose one class is ``every``. For each block ``l``,
with ``T = reading.table("every")`` and ``rng = numpy.random.default_rng([SEEDS["reference"],
l])``:

* **Names:** ``reference_sample(T, REFERENCE["latents"], rng)``, and each latent's top candidate,
  steps 1 and 2 of D49, named where its F1 is at least ``sets.FLOOR``. ``"names"``: ``{"width",
  "active", "sampled", "named", "by_description", "coverage_median", "seconds"}``: the site's
  active width; how many latents are active on some row; how many were sampled; the share of them
  named; for each description that counts at ``every``, the share of them named as it; the median,
  over the named ones, of ``name.coverage(reading, [latent], value)`` at the name's own
  description; and the seconds it took.
* **Steering:** ``reference_values(T, "token(p+1)", REFERENCE["values"], rng)``, the same
  ``rng`` drawing after the sample, and each value steered with ``name.steer(reading, "every",
  "token(p+1)", v, instances=REFERENCE["per_value"], rng=numpy.random.default_rng(SEEDS["reference"]))``.
  ``"steering"``: ``{"values_named", "values", "by_value", "pooled", "seconds"}``: how many values
  some latent names, before the cut; the values steered; each one's steering; ``pool`` of them;
  the seconds.
* **Open-this-up,** at blocks 1 and above: ``OpenCandidate(window, position,
  T.set_for("token(p+1)", v), None, None)`` at every held-out window and position, window by window
  and positions ascending, where ``token(p+1)`` is a steered value ``v`` and that set's codes there
  sum above zero; then ``open_targets`` of the reading at block ``l - 1``, the reading at ``l`` and
  those candidates, with ``instances=REFERENCE["open"]`` and ``seed=SEEDS["reference"]``.
  ``"open"``: its result and ``"seconds"``; None at block 0.

``blocks``: ``{"<l>": {"names", "steering", "open"}}``. ``seconds``: ``{"total"}`` and each
block's. Measurements: REQUIRED_MEASUREMENTS["G6.6"]. Its status is always ``pass``: it records,
not judges. Round 3 named 30.1%, 27.0%, 18.4% and 8.6% of sampled latents at blocks 0 to 3, mostly
as ``token(p)``, at a median coverage of 0.72 to 0.84; steered 8 to 11 named ``token(p+1)``
values, following 3.2% to 22.7%; and opened their sets within 128 parts for 25%, 45% and 91% of
instances at blocks 1 to 3. Its own definitions differ in places (``name.steer`` draws ``w`` from
every value, not only named ones), so its readings needn't match.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase6/g66_reference.py``.
#   Takes --overwrite and --allow-dirty. It reads G1.4's run through Phase 5's ``reference_model`` after
#   ``require`` of G4.1, G5.5 and G6.1; every dictionary comes from the Phase 5 cache (``_common.topk``). It
#   writes nothing but its report.
# * Each block's site is read once over the held-out windows, at the final checkpoint's params, and kept
#   for the next block's open-this-up, which reads it as the upstream site.
# * ``"names"``: a sampled latent is named where its top candidate's F1 is at least ``sets.FLOOR``.
#   ``"named"`` and each ``"by_description"`` share are over the sampled latents, None with none sampled;
#   ``"coverage_median"`` None with none named.
# * ``"steering"``: ``values_named`` counts the values of ``token(p+1)`` whose best latent's F1 is at least
#   ``sets.FLOOR``, ``reference_values``'s rule before its cut; ``by_value`` is ``[{"value", "steering"}]``,
#   ascending, each ``name.steer``'s result (None where it finds no instance).
# * ``"seconds"`` of each part is wall-clock time from ``time.perf_counter``; a block's in ``seconds`` covers
#   its reading as well.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G6.6"
REPORT_NAME = "g66_reference.json"
NEEDS = ("G4.1", "G5.5", "G6.1")
DESCRIPTION = "token(p+1)"
CLASS = "every"


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, _ = _common.start(args, REPORT_NAME, NEEDS)
    m = _common.reference_model()
    measurements = measure(m)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """Always pass: G6.6 records, it doesn't judge (D44)."""
    return "pass"


def measure(m: Any) -> dict[str, Any]:
    start = time.perf_counter()
    _common.say(f"{GATE_ID}: reference run {m.run_id}, final step {m.step:,}; {len(m.held_out)} held-out windows of"
                f" {m.config.seq} tokens")
    dictionaries = [_common.topk(m, l) for l in range(m.n_layers)]
    blocks: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    below = None
    for l, dictionary in enumerate(dictionaries):
        began = time.perf_counter()
        reading = _common.read(m, m.params, dictionary, f"{GATE_ID}: block {l}: reading the site")
        table = reading.table(CLASS)
        rng = np.random.default_rng([_common.SEEDS["reference"], l])
        names = naming(reading, table, rng, l)
        steering, values = steered(reading, table, rng, l)
        opened = None if below is None else opening(below, reading, table, values, l)
        blocks[str(l)] = {"names": names, "steering": steering, "open": opened}
        seconds[str(l)] = time.perf_counter() - began
        below = reading
    seconds["total"] = time.perf_counter() - start
    return {
        "blocks": blocks,
        "seconds": seconds,
        "settings": _common.REFERENCE,
        "seed": _common.SEEDS["reference"],
        "run_id": m.run_id,
        "step": m.step,
        "held_out_windows": int(len(m.held_out)),
        "held_out_sha256": _batch_hash(m.held_out),
        "dictionaries": [_common.core.content_hash(d) for d in dictionaries],
    }


def naming(reading: Any, table: Any, rng: np.random.Generator, l: int) -> dict[str, Any]:
    """Steps 1 and 2 of D49 for ``REFERENCE["latents"]`` latents drawn with ``rng``, and the named ones' coverage."""
    from nablaframe.gestures import name, sets

    began = time.perf_counter()
    sample = _common.reference_sample(table, _common.REFERENCE["latents"], rng)
    named: list[tuple[int, str, int]] = []
    with _common.ticking(f"{GATE_ID}: block {l}: naming {len(sample)} latents", write=_common.say):
        for latent in sample:
            found = table.candidates(latent)
            if found and found[0][2] >= sets.FLOOR:
                named.append((latent, found[0][0], int(found[0][1])))
        coverage = [name.coverage(reading, [latent], value)[d] for latent, d, value in named]
    coverage = [c for c in coverage if c is not None]
    share = (lambda n: float(n) / len(sample)) if sample else (lambda n: None)
    out = {"width": int(reading.width), "active": int(np.count_nonzero(np.asarray(table.active) > 0)),
           "sampled": len(sample), "named": share(len(named)),
           "by_description": {d: share(sum(1 for _, nd, _ in named if nd == d)) for d in table.descriptions},
           "coverage_median": float(np.median(coverage)) if coverage else None,
           "seconds": time.perf_counter() - began}
    _common.say(f"{GATE_ID}: block {l}: {out['active']} of {out['width']} latents active, {out['sampled']} sampled,"
                f" named {_common.show(out['named'])}; coverage median {_common.show(out['coverage_median'])}")
    return out


def steered(reading: Any, table: Any, rng: np.random.Generator, l: int) -> tuple[dict[str, Any], list[int]]:
    """D49's steering for up to ``REFERENCE["values"]`` named values of ``token(p+1)``, ``rng`` drawing after the sample."""
    from nablaframe.gestures import name, sets

    began = time.perf_counter()
    values_named = int(np.count_nonzero(table.f1(DESCRIPTION).max(axis=0) >= sets.FLOOR))
    values = _common.reference_values(table, DESCRIPTION, _common.REFERENCE["values"], rng)
    by_value = []
    progress = p1.Progress(f"{GATE_ID}: block {l}: steering", len(values))
    with _common.ticking(f"{GATE_ID}: block {l}: steering {len(values)} values", write=_common.say):
        for v in values:
            by_value.append({"value": v, "steering": name.steer(
                reading, CLASS, DESCRIPTION, v, instances=_common.REFERENCE["per_value"],
                rng=np.random.default_rng(_common.SEEDS["reference"]))})
            progress.advance()
    pooled = _common.pool([entry["steering"] for entry in by_value])
    out = {"values_named": values_named, "values": values, "by_value": by_value, "pooled": pooled,
           "seconds": time.perf_counter() - began}
    _common.say(f"{GATE_ID}: block {l}: {values_named} values named, {len(values)} steered; follows"
                f" {_common.show(pooled['follows'])}, null {_common.show(pooled['null_follows'])}")
    return out, values


def opening(up: Any, down: Any, table: Any, values: list[int], l: int) -> dict[str, Any]:
    """Open-this-up of the steered values' sets at block ``l``, into block ``l - 1``, wherever each set fires."""
    began = time.perf_counter()
    held = down.described[DESCRIPTION]
    sets_for = {v: table.set_for(DESCRIPTION, v) for v in values}
    candidates = []
    for i in range(len(down.windows)):
        for p in range(down.codes.shape[1]):
            v = int(held[i, p])
            if v in sets_for and down.codes[i, p, list(sets_for[v])].astype(np.float64).sum() > 0:
                candidates.append(_common.OpenCandidate(i, p, sets_for[v], None, None))
    with _common.ticking(f"{GATE_ID}: block {l}: opening {len(candidates)} candidates into block {l - 1}",
                         write=_common.say):
        out = _common.open_targets(up, down, candidates, instances=_common.REFERENCE["open"],
                                   seed=_common.SEEDS["reference"])
    out["seconds"] = time.perf_counter() - began
    _common.say(f"{GATE_ID}: block {l}: opened {out['measured']} of {out['instances']} instances; within 128"
                f" {_common.show(out['orders']['absolute']['within'])}")
    return out


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
