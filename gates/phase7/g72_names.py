"""G7.2: the modular-addition specimen's dictionary and its names at ``answer``, recorded, not judged (C11, D57).

Spec: docs/specs/phase-7.md, "The names on modular addition"; helpers in ``_common.py``. Report:
``reports/phase7/g72_names.json``. Run in nf-jax, on the Mac, after G7.1 passes.

On G7.1's run at its final checkpoint, and on its twin:

* **The dictionary:** a TopK SAE at block 0's ``hook_resid_post`` with Phase 5's settings and
  ``DICTIONARY``'s steps and batch: ``core.Settings("topk", p5.WIDTH_FACTOR * 32, p5.K,
  DICTIONARY["steps"], DICTIONARY["batch"], lr=p5.LR, seed=p5.DICT_SEED,
  jac_batch=p5.JAC_BATCH)``, width 256 and ``k`` 16. ``p5.settings`` takes only Phase 5's runs,
  so the gate builds it. Trained on the activations of the training pairs, every position,
  through ``p5.train_dictionary`` and ``p5.cached`` into ``runs/phase7/dictionaries/``, under
  ``p5.dictionary_key`` of the run's final state, the site, the settings and the tokens the model
  reads of the training pairs. The twin's is trained the same way on the twin's
  activations, keyed by ``p5.twin_state``. ``fvu`` and ``twin_fvu``: each on the held-out pairs'
  activations, as Phase 5 measures it. ``meters``: ``{"fvu", "twin_fvu"}``, the two again, the
  mapping modular addition's features payload cites (D38, docs/deltas/phase-7-delta-4.md).
* **Names:** for each of ``NAMES`` at class ``answer`` on the held-out pairs, Phase 6's
  ``planted_name`` with ``INSTANCES["name"]`` and ``SEEDS["name"]``, trained and twin.
  ``names``: ``{description: {"trained": ..., "twin": ...}}``, each that mapping without its
  ``by_value``. This is what C11 rules on: the spec expects the sum named as planted for about a
  quarter of its values.
* **Records:** for each of ``NAMES`` and each of its values at ``answer``, a name record of that
  value's best latent by F1, ``ranked(d, v)[0]``, at the final checkpoint, with
  ``INSTANCES["name"]`` and ``SEEDS["name"]``, through ``cli.record`` into
  ``runs/phase7/gestures/modular_addition/``. The run's ``cli.Context`` holds G7.1's checkpoints,
  the held-out pairs and the trained dictionary. A latent best for two values is recorded once,
  and the twin, which isn't a checkpoint of its run, gets no records. G7.4 cites them, and G7.5 builds them again.
  ``records``: how many were written or found.

Measurements: REQUIRED_MEASUREMENTS["G7.2"]. Its status is always ``pass``: it records, not judges.
``seconds`` holds each stage's time.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase7/g72_names.py``.
#   Takes --overwrite and --allow-dirty. It reads G7.1's run after ``require`` of G7.1 and ``modadd_run``'s
#   checks. It writes its dictionaries, through the cache, and its records, nothing else but its report.
# * Each dictionary's activations are collected in chunks of ``_common.CHUNK`` pairs, every position, as Phase 5
#   collects them. ``fvu`` and ``twin_fvu`` are ``measure.fvu`` over every position of the held-out pairs.
# * Each model's site is read once over the held-out pairs (``sets.read_site`` at ``cli.READ_CHUNK``, through
#   Phase 6's ``read``); the three descriptions are named on that one reading.
# * The records: one ``cli.Context`` of the run's checkpoints, the held-out pairs and the trained dictionary. The
#   best latents are gathered over every description and value first, so a latent best for several is one
#   request. ``record_latents``: ``{description: {value: latent}}``; ``record_keys``: each latent's record key.

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1, p5, p6 = _common.p1, _common.p5, _common.p6

GATE_ID = "G7.2"
REPORT_NAME = "g72_names.json"
NEEDS = ("G7.1",)
CLASS = "answer"


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, reports = _common.start(args, REPORT_NAME, NEEDS)
    measurements = measure(reports["G7.1"])
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, "pass", measurements, report, overwrite=args.overwrite)


def measure(g71: dict[str, Any]) -> dict[str, Any]:
    from nablaframe.dictionaries import core

    m = _common.modadd_model(g71)
    _common.say(f"{GATE_ID}: run {m.run_id}, final step {m.step:,}; {len(m.fit):,} training and {len(m.held_out):,}"
                " held-out pairs")
    seconds: dict[str, float | None] = {}

    trained, seconds["dictionary"] = _common.dictionary(m, train=True)
    twin, seconds["twin_dictionary"] = _common.dictionary(m, twin=True, train=True)

    started = time.monotonic()
    tap = trained.sites[0][0]
    fvu = site_fvu(m, m.params, trained, tap, "trained")
    twin_fvu = site_fvu(m, m.twin, twin, tap, "twin")
    seconds["fvu"] = time.monotonic() - started

    started = time.monotonic()
    readings = {"trained": _common.p6.read(m, m.params, trained, f"{GATE_ID}: reading the trained model's site"),
                "twin": _common.p6.read(m, m.twin, twin, f"{GATE_ID}: reading the twin's site")}
    names: dict[str, Any] = {}
    for d in _common.NAMES:
        names[d] = {}
        for who, reading in readings.items():
            with _common.ticking(f"{GATE_ID}: {d} at {CLASS}, {who}: naming and steering every value",
                                 write=_common.say):
                named = _common.p6.planted_name(reading, CLASS, d, instances=_common.p6.INSTANCES["name"],
                                                seed=_common.p6.SEEDS["name"])
            named.pop("by_value")
            names[d][who] = named
            _common.say(f"{GATE_ID}: {d} at {CLASS}, {who}: {named['values']} values, named {named['named']:.3f}, as"
                        f" planted {named['named_as_planted']:.3f}; follows"
                        f" {_common.show(named['steering']['follows'])},"
                        f" null {_common.show(named['steering']['null_follows'])}")
    seconds["names"] = time.monotonic() - started

    started = time.monotonic()
    latents, keys = write_records(m, trained, readings["trained"])
    seconds["records"] = time.monotonic() - started

    out = {
        "fvu": fvu,
        "twin_fvu": twin_fvu,
        "meters": {"fvu": fvu, "twin_fvu": twin_fvu},
        "names": names,
        "records": len(keys),
        "seconds": seconds,
        "record_latents": latents,
        "record_keys": keys,
        "record_dir": p1.rel(_common.record_dir(_common.RUN_NAME)),
        "dictionaries": {"trained": core.content_hash(trained), "twin": core.content_hash(twin)},
        "dictionary_keys": {"trained": trained.key, "twin": twin.key},
        "instances": _common.p6.INSTANCES["name"],
        "seed": _common.p6.SEEDS["name"],
        "class": CLASS,
        "run_id": m.run_id,
        "step": m.step,
        "twin_seed": m.twin_seed,
    }
    _common.say(f"{GATE_ID}: FVU {fvu:.4f} (twin {twin_fvu:.4f}); the sum named as planted for"
                f" {names['token(p+1)']['trained']['named_as_planted']:.3f} of its values; {len(keys)} name records")
    return out


def site_fvu(m: Any, params: Any, dictionary: Any, tap: str, who: str) -> float:
    """``measure.fvu`` of the dictionary's site over every position of the held-out pairs' activations."""
    from nablaframe.dictionaries import measure

    acts = p5.activations(m, params, m.held_tokens, [tap], f"{GATE_ID}: the {who} model's held-out activations")
    x = p5.rows(acts[tap])
    return float(measure.fvu(dictionary, 0, x, x))


def write_records(m: Any, dictionary: Any, reading: Any) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """A name record of each value's best latent by F1 at ``answer``, for each of ``NAMES``, each latent once."""
    from nablaframe.dictionaries import core
    from nablaframe.gestures import cli
    from nablaframe.gestures.records import GestureRequest

    table = reading.table(CLASS)
    latents: dict[str, dict[str, int]] = {}
    for d in _common.NAMES:
        latents[d] = {str(int(v)): int(table.ranked(d, int(v))[0]) for v in table.values(d)}
    distinct = sorted({latent for by_value in latents.values() for latent in by_value.values()})
    ctx = _common.context(m, dictionaries=[dictionary])
    content = core.content_hash(dictionary)
    folder = _common.record_dir(_common.RUN_NAME)
    _common.say(f"{GATE_ID}: {sum(len(v) for v in latents.values())} values' best latents, {len(distinct)} distinct:"
                f" one name record each, into {p1.rel(folder)}")
    keys: dict[str, str] = {}
    progress = p1.Progress(f"{GATE_ID}: name records", len(distinct), write=_common.say)
    with _common.ticking(f"{GATE_ID}: writing the name records", write=_common.say):
        for latent in distinct:
            request = GestureRequest("name", m.run_id, m.step,
                                     {"dictionary": content, "site": 0, "class": CLASS, "latent": latent},
                                     _common.p6.INSTANCES["name"], _common.p6.SEEDS["name"])
            keys[str(latent)] = cli.record(folder, request, ctx).key
            progress.advance()
    return latents, keys


if __name__ == "__main__":
    raise SystemExit(main())
