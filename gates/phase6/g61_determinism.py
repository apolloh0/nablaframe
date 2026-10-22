"""G6.1: ``dsteer`` keeps D48's identities at every TopK site, requests hash apart, and each kind of record rebuilds byte for byte (D47, D48).

Spec: docs/specs/phase-6.md, D47, D48 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase6/g61_determinism.json``. Run in nf-jax, on the Mac, after G4.1, G5.2 and G5.3 pass.

On G4.1's runs at their final checkpoints, with the Phase 5 TopK SAEs from the cache:

* **Identities:** for both zoo models, at every block's TopK SAE, ``dsteer_identities`` on the
  tokens of the first ``IDENTITY_WINDOWS`` held-out windows. ``identities_by_site``:
  ``{"<run> topk <output tap>": [one bool per identity, in IDENTITIES order]}``;
  ``identities_checked`` and ``identities_exact``, counts over both models.
* **Request hashes:** on the two-hop run, ``request_hashes(requests(run id, its checkpoint
  steps, (block 2's TopK content hash, block 3's)))``: ``request_hashes_checked``,
  ``request_hashes_distinct`` and ``request_hashes_stable``.
* **Records:** on the two-hop run's ``cli.Context``: its held-out windows, every checkpoint, the
  TopK SAE at every block, and ``ROBUST_SEEDS``'s TopK SAEs at every block as its seeds. One
  request of each kind, at the final checkpoint, with ``RECORD_INSTANCES`` instances and seed 0,
  ``v`` being the least value of ``b`` at block 2's TopK SAE at ``query``:

  - **name:** block 2's TopK SAE, site 0, class ``query``, the latent ``ranked("b", v)[0]``
  - **edge:** ``hop``, from block 2's TopK SAE to block 3's, sites 0
  - **open:** block 2's TopK SAE at class ``query``, the name ``b = v``, into block 1's
  - **generalize:** ``b`` at block 2's TopK SAE, class ``query``

  Each is built twice with ``cli.build``, never through a cache, and once more by the command
  line's code: ``cli.main``, called in this process, with ``--run two_hop``, files written under
  ``runs/phase6/g61_determinism/`` (the config, the held-out windows, the seeds' map, the
  request, and a fresh ``--out``), G4.1's checkpoints folder and the Phase 5 dictionary cache.
  ``records_by_kind``: ``{kind: [the two builds' canonical JSON is identical, the command line's
  file is that and a newline]}``. ``records_identical``: all four kinds checked, every value true.

Measurements: REQUIRED_MEASUREMENTS["G6.1"]. Pass needs at least one identity checked and every one
exact, the request hashes distinct and stable, and ``records_identical``. The working folder,
``runs/phase6/g61_determinism/``, is started fresh, and ``--overwrite`` lets it replace one.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase6/g61_determinism.py``. Takes --overwrite and --allow-dirty. It reads G4.1's runs after
#   ``require`` of G4.1, G5.2 and G5.3 and ``zoo_run``'s checks; every dictionary comes from the Phase 5
#   cache (``_common.topk``). It writes nothing outside its report but its working folder.
# * ``identities_by_site`` names each site ``"<run> topk <output tap>"``, ``<run>`` the run's name.
# * The records: one ``cli.Context`` of the two-hop run, ``seeds`` mapping each block's seed-0 TopK SAE
#   to each robustness seed's. The two in-process builds and the command line's share nothing but their
#   inputs: each ``cli.build`` reads every site again, and ``cli.main`` loads the checkpoints and the
#   dictionaries from their files. Each command line request writes to ``records/``, which starts empty, so
#   no record comes from a cache. ``records_by_kind``: ``{kind: [builds identical, command line
#   identical]}``, and ``record_keys``: ``{kind: key}`` beside it.

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1, p4 = _common.p1, _common.p4

GATE_ID = "G6.1"
REPORT_NAME = "g61_determinism.json"
NEEDS = ("G4.1", "G5.2", "G5.3")
WORK_NAME = "g61_determinism"
KINDS = ("name", "edge", "open", "generalize")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report, g41 = _common.start(args, REPORT_NAME, NEEDS)
    work = _common.fresh_dir(Path(_common.RUN_DIR) / WORK_NAME, overwrite=args.overwrite)
    measurements = measure(g41, work)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G6.1 rule holds."""
    n = _common.number
    ok = (n(m["identities_checked"]) and m["identities_checked"] >= 1
          and n(m["identities_exact"]) and m["identities_exact"] == m["identities_checked"]
          and m["request_hashes_distinct"] is True and m["request_hashes_stable"] is True
          and m["records_identical"] is True)
    return "pass" if ok else "fail"


def measure(g41: dict[str, Any], work: Path) -> dict[str, Any]:
    models = {name: _common.zoo_model(name, g41) for name in _common.ZOO}
    dictionaries = {name: [_common.topk(m, l) for l in range(m.n_layers)] for name, m in models.items()}

    by_site = identities(models, dictionaries)
    flags = [flag for site in by_site.values() for flag in site]

    m = models["two_hop"]
    hashes = [_common.core.content_hash(d) for d in dictionaries["two_hop"]]
    reqs = _common.requests(m.run_id, m.run.steps, (hashes[2], hashes[3]))
    request_hashes = _common.request_hashes(reqs)
    _common.say(f"{GATE_ID}: {request_hashes['checked']} requests, hashes distinct {request_hashes['distinct']},"
                f" stable {request_hashes['stable']}")

    by_kind, keys = rebuilt_records(m, dictionaries["two_hop"], work)
    out = {
        "identities_checked": len(flags),
        "identities_exact": sum(bool(flag) for flag in flags),
        "identities_by_site": by_site,
        "identities": list(_common.IDENTITIES),
        "identity_windows": _common.IDENTITY_WINDOWS,
        "request_hashes_checked": request_hashes["checked"],
        "request_hashes_distinct": request_hashes["distinct"],
        "request_hashes_stable": request_hashes["stable"],
        "records_identical": len(by_kind) == len(KINDS) and all(all(v) for v in by_kind.values()),
        "records_by_kind": by_kind,
        "record_keys": keys,
        "record_instances": _common.RECORD_INSTANCES,
        "run_ids": {name: m.run_id for name, m in models.items()},
        "steps": {name: m.step for name, m in models.items()},
        "dictionaries": {name: [_common.core.content_hash(d) for d in ds] for name, ds in dictionaries.items()},
        "held_out_sha256": {name: _batch_hash(m.held_out) for name, m in models.items()},
    }
    _common.say(f"{GATE_ID}: identities exact {out['identities_exact']} of {out['identities_checked']}; records"
                f" identical {out['records_identical']}")
    return out


def identities(models: dict[str, Any], dictionaries: dict[str, list[Any]]) -> dict[str, list[bool]]:
    """D48's three identities at every TopK site of both zoo models, on the first held-out windows."""
    by_site = {}
    for name, m in models.items():
        tokens = m.held_tokens[:_common.IDENTITY_WINDOWS]
        for d in dictionaries[name]:
            site = f"{name} topk {d.sites[0][1]}"
            with _common.ticking(f"{GATE_ID}: identities at {site}", write=_common.say):
                by_site[site] = _common.dsteer_identities(m.model, m.params, n_layers=m.n_layers, dictionary=d,
                                                          tokens=tokens)
            _common.say(f"{GATE_ID}: {site}: " + ", ".join(f"{identity} {flag}" for identity, flag
                                                           in zip(_common.IDENTITIES, by_site[site])))
    return by_site


def record_requests(m: Any, dictionaries: list[Any], ctx: Any) -> dict[str, Any]:
    """One request of each kind on the two-hop run at its final checkpoint, ``v`` the least value of ``b``."""
    from nablaframe.gestures.records import GestureRequest

    hashes = [_common.core.content_hash(d) for d in dictionaries]
    reading = _common.read(m, m.params, dictionaries[2], f"{GATE_ID}: reading block 2's TopK SAE")
    table = reading.table("query")
    v = int(table.values("b")[0])
    latent = int(table.ranked("b", v)[0])
    _common.say(f"{GATE_ID}: the records' value of b is {v}, its best latent {latent}")
    edge = {"relation": "hop", "up": hashes[2], "up_site": 0, "down": hashes[3], "down_site": 0}
    args = {
        "name": {"dictionary": hashes[2], "site": 0, "class": "query", "latent": latent},
        "edge": edge,
        "open": {"dictionary": hashes[2], "site": 0, "class": "query", "upstream": hashes[1], "upstream_site": 0,
                 "description": "b", "value": v},
        "generalize": {"dictionary": hashes[2], "site": 0, "class": "query", "description": "b"},
    }
    return {kind: GestureRequest(kind, ctx.run, m.step, args[kind], _common.RECORD_INSTANCES, 0) for kind in KINDS}


def rebuilt_records(m: Any, dictionaries: list[Any], work: Path) -> tuple[dict[str, list[bool]], dict[str, str]]:
    """Each kind's record built twice with ``cli.build`` and once by ``cli.main``, compared byte for byte."""
    from nablaframe.gestures import cli, records
    from nablaframe.trace.schema import canonical_json

    seeds = {s: [_common.topk(m, l, seed=s) for l in range(m.n_layers)] for s in _common.p5.ROBUST_SEEDS}
    content = _common.core.content_hash
    seed_map = {s: {content(a): content(b) for a, b in zip(dictionaries, ds)} for s, ds in seeds.items()}
    ctx = _common.context(m, dictionaries + [d for ds in seeds.values() for d in ds], seed_map)
    reqs = record_requests(m, dictionaries, ctx)

    files = {"config": work / "config.json", "windows": work / "held_out.npy", "seeds": work / "seeds.json"}
    files["config"].write_text(json.dumps(asdict(m.config), sort_keys=True) + "\n", encoding="utf-8")
    np.save(files["windows"], m.held_out, allow_pickle=False)
    files["seeds"].write_text(json.dumps({str(s): c for s, c in seed_map.items()}, sort_keys=True) + "\n",
                              encoding="utf-8")
    out = work / "records"
    out.mkdir()

    by_kind, keys = {}, {}
    for kind, request in reqs.items():
        built = []
        for attempt in (1, 2):
            with _common.ticking(f"{GATE_ID}: building the {kind} record, build {attempt}", write=_common.say):
                built.append(cli.build(request, ctx))
        texts = [canonical_json(records.record_json(r)) for r in built]
        path = work / f"request-{kind}.json"
        path.write_text(json.dumps(records.request_json(request), sort_keys=True) + "\n", encoding="utf-8")
        argv = ["--run", "two_hop", "--config", str(files["config"]), "--checkpoints", str(m.run.folder / "checkpoints"),
                "--windows", str(files["windows"]), "--dictionaries", str(_common.p5.DICTIONARY_DIR), "--request",
                str(path), "--out", str(out), "--seeds", str(files["seeds"])]
        with _common.ticking(f"{GATE_ID}: the command line's {kind} record", write=_common.say):
            code = cli.main(argv)
        written = out / f"gesture-{built[0].key}.json"
        by_command = code == 0 and written.is_file() and written.read_bytes() == (texts[0] + "\n").encode("ascii")
        by_kind[kind] = [texts[0] == texts[1], bool(by_command)]
        keys[kind] = built[0].key
        _common.say(f"{GATE_ID}: the {kind} record, key {built[0].key[:12]}: builds identical {by_kind[kind][0]},"
                    f" the command line's identical {by_kind[kind][1]}")
    return by_kind, keys


def _batch_hash(windows: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(windows)


if __name__ == "__main__":
    raise SystemExit(main())
