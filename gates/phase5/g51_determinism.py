"""G5.1: dictionaries train bit for bit, their cache keys are distinct, and ``dpatch`` keeps D25's identities.

Spec: docs/specs/phase-5.md, D40, D45 and "Gates"; helpers in ``_common.py``. Report:
``reports/phase5/g51_determinism.json``. Run in nf-jax, on the Mac, after G4.1 passes.

On G4.1's runs at their final checkpoints:

* **Determinism:** on the induction model, each of ``topk`` (block 0), ``staircase``,
  ``transcoder`` (block 0) and ``jsae`` (block 0, ``lam`` 1.0) is trained twice from one key with
  ``settings(..., short=True)`` on ``zoo_windows(task, "induction", "train")``, each time from
  scratch through ``train_dictionary``, never from the cache, and saved into a folder of its own.
  ``identical_by_kind``: whether both ``.safetensors`` files and both ``.json`` files are
  byte-identical; ``files_identical``, whether every kind's are; ``kinds_checked``, the kinds.
* **Cache keys:** ``dictionary_key`` for a ``topk`` at the induction model's block 0, over the 64
  combinations of two values each of width (``WIDTH_FACTOR`` and half of it, times d_model),
  ``k`` (16 and 8), ``steps`` (3,000 and 2,000), ``batch`` (4,096 and 2,048), ``lr`` (5e-4 and
  1e-3) and ``seed`` (0 and 1), then four more keys from the first, each with one other input
  changed: the state hash (the twin's), the sites (block 1), the training windows (the held-out
  ones) and the code (the blob hash of ``core.py`` instead of ``train.py``).
  ``cache_keys_checked`` (68) and ``cache_keys_distinct``.
* **Identities:** for both zoo models, one dictionary of each shipped kind at every site, trained
  with ``settings(..., short=True)`` through ``cached`` (``topk`` at every block, the
  ``staircase`` at each of its sites, a ``transcoder`` at every MLP). At each site's output tap, on
  the first 64 held-out windows, two ``dpatch`` edits at every position through
  ``run_intervened``: from a source equal to the target, and with ``latents=[]`` from the second 64
  held-out windows. Each must give logits bit for bit equal to the unintervened pass's.
  ``identities_by_site``: ``{"<model> <kind> <output tap>": [same source, no latents]}``, each
  true or false; ``identities_checked`` and ``identities_exact``, counts over both models.

Measurements: REQUIRED_MEASUREMENTS["G5.1"]. Pass needs ``files_identical``,
``cache_keys_distinct``, at least one identity checked and every one exact.
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase5/g51_determinism.py``. Takes --overwrite and --allow-dirty; --overwrite also starts
#   its working folder, ``runs/phase5/g51_determinism/``, over.
# * Determinism: each kind's two dictionaries are saved with ``save_dictionary`` into
#   ``runs/phase5/g51_determinism/<kind>/first/`` and ``.../second/``. Since each file is named
#   by its content hash, two folders are identical when they hold the same file names with the
#   same bytes. The keys are ``dictionary_key``s of the checkpoint's state hash, the kind's
#   sites, the short settings and the tokens the model reads of the training windows.
# * Cache keys: the combinations run in ``itertools.product`` order, width first, from
#   ``settings("induction", "topk", d_model)`` with those six fields replaced; the first
#   combination is the spec's settings. The changed code identity is ``core.py``'s git blob hash,
#   as ``train.code_identity`` computes ``train.py``'s.
# * Identities: every pass goes through ``run_intervened``, the unintervened one with an empty
#   spec, each identity one ``dpatch`` edit at ``"all"`` positions of the site's output tap. Bit
#   for bit means equal shapes, dtypes and bytes.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G5.1"
REPORT_NAME = "g51_determinism.json"
WORK_NAME = "g51_determinism"
DETERMINISM = (("topk", 0, 0.0), ("staircase", None, 0.0), ("transcoder", 0, 0.0), ("jsae", 0, 1.0))
KEY_VALUES = {"width": None, "k": (16, 8), "steps": (3000, 2000), "batch": (4096, 2048), "lr": (5e-4, 1e-3),
              "seed": (0, 1)}
IDENTITY_WINDOWS = 64


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    g41 = _common.require_g41()
    work = _common.fresh_dir(Path(_common.RUN_DIR) / WORK_NAME, overwrite=args.overwrite)
    models = {name: _common.zoo_model(name, g41) for name in _common.ZOO}
    measurements = measure(models, work)
    measurements["versions"] = p1.phase0.package_versions(_common.VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G5.1 rule holds."""
    checked, exact = m["identities_checked"], m["identities_exact"]
    ok = (m["files_identical"] is True and m["cache_keys_distinct"] is True and _number(checked) and checked >= 1
          and _number(exact) and exact == checked)
    return "pass" if ok else "fail"


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def measure(models: dict[str, Any], work: Path) -> dict[str, Any]:
    induction = models["induction"]
    measured: dict[str, Any] = {}
    measured |= determinism(induction, work)
    measured |= cache_keys(induction)
    measured |= identities(models)
    measured |= {
        "run_ids": {name: m.run_id for name, m in models.items()},
        "steps": {name: m.step for name, m in models.items()},
        "short": _common.SHORT,
        "identity_windows": IDENTITY_WINDOWS,
        "train_seconds": {name: m.train_seconds for name, m in models.items()},
    }
    return measured


def determinism(m: Any, work: Path) -> dict[str, Any]:
    """Each kind trained twice from one key at the short settings, from scratch, and the files compared."""
    from nablaframe.dictionaries.core import save_dictionary
    from nablaframe.dictionaries.train import mlp_weights

    tokens = m.train_tokens()
    identical: dict[str, bool] = {}
    keys: dict[str, str] = {}
    for kind, layer, lam in DETERMINISM:
        sites = _common.sites(kind, m.n_layers, layer)
        settings = _common.settings(m.name, kind, m.d_model, lam=lam, short=True)
        key = keys[kind] = _common.dictionary_key(m.state, sites, settings, tokens)
        weights = mlp_weights(m.params, layer) if kind == "jsae" else None
        folders = []
        for attempt in ("first", "second"):
            trained = _common.train_dictionary(m.model, m.params, tokens, settings, sites, key=key, chunk=m.chunk,
                                               label=f"{GATE_ID}: {kind}, {attempt} training", mlp_weights=weights)
            folders.append(save_dictionary(trained, work / kind / attempt).parent)
        identical[kind] = _same_files(*folders)
        print(f"{GATE_ID}: {kind} trained twice from key {key[:12]}:"
              f" {'byte-identical' if identical[kind] else 'the files differ'}", flush=True)
    return {
        "files_identical": all(identical.values()),
        "identical_by_kind": identical,
        "kinds_checked": [kind for kind, _, _ in DETERMINISM],
        "determinism_keys": keys,
        "determinism_dir": p1.rel(work),
    }


def _same_files(first: Path, second: Path) -> bool:
    a = {path.name: path.read_bytes() for path in sorted(Path(first).iterdir())}
    b = {path.name: path.read_bytes() for path in sorted(Path(second).iterdir())}
    return len(a) == 2 and a == b


def cache_keys(m: Any) -> dict[str, Any]:
    """68 keys for a TopK SAE at block 0: 64 setting combinations, then four from the first, one input changed each."""
    import dataclasses
    import itertools

    from nablaframe.dictionaries import core

    values = dict(KEY_VALUES, width=(_common.WIDTH_FACTOR * m.d_model, _common.WIDTH_FACTOR * m.d_model // 2))
    base = _common.settings(m.name, "topk", m.d_model)
    sites = _common.sites("topk", m.n_layers, 0)
    tokens = m.train_tokens()
    keys = [_common.dictionary_key(m.state, sites, dataclasses.replace(base, **dict(zip(values, combination))), tokens)
            for combination in itertools.product(*values.values())]
    first = dataclasses.replace(base, **{name: options[0] for name, options in values.items()})
    keys += [
        _common.dictionary_key(m.twin_state, sites, first, tokens),
        _common.dictionary_key(m.state, _common.sites("topk", m.n_layers, 1), first, tokens),
        _common.dictionary_key(m.state, sites, first, m.tokens(m.held_out)),
        core.dictionary_key(state_sha256=m.state, sites=sites, settings=first,
                            batch_hashes=[_batch_hash(tokens)], code=_blob_hash(Path(core.__file__))),
    ]
    distinct = len(set(keys)) == len(keys)
    print(f"{GATE_ID}: {len(keys)} cache keys, {len(set(keys))} distinct", flush=True)
    return {"cache_keys_checked": len(keys), "cache_keys_distinct": distinct}


def _blob_hash(path: Path) -> str:
    """A file's git blob hash, as ``git hash-object`` prints it."""
    import hashlib

    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _batch_hash(tokens: np.ndarray) -> str:
    from nablaframe.data.corpus import batch_hash

    return batch_hash(tokens)


def identities(models: dict[str, Any]) -> dict[str, Any]:
    """D25's identities for dpatch at every site of every shipped kind on both zoo models."""
    by_site: dict[str, list[bool]] = {}
    for name, m in models.items():
        held = m.held_tokens
        if len(held) < 2 * IDENTITY_WINDOWS:
            raise SystemExit(f"{name} has {len(held)} held-out windows, fewer than {2 * IDENTITY_WINDOWS}."
                             " No report written.")
        target, other = held[:IDENTITY_WINDOWS], held[IDENTITY_WINDOWS:2 * IDENTITY_WINDOWS]
        clean, _ = _common.p3.unintervened(m.model, m.params, target, n_layers=m.n_layers)
        shipped = [("topk", layer) for layer in range(m.n_layers)] + [("staircase", None)]
        shipped += [("transcoder", layer) for layer in range(m.n_layers)]
        for kind, layer in shipped:
            d = _common.dictionary(m, kind, layer, short=True)
            for _, writes in d.sites:
                key = f"{name} {_common.site_name(kind, writes)}"
                by_site[key] = [_exact(m, d, writes, target, target, None, clean),
                                _exact(m, d, writes, target, other, [], clean)]
                print(f"{GATE_ID}: {key}: same source {'exact' if by_site[key][0] else 'NOT exact'},"
                      f" no latents {'exact' if by_site[key][1] else 'NOT exact'}", flush=True)
    checked = 2 * len(by_site)
    exact = sum(int(ok) for pair in by_site.values() for ok in pair)
    print(f"{GATE_ID}: {exact} of {checked} identities bit for bit", flush=True)
    return {"identities_checked": checked, "identities_exact": exact, "identities_by_site": by_site}


def _exact(m: Any, d: Any, tap: str, target: np.ndarray, source: np.ndarray, latents: list[int] | None,
           clean: np.ndarray) -> bool:
    """Whether one dpatch edit at every position of ``tap`` gives logits bit for bit equal to ``clean``."""
    from nablaframe.dictionaries.core import content_hash
    from nablaframe.interventions.spec import InterventionSpec, edit

    objects = _common.p3.Objects()
    source_hash = objects.batch(source)
    objects.batch(target)
    digest = content_hash(d)
    args = {"dictionary": digest, "source": source_hash} | ({} if latents is None else {"latents": latents})
    store = dict(objects.arrays) | {digest: d}
    logits, _ = _common.p3.run(m.model, m.params, target, InterventionSpec((edit(tap, "all", "dpatch", **args),)),
                               n_layers=m.n_layers, resolve=store.__getitem__)
    return bool(logits.shape == clean.shape and logits.dtype == clean.dtype
                and logits.tobytes() == clean.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
