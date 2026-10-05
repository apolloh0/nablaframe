"""Phase 3 package 7: the compare payload and the command line (D24, D27).

Spec: docs/specs/phase-3.md. Needs the jax lane.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from nablaframe.data.corpus import batch_hash  # noqa: E402
from nablaframe.interventions.apply import code_identity, run_intervened  # noqa: E402
from nablaframe.interventions.cli import compare_payload, main  # noqa: E402
from nablaframe.interventions.effects import effect_payload, intervention_key, load_result  # noqa: E402
from nablaframe.interventions.spec import InterventionSpec, edit, spec_hash, spec_json  # noqa: E402
from nablaframe.trace.schema import canonical_json  # noqa: E402
from nablaframe.train.checkpoint import save_params, state_hash  # noqa: E402
from interventions_fixtures import MODEL, OTHER, PARAMS, PARAMS_AT, STORE, TOKENS, base_logits  # noqa: E402
from phase2_fixtures import TINY  # noqa: E402

TAPS = ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]


def _payload(spec, taps=TAPS):
    return compare_payload(MODEL, PARAMS, TOKENS, spec, n_layers=TINY.n_layers, resolve=STORE.__getitem__,
                           params_at=PARAMS_AT.__getitem__, taps=taps)


def test_compare_payload_pairs_both_passes():
    spec = InterventionSpec((edit("blocks.0.hook_resid_post", [3], "zero"),))
    payload = _payload(spec)
    intervened = np.asarray(run_intervened(MODEL, PARAMS, TOKENS, spec, n_layers=TINY.n_layers,
                                           resolve=STORE.__getitem__, params_at=PARAMS_AT.__getitem__)[0])
    for key, value in effect_payload(spec, base_logits(), intervened, TOKENS).items():
        assert payload[key] == value, key
    assert payload["taps"] == TAPS and payload["tokens"] == TOKENS.tolist()
    after = np.asarray(payload["tap_changes"]["blocks.1.hook_resid_post"])
    assert after.shape == TOKENS.shape and (after[:, :3] == 0).all() and (after[:, 3:] > 0).all(), "causality"
    _, taps = MODEL.instrumented(PARAMS, TOKENS, ["blocks.0.hook_resid_post"])
    zeroed = np.linalg.norm(np.asarray(taps["blocks.0.hook_resid_post"], np.float64)[:, 3], axis=-1)
    at = np.asarray(payload["tap_changes"]["blocks.0.hook_resid_post"])
    assert np.allclose(at[:, 3], zeroed, rtol=1e-6) and (np.delete(at, 3, axis=1) == 0).all(), "the L2 norm of the change"
    null = _payload(InterventionSpec((edit("blocks.0.hook_resid_post", [], "zero"),)))
    assert all((np.asarray(v) == 0).all() for v in null["tap_changes"].values()) and null["kl_mean"] == 0.0


def _setup(tmp_path, spec):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dataclasses.asdict(TINY)))
    checkpoints = tmp_path / "checkpoints"
    save_params(PARAMS, checkpoints, step=0, run_id=None)
    save_params(OTHER, checkpoints, step=4, run_id=None)
    tokens = tmp_path / "tokens.npy"
    np.save(tokens, TOKENS)
    objects = tmp_path / "objects"
    objects.mkdir()
    for h, array in STORE.items():
        np.save(objects / f"{h}.npy", array)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(canonical_json(spec_json(spec)))
    args = ["--config", str(config), "--checkpoints", str(checkpoints), "--step", "0", "--tokens", str(tokens),
            "--spec", str(spec_file), "--objects", str(objects), "--out", str(tmp_path / "out"),
            "--taps", "blocks.1.hook_resid_post"]
    return args, objects


def test_cli_writes_payload_cache_and_index(tmp_path):
    spec = InterventionSpec((edit("blocks.0.hook_resid_post", [3], "zero"), edit("blocks.1.mlp", "all", "swap", step=4)))
    args, objects = _setup(tmp_path, spec)
    assert main(args) == 0
    out = tmp_path / "out"
    name = f"intervention-{spec_hash(spec)}-step-0.json"
    expected = {**_payload(spec, ["blocks.1.hook_resid_post"]), "step": 0}
    written = (out / name).read_text()
    assert written == canonical_json(expected)
    key = intervention_key(checkpoint_sha256=state_hash(PARAMS), batch_hash=batch_hash(TOKENS),
                           spec_hash=spec_hash(spec), code=code_identity())
    intervened = np.asarray(run_intervened(MODEL, PARAMS, TOKENS, spec, n_layers=TINY.n_layers,
                                           resolve=STORE.__getitem__, params_at=PARAMS_AT.__getitem__)[0])
    assert np.array_equal(load_result(out / "results", key), intervened), "the intervened logits, cached under D24's key"
    index = json.loads((out / "index.json").read_text())
    assert index == {"interventions": [{"spec_hash": spec_hash(spec), "step": 0, "file": name}]}
    assert main(args) == 0 and (out / name).read_text() == written, "a rerun is byte-identical"
    assert json.loads((out / "index.json").read_text()) == index, "and the index lists it once"


def test_cli_refuses_a_missing_object(tmp_path):
    spec = InterventionSpec((edit("blocks.0.hook_resid_post", "all", "patch", source="0" * 64),))
    args, _ = _setup(tmp_path, spec)
    assert main(args) != 0
    assert not (tmp_path / "out" / f"intervention-{spec_hash(spec)}-step-0.json").exists(), "nothing written"
