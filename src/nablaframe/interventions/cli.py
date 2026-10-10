"""The compare payload and the command line (D24, D27).

Spec: docs/specs/phase-3.md. Both passes always go through ``run_intervened``: edit values
must enter the compiled program as inputs, and hooks built elsewhere could bake them in as
constants, which XLA may fold, breaking D25's identities (found in PR #38).

* ``compare_payload(model, params, tokens, spec, *, n_layers, resolve, params_at=None,
  taps=())``: ``effect_payload(spec, unintervened logits, intervened logits, tokens)`` with
  three more keys: ``"taps"`` (the requested taps, in order), ``"tokens"`` (nested ints) and
  ``"tap_changes"``, mapping each requested tap to the L2 norm over its trailing axes of the
  intervened value minus the unintervened one, (Batch, Pos), in float64. The unintervened
  pass is ``run_intervened`` with an empty spec.
* ``top_tokens(logits, k=5)``: ``(tokens, probs)``, nested lists (Batch, Pos, k): each position's k most probable tokens, best first, ties to the lower id, and their softmax probabilities in float64. The same rule as the logit lens payload's top tokens.
* ``compare_payload`` gains ``"top_tokens"`` and ``"top_probs"``, each ``{"unintervened": ..., "intervened": ...}``, from ``top_tokens`` of each pass's logits. So the viewer can show what each pass predicts.
* ``main(argv)``: ``--config`` (a JSON of ``GPT2Config``'s fields), ``--checkpoints`` (a
  folder of ``step-<n>.safetensors``), ``--step``, ``--tokens`` (an int32 ``.npy``), ``--spec``
  (the spec's JSON), ``--objects`` (a folder of ``<hash>.npy`` files that ``resolve`` reads),
  ``--out``, and ``--taps`` (comma-separated, optional). It writes, in ``--out``:

  - ``intervention-<spec_hash>-step-<step>.json``: the compare payload plus ``"step"``,
    through ``export_view``.
  - ``results/<key>.npy``: the intervened logits, through ``save_result``, under
    ``intervention_key`` with the checkpoint's ``state_hash``, ``batch_hash(tokens)``, the
    spec hash and ``apply.code_identity()`` (D24).
  - ``index.json``, rewritten last as canonical JSON: ``{"interventions": [{"spec_hash",
    "step", "file"}, ...]}``, one entry per payload file in ``--out``, sorted by spec hash
    then step.

  Returns 0. Any error, such as a missing object, prints one line and returns 1 having
  written nothing new.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.corpus import batch_hash
from ..trace.schema import canonical_json
from ..views.cache import _write_atomic
from ..views.export import export_view
from .apply import code_identity, editable_taps, run_intervened
from .effects import effect_payload, intervention_key, load_result, save_result
from .spec import InterventionSpec, spec_from_json, spec_hash

_PAYLOAD = re.compile(r"intervention-([0-9a-f]{64})-step-(0|[1-9][0-9]*)\.json")


def _array(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "array", value))


def top_tokens(logits: Any, k: int = 5) -> tuple[list, list]:
    """Each position's k most probable tokens, best first, ties to the lower id, and their float64 softmax."""
    shifted = _array(logits).astype(np.float64)
    shifted -= shifted.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    top = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    return top.tolist(), np.take_along_axis(p, top, axis=-1).tolist()


def _compare(model: Any, params: Any, tokens: np.ndarray, spec: Any, *, n_layers: int,
             resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None,
             taps: Sequence[str]) -> tuple[dict[str, Any], np.ndarray]:
    """The compare payload and the intervened logits, from two ``run_intervened`` passes."""
    taps = list(taps)
    # Tap changes are per position, so every requested tap must be (Batch, Pos, ...): that rules out
    # attn.hook_pattern, whose axes are (Batch, Head, QPos, KPos).
    positional = set(editable_taps(n_layers))
    refused = [tap for tap in taps if tap not in positional]
    if refused:
        raise ValueError(f"tap changes need taps whose axes start (Batch, Pos); not one of those: {refused}")
    common = {"n_layers": n_layers, "resolve": resolve, "params_at": params_at, "taps": taps}
    base_logits, base_taps = run_intervened(model, params, tokens, InterventionSpec(()), **common)
    logits, moved_taps = run_intervened(model, params, tokens, spec, **common)
    base_logits, logits = _array(base_logits), _array(logits)
    tap_changes = {}
    for tap in taps:
        change = _array(moved_taps[tap]).astype(np.float64) - _array(base_taps[tap]).astype(np.float64)
        tap_changes[tap] = np.sqrt((change * change).sum(axis=tuple(range(2, change.ndim)))).tolist()
    payload = effect_payload(spec, base_logits, logits, tokens)
    base_top, base_probs = top_tokens(base_logits)
    top, probs = top_tokens(logits)
    payload.update(top_tokens={"unintervened": base_top, "intervened": top},
                   top_probs={"unintervened": base_probs, "intervened": probs})
    payload.update(taps=taps, tokens=np.asarray(tokens).tolist(), tap_changes=tap_changes)
    return payload, logits


def compare_payload(model: Any, params: Any, tokens: np.ndarray, spec: Any, *, n_layers: int,
                    resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None = None,
                    taps: Sequence[str] = ()) -> dict[str, Any]:
    return _compare(model, params, tokens, spec, n_layers=n_layers, resolve=resolve, params_at=params_at,
                    taps=taps)[0]


class _Parser(argparse.ArgumentParser):
    """Raises on a bad command line, so ``main`` reports it in one line and returns 1."""

    def error(self, message: str) -> Any:
        raise ValueError(message)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = _Parser(prog="python -m nablaframe.interventions.cli",
                     description="Run an intervention spec on a checkpoint and export its compare payload.")
    parser.add_argument("--config", type=Path, required=True, help="a JSON of GPT2Config's fields")
    parser.add_argument("--checkpoints", type=Path, required=True, help="a folder of step-<n>.safetensors")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tokens", type=Path, required=True, help="an int32 .npy, (batch, pos)")
    parser.add_argument("--spec", type=Path, required=True, help="the spec's JSON")
    parser.add_argument("--objects", type=Path, required=True, help="a folder of <hash>.npy files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--taps", default="", help="comma-separated tap names")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> None:
    """Computes and checks everything, then writes the payload, the cached logits and the index."""
    from ..model.gpt2 import GPT2Config, GPT2Model, init_params
    from ..train.checkpoint import load_params, state_hash

    config = GPT2Config(**json.loads(args.config.read_text(encoding="utf-8")))
    tokens = np.load(args.tokens, allow_pickle=False)
    if tokens.dtype != np.int32:
        raise ValueError(f"{args.tokens} holds {tokens.dtype}, not int32 token ids")
    spec = spec_from_json(json.loads(args.spec.read_text(encoding="utf-8")))
    taps = [tap for tap in args.taps.split(",") if tap]

    template = init_params(config, seed=0)
    loaded: dict[int, Any] = {}

    def params_at(step: int) -> Any:
        if step not in loaded:
            path = args.checkpoints / f"step-{step}.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"no checkpoint {path}")
            loaded[step] = load_params(path, like=template)
        return loaded[step]

    def resolve(digest: str) -> np.ndarray:
        path = args.objects / f"{digest}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"no object {path}")
        return np.load(path, allow_pickle=False)

    params = params_at(args.step)
    payload, logits = _compare(GPT2Model(config), params, tokens, spec, n_layers=config.n_layers,
                               resolve=resolve, params_at=params_at, taps=taps)
    payload["step"] = args.step
    canonical_json(payload)  # refuses NaN or infinity before anything is written

    out, results = args.out, args.out / "results"
    for folder in (out, results):
        if folder.exists() and not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a folder")
    name = f"intervention-{spec_hash(spec)}-step-{args.step}"
    key = intervention_key(checkpoint_sha256=state_hash(params), batch_hash=batch_hash(tokens),
                           spec_hash=spec_hash(spec), code=code_identity())
    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise ValueError(f"the intervened logits are {logits.dtype}, and a cached result must be float32")
    cached = load_result(results, key)
    if cached is not None and (cached.shape != logits.shape or cached.tobytes() != logits.tobytes()):
        raise ValueError(f"results/{key}.npy holds a different result; a cache key never changes meaning")

    files = {f"{name}.json"} | ({p.name for p in out.iterdir() if p.is_file()} if out.is_dir() else set())
    entries = []
    for file in files:
        match = _PAYLOAD.fullmatch(file)
        if match:
            entries.append({"spec_hash": match.group(1), "step": int(match.group(2)), "file": file})
    entries.sort(key=lambda entry: (entry["spec_hash"], entry["step"]))
    index = canonical_json({"interventions": entries}).encode("ascii")

    export_view(out, name, payload)
    save_result(results, key, logits)
    _write_atomic(out / "index.json", index)


def main(argv: list[str] | None = None) -> int:
    try:
        _run(_arguments(argv))
    except Exception as exc:  # noqa: BLE001 - every error is one line and exit code 1
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
