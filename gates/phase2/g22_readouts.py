"""G2.2: readout fidelity.

Computes the readout views (``attention_pattern``, ``residual_norms``,
``per_token_loss``) for every trace step twice, with two separate ``TraceReader``s, and
compares a seeded sample of 200 displayed values with the traced tensors read
independently through ``TraceStore.load_tensor``.

Measurements: ``sample_size``, ``sample_matches`` (values equal bit for bit) and
``builds_identical`` (the two passes give byte-identical arrays).
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g22_readouts.py``. Takes --overwrite and --allow-dirty. It reads
#   G1.6's store after G2.1 has added the logits, and writes nothing outside its report.
# * A pass computes, at every step, ``attention_pattern`` for every layer, head and
#   probe sequence, ``residual_norms`` and ``per_token_loss``, and keeps the SHA-256 of
#   each array's bytes, dtype and shape. ``builds_identical`` is whether the two passes'
#   lists are equal.
# * The sample is drawn before either pass, with ``default_rng(0)``: for each of the 200
#   draws a view (one of the three, uniformly), a step, then one displayed value of that
#   view. The first pass keeps the drawn values.
# * Each sampled value is checked against a reference computed only from the tensor that
#   ``TraceStore.load_tensor`` returns, through a store this gate opens itself:
#   - an attention value is the traced ``blocks.<l>.attn.hook_pattern`` element itself;
#   - a residual norm is that one position's L2 norm over Embed, in float64, then float32,
#     as readouts.py's contract states;
#   - a loss is ``-log softmax`` of that one position's traced logits in float64 at the
#     next probe token, then float32; at the last position, NaN.
#   "Bit for bit" compares the float32 bit patterns, so a NaN matches only a NaN with
#   the same bits.
# * Every value the readouts display is a pure numpy decode, so nothing here depends on
#   the jax lane beyond the store's own imports.

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.2"
REPORT_NAME = "g22_readouts.json"
SAMPLE_SIZE = 200
SAMPLE_SEED = 0
READOUTS = ("attention_pattern", "residual_norms", "per_token_loss")
VERSIONS = ("numpy", "safetensors")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    probe = p1.probe(p1.load_split("validation"), ref.config.seq)
    store_root = _common.g16_store()
    _common.open_reader(store_root, ref.steps, probe)
    measurements = measure(store_root, ref.steps, ref.config, probe)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.2 rule holds."""
    ok = m["sample_matches"] == m["sample_size"] and m["builds_identical"] is True
    return "pass" if ok else "fail"


def measure(store_root: Path, steps: tuple[int, ...], config: Any, probe: np.ndarray, *,
            sample_size: int = SAMPLE_SIZE) -> dict[str, Any]:
    from nablaframe.views.readouts import TraceReader

    probe = np.asarray(probe)
    sample = draw_sample(steps, config, probe.shape, sample_size)
    first = run_pass(TraceReader(store_root), steps, config, probe, "first pass", keep=sample)
    second = run_pass(TraceReader(store_root), steps, config, probe, "second pass", keep=[])
    identical = first["digests"] == second["digests"]
    print(f"{GATE_ID}: the two passes are {'byte-identical' if identical else 'different'}"
          f" over {len(first['digests']):,} arrays", flush=True)

    checked = check_sample(store_root, sample, first["kept"], probe, config.n_layers)
    matches = sum(1 for entry in checked if entry["matches"])
    print(f"{GATE_ID}: {matches} of {len(checked)} sampled values equal the traced tensor bit for bit", flush=True)
    return {
        "sample_size": len(checked),
        "sample_matches": matches,
        "builds_identical": identical,
        "arrays_compared": len(first["digests"]),
        "arrays_differing": sum(1 for a, b in zip(first["digests"], second["digests"]) if a != b)
                            + abs(len(first["digests"]) - len(second["digests"])),
        "steps": list(steps),
        "sample_seed": SAMPLE_SEED,
        "sample": checked,
        "store": p1.rel(store_root),
    }


def draw_sample(steps: tuple[int, ...], config: Any, probe_shape: tuple[int, int], n: int) -> list[dict[str, Any]]:
    """``n`` displayed values, each a readout, a step and an index into that readout's array."""
    rng = np.random.default_rng(SAMPLE_SEED)
    batch, seq = probe_shape
    sample = []
    for _ in range(n):
        view = READOUTS[int(rng.integers(len(READOUTS)))]
        step = int(steps[int(rng.integers(len(steps)))])
        if view == "attention_pattern":
            layer, head, sequence = (int(rng.integers(k)) for k in (config.n_layers, config.n_heads, batch))
            query = int(rng.integers(seq))
            key = int(rng.integers(query + 1))  # where the causal pattern can be non-zero
            sample.append({"view": view, "step": step, "layer": layer, "head": head, "sequence": sequence,
                           "index": [query, key]})
        elif view == "residual_norms":
            sample.append({"view": view, "step": step,
                           "index": [int(rng.integers(config.n_layers + 1)), int(rng.integers(batch)),
                                     int(rng.integers(seq))]})
        else:
            sample.append({"view": view, "step": step, "index": [int(rng.integers(batch)), int(rng.integers(seq))]})
    return sample


def _key(entry: dict[str, Any]) -> tuple:
    if entry["view"] == "attention_pattern":
        return (entry["view"], entry["step"], entry["layer"], entry["head"], entry["sequence"])
    return (entry["view"], entry["step"])


def _digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    head = f"{array.dtype.str}{list(array.shape)}".encode("ascii")
    return hashlib.sha256(head + array.tobytes()).hexdigest()


def run_pass(reader: Any, steps: tuple[int, ...], config: Any, probe: np.ndarray, label: str, *,
             keep: list[dict[str, Any]]) -> dict[str, Any]:
    """One pass over every readout at every step: each array's digest, in order, and the values in ``keep``."""
    from nablaframe.views.readouts import attention_pattern, per_token_loss, residual_norms

    wanted: dict[tuple, list[int]] = {}
    for i, entry in enumerate(keep):
        wanted.setdefault(_key(entry), []).append(i)
    kept: list[Any] = [None] * len(keep)

    def record(key: tuple, array: np.ndarray) -> None:
        digests.append([*key, _digest(array)])
        for i in wanted.get(key, ()):
            kept[i] = array[tuple(keep[i]["index"])].copy()

    digests: list[list[Any]] = []
    batch = probe.shape[0]
    progress = p1.Progress(f"readouts, {label}", len(steps))
    for step in steps:
        for layer in range(config.n_layers):
            for head in range(config.n_heads):
                for sequence in range(batch):
                    record(("attention_pattern", step, layer, head, sequence),
                           attention_pattern(reader, step, layer, head, sequence))
        record(("residual_norms", step), residual_norms(reader, step, config.n_layers))
        record(("per_token_loss", step), per_token_loss(reader, step, probe))
        progress.advance()
    return {"digests": digests, "kept": kept}


def check_sample(store_root: Path, sample: list[dict[str, Any]], shown: list[Any], probe: np.ndarray,
                 n_layers: int) -> list[dict[str, Any]]:
    """Each sampled value against its reference, computed from ``TraceStore.load_tensor`` alone."""
    from nablaframe.trace.store import TraceStore

    checked = []
    with TraceStore(store_root, mode="r") as store:
        current = {(event.step, event.tap): event for event in store.current() if event.tensor_ref is not None}

        def load(step: int, tap: str) -> np.ndarray:
            return store.load_tensor(current[(step, tap)])

        progress = p1.Progress("checking sampled values", len(sample))
        for entry, value in zip(sample, shown):
            expected = reference_value(load, entry, probe, n_layers)
            value = np.float32(value)
            checked.append({**entry, "shown": _float(value), "traced": _float(expected),
                            "matches": bool(value.view(np.uint32) == expected.view(np.uint32))})
            progress.advance()
    return checked


def reference_value(load: Any, entry: dict[str, Any], probe: np.ndarray, n_layers: int) -> np.float32:
    """The value ``entry`` should display, from the traced tensor only."""
    step, index = entry["step"], entry["index"]
    if entry["view"] == "attention_pattern":
        pattern = load(step, f"blocks.{entry['layer']}.attn.hook_pattern")
        return np.float32(pattern[entry["sequence"], entry["head"], index[0], index[1]])
    if entry["view"] == "residual_norms":
        row, b, t = index
        tap = f"blocks.{row}.hook_resid_pre" if row < n_layers else f"blocks.{n_layers - 1}.hook_resid_post"
        vector = load(step, tap)[b, t].astype(np.float64)
        return np.float32(np.linalg.norm(vector, axis=-1))
    b, t = index
    if t == probe.shape[1] - 1:
        return np.float32(np.nan)
    row = load(step, "logits")[b, t].astype(np.float64)
    shifted = row - row.max()
    return np.float32(-(shifted[int(probe[b, t + 1])] - np.log(np.exp(shifted).sum())))


def _float(value: np.floating) -> float | None:
    return p1.finite_or_none(value)


if __name__ == "__main__":
    raise SystemExit(main())
