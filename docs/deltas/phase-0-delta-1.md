# Phase 0, delta 1: contracts, stubs and tests

Date: 2026-09-27. Spec: [docs/specs/phase-0.md](../specs/phase-0.md).

This delta carries every Phase 0 contract as a stub, plus the twelve named tests. All 26 test cases fail on the stubs, which is the starting point for the implementation threads. Nothing here overwrites an existing file.

## Apply

Unzip at the repo root, then:

```
uv sync
uv run pytest            # expect 26 failed: the stubs raise NotImplementedError
```

Commit the new files, then make the edits below. Edits 1 to 7 assume D1 and D2 are ruled as proposed.

## New files

| Path | What it is |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Core deps (numpy, safetensors), lane extras `jax`, `metal`, `torch`, dev group with pytest |
| `src/nablaframe/trace/schema.py` | RunManifest, TapSpec, TraceEvent; canonical JSON and hashing rules |
| `src/nablaframe/trace/registry.py` | The 19 canonical tap names for a 2-layer model, and their axes and sites |
| `src/nablaframe/trace/store.py` | Append-only store: JSONL events, safetensors shards, superseding corrections |
| `src/nablaframe/oracle/tolerance.py` | ToleranceRecord: a tolerance with the report it was measured in |
| `src/nablaframe/oracle/mode_equality.py` | The oracle, both policies |
| `src/nablaframe/reports.py` | Gate report schema; rejects a status that contradicts its own measurements |
| `src/nablaframe/lanes/jax_ref/toy.py` | The G0.6 toy model contract |
| `gates/g01…g06_*.py`, `gates/summarize.py` | Gate and summary stubs, each quoting its spec row |
| `gates/configs/gpt2_nano.yaml` | Levanter's tiny GPT-2 config, vendored verbatim at Marin commit `dba12e3` |
| `scripts/check_tl_names.py` | Runtime check that every tap name resolves on TransformerLens |
| `tests/` | The twelve named tests, a numpy toy model, the vendored TransformerLens name list |

## Environments

```
UV_PROJECT_ENVIRONMENT=.venv-jax   uv sync --extra jax     # nf-jax
UV_PROJECT_ENVIRONMENT=.venv-metal uv sync --extra metal   # nf-metal: nf-jax plus metaljax
UV_PROJECT_ENVIRONMENT=.venv-torch uv sync --extra torch   # nf-torch
```

Prefix `uv run` with the same variable to run inside a lane. Unit tests need no lane: plain `uv run pytest`.

## Proposed rulings

Add to the Phase decisions table in `docs/decisions.md`, after the D0 row, once ruled:

```
| D1 | 2026-09-27 | Levanter and Haliax come from PyPI as `marin-levanter` and `marin-haliax` dev builds, pinned by `uv.lock`. The jax and metal extras conflict with the torch extra. The lock targets macOS arm64 and Linux x86_64 | The `levanter` package stopped in October 2025; one shared resolution dropped TransformerLens to 1.14; the newest build of one Marin dependency shipped an Intel-Mac wheel only |
| D2 | 2026-09-27 | Tap names are TransformerLens legacy names, resolved as aliases on `TransformerBridge`. G0.4 loads through the bridge without compatibility mode | TransformerLens 4.0 removed `HookedTransformer`; compatibility mode reproduces its weight processing, so activations would no longer match the raw weights Levanter exports |
```

## Edits to docs/specs/phase-0.md

**Edit 1**, Environments table. Replace:

```
| `nf-jax` | jax (CPU), equinox, haliax, levanter, optax, numpy | Reference lane: G0.1, G0.2, G0.3, G0.6 |
```

with:

```
| `nf-jax` | jax (CPU), equinox, optax, marin-levanter and marin-haliax with their marin-* dependencies, numpy | Reference lane: G0.1, G0.2, G0.3, G0.6 |
```

**Edit 2**, the sentence after that table. Replace:

```
Record, don't assume, whether levanter and metaljax resolve to a shared jax version. If they do, `nf-metal` folds into `nf-jax`.
```

with:

```
The lock resolves one jax version (0.11.2) for marin-levanter and metaljax, so `nf-metal` is the jax lane plus metaljax (`uv sync --extra metal`). Confirm it installs and runs on the Mac before trusting G0.5.
```

**Edit 3**, replace the whole code block under "Files added in Phase 0" with:

```
pyproject.toml
uv.lock
src/nablaframe/
  reports.py              # gate report schema and status rules
  trace/
    schema.py             # RunManifest, TapSpec, TraceEvent (frozen dataclasses)
    registry.py           # canonical tap names, TransformerLens-compatible
    store.py              # append-only event log plus tensor shards
  oracle/
    mode_equality.py      # fast vs instrumented comparison
    tolerance.py          # pinned tolerance records
  lanes/
    jax_ref/              # the G0.6 toy model
    torch_ref/            # cross-check and fallback
gates/
  configs/gpt2_nano.yaml  # vendored from the Marin monorepo
  g01_cpu_train.py
  g02_determinism.py
  g03_throughput.py
  g04_crosscheck.py
  g05_metal_probe.py
  g06_mode_probe.py
  summarize.py
scripts/
  check_tl_names.py       # runtime check of tap names, nf-torch
tests/
  data/transformerlens_legacy_hook_names.txt
reports/phase0/           # generated JSON and markdown, committed
```

**Edit 4**, G0.1 row. Replace:

```
Levanter's own tiny GPT-2 config trains in `nf-jax` on CPU for 1,000 steps
```

with:

```
Levanter's own tiny GPT-2 config (`gates/configs/gpt2_nano.yaml`, run in float32) trains in `nf-jax` on CPU for 1,000 steps
```

**Edit 5**, G0.4 row. Replace:

```
loaded in TransformerLens in `nf-torch`
```

with:

```
loaded through TransformerLens's `TransformerBridge` in `nf-torch` without compatibility mode
```

**Edit 6**, after the paragraph beginning "Every report carries", add:

```
`src/nablaframe/reports.py` also rejects a report whose status contradicts its own measurements, such as a G0.3 pass below 25,000 tokens/s.
```

**Edit 7**, replace the TraceEvent paragraph with:

```
**TraceEvent**, one per recorded tensor. Callers set `run_id`, `step` (training step of the weights), `kind` (forward, grad, update, checkpoint, intervention), `tap`, `batch_hash` and `mode` (fast or instrumented). The store fills `sha256` of the tensor bytes, `supersedes` (the event a correction replaces), `event_id` (the hash of the event's content) and `tensor_ref` (its shard file).
```

## Verification record

**Verified** in a Linux x86_64 sandbox:

- All 26 test cases fail on the stubs, and collection is clean.
- All 26 pass against a private reference implementation, so every test is satisfiable. The reference is not shipped.
- 18 planted bugs in the reference are each caught by the test named for them. The first run missed tensor-less duplicate events, and `test_store_rejects_rewrite` was tightened.
- `uv sync` then `uv run pytest` works on a fresh checkout.
- The jax lane installs; jax 0.11.2 runs on CPU; Levanter imports; Haliax builds a named array.
- `uv.lock` resolves all three lanes: marin-levanter 0.2.129.dev36313914471, jax 0.11.2, metaljax 0.11.8, TransformerLens 4.0.0, torch 2.14.0.
- The 19 tap names were checked against the TransformerLens source at commit `24b039f`.

**Not yet verified:**

- Installation on the Mac, for any lane.
- The torch lane's install, and `scripts/check_tl_names.py`, which has never run.
- Whether `levanter.main.train_lm` accepts the G0.1 overrides as written, and whether the `float32` precision string is accepted.

## Implementation threads

Run 1 and 2 in parallel; they share no files. Run 3 after 1 merges; it uses the schema. Run 4 after 1 to 3 merge.

| # | Files | Tests that must pass | Model, effort |
| --- | --- | --- | --- |
| 1 | `trace/schema.py`, `trace/registry.py` | `test_schema.py`, `test_registry.py` | Sonnet, medium |
| 2 | `oracle/`, `reports.py` | `test_oracle.py`, `test_reports.py` | Opus, high |
| 3 | `trace/store.py` | `test_store.py` | Sonnet, high |
| 4 | `gates/`, `lanes/jax_ref/toy.py` | All 26, plus each gate importing cleanly | Opus, high |

Each thread changes only its listed files and leaves every other test as it found it. Thread 4 writes the gate scripts but never runs a gate for results.

## Gate runs, on the Mac, after thread 4 merges

G0.4 reads G0.1's checkpoint, and G0.5 compares against G0.1's CPU losses, so G0.1 goes first.

```
UV_PROJECT_ENVIRONMENT=.venv-jax   uv run python gates/g01_cpu_train.py
UV_PROJECT_ENVIRONMENT=.venv-jax   uv run python gates/g02_determinism.py
UV_PROJECT_ENVIRONMENT=.venv-jax   uv run python gates/g03_throughput.py
UV_PROJECT_ENVIRONMENT=.venv-jax   uv run python gates/g06_mode_probe.py
UV_PROJECT_ENVIRONMENT=.venv-metal uv run python gates/g05_metal_probe.py
UV_PROJECT_ENVIRONMENT=.venv-torch uv run python scripts/check_tl_names.py
UV_PROJECT_ENVIRONMENT=.venv-torch uv run python gates/g04_crosscheck.py
uv run python gates/summarize.py
```

Commit `reports/phase0/`, then rule D0 at human gate H0.
