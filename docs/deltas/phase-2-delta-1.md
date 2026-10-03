# Phase 2, delta 1: stubs, tests and contract

Date: 2026-10-03. Spec: [../specs/phase-2.md](../specs/phase-2.md).

This delta ships stubs for work packages 2 to 7, 20 new tests (one shared helper, `tests/phase2_fixtures.py`), and the contract text for work packages 0 and 1, which change five existing files. Stubs for the Phase 2 gates come in a later delta, once the code they drive exists.

## Apply

Unzip at the repo root, then run the edit script from the repo root:

```
python3 apply_phase2_delta1.py
git diff
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect `19 failed, 93 passed`: the 92 existing tests, the viewer's offline rule (which the stub already meets), and 19 new cases waiting on their work packages. Without JAX, plain `uv run pytest` gives `3 failed, 66 passed, 13 skipped`.

## Contract changes for work package 0

**`gates/phase1/g11_tokenizer.py`**

- "`download_url(repo, name, revision)` returns `https://huggingface.co/datasets/<repo>/resolve/<revision>/<name>`. G1.1 takes `--revision`. Without it, G1.1 first resolves the current commit of `main` from Hugging Face's `X-Repo-Commit` header, then downloads both files from that commit, so both always come from one recorded revision. Raw files already in place are still reused."

**`gates/phase1/g12_parity.py`**

- "Before building the export, G1.2 sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in its own process. If the export then fails for want of something online, the gate writes a fail report with `error_stage: "export"`, naming what was requested, so a network dependency is stated rather than silent. A passing report records `hf_offline: true`."

## Contract changes for work package 1 (D12)

**`src/nablaframe/trace/schema.py`**

- `SITES` gains `"unembed"`: "The site of the model's logits (D12)."

**`src/nablaframe/trace/registry.py`**

- "`LOGITS = "logits"`: the model's output, recorded by the engine (D12). It is the one canonical tap that isn't a TransformerLens hook, so `tap_names` does not include it; those stay TransformerLens names only."
- `tap_spec` gains a keyword `vocab: int | None = None`: "For `LOGITS`, the axes are Batch, Pos and Vocab, and the site is `unembed`; `vocab` is required, and omitting it raises ValueError. Other names ignore it."

**`src/nablaframe/trace/engine.py`**

- "`record` and `rematerialize` also accept `LOGITS`. Its tensor is the logits output of the same instrumented pass that produces every other tap, with every canonical tap requested, so recording logits changes neither the compiled program nor any existing event. A store recorded before D12 gains only the `logits` events on a retrace."

## Threads

Paste into the Project. The carry-overs and the logits share no files, and neither do the caches, so these run first, two at a time:

```
Start two threads now, without proposing them first. They share no files.

Thread "[P2] Logits in the trace", on Opus at high effort:
Implement the "Contract changes for work package 1" section of docs/deltas/phase-2-delta-1.md. Write each bullet into the named file's docstring verbatim, then implement it. Change only src/nablaframe/trace/schema.py, src/nablaframe/trace/registry.py and src/nablaframe/trace/engine.py. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase2_carryovers.py -k "not g11"` passes and every test that passed before still does.

Thread "[P2] Phase 1 carry-overs", on Sonnet at medium effort:
Implement the "Contract changes for work package 0" section of docs/deltas/phase-2-delta-1.md. Write each bullet into the named file's docstring verbatim, then implement it. Change only gates/phase1/g11_tokenizer.py and gates/phase1/g12_parity.py. Exercise both changes in a scratch folder, as the gates thread did, and say in the PR what that covered. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_phase2_carryovers.py -k g11` passes and every test that passed before still does.
```

Then, as slots free up and once the logits thread has merged, one thread per package. Each changes only its own files and is done when its test file passes with no skips:

| Thread | Files | Tests | Model, effort | Needs |
| --- | --- | --- | --- | --- |
| [P2] Caches and export | `views/cache.py`, `views/export.py` | `test_views_cache.py` | Sonnet, medium | Nothing |
| [P2] Readouts | `views/readouts.py` | `test_views_readouts.py` | Sonnet, high | Logits |
| [P2] Logit lens | `views/lens.py` | `test_views_lens.py` | Sonnet, high | Logits |
| [P2] Attribution | `views/attribution.py` | `test_views_attribution.py` | Sonnet, high | Logits, lens |
| [P2] J-lens probe | `views/jlens_probe.py` | `test_jlens_probe.py` | Opus, high | Nothing |
| [P2] Viewer | `views/shards.py`, `viewer/index.html`, `viewer/app.js` | `test_viewer.py` | Opus, high | Logits, readouts, caches |

The viewer thread implements the page against the contract in `viewer/app.js`. Its tests cover the hash slice and the offline rule; the page itself is checked by hand at H2.

## Verification record

**Verified** from a fresh clone of my mirror of `main` at `ed9f1fc`, with the jax lane built from the lockfile:

- **On the stubs:** 19 new cases fail and 93 pass, with a clean collection.
- **On a private reference implementation, not shipped** (D12 applied to the merged engine as its contract says, plus every new module): all 112 pass.
- **Planted bugs:** each of 18 is caught by the test aimed at it, as a test failure rather than a collection error. Two came from tightening my own tests before the run. The tiny trace now perturbs every parameter, because at initialization the layer norm's weight is exactly 1 and its bias exactly 0, so a lens ignoring them would have passed. The meter test now checks layer 0's KL against an independent computation, because at the last layer either direction of KL is zero.

**Found while building the reference, and now part of the contract:**

- **Logits come from the existing pass.** The engine already receives the logits on every recorded pass. Storing those, rather than requesting logits as an extra output, keeps the compiled program unchanged, so the 980 events already in the reference store still recompute bit for bit.
- **`tap_names` stays TransformerLens-only.** The committed registry test checks every name against TransformerLens's list.

**Not verified:** anything on the Mac, the viewer page itself, and G1.2 in offline mode.
