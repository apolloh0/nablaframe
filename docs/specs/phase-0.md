# Phase 0 spec: gates, trace schema, oracle harness

Status: ready for implementation. Charter: [../charter.md](../charter.md). Decisions: [../decisions.md](../decisions.md).

## Purpose and decision

Phase 0 produces one recorded decision, D0: which lane Phase 1 is specified against. It answers whether the reference stack trains small models on this Mac deterministically, fast enough, and in agreement with an independent codebase.

The gates are measurements, not builds. Phase 0 writes no interpretability code beyond the trace schema and the oracle harness, which both lanes share.

| Outcome | Condition | Phase 1 lane |
| --- | --- | --- |
| JAX lane | G0.1 through G0.4 pass | JAX + Haliax + Equinox, JAX CPU float32 as reference |
| PyTorch lane | G0.1 or G0.2 fails | nanochat-style PyTorch, CPU float32 as reference, same trace schema |
| Revisit | G0.3 or G0.4 fails in both lanes | Stop and review scale targets before Phase 1 |

G0.5 (metaljax) never decides the lane. It only decides whether an accelerator is available, and a failure there costs speed, not correctness.

## Environments

Three uv environments keep the lanes from fighting over versions. Python 3.12 everywhere, since metaljax requires 3.12 or later. Exact versions live in each environment's lockfile, and the Phase 0 report records what actually resolved.

| Env | Contents | Used by |
| --- | --- | --- |
| `nf-jax` | jax (CPU), equinox, optax, marin-levanter and marin-haliax with their marin-* dependencies, numpy | Reference lane: G0.1, G0.2, G0.3, G0.6 |
| `nf-metal` | `nf-jax` contents plus metaljax on the jax line it pins | Accelerator probe: G0.5 |
| `nf-torch` | torch (CPU and MPS), transformer_lens, safetensors | Cross-check lane and fallback: G0.4 |

The lock resolves one jax version (0.11.2) for marin-levanter and metaljax, so `nf-metal` is the jax lane plus metaljax (`uv sync --extra metal`). Confirm it installs and runs on the Mac before trusting G0.5.

## Files added in Phase 0

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

## Gates

Six gates, each a script under `gates/` that writes one JSON report. G0.1 through G0.4 decide the lane; G0.5 and G0.6 set policy and never block.

| Gate | Measures | Pass | On failure |
| --- | --- | --- | --- |
| G0.1 CPU training | Levanter's own tiny GPT-2 config (`gates/configs/gpt2_nano.yaml`, run in float32) trains in `nf-jax` on CPU for 1,000 steps | Completes with no NaN or Inf; eval loss at step 1,000 is at least 30% below step 0 | PyTorch lane |
| G0.2 Determinism | Same seed twice, plus a run killed at step 500 and resumed; SHA-256 over serialized parameter leaves at step 1,000 | All three hashes identical | Retry single-threaded (for example `--xla_cpu_multi_thread_eigen=false`). Still different: max abs parameter diff ≤ 1e-6 is a conditional pass with a pinned tolerance; larger is a fail |
| G0.3 Throughput | Training tokens/s on CPU at size S (4 layers, d=256, batch 32×256) and size M (4 layers, d=512, batch 64×256) | S ≥ 25,000 tokens/s. M recorded only | Revisit scale targets |
| G0.4 Cross-check | G0.1 checkpoint exported to Hugging Face format, loaded through TransformerLens's `TransformerBridge` in `nf-torch` without compatibility mode, logits compared on a fixed 8×128 batch | Max abs logit diff ≤ 1e-3, and identical top-1 tokens wherever the top-2 margin exceeds 1e-3 | Diagnose export or load path before D0 |
| G0.5 Metal probe | G0.1 config in `nf-metal`: declined ops, loss vs CPU over the first 200 steps, tokens/s | Runs, with loss within 2% relative of CPU at every logged step | No accelerator; CPU only |
| G0.6 Mode probe | A 2-layer, d=128 model under `jit`, forward returning logits only vs logits plus all taps, 3 seeds | Always passes: records bitwise equality or the measured max abs diff | Not applicable; the result sets the oracle policy |

Every report carries the gate id, pass or fail, the measurements, the env lockfile hash, `XLA_FLAGS`, thread count, chip model, and a UTC timestamp. The Phase 0 summary is generated from the six JSON files, never written by hand.

`src/nablaframe/reports.py` also rejects a report whose status contradicts its own measurements, such as a G0.3 pass below 25,000 tokens/s.

## Trace schema v0

The schema is the contract between lanes, so Phase 0 fixes it even though Phase 1 fills it. Three records, all frozen dataclasses with a JSON form, and one rule: any tensor in a trace can be recomputed from its manifest and a checkpoint.

**RunManifest**, one per run. Fields: model config, optimizer and schedule, seeds (parameter init, data order, dropout), corpus id and hash, tokenizer id and hash, git commit and dirty flag, env lockfile hash, backend, `XLA_FLAGS`, checkpoint schedule. `run_id` is the SHA-256 of every field except the creation timestamp, so two runs with one `run_id` must produce identical traces. G0.2 is the first test of that promise.

**TapSpec**, one per tap point. Fields: `name`, `axes` (ordered named axes with sizes), `dtype`, `site` (residual, attention, MLP, embedding, final norm). Names follow TransformerLens hook names, so traces cross-check without a translation table.

**TraceEvent**, one per recorded tensor. Callers set `run_id`, `step` (training step of the weights), `kind` (forward, grad, update, checkpoint, intervention), `tap`, `batch_hash` and `mode` (fast or instrumented). The store fills `sha256` of the tensor bytes, `supersedes` (the event a correction replaces), `event_id` (the hash of the event's content) and `tensor_ref` (its shard file).

| Tap (per layer L) | Axes | Needed for |
| --- | --- | --- |
| `hook_embed`, `hook_pos_embed` | Batch, Pos, Embed | Input side of every lens |
| `blocks.L.hook_resid_pre` | Batch, Pos, Embed | Logit lens, J-lens, patching |
| `blocks.L.hook_resid_mid` | Batch, Pos, Embed | Attention vs MLP attribution |
| `blocks.L.hook_resid_post` | Batch, Pos, Embed | Logit lens, J-lens, patching |
| `blocks.L.attn.hook_pattern` | Batch, Head, QPos, KPos | Attention views, induction detection |
| `blocks.L.attn.hook_z` | Batch, Pos, Head, HeadDim | Head-level patching, OV analysis |
| `blocks.L.hook_attn_out` | Batch, Pos, Embed | Per-component attribution |
| `blocks.L.ln2.hook_normalized` | Batch, Pos, Embed | MLP input for SAEs, transcoders, Jacobian SAEs |
| `blocks.L.hook_mlp_out` | Batch, Pos, Embed | MLP output for the same dictionaries |
| `ln_final.hook_normalized` | Batch, Pos, Embed | Unembedding side of every lens |

The store is append-only: events in JSONL, tensors in safetensors shards. Nothing is edited in place; a correction appends a superseding event that names the one it replaces. One writer per store at a time. An append interrupted between its shard and its log line leaves an orphan shard; retrying reuses it when its content matches and raises IntegrityError when it doesn't.

## Mode-equality oracle

The oracle asserts that turning instrumentation on never changes what the model computes. It compares `fast(params, tokens) -> logits` against `instrumented(params, tokens, taps) -> (logits, tensors)` on the same inputs, and G0.6 decides which policy it enforces.

| Policy | Chosen when | Assertion | Consequence |
| --- | --- | --- | --- |
| Bitwise | G0.6 finds identical logits on all 3 seeds | `array_equal` on logits | Either mode may serve views |
| Instrumented canonical | G0.6 finds any difference | Max abs logit diff ≤ τ, with τ pinned from the G0.6 measurement | Views read only from instrumented mode; fast mode is for training |

A pinned tolerance is a record, not a constant: its value, the report it was measured in, the backend and the dtype. Any change to a policy or τ needs a D-entry in `docs/decisions.md` plus a discrimination test, never a unilateral edit by the implementing agent. A record applies only to logits of its own dtype, and its source report is a committed gate report at `reports/phase<N>/<name>.json` (D3).

The oracle is itself tested against planted defects, so a green oracle means something:

- A no-op tap passes.
- A tap that adds 1e-3 to `blocks.0.hook_resid_post` in place fails under both policies.
- Under the bitwise policy, a single logit perturbed by one ulp fails.

In Phase 0 the harness runs against the G0.6 toy model only. Wiring it to the real model is Phase 1's first task.

## Deliverables and tests

This is the first code delta, so every file is new. From the next delta on, deltas ship new files only, and changes to existing files arrive as explicit append-payload instructions, never as overwritten copies.

| Deliverable | Contents | Done when |
| --- | --- | --- |
| `trace/schema.py`, `trace/registry.py` | RunManifest, TapSpec, TraceEvent; canonical tap names | Schema and registry tests green |
| `trace/store.py` | Append-only JSONL events, safetensors shards, superseding events | Store tests green |
| `oracle/` | Both policies, tolerance records, planted-defect harness | Oracle tests green |
| `gates/g01` to `g06` | One script per gate, one JSON report each | Each runs end to end and its report validates |
| `reports/phase0/summary.md` | Generated from the six JSON reports | Regenerates identically from the same reports |
| D0 draft in `docs/decisions.md` | Drafted from the summary | Ruled at human gate H0 |

Tests Claude Code implements against, stubs first:

- `test_manifest_run_id_excludes_timestamp`
- `test_manifest_run_id_changes_with_each_field`, parametrized over every field
- `test_tapspec_json_roundtrip`
- `test_registry_matches_transformerlens_names`, against a vendored name list, so `nf-jax` never imports TransformerLens
- `test_store_rejects_rewrite`
- `test_store_supersede_links_original`
- `test_event_sha256_matches_tensor_bytes`
- `test_oracle_noop_tap_passes`
- `test_oracle_mutating_tap_fails`
- `test_oracle_one_ulp_fails_under_bitwise`
- `test_tolerance_record_requires_source_report`
- `test_gate_reports_validate`

**Human gate H0:** apolloh0 reviews the generated summary and rules D0. Nothing in Phase 1 starts before that ruling.

## Open questions

- [ ] Vocabulary size: 4,096 or 8,192 words. The larger one lets the lens name more concepts but costs about 1M extra parameters at d=256 with tied embeddings.
- [ ] G0.3's 25,000 tokens/s threshold is derived from published M5 Max numbers. Confirm the chip before trusting it.
- [ ] If Levanter's Hugging Face export does not load cleanly into TransformerLens, is a hand-written weight map acceptable inside G0.4?
- [ ] G0.1 runs on whatever corpus Levanter's tiny config ships with. Should the TinyStories tokenizer land in Phase 0 or Phase 1?
