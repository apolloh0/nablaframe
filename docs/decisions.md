# Decisions

Every ruling that shapes Nablaframe lands here, with its reason and its evidence. Entries are never edited after they are ruled; a later entry supersedes an earlier one by naming it.

- **C-entries** change the [charter](charter.md).
- **D-entries** are phase rulings: lanes, tolerances, oracle policy, test-defect rulings.
- Oracle changes need a D-entry plus a discrimination test, never a unilateral edit by the implementing agent.

## Charter decisions

| ID | Date | Decision | Reason |
| --- | --- | --- | --- |
| C1 | 2026-09-25 | Name is Nablaframe; repo and package `nablaframe`, imported as `nf` | Coined, so free of collisions; ∇ plus frame names the mechanism |
| C2 | 2026-09-25 | Levanter is a pip dependency, never a fork | Its monorepo now targets far larger models |
| C3 | 2026-09-25 | JAX on CPU is the reference backend; metaljax is optional | Official and reproducible; the community backend is young |
| C4 | 2026-09-25 | TinyStories uses a word-level tokenizer of a few thousand tokens | The J-lens needs vocabulary larger than d_model and single-token concepts to be readable |
| C5 | 2026-09-25 | Tap names follow TransformerLens and include MLP input and output | Free cross-checks; any dictionary can train later from stored traces |
| C6 | 2026-09-25 | Oracle changes need a ruling and a discrimination test | Carried over from Glasshouse |
| C7 | 2026-09-25 | Workspace-style results are hypotheses, with nulls expected and shown | The source paper's evidence points to scale |
| C8 | 2026-09-25 | The mark is two neurons whose processes outline the ∇ and meet at synapses; the dark tile is the default in docs and as the icon | The ∇ sits in the synaptic gaps, where learning happens; the dark tile holds up at icon sizes |

## Phase decisions

| ID | Date | Decision | Evidence |
| --- | --- | --- | --- |
| D0 | 2026-09-29 | Phase 1 is specified against the JAX lane: JAX, Haliax and Equinox, with JAX on CPU in float32 as the reference | G0.1 pass; G0.2 bit-identical parameters across two fresh runs and a killed-and-resumed run, on all 12 threads; G0.4 pass, max abs logit diff 6.7e-6 and top-1 agreement 1.0 over 1,024 positions; G0.3 ruled in D5. The PyTorch lane could use the GPU through MPS, but bitwise replay is what the charter ranks first. `reports/phase0/summary.md` |
| D1 | 2026-09-27 | Levanter and Haliax come from PyPI as `marin-levanter` and `marin-haliax` dev builds, pinned by `uv.lock`. The jax and metal extras conflict with the torch extra. The lock targets macOS arm64 and Linux x86_64 | The `levanter` package stopped in October 2025; one shared resolution dropped TransformerLens to 1.14; the newest build of one Marin dependency shipped an Intel-Mac wheel only |
| D2 | 2026-09-27 | Tap names are TransformerLens legacy names, resolved as aliases on `TransformerBridge`. G0.4 loads through the bridge without compatibility mode | TransformerLens 4.0 removed `HookedTransformer`; compatibility mode reproduces its weight processing, so activations would no longer match the raw weights Levanter exports |
| D3 | 2026-09-28 | A tolerance record applies only to logits of its own dtype: the canonical policy raises ValueError otherwise. Its source report must match `reports/phase<N>/<name>.json` | A tolerance measured on float32 logits means nothing on float16 ones; the pattern merged in PR #2 accepted `reports/../x.json`. Discrimination tests: `tests/test_oracle_d3.py`, which fails on the merged oracle |
| D4 | 2026-09-28 | Test defect: two G0.2 fixtures in `tests/test_reports.py` claimed `identical: false` while their three hashes agreed. Corrected to differing hashes | The fixtures contradicted themselves; the new G0.2 consistency rule rightly rejects them. The corrected test passes on the validator merged in PR #2 |
| D5 | 2026-09-29 | G0.3's threshold of 25,000 tokens/s is replaced by a needs-based criterion: an S-size run of 100M tokens finishes within 8 hours on the reference Mac, which means at least 3,500 tokens/s at vocabulary 8,192. G0.3's report stays a fail against the original threshold. Routine work stays at S to M sizes; the vocabulary choice stays open for Phase 1 | The 25,000 figure came from published M5 Max numbers; the reference Mac is an M2 Max, as the spec's open question warned. Measured: S 12,260 tokens/s at vocabulary 8,192 and 15,255 at 4,096; M 4,898 and 5,415. `reports/phase0/g03_throughput.json` |
| D6 | 2026-09-29 | The oracle enforces the bitwise policy on the JAX CPU lane; no tolerance is pinned. A change of lockfile, backend or logits dtype re-runs G0.6 before the policy carries over | G0.6 found fast and instrumented logits bit-identical on all three seeds. `reports/phase0/g06_mode_probe.json` |
| D7 | 2026-09-29 | Nablaframe owns its reference model (GPT-2 in JAX, taps built in), training loop, batch order and checkpoints. Levanter stays a dependency as the parity reference (G1.2) and for Hugging Face export; the Phase 0 gate scripts keep using its trainer | Taps can't be added to Levanter's model without patching, and the model you read should be code you own. Batch k depending only on the seed and k makes replay exact by construction. Costs: determinism is proven again (G1.3), and continuity is kept by logit parity with Levanter (G1.2) |
| D8 | 2026-09-29 | The vocabulary is 4,096 if G1.1 measures an unknown-word rate of at most 1.0% on the TinyStories validation split at 4,096, otherwise 8,192. The rule is fixed before the measurement | 4,096 is 24% faster at size S (G0.3: 15,255 against 12,260 tokens/s) and gives embeddings 25% of the parameters instead of 39%; the J-lens can't name a word mapped to `<unk>` |
