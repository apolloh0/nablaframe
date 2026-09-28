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
| D0 | Pending | Phase 1 lane: JAX or PyTorch | `reports/phase0/summary.md`, ruled at human gate H0 |
| D1 | 2026-09-27 | Levanter and Haliax come from PyPI as `marin-levanter` and `marin-haliax` dev builds, pinned by `uv.lock`. The jax and metal extras conflict with the torch extra. The lock targets macOS arm64 and Linux x86_64 | The `levanter` package stopped in October 2025; one shared resolution dropped TransformerLens to 1.14; the newest build of one Marin dependency shipped an Intel-Mac wheel only |
| D2 | 2026-09-27 | Tap names are TransformerLens legacy names, resolved as aliases on `TransformerBridge`. G0.4 loads through the bridge without compatibility mode | TransformerLens 4.0 removed `HookedTransformer`; compatibility mode reproduces its weight processing, so activations would no longer match the raw weights Levanter exports |
