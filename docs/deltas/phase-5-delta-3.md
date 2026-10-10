# Phase 5, delta 3: the edge-sparsity meter

Date: 2026-10-10. Spec: [../specs/phase-5.md](../specs/phase-5.md). Rulings: D42 and D43.

This delta ships:

- **A stub for `dictionaries/edges.py`:** SCALAR's edge scores and ablation curve between two dictionaries, the meter's reading, and the Jacobian across an MLP with its recorded statistics. Its docstring is its contract.
- **10 tests** in one new file.
- **One sentence in D42's spec text:** how an MLP pair's root mean square is taken.
- **Delta 3's tests,** listed in the spec's Tests section.

The Jacobian SAE's penalty is already in `train.py` (delta 1). The gates come next.

## Apply

Every file in the zip is new. From the repo root:

```
unzip -n nablaframe-phase5-delta3.zip
python3 apply_phase5_delta3.py
git status
UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest
```

Expect the 10 edge tests to fail, beside whatever the measurements thread has left: `21 failed, 231 passed` before its PR merges, `10 failed, 242 passed` after.

## What `edges.py` holds

A side is a `(dictionary, site)` pair, so the same functions take two TopK SAEs, a Jacobian SAE's two sites, or two blocks of one Staircase SAE.

| Function | Gives |
| --- | --- |
| `mlp_edge_scores`, `block_edge_scores` | every upstream-to-downstream latent pair's score, by integrated gradients from the origin in five steps |
| `mlp_curve`, `block_curve` | the ablation curve at 14 edge counts, the pair spliced in whole, and the count of scored edges |
| `meter` | the reading: the fewest edges after which the KL stays within 10% of the way from one edge's KL to the floor; None if it never settles |
| `mlp_jacobian`, `jacobian_stats` | the Jacobian between two sides' active latents across an MLP, and what D43 records of it |
| `edge_counts`, `kl` | the curve's edge counts and its KL |

The tests check every derivative and ablation against a direct computation written out in the test: forward-mode derivatives in float64, and each downstream latent recomputed one at a time. The meter is checked on round 2's own curves: induction's MLPs read 606 and 252 edges, against their twins' 58,391 and 22,363, as in D42.

### Settled in the contract

These follow the feasibility program the bars were set from:

- **An MLP pair is scored per position.** Each position's MLP is its own computation, so the root mean square is over every position of every window. A block pair, where attention mixes positions, sums each window's attribution over positions and takes the root mean square over windows. The apply script adds this sentence to D42's spec text, which said only "over windows".
- **Ties in the ranking go to the lower upstream id, then the lower downstream id.**
- **A downstream latent with a kept edge whose upstream latents are all silent at a position still takes its value there,** from the decoder's bias alone. A latent with no kept edge is zero.
- **A curve that never settles is read as None.** D43 records this on the reference run's MLPs for the Jacobian SAE. The gates decide what None means for a bar.
- **The Jacobian statistics count a row of zeros as needing no entries.** Feasibility counted every entry for such a row; a row of zeros is rare and has no mass to hold.

## Threads

Paste into the Project:

```
Start one thread now, without proposing it first.

Thread "[P5] Edge-sparsity meter", on Opus at high effort:
Implement src/nablaframe/dictionaries/edges.py against its docstring and docs/specs/phase-5.md (D42, D43). Change only that file. Every replacement enters the compiled program as an argument, as PR #38 established. The gates run the meter on 8 two-hop windows with 1,024 latents a side, and on 2 reference windows of 256 tokens with 2,048: compile each kind of pass once per pair, not once per downstream latent or edge count, and say in the PR how long scoring and the curve take for one pair at two-hop's size. Done when `UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax pytest tests/test_dictionaries_edges.py` passes with no skips, and every test that passed before still does.
```

It shares no files with "[P5] Dictionary measurements", so the two can run at once.

## Verification record

**Verified** from a clone of my mirror of `main` at delta 2:

- **On the stub:** the 10 new tests fail, the 11 measurement tests fail as they did before, and the other 231 pass.
- **On private reference implementations of both modules, not shipped:** all 252 pass with no skips. At two-hop's size on random dictionaries (8 windows, 1,024 latents a side), it scored an MLP pair in 3 seconds and ran its curve in 30, and scored a block pair in 24 and ran its curve in 25, on this container's CPU.
- **Planted bugs:** all 25 are caught by the test aimed at them. They include:
  - edge counts floored instead of rounded, and the KL taken the wrong way round
  - a meter whose floor is only the pair or only the last point, that needs points strictly below its bound, that stops at the first point below, or that lets a curve starting under the floor lower the bound
  - a Jacobian without its mask on silent latents or without GELU's slope, and statistics one entry short, needing more than 90%, or averaged instead of taking the median
  - scores not multiplied by the code, taken at the path's endpoints instead of its midpoints, averaged instead of root mean squared, summed in square over positions, or taken at the code itself rather than along the path
  - a ranking whose ties go to the higher ids, unscored edges counted, or the curve's point count ignored
  - unkept latents keeping their values, the pair spliced through the true MLP output, or the block's upstream left clean
  - a pair read backwards, or across another block's MLP

  The tests also reproduce round 2's readings from its curves (606, 58,391, 252 and 22,363 edges) and its edge counts.

**Not verified:** anything on the Mac, and any meter on trained dictionaries. That's G5.4.
