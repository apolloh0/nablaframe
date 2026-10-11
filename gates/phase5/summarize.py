"""The Phase 5 summary: reports/phase5/summary.md, generated from the six Phase 5 reports.

Same rules as gates/phase4/summarize.py: never written by hand, a missing or invalid report says
so rather than being skipped, and every number comes from a report. Its H5 evidence:

* G5.1: whether each kind trained byte-identically, the cache keys checked and whether they were
  distinct, and the identities exact of those checked, by site
* G5.2 and G5.3, by model: each dictionary's meters beside the bars (its FVU and its twin's, and the
  splice's recovery, or the change it makes where an ablation costs too little to judge); the
  faithful code beside the residual patch, by kind and test, with the single-latent patch beside
  them; the selective latents, trained against twin, with the margin; and, recorded, readability
  and seed robustness. G5.3 names the best swap block it read from G4.3
* G5.4: each MLP's edge meter, trained against twin, with its share of the scored edges and, where
  judged, its ratio beside the bar; the block pairs; the Jacobian SAE at each ``λ`` beside the TopK
  pair, with its Jacobian statistics and both sides' FVU
* G5.5: the reference run's meters, its two MLP pairs' meters, and each dictionary's training time
* G5.6: the badges, each with its status and facts, as Phase 4's summary lists them; whether the
  builds matched; each run's features payloads; and ``query_latents``, the latents active at ``⇒``
  in the two-hop TopK at the best swap block, with the intermediate of their top contexts

The verdict covers G5.1 to G5.4 and G5.6. G5.5 always passes, so its numbers are listed without a
verdict of their own.
"""

# summarize() raises ValueError, naming every problem, when a report is unreadable, invalid
# (validate_report, which includes the status rules), not a Phase 5 report, or when the inputs
# are not exactly one report per gate. main() prints the problems, writes nothing and returns 1.
#
# A pure function of the report files: no timestamps of its own, rows in gate order. Reports
# written from a dirty tree, or from different commits or lockfiles, are named as such. It never
# states a result that is not in a report. apolloh0 rules H5.
#
# Reports are canonical JSON, whose keys come back sorted, so the summary puts them back in the
# spec's order: kinds as topk, staircase, transcoder and jsae; sites and blocks by number; variables
# as the gates list them. Anything else follows, sorted. A rate, an FVU or a share is shown as a
# percentage; an F1, a ratio or a margin to three decimals; a loss or a change in nats to three
# decimals; an edge count or a time as a number. Badges follow views/badges.py's SUBJECTS, then
# views/dictionary_badges.py's, each fact in the badge's own order, its value shown by its unit.
#
# Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
# gates/phase5/summarize.py``.

from __future__ import annotations

from pathlib import Path


def summarize(report_paths: list[Path]) -> str:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
