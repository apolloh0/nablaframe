"""The Phase 6 summary: reports/phase6/summary.md, generated from the seven Phase 6 reports.

Same rules as gates/phase5/summarize.py: never written by hand, a missing or invalid report says
so rather than being skipped, and every number comes from a report. Its H6 evidence:

* G6.1: the identities exact of those checked, by site; the request hashes checked, and whether
  they were distinct and stable; and for each kind, whether its record built twice, and through the
  command line, was identical
* G6.2: each place, judged or recorded, trained beside twin: its values, the shares named and named
  as planted, steering's follows beside the null's with unchanged and ablated, the largest-code
  latents' shares, the set's effect (the shares of values ranked first and last, and the mean rank
  from the top), and coverage at the name's own description
* G6.3: each edge, trained beside twin: follows and kept beside their nulls, the output, through
  the residual alone, the label, and the single latents; the judged edges' bounds beside the bars
* G6.4: each target, trained beside twin: the reading beside its bar, the share within 128 parts,
  random order's share, and the planted set's median rank and share in the top 8
* G6.5: each judged name: the slices' F1 and follows ranges, each seed's F1, follows and cosine,
  the envelope, where the probe reads from, the start offset and step 0; then a table of every
  checkpoint's F1, follows, the null's follows, the probe, the patch, the accuracy and whether the
  name holds
* G6.6: each block of the reference run: the latents named, by description, with their coverage;
  the values named and steered, and the pooled steering; open-this-up's share within 128; and the
  seconds
* G6.7: the badges, each with its status and facts, as Phase 5's summary lists them; whether the
  builds matched; each run's records by kind; whether the reports matched; and the two-hop
  intermediate: its generalize record's envelope, and each value's name record with its label, its
  name and the rank of the named value from the bottom of its set's effect

The verdict covers G6.1 to G6.5 and G6.7. G6.6 always passes, so its numbers are listed without a
verdict of their own.
"""

# summarize() raises ValueError, naming every problem, when a report is unreadable, invalid
# (validate_report, which includes the status rules), not a Phase 6 report, or when the inputs
# are not exactly one report per gate. main() prints the problems, writes nothing and returns 1.
#
# A pure function of the report files: no timestamps of its own, rows in gate order. Reports
# written from a dirty tree, or from different commits or lockfiles, are named as such. It never
# states a result that is not in a report. apolloh0 rules H6.
#
# Reports are canonical JSON, whose keys come back sorted, so the summary puts them back in the
# spec's order: runs as two-hop then induction, places by block, checkpoints by step, blocks by
# number. A rate, a share or a fraction is shown as a percentage; an F1, a ratio or a cosine to three
# decimals; a reading in parts, a rank, a step, a count or a time as a number. Badges follow
# views/badges.py's SUBJECTS, then views/dictionary_badges.py's, then views/gesture_badges.py's, each
# fact in the badge's own order, its value shown by its unit.
#
# Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
# gates/phase6/summarize.py``.

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
