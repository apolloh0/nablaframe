"""G3.5: the viewer's compare mode and its offline rule (D27).

Runs the command line twice for 6 example specs (one per operation) on the reference run's
final checkpoint and probe, into two folders; the second stays at ``runs/phase3/interventions/``
for the viewer. Measurements: ``builds_identical`` (every file byte-identical),
``external_urls`` (as G2.6 counts them) and ``payloads``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
