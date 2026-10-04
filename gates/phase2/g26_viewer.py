"""G2.6: the viewer's data and its offline rule (D13).

Runs ``build_views`` for the reference run twice, into two folders, with the vocabulary
from the run's tokenizer and G1.4's validation-loss curve, printing a progress line at
least every 30 seconds. The second build stays at ``runs/phase2/views/`` for the viewer.

Measurements: ``builds_identical`` (every file byte-identical), ``external_urls``
(matches in ``viewer/`` for the pattern in ``tests/test_viewer.py``) and ``export_bytes``
(the export's total size).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
