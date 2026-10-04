"""G2.1: logits in the trace (D12).

Opens a ``TraceRun`` on the reference store with G1.4's manifest and checkpoints and
G1.6's probe, and records every canonical tap plus ``LOGITS`` at each of the 28 trace
steps. Identical events are skipped, so only logits are added, and a rerun adds nothing.
A ``NotReproducible`` is a finding: a fail report with ``error``.

Measurements: ``logits_events`` (current logits events after the run),
``logits_events_expected`` (the number of trace steps), ``other_events_written`` (events
this run wrote for any other tap), ``sample_size`` (20) and ``sample_matches`` (a seeded
sample of logits events that verify bit for bit), ``logits_bytes`` (the total size of the
logits events' shard files) and ``logits_bytes_expected``
(``logits_bytes_expected(28, 8, 256, 4096)``). Counting totals rather than new events
makes a rerun after success pass again.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
