"""One progress line about every ``interval_s`` seconds, so a long run never looks hung.

Spec: docs/specs/phase-1.md, "Trainer and checkpoints".

``tick(step, loss, tokens)`` is called after every update with the tokens that
update processed. When at least ``interval_s`` have passed since the last line
(or since construction), it writes one line:

    step 1,200/12,208 | loss 2.431 | 12,100 tokens/s | about 15 min left

The rate covers tokens since the last line. The time left assumes that rate holds.
Callers never call ``tick`` inside a timed measurement.
"""

from __future__ import annotations

import sys
import time
from typing import Callable


class Heartbeat:
    def __init__(self, total_steps: int, interval_s: float = 30.0, *,
                 clock: Callable[[], float] = time.monotonic,
                 write: Callable[[str], object] = lambda line: print(line, file=sys.stderr, flush=True)) -> None:
        raise NotImplementedError

    def tick(self, *, step: int, loss: float, tokens: int) -> None:
        raise NotImplementedError
