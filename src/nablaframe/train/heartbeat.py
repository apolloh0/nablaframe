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
        if total_steps < 1:
            raise ValueError(f"total_steps must be at least 1, got {total_steps}")
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self.total_steps = total_steps
        self.interval_s = interval_s
        self._clock = clock
        self._write = write
        self._last_time = clock()
        self._tokens = 0
        self._updates = 0

    def tick(self, *, step: int, loss: float, tokens: int) -> None:
        self._tokens += tokens
        self._updates += 1
        now = self._clock()
        elapsed = now - self._last_time
        if elapsed < self.interval_s:
            return
        rate = self._tokens / elapsed
        seconds_per_update = elapsed / self._updates
        left = max(self.total_steps - step, 0) * seconds_per_update
        self._write(f"step {step:,}/{self.total_steps:,} | loss {loss:.3f} | {rate:,.0f} tokens/s"
                    f" | about {_duration(left)} left")
        self._last_time = now
        self._tokens = 0
        self._updates = 0


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:,.0f} min"
