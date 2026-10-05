"""G3.2: determinism and caching of intervention results (D24).

A seeded sample of 12 specs, covering every operation, run on the reference run's final
checkpoint and probe through the command line's code twice, and once more after deleting the
results folder. Measurements: ``sample_size``, ``recompute_identical`` (how many gave
byte-identical results every time) and ``cache_keys_distinct`` (every spec, checkpoint and
batch gave a different key, and the same ones the same key).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
