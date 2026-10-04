"""G2.2: readout fidelity.

Computes the readout views (``attention_pattern``, ``residual_norms``,
``per_token_loss``) for every trace step twice, with two separate ``TraceReader``s, and
compares a seeded sample of 200 displayed values with the traced tensors read
independently through ``TraceStore.load_tensor``.

Measurements: ``sample_size``, ``sample_matches`` (values equal bit for bit) and
``builds_identical`` (the two passes give byte-identical arrays).
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
