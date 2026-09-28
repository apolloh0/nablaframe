"""Mode-equality oracle: turning instrumentation on never changes what the model computes.

Spec: docs/specs/phase-0.md, "Mode-equality oracle".

The oracle is backend-agnostic: it compares arrays after ``np.asarray``, so JAX
and PyTorch lanes use the same code. Gate G0.6 decides which policy it enforces,
and that decision is recorded as a D-entry. Do not change a policy or a pinned
tolerance here without one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .tolerance import ToleranceRecord

POLICIES: tuple[str, ...] = ("bitwise", "instrumented_canonical")


class InstrumentedModel(Protocol):
    """What the oracle needs from a model, in either lane."""

    def fast(self, params: Any, tokens: Any) -> Any:
        """Logits with no instrumentation."""
        ...

    def instrumented(
        self, params: Any, tokens: Any, taps: Sequence[str]
    ) -> tuple[Any, dict[str, Any]]:
        """Logits, plus one recorded tensor per requested tap name."""
        ...


@dataclass(frozen=True)
class OracleResult:
    """``max_abs_diff`` is ``inf`` when shapes or dtypes differ or any value is
    non-finite. ``n_mismatched`` counts elements whose bit patterns differ.
    ``reason`` is ``""`` when ``passed`` and says why otherwise."""

    passed: bool
    policy: str
    max_abs_diff: float
    n_mismatched: int
    reason: str


def compare_logits(
    fast: Any,
    instrumented: Any,
    policy: str,
    tolerance: ToleranceRecord | None = None,
) -> OracleResult:
    """Compare fast-mode and instrumented-mode logits under ``policy``.

    * Unknown ``policy``: ValueError.
    * ``"bitwise"`` takes no tolerance: passing one is a ValueError. It passes only
      when every element's bit pattern is identical, so ``-0.0`` and ``0.0`` differ.
    * ``"instrumented_canonical"`` requires a ToleranceRecord: omitting it is a
      ValueError. It passes when ``max_abs_diff <= tolerance.value``.
    * Under either policy, differing shapes or dtypes, or any NaN or infinity in
      either array, fail.
    """
    raise NotImplementedError


def check_mode_equality(
    model: InstrumentedModel,
    params: Any,
    tokens: Any,
    taps: Sequence[str],
    policy: str,
    tolerance: ToleranceRecord | None = None,
) -> OracleResult:
    """Run ``model.fast`` and ``model.instrumented`` on the same inputs and compare.

    Fails, without comparing logits, when the recorded tap names are not exactly
    the requested ``taps``. Otherwise returns ``compare_logits`` on the two logit
    arrays with the same ``policy`` and ``tolerance``.
    """
    raise NotImplementedError
