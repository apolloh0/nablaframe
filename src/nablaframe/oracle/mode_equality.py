"""Mode-equality oracle: turning instrumentation on never changes what the model computes.

Spec: docs/specs/phase-0.md, "Mode-equality oracle".

The oracle is backend-agnostic: it compares arrays after ``np.asarray``, so JAX
and PyTorch lanes use the same code. Gate G0.6 decides which policy it enforces,
and that decision is recorded as a D-entry. Do not change a policy or a pinned
tolerance here without one.
"""

from __future__ import annotations

import math
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
    _check_policy(policy, tolerance)
    a = np.asarray(fast)
    b = np.asarray(instrumented)

    if a.shape != b.shape or a.dtype != b.dtype:
        return OracleResult(
            passed=False,
            policy=policy,
            max_abs_diff=math.inf,
            n_mismatched=max(a.size, b.size),
            reason=f"fast is {a.dtype}{list(a.shape)}, instrumented is {b.dtype}{list(b.shape)}",
        )

    try:
        n_nonfinite = int(np.count_nonzero(~np.isfinite(a)) + np.count_nonzero(~np.isfinite(b)))
    except TypeError:
        return OracleResult(False, policy, math.inf, a.size, f"cannot compare logits of dtype {a.dtype}")

    n_mismatched = _count_bit_mismatches(a, b)
    if n_nonfinite:
        return OracleResult(False, policy, math.inf, n_mismatched, f"{n_nonfinite} non-finite logit values")

    max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0

    if policy == "bitwise":
        if n_mismatched:
            reason = f"{n_mismatched} of {a.size} logits differ bitwise (max abs diff {max_abs_diff:.3e})"
            return OracleResult(False, policy, max_abs_diff, n_mismatched, reason)
        return OracleResult(True, policy, max_abs_diff, n_mismatched, "")

    assert tolerance is not None
    if max_abs_diff <= tolerance.value:
        return OracleResult(True, policy, max_abs_diff, n_mismatched, "")
    reason = (
        f"max abs logit diff {max_abs_diff:.3e} exceeds pinned tolerance {tolerance.value:.3e}"
        f" from {tolerance.source_report}"
    )
    return OracleResult(False, policy, max_abs_diff, n_mismatched, reason)


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
    _check_policy(policy, tolerance)
    # Copy, so nothing the instrumented run does in place can reach the fast logits.
    fast_logits = np.array(model.fast(params, tokens), copy=True)
    instrumented_logits, recorded = model.instrumented(params, tokens, taps)

    requested, got = set(taps), set(recorded)
    if got != requested:
        missing, unexpected = sorted(requested - got), sorted(map(str, got - requested))
        return OracleResult(
            passed=False,
            policy=policy,
            max_abs_diff=math.inf,
            n_mismatched=0,
            reason=f"recorded taps differ from requested: missing {missing}, unexpected {unexpected}",
        )
    return compare_logits(fast_logits, instrumented_logits, policy, tolerance)


def _check_policy(policy: str, tolerance: ToleranceRecord | None) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if policy == "bitwise" and tolerance is not None:
        raise ValueError("the bitwise policy takes no tolerance")
    if policy == "instrumented_canonical" and not isinstance(tolerance, ToleranceRecord):
        raise ValueError(f"the instrumented_canonical policy requires a ToleranceRecord, got {tolerance!r}")


def _count_bit_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Elements whose bytes differ. ``a`` and ``b`` share shape and dtype."""
    if a.size == 0:
        return 0
    width = a.dtype.itemsize
    a_bytes = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(-1, width)
    b_bytes = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(-1, width)
    return int(np.count_nonzero(np.any(a_bytes != b_bytes, axis=1)))
