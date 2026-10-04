"""G2.4: direct logit attribution (D14).

* Planted, at decoder level, at size S: two equally strong parts, one along each target
  token's direction and one exactly perpendicular to it, as in
  ``tests/test_views_attribution.py``. ``planted_aligned_credit`` is the fraction of
  positions where the aligned part's contribution is positive;
  ``planted_perpendicular_quiet`` the fraction where the perpendicular part's magnitude
  is at most 1e-3 of the aligned part's.
* The decomposition error on the reference run's probe at every trace step:
  ``decomposition_error_by_step`` and its maximum, ``decomposition_error_max``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
