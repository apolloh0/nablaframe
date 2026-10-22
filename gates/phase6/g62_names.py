"""G6.2: every planted name, on the trained zoo models and their twins (judged; D49).

Spec: docs/specs/phase-6.md, D49, "What each run declares" and "Discrimination tests on the zoo";
helpers in ``_common.py``. Report: ``reports/phase6/g62_names.json``. Run in nf-jax, on the Mac,
after G4.1, G5.2, G5.3 and G6.1 pass.

On G4.1's runs at their final checkpoints and on their twins (``twin_params``), each with its own
Phase 5 TopK SAEs from the cache, read over the held-out windows. For every planted name the run
declares (``declaration.planted``: two-hop's ``b`` after blocks 2 and 1 and ``token(p+1)`` after
block 3, at ``query``; induction's ``token(p)`` after block 0 at ``every`` and ``token(p+1)`` after
block 1 at ``mechanism``), on each model: ``planted_name(reading, class, description,
instances=INSTANCES["name"], seed=SEEDS["name"])``, the reading being the block's TopK SAE's.

``names``: ``{"<run> <description> after block <l> at <class>": {"judged", "trained",
"twin"}}``, ``judged`` the declaration's flag and the other two ``planted_name``'s results. Over the
judged names, each through ``least`` or ``greatest``, so a reading that wasn't made fails:

* ``named_as_planted_min``: the least trained ``named_as_planted``
* ``follows_min`` and ``null_follows_max``: the least trained ``steering["follows"]`` and the
  greatest trained ``steering["null_follows"]``
* ``twin_named_max`` and ``twin_follows_max``: the greatest twin ``named`` and twin
  ``steering["follows"]``

Measurements: REQUIRED_MEASUREMENTS["G6.2"]. Pass needs ``named_as_planted_min >=
BARS["named_as_planted"]``, ``follows_min >= BARS["follows"]``, ``null_follows_max <=
BARS["null_follows"]``, ``twin_named_max <= BARS["twin_named"]`` and ``twin_follows_max <=
BARS["twin_follows"]``. Round 3 read, for two-hop's ``b`` and answer and induction's current and
next token: named as planted 0.812, 1.000, 0.969 and 1.000; follows 0.939, 1.000, 1.000 and 0.951,
the null at most 0.008; on the twins, no value named and follows at most 0.018.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
