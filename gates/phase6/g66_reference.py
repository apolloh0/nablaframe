"""G6.6: the gestures on the reference run, at its final checkpoint (recorded; D44).

Spec: docs/specs/phase-6.md, D44 and "The reference run"; helpers in ``_common.py``. Report:
``reports/phase6/g66_reference.json``. Run in nf-jax, on the Mac, after G4.1, G5.5 and G6.1 pass,
with G1.4's run and the reference corpus in place.

On G1.4's run at its final checkpoint, with the Phase 5 TopK SAE at every block from the cache, as
G5.5 trained them, each read over the reference run's held-out windows with
``cli.declaration_for("reference", config)``, whose one class is ``every``. For each block ``l``,
with ``T = reading.table("every")`` and ``rng = numpy.random.default_rng([SEEDS["reference"],
l])``:

* **Names:** ``reference_sample(T, REFERENCE["latents"], rng)``, and each latent's top candidate,
  steps 1 and 2 of D49, named where its F1 is at least ``sets.FLOOR``. ``"names"``: ``{"width",
  "active", "sampled", "named", "by_description", "coverage_median", "seconds"}``: the site's
  active width; how many latents are active on some row; how many were sampled; the share of them
  named; for each description that counts at ``every``, the share of them named as it; the median,
  over the named ones, of ``name.coverage(reading, [latent], value)`` at the name's own
  description; and the seconds it took.
* **Steering:** ``reference_values(T, "token(p+1)", REFERENCE["values"], rng)``, the same
  ``rng`` drawing after the sample, and each value steered with ``name.steer(reading, "every",
  "token(p+1)", v, instances=REFERENCE["per_value"], rng=numpy.random.default_rng(SEEDS["reference"]))``.
  ``"steering"``: ``{"values_named", "values", "by_value", "pooled", "seconds"}``: how many values
  some latent names, before the cut; the values steered; each one's steering; ``pool`` of them;
  the seconds.
* **Open-this-up,** at blocks 1 and above: ``OpenCandidate(window, position,
  T.set_for("token(p+1)", v), None, None)`` at every held-out window and position, window by window
  and positions ascending, where ``token(p+1)`` is a steered value ``v`` and that set's codes there
  sum above zero; then ``open_targets`` of the reading at block ``l - 1``, the reading at ``l`` and
  those candidates, with ``instances=REFERENCE["open"]`` and ``seed=SEEDS["reference"]``.
  ``"open"``: its result and ``"seconds"``; None at block 0.

``blocks``: ``{"<l>": {"names", "steering", "open"}}``. ``seconds``: ``{"total"}`` and each
block's. Measurements: REQUIRED_MEASUREMENTS["G6.6"]. Its status is always ``pass``: it records,
not judges. Round 3 named 30.1%, 27.0%, 18.4% and 8.6% of sampled latents at blocks 0 to 3, mostly
as ``token(p)``, at a median coverage of 0.72 to 0.84; steered 8 to 11 named ``token(p+1)``
values, following 3.2% to 22.7%; and opened their sets within 128 parts for 25%, 45% and 91% of
instances at blocks 1 to 3. Its own definitions differ in places (``name.steer`` draws ``w`` from
every value, not only named ones), so its readings needn't match.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
