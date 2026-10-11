"""G5.4: the edge-sparsity meter on the zoo, against the twins (judged across the MLPs that matter; D42, D43).

Spec: docs/specs/phase-5.md, D42, D43 and "The edge-sparsity meter"; helpers in ``_common.py``.
Report: ``reports/phase5/g54_edges.json``. Run in nf-jax, on the Mac, after G4.1 and G5.1 pass.

On G4.1's runs at their final checkpoints and on their twins, every dictionary through
``cached`` as in G5.2 and G5.3, the meter's ``tokens`` being the first ``METER_WINDOWS`` of each
model's held-out windows.

* **Across every MLP:** a TopK pair, the ``jsae`` at ``lam`` 0.0 at that block, its sites 0 and 1
  the two sides, on both models, with ``train.mlp_weights`` of that block. The MLP's ``cost``:
  ``readout_loss`` at the model's ``READOUT`` of ``ablated_logits`` at the block's
  ``hook_mlp_out``, minus the unintervened pass's, on all its held-out windows. It is judged where
  ``cost >= BARS["ablation_cost"]``. For each model: ``mlp_edge_scores``, ``mlp_curve`` and
  ``meter``. ``edge_pairs``: ``{"<run> mlp <l>": {"cost", "judged", "trained": {"meter",
  "share", "scored_edges", "pair_kl", "curve"}, "twin": {...}, "ratio"}}``, ``share`` being
  ``meter / scored_edges`` (None with the meter) and ``ratio`` ``edge_ratio`` of the two.
  ``judged_mlps``: the judged keys. ``edge_ratio_max``: the greatest ``ratio`` over them, or None
  if any is None or none is judged.
* **Recorded, across every block:** on both models, ``block_edge_scores``, ``block_curve`` and
  ``meter`` for the ``topk`` pair at blocks ``l`` and ``l + 1`` and for the ``staircase``'s sites
  ``l`` and ``l + 1``: ``block_pairs``, ``{"<run> <kind> <l>-><l + 1>": {"trained": {...},
  "twin": {...}}}`` as above.
* **Recorded, the Jacobian SAE:** on the trained models, the ``jsae`` at each of ``LAMBDAS``
  across every MLP, with the ``lam`` 0.0 pair beside it: its ``meter`` as above; ``fvu`` of each
  side on the held-out windows; and ``jacobian_stats`` of ``mlp_jacobian`` on the codes of every
  position of the meter's windows. ``jsae``: ``{"<run> mlp <l>": {"<lam>": {"meter", "share",
  "scored_edges", "pair_kl", "fvu_in", "fvu_out", "jacobian"}}}``, ``"<lam>"`` written as ``"0"``,
  ``"1"`` and ``"4"``.

Measurements: REQUIRED_MEASUREMENTS["G5.4"]. Pass needs ``edge_ratio_max <= BARS["edge_ratio"]``.
Round 2 read, trained against twin: induction's MLP 0, 606 edges against 58,391; its MLP 1, 252
against 22,363; two-hop's MLP 0, 263 against 3,944 (D42).
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
