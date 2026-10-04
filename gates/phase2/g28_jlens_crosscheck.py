"""G2.8: the J-lens cross-check against anthropics/jacobian-lens, in the torch lane (D17).

Run with ``UV_PROJECT_ENVIRONMENT=.venv-torch uv run --extra torch``. Refuses to start without
G2.7's outputs. Loads G2.7's export through the reference's own loader, then:

* Computes the reference's J for layer 0 with ``jlens.fitting.jacobian_for_prompt`` on the same
  8 windows, with ``max_seq_len=256``, ``skip_first=16``, and the final block as target, and
  averages the per-window J's equally, as its ``fit`` does. The reference reads text, so it
  must see our exact token ids: either through an adapter that supplies them, or through
  decoded text whose re-tokenization is checked id for id. ``tokens_identical`` records it.
* ``jacobian_relative_difference``: the relative Frobenius difference between the reference's
  J, in float32, never through its float16 save, and G2.7's.
* ``export_logit_max_abs_diff``: the export's logits on G1.6's probe, through the Hugging
  Face model, against the traced logits at step 12,208.

Measurements: ``windows``, ``tokens_identical``, ``export_logit_max_abs_diff``,
``jacobian_relative_difference``.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
