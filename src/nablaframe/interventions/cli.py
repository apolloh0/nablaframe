"""The compare payload and the command line (D24, D27).

Spec: docs/specs/phase-3.md. Both passes always go through ``run_intervened``: edit values
must enter the compiled program as inputs, and hooks built elsewhere could bake them in as
constants, which XLA may fold, breaking D25's identities (found in PR #38).

* ``compare_payload(model, params, tokens, spec, *, n_layers, resolve, params_at=None,
  taps=())``: ``effect_payload(spec, unintervened logits, intervened logits, tokens)`` with
  three more keys: ``"taps"`` (the requested taps, in order), ``"tokens"`` (nested ints) and
  ``"tap_changes"``, mapping each requested tap to the L2 norm over its trailing axes of the
  intervened value minus the unintervened one, (Batch, Pos), in float64. The unintervened
  pass is ``run_intervened`` with an empty spec.
* ``main(argv)``: ``--config`` (a JSON of ``GPT2Config``'s fields), ``--checkpoints`` (a
  folder of ``step-<n>.safetensors``), ``--step``, ``--tokens`` (an int32 ``.npy``), ``--spec``
  (the spec's JSON), ``--objects`` (a folder of ``<hash>.npy`` files that ``resolve`` reads),
  ``--out``, and ``--taps`` (comma-separated, optional). It writes, in ``--out``:

  - ``intervention-<spec_hash>-step-<step>.json``: the compare payload plus ``"step"``,
    through ``export_view``.
  - ``results/<key>.npy``: the intervened logits, through ``save_result``, under
    ``intervention_key`` with the checkpoint's ``state_hash``, ``batch_hash(tokens)``, the
    spec hash and ``apply.code_identity()`` (D24).
  - ``index.json``, rewritten last as canonical JSON: ``{"interventions": [{"spec_hash",
    "step", "file"}, ...]}``, one entry per payload file in ``--out``, sorted by spec hash
    then step.

  Returns 0. Any error, such as a missing object, prints one line and returns 1 having
  written nothing new.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def compare_payload(model: Any, params: Any, tokens: np.ndarray, spec: Any, *, n_layers: int,
                    resolve: Callable[[str], np.ndarray], params_at: Callable[[int], Any] | None = None,
                    taps: Sequence[str] = ()) -> dict[str, Any]:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
