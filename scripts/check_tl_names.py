"""Confirm every canonical tap name resolves on a TransformerLens TransformerBridge.

Run in nf-torch:  uv run python scripts/check_tl_names.py

TransformerLens 4.0 removed HookedTransformer. The legacy hook names in
nablaframe.trace.registry survive as aliases on TransformerBridge for pre-norm
GPT-2-style models. This script boots GPT-2 through the bridge, runs one prompt
with a cache, and looks up each name from registry.tap_names(2) (blocks 0 and 1).

Exit code 0 when every name resolves, 1 otherwise, printing the missing names.
A missing name is a finding for apolloh0, not something to patch around: report it.

This file was written against the TransformerLens source at commit 24b039f
(2026-09-21) and has not been executed yet.
"""

from __future__ import annotations

import sys


def main() -> int:
    from transformer_lens.model_bridge import TransformerBridge

    from nablaframe.trace.registry import tap_names

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    _, cache = bridge.run_with_cache("Hello World")

    missing = []
    for name in tap_names(2):
        try:
            cache[name]
        except KeyError:
            missing.append(name)

    if missing:
        print("Names that did not resolve on TransformerBridge:")
        for name in missing:
            print(f"  {name}")
        return 1
    print(f"All {len(tap_names(2))} canonical tap names resolve on TransformerBridge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
