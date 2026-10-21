#!/usr/bin/env python3
"""Apply Phase 6 delta 1's edits: the dsteer and array_hash contract.

Run from the repo root:

    python3 apply_phase6_delta1.py

Run it after unzipping the delta. It edits:
  src/nablaframe/interventions/spec.py   the module docstring gains dsteer's arguments and array_hash (D48)
  src/nablaframe/interventions/apply.py  the module docstring gains how dsteer applies (D48)

Only docstrings change; no code does.

Each edit replaces one exact passage, and each new file must not exist yet. Everything is checked
before anything is written: if a passage is missing or appears more than once, or a new file is
already there, the script stops and writes nothing. A second run stops too.
"""

from __future__ import annotations

import sys
from pathlib import Path

EDITS = {'src/nablaframe/interventions/spec.py': [("spec.py's dsteer contract", '  stored sorted, so two specs that list the same latents in another order hash equal.\n', '  stored sorted, so two specs that list the same latents in another order hash equal.\n* ``OPERATIONS`` gains ``dsteer`` (D48), taking ``dictionary``, ``remove``, ``add`` and\n  ``amounts``, each a content hash: the dictionary\'s ``content_hash``, and each array\'s\n  ``array_hash``.\n* ``array_hash(array)``: the SHA-256 of the canonical JSON of ``{"dtype", "shape", "bytes"}``:\n  the array\'s numpy dtype name (``"int32"``, ``"float32"``), its shape as a list, and its\n  ``tensor_sha256``. Arrays equal in dtype, shape and values hash equal, whatever their byte\n  order. Unlike the byte hash alone, it tells apart arrays whose bytes agree, such as an empty\n  ``remove`` and an empty ``amounts``. ValueError for an array that isn\'t of integers or floats.\n')], 'src/nablaframe/interventions/apply.py': [("apply.py's dsteer contract", '    adds no operation at all, so it reproduces the unintervened pass bit for bit (D25).\n', "    adds no operation at all, so it reproduces the unintervened pass bit for bit (D25).\n  - ``dsteer`` (D48) applies at one of its dictionary's output taps, which picks the site, as\n    ``dpatch``'s does, and ``resolve`` returns its ``dictionary`` as for ``dpatch``. ``resolve``\n    returns its three arrays too, each checked with ``array_hash``: ``remove`` and ``add``, int32\n    arrays shaped (n, 3) whose rows are ``[sequence, position, latent]``, and ``amounts``, float32\n    shaped (len(add),). Each of these raises ValueError before anything runs, beside ``dpatch``'s\n    refusals of the tap and the dictionary: an array of another dtype or shape; a row whose\n    sequence isn't in ``range(batch)``, whose position isn't one the edit selects, or whose latent\n    isn't in ``range(active_width(site))``; a row repeated within ``remove`` or within ``add``; an\n    amount that isn't finite. The codes ``f`` are ``encode`` of the target batch's values at the\n    site's input tap, read as ``dpatch`` reads ``f_target``, and a tap whose values aren't shaped\n    (batch, pos, the dictionary's width there) raises ValueError. The change starts at zero for\n    every sequence, position and latent; each ``remove`` row subtracts ``f`` there, and each\n    ``add`` row adds its amount there, so a latent both removed and added ends at its amount. At\n    the selected positions it adds ``decode_delta`` of the change, so the target's error term is\n    kept. Where the change is zero at every selected position, as with empty ``remove`` and\n    ``add``, removing only latents whose code is zero, or adding only zero amounts, it adds no\n    operation at all, so it reproduces the unintervened pass bit for bit (D25).\n")]}
NEW = {}


def main() -> int:
    root = Path.cwd()
    if not (root / "docs" / "charter.md").is_file():
        print("STOP: run this from the repo root (no docs/charter.md here); nothing written.")
        return 1
    for rel in NEW:
        if (root / rel).exists():
            print(f"STOP: {rel} already exists; nothing written.")
            return 1
    updated: dict[str, tuple[str, int]] = {}
    for rel, edits in EDITS.items():
        path = root / rel
        if not path.is_file():
            print(f"STOP: {rel} not found; nothing written.")
            return 1
        text = path.read_text(encoding="utf-8")
        for label, old, new in edits:
            if new in text:
                print(f"STOP: {label} is already applied in {rel}; nothing written.")
                return 1
            if text.count(old) != 1:
                print(f"STOP: the passage for {label} appears {text.count(old)} times in {rel}, not once; nothing written.")
                return 1
            text = text.replace(old, new, 1)
        updated[rel] = (text, len(edits))
    for rel, text in NEW.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
        print(f"wrote {rel}")
    for rel, (text, count) in updated.items():
        (root / rel).write_text(text, encoding="utf-8")
        print(f"updated {rel}: {count} edit(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
