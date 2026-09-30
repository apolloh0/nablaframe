"""Gate G1.1: Tokenizer and corpus.

Spec: docs/specs/phase-1.md, Gates table, G1.1 row. Report: reports/phase1/g11_tokenizer.json.

Downloads ``TinyStoriesV2-GPT4-train.txt`` and ``TinyStoriesV2-GPT4-valid.txt`` from
``roneneldan/TinyStories`` on Hugging Face into ``data/tinystories/raw/`` and records
each file's SHA-256. Cleans both with ``clean_stories``. Then:

* Builds the tokenizer twice at 4,096 and twice at 8,192 from the cleaned train
  stories; ``builds_identical`` is whether each pair has equal JSON.
* ``unk_rate_<size>``: the fraction of validation tokens, not counting the ``<eos>``
  that ``encode`` appends, that encode as ``<unk>``.
* ``unk_rate_4096_cased``: the same at 4,096 without lowercasing. Fold to ASCII
  only, match words with ``[A-Za-z]+(?:'[A-Za-z]+)*`` and everything else one
  character at a time, and build from the 4,094 most frequent cased types. It is
  a measurement only; the tokenizer stays lowercased.
* ``vocab_selected``: 4,096 if ``unk_rate_4096 <= 0.01``, else 8,192 (D8).
* Writes the selected tokenizer to ``data/tinystories/tokenizer.json`` and both
  encoded splits to ``train.bin`` and ``validation.bin`` (little-endian uint16),
  and records their SHA-256s, the token counts and ``stories_dropped``.

Progress: a ``Progress`` line while cleaning, counting and encoding.
Measurements: REQUIRED_MEASUREMENTS["G1.1"], plus the file hashes and story counts.


Every gate: runs on the Mac only, in nf-jax
(``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python gates/phase1/<script>``);
refuses a dirty tree unless ``--allow-dirty``; refuses to overwrite its report unless
``--overwrite``; writes one schema 2 report with ``write_report``, measurements
passed through ``plain``; prints progress at least every 30 s during anything
longer than a minute; returns exit code 0 whenever it wrote a valid report,
whatever the gate's status, and non-zero only when it could not produce one.
"""

from __future__ import annotations

import sys

GATE_ID = "G1.1"
REPORT_PATH = "reports/phase1/g11_tokenizer.json"


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
