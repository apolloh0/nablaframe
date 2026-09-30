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

# How it measures:
#
# * Each raw file is downloaded to ``<name>.tmp`` and renamed into place when
#   complete, beside a ``<name>.source.json`` recording the URL and the dataset
#   revision Hugging Face resolved it to. A file already in place is reused, and
#   its hash is what the report records either way.
# * Cleaning streams each file in blocks cut just after a ``<|endoftext|>``, so it
#   gives exactly what ``clean_stories`` gives on the whole file, without holding
#   the raw text in memory twice.
# * The builds, the unknown-word rates and the encoding all use the cleaned
#   stories. Each unknown-word rate uses the first build at its size.
# * ``corpus_sha256`` is ``_common.corpus_sha256()`` of the two streams written,
#   the value G1.4 puts in its manifest.
# * status is pass when both pairs of builds are identical. vocab_selected always
#   follows D8's rule, so a pass also means D8 selected a vocabulary.
# * A download, read or write error means the gate could not measure: no report,
#   exit code 1.

from __future__ import annotations

import codecs
import json
import re
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import _common  # noqa: E402

GATE_ID = "G1.1"
REPORT_PATH = "reports/phase1/g11_tokenizer.json"

REPO = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "validation": "TinyStoriesV2-GPT4-valid.txt"}
URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"
SIZES = (4096, 8192)
UNK_LIMIT = 0.01  # D8
CASED_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\S")
BLOCK_BYTES = 1 << 26
ENCODE_CHUNK = 20_000  # stories per encode_split call, so progress shows while encoding


def main(argv: list[str] | None = None) -> int:
    args = _common.arg_parser(__doc__).parse_args(argv)
    report = _common.start(args, REPORT_PATH)
    outputs = [_common.tokenizer_path(), _common.split_path("train"), _common.split_path("validation")]
    existing = [_common.rel(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{', '.join(existing)} already exist. Pass --overwrite to rebuild them.")
    try:
        raw = {split: download(REPO, name, raw_dir()) for split, name in FILES.items()}
        measurements = measure(raw["train"], raw["validation"], _common.DATA_DIR)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"{GATE_ID} could not measure: {type(exc).__name__}: {exc}. No report written.", file=sys.stderr)
        return 1
    return _common.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def raw_dir() -> Path:
    return _common.DATA_DIR / "raw"


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G1.1 rule holds."""
    rule_ok = m["vocab_selected"] == select_vocab(m["unk_rate_4096"])
    return "pass" if m["builds_identical"] is True and rule_ok else "fail"


def select_vocab(unk_rate_4096: float | None) -> int | None:
    """D8: 4,096 if the unknown-word rate at 4,096 is at most 1.0%, otherwise 8,192."""
    if unk_rate_4096 is None:
        return None
    return 4096 if unk_rate_4096 <= UNK_LIMIT else 8192


def measure(raw_train: Path, raw_validation: Path, out_dir: Path) -> dict[str, Any]:
    """What G1.1 measures from the two raw files; writes the selected tokenizer and both streams to ``out_dir``."""
    from nablaframe.data.corpus import encode_split
    from nablaframe.data.tokenizer import UNK_ID, build_tokenizer

    sizes = small, large = SIZES
    train, train_dropped = clean_file(raw_train, "cleaning train")
    validation, validation_dropped = clean_file(raw_validation, "cleaning validation")
    print(f"{GATE_ID}: {len(train):,} train and {len(validation):,} validation stories kept;"
          f" {train_dropped:,} and {validation_dropped:,} dropped", flush=True)

    builds: dict[int, list[Any]] = {}
    for size in sizes:
        builds[size] = [build_tokenizer(_counting(train, f"building {size:,} ({attempt} of 2)"), size)
                        for attempt in (1, 2)]
    identical = {size: builds[size][0].to_json() == builds[size][1].to_json() for size in sizes}

    unk_rates, unk_counts = {}, {}
    for size in sizes:
        unk, total = unknown_rate(builds[size][0], validation, UNK_ID)
        unk_rates[size], unk_counts[size] = (unk / total if total else None), (unk, total)
    cased_unk, cased_total, cased_vocab_types = cased_unknown_rate(train, validation, small - 2)
    rate_small = unk_rates[small]
    if rate_small is None:
        raise ValueError("the validation split encodes to no tokens, so no unknown-word rate can be measured")
    selected = select_vocab(rate_small)
    tokenizer = builds[selected][0]

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_file, train_file, validation_file = (
        out_dir / "tokenizer.json", out_dir / "train.bin", out_dir / "validation.bin")
    token_counts = {}
    for split, stories, path in (("train", train, train_file), ("validation", validation, validation_file)):
        progress = _common.Progress(f"encoding {split}", len(stories))
        chunks = []
        for i in range(0, len(stories), ENCODE_CHUNK):
            chunks.append(encode_split(stories[i:i + ENCODE_CHUNK], tokenizer))
            progress.advance(min(ENCODE_CHUNK, len(stories) - i))
        stream = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint16)
        del chunks
        _common.write_atomic(path, stream.astype("<u2").tobytes())
        token_counts[split] = int(len(stream))
        del stream
    _common.write_atomic(tokenizer_file, tokenizer.to_json().encode("utf-8"))

    return {
        "vocab_selected": selected,
        "unk_rate_4096": rate_small,
        "unk_rate_8192": unk_rates[large],
        "unk_rate_4096_cased": cased_unk / cased_total if cased_total else None,
        "builds_identical": all(identical.values()),
        "train_tokens": token_counts["train"],
        "validation_tokens": token_counts["validation"],
        "stories_dropped": train_dropped + validation_dropped,
        "stories_dropped_train": train_dropped,
        "stories_dropped_validation": validation_dropped,
        "stories_kept_train": len(train),
        "stories_kept_validation": len(validation),
        "builds_identical_by_size": {str(size): identical[size] for size in sizes},
        "vocab_sizes_built": {str(size): len(builds[size][0].vocab) for size in sizes},
        "unk_counts": {str(size): {"unknown": unk_counts[size][0], "tokens": unk_counts[size][1]} for size in sizes},
        "unk_counts_4096_cased": {"unknown": cased_unk, "tokens": cased_total, "vocab_types": cased_vocab_types},
        "unk_limit": UNK_LIMIT,
        "dataset_files": {split: raw_file_record(path) for split, path in
                          (("train", raw_train), ("validation", raw_validation))},
        "tokenizer_sha256": tokenizer.sha256,
        "tokenizer_json_sha256": _common.sha256_file(tokenizer_file),
        "train_bin_sha256": _common.sha256_file(train_file),
        "validation_bin_sha256": _common.sha256_file(validation_file),
        "corpus_sha256": _common.corpus_sha256(train_file, validation_file),
        "outputs": {"tokenizer": _common.rel(tokenizer_file), "train": _common.rel(train_file),
                    "validation": _common.rel(validation_file)},
    }


def clean_file(path: Path, label: str) -> tuple[list[str], int]:
    """``clean_stories`` over the whole file, read in blocks that end just after a separator."""
    from nablaframe.data.corpus import SEPARATOR, clean_stories

    stories: list[str] = []
    dropped = 0
    progress = _common.Progress(label, path.stat().st_size)
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending = ""
    with open(path, "rb") as f:
        while True:
            block = f.read(BLOCK_BYTES)
            final = not block
            pending += decoder.decode(block, final=final)
            progress.advance(len(block))
            if final:
                end = len(pending)
            else:
                cut = pending.rfind(SEPARATOR)
                if cut < 0:
                    continue
                end = cut + len(SEPARATOR)
            kept, n_dropped = clean_stories(pending[:end])
            stories += kept
            dropped += n_dropped
            pending = pending[end:]
            if final:
                return stories, dropped


def _counting(stories: list[str], label: str) -> Iterator[str]:
    progress = _common.Progress(label, len(stories))
    for story in stories:
        yield story
        progress.advance()


def unknown_rate(tokenizer: Any, stories: Iterable[str], unk_id: int) -> tuple[int, int]:
    """Validation tokens encoded as ``<unk>``, and all validation tokens, leaving out the ``<eos>`` encode appends."""
    unknown = total = 0
    for story in stories:
        ids = tokenizer.encode(story)[:-1]
        total += len(ids)
        unknown += sum(1 for i in ids if i == unk_id)
    return unknown, total


def cased_tokens(story: str) -> list[str]:
    """Folded to ASCII but not lowercased; cased words, and every other non-space character on its own."""
    from nablaframe.data.tokenizer import fold_to_ascii

    return CASED_WORD.findall(fold_to_ascii(story))


def cased_unknown_rate(train: list[str], validation: list[str], n_types: int) -> tuple[int, int, int]:
    """Unknown validation tokens and all validation tokens, against the ``n_types`` most frequent cased train types.

    Ties break alphabetically, as the tokenizer's do. Also returns how many types the vocabulary holds.
    """
    counts: Counter[str] = Counter()
    for story in _counting(train, "counting cased types"):
        counts.update(cased_tokens(story))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    known = {token for token, _ in ranked[:n_types]}
    unknown = total = 0
    for story in validation:
        tokens = cased_tokens(story)
        total += len(tokens)
        unknown += sum(1 for token in tokens if token not in known)
    return unknown, total, len(known)


def raw_file_record(path: Path) -> dict[str, Any]:
    source = path.with_name(path.name + ".source.json")
    record = {"name": path.name, "bytes": path.stat().st_size, "sha256": _common.sha256_file(path),
              "url": None, "revision": None}
    if source.exists():
        recorded = json.loads(source.read_text(encoding="utf-8"))
        record["url"], record["revision"] = recorded.get("url"), recorded.get("revision")
    return record


class _RevisionRecorder(urllib.request.HTTPRedirectHandler):
    """Keeps the ``X-Repo-Commit`` header Hugging Face sends on the redirect to its file store."""

    def __init__(self) -> None:
        self.revision: str | None = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001  urllib's signature
        self.revision = self.revision or headers.get("X-Repo-Commit")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(repo: str, name: str, folder: Path) -> Path:
    """``folder/name``, downloaded from URL unless it is already there."""
    path = folder / name
    if path.exists():
        print(f"{GATE_ID}: using {_common.rel(path)}, already downloaded", flush=True)
        return path
    folder.mkdir(parents=True, exist_ok=True)
    url = URL.format(repo=repo, name=name)
    recorder = _RevisionRecorder()
    opener = urllib.request.build_opener(recorder)
    tmp = path.with_name(path.name + ".tmp")
    print(f"{GATE_ID}: downloading {url}", flush=True)
    try:
        with opener.open(url, timeout=60) as response, open(tmp, "wb") as out:
            revision = recorder.revision or response.headers.get("X-Repo-Commit")
            length = int(response.headers.get("Content-Length") or 0)
            progress = _common.Progress(f"downloading {name}", length)
            while block := response.read(1 << 20):
                out.write(block)
                progress.advance(len(block))
        if length and tmp.stat().st_size != length:
            raise OSError(f"{url} ended after {tmp.stat().st_size:,} of {length:,} bytes")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    source = {"url": url, "revision": revision}
    path.with_name(path.name + ".source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
