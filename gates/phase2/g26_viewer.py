"""G2.6: the viewer's data and its offline rule (D13).

Runs ``build_views`` for the reference run twice, into two folders, with the vocabulary
from the run's tokenizer and G1.4's validation-loss curve, printing a progress line at
least every 30 seconds. The second build stays at ``runs/phase2/views/`` for the viewer.

Measurements: ``builds_identical`` (every file byte-identical), ``external_urls``
(matches in ``viewer/`` for the pattern in ``tests/test_viewer.py``) and ``export_bytes``
(the export's total size).
"""

# How it measures:
#
# * Run in nf-jax: ``UV_PROJECT_ENVIRONMENT=.venv-jax uv run --extra jax python
#   gates/phase2/g26_viewer.py``. Takes --overwrite and --allow-dirty. It reads G1.6's
#   store after G2.1 has added the logits. --overwrite also lets it replace both build
#   folders.
# * The first build goes to runs/phase2/g26_viewer/first-build/ and stays there, so a
#   difference can be inspected; the second to runs/phase2/views/. Each folder starts
#   empty, so the comparison covers exactly the files each build wrote.
# * The validation-loss curve is ``val_loss_curve`` from G1.4's committed report,
#   reports/phase1/g14_reference.json, and the vocabulary is the tokenizer's, from
#   data/tinystories/tokenizer.json. Parameters for each step come from G1.4's trace
#   checkpoints.
# * ``builds_identical``: both builds wrote the same file names, and each file's bytes
#   are equal. ``files_differing`` names the files that are not.
# * ``external_urls`` counts every match of ``EXTERNAL_URL``, copied from
#   tests/test_viewer.py, in every file directly in viewer/, as that test reads them.
#   ``external_url_matches`` lists them with their file. The page's own files are
#   whatever viewer/ holds when the gate runs.
# * ``export_bytes`` sums the sizes of the second build's files, index.json included.

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common  # noqa: E402

p1 = _common.p1

GATE_ID = "G2.6"
REPORT_NAME = "g26_viewer.json"
# The offline rule's pattern, as tests/test_viewer.py states it.
EXTERNAL_URL = re.compile(r"https?://|src=[\"']//|import\s*\(\s*[\"']//")
G14_REPORT = "reports/phase1/g14_reference.json"
VERSIONS = ("jax", "jaxlib", "equinox", "marin-haliax", "safetensors", "numpy")


def main(argv: list[str] | None = None) -> int:
    args = p1.arg_parser(__doc__).parse_args(argv)
    report = p1.start(args, _common.report_relative(REPORT_NAME))
    p1.require_data()
    ref = _common.reference()
    probe = p1.probe(p1.load_split("validation"), ref.config.seq)
    store_root = _common.g16_store()
    _common.open_reader(store_root, ref.steps, probe)
    curve = p1.phase0.load_report(p1.REPO_ROOT / G14_REPORT)["measurements"].get("val_loss_curve")
    if not curve:
        raise SystemExit(f"{G14_REPORT} has no val_loss_curve. No report written.")
    first = _common.fresh_dir(_common.RUN_DIR / "g26_viewer" / "first-build", overwrite=args.overwrite)
    second = _common.fresh_dir(_common.RUN_DIR / "views", overwrite=args.overwrite)

    vocab = p1.load_tokenizer().vocab
    builds = {label: build(store_root, out, ref, probe, vocab=vocab, curve=curve, label=label)
              for label, out in (("first build", first), ("second build", second))}
    measurements = compare(first, second)
    measurements |= offline(p1.REPO_ROOT / "viewer")
    measurements["export_bytes"] = _common.file_bytes(builds["second build"])
    measurements["files_written"] = len(builds["second build"])
    measurements["views_dir"] = p1.rel(second)
    measurements["first_build_dir"] = p1.rel(first)
    measurements["run_id"] = ref.manifest.run_id
    measurements["versions"] = p1.phase0.package_versions(VERSIONS)
    return p1.write(GATE_ID, status(measurements), measurements, report, overwrite=args.overwrite)


def status(m: dict[str, Any]) -> str:
    """pass exactly when reports.py's G2.6 rule holds."""
    return "pass" if m["builds_identical"] is True and m["external_urls"] == 0 else "fail"


def build(store_root: Path, out: Path, ref: Any, probe: Any, *, vocab: Any, curve: Any, label: str) -> list[Path]:
    """One ``build_views`` of the reference run into ``out``, with a fresh reader and a progress line."""
    from nablaframe.views.build import build_views
    from nablaframe.views.readouts import TraceReader

    reader = TraceReader(store_root)
    progress = p1.Progress(f"views, {label}, steps", len(reader.steps()))
    paths = build_views(reader, out, params_for=ref.params_for, probe=probe, n_layers=ref.config.n_layers,
                        run_id=ref.manifest.run_id, vocab=vocab, val_loss_curve=curve,
                        progress=lambda done, total: progress.advance())
    print(f"{GATE_ID}: {label}: {len(paths):,} files written to {p1.rel(out)}", flush=True)
    return paths


def compare(first: Path, second: Path) -> dict[str, Any]:
    """Whether two build folders hold the same file names with the same bytes."""
    names_a = sorted(path.name for path in Path(first).iterdir())
    names_b = sorted(path.name for path in Path(second).iterdir())
    differing = sorted(set(names_a) ^ set(names_b))
    differing += [name for name in sorted(set(names_a) & set(names_b))
                  if (Path(first) / name).read_bytes() != (Path(second) / name).read_bytes()]
    identical = not differing and bool(names_a)
    print(f"{GATE_ID}: the two builds are {'byte-identical' if identical else 'different'} over {len(names_b):,}"
          " files", flush=True)
    return {"builds_identical": identical, "files_compared": len(set(names_a) | set(names_b)),
            "files_differing": sorted(differing)}


def offline(viewer: Path) -> dict[str, Any]:
    """Matches of the offline rule's pattern in the viewer's files."""
    matches = []
    for path in sorted(Path(viewer).glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            matches += [{"file": path.name, "match": match.group(0)} for match in EXTERNAL_URL.finditer(text)]
    files = sorted(path.name for path in Path(viewer).glob("*") if path.is_file())
    print(f"{GATE_ID}: {len(matches)} external references in {', '.join(files) or 'no files'}", flush=True)
    return {"external_urls": len(matches), "external_url_matches": matches, "viewer_files": files}


if __name__ == "__main__":
    raise SystemExit(main())
