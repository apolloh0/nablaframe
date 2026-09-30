"""Follow the most recently written log under a folder, refreshing the screen.

For long-running jobs that write to log files, such as the gate scripts. Run it in
a second terminal, from the repo root:

    python3 scripts/watch.py                 # watches runs/
    python3 scripts/watch.py runs/phase1     # a narrower folder
    python3 scripts/watch.py --lines 12 --every 5

It switches automatically to whichever log was written most recently, so it
follows a job from run to run. The header shows how long ago that log last
changed; if nothing has been written for --stale seconds, it says so. That's
the quickest way to tell a quiet step from a stuck job.

Press Ctrl-C to stop. It only reads files, so stopping it is always safe, even in
the middle of a run. Standard library only; no environment needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def newest_log(folder: Path) -> Path | None:
    logs = [p for p in folder.rglob("*.log") if p.is_file()]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def last_lines(path: Path, n: int, max_bytes: int = 65536) -> list[str]:
    """The last n non-blank lines, treating progress-bar carriage returns as line breaks."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - max_bytes))
        text = f.read().decode("utf-8", errors="replace")
    lines = [line.rstrip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return lines[-n:]


def ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def screen(folder: Path, lines: int, stale: float) -> list[str]:
    log = newest_log(folder) if folder.is_dir() else None
    if log is None:
        return [f"No .log files under {folder}/ yet. Waiting for a job to start writing one."]
    idle = time.time() - log.stat().st_mtime
    out = [f"{log}", f"last written {ago(idle)} ago"]
    if idle > stale:
        out.append(f"!! Nothing new for {ago(idle)}. It may be a long quiet step, or the job may have stopped.")
    out.append("-" * 60)
    return out + last_lines(log, lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Follow the newest log file under a folder.")
    parser.add_argument("folder", nargs="?", default="runs", help="folder to search for *.log files (default: runs)")
    parser.add_argument("--lines", type=int, default=8, help="lines of the log to show (default: 8)")
    parser.add_argument("--every", type=float, default=10.0, help="seconds between refreshes (default: 10)")
    parser.add_argument("--stale", type=float, default=300.0,
                        help="warn when the log hasn't changed for this many seconds (default: 300)")
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    try:
        while True:
            width = shutil.get_terminal_size().columns
            body = [line[:width] for line in screen(folder, args.lines, args.stale)]
            body += ["", f"Refreshing every {args.every:g} s. Press Ctrl-C to stop."]
            sys.stdout.write("\033[2J\033[H" + "\n".join(body) + "\n")
            sys.stdout.flush()
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
