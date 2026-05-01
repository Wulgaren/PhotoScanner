#!/usr/bin/env python3
"""
Write announcements_summary.txt from announcements.txt using Apple Foundation Models
(Swift CLI in tools/AnnouncementsSummarizer/). PCC routing is system-controlled, not app-controlled.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"
SUMMARY_NAME = "announcements_summary.txt"


def default_binary() -> Path:
    env = os.environ.get("PHOTOSCANNER_ANNOUNCEMENTS_SUMMARIZER")
    if env:
        return Path(env).expanduser().resolve()
    arch = platform.machine()  # arm64 or x86_64 on Mac
    triple = f"{arch}-apple-macosx"
    return (
        REPO_ROOT
        / "tools"
        / "AnnouncementsSummarizer"
        / ".build"
        / triple
        / "release"
        / "announcements-summarizer"
    )


def load_announcements_path() -> Path | None:
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    save_dir = cfg.get("save_directory", "~/Pictures/TwitterImages")
    base = Path(save_dir).expanduser()
    return base / "announcements" / "announcements.txt"


def write_announcements_summary(
    announcements_path: Path,
    *,
    binary: Path | None = None,
    dry_run: bool = False,
) -> bool | None:
    """
    Write <same-dir>/announcements_summary.txt from the announcement log, or print when dry_run.

    Returns:
        True if a summary was written (or printed in dry_run),
        None if nothing to do (not macOS, no file, or empty),
        False on failure (I/O, missing binary, summarizer process error).
    """
    if platform.system() != "Darwin":
        return None
    if not announcements_path.is_file():
        return None
    try:
        raw = announcements_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"⚠️  Could not read {announcements_path}: {e}", file=sys.stderr)
        return False
    if not raw.strip():
        return None
    bin_path = binary or default_binary()
    if not bin_path.is_file():
        print(
            "⚠️  announcements-summarizer not found; build: "
            "cd tools/AnnouncementsSummarizer && swift build -c release",
            file=sys.stderr,
        )
        return False
    proc = subprocess.run(
        [str(bin_path), str(announcements_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        print(f"⚠️  Summarizer failed: {err}", file=sys.stderr)
        return False
    body = proc.stdout.rstrip() + "\n"
    if dry_run:
        sys.stdout.write(body)
        return True
    out = announcements_path.parent / SUMMARY_NAME
    out.write_text(body, encoding="utf-8")
    print(f"📝 Wrote {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write announcements_summary.txt from announcements.txt (Apple Foundation Models)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to announcements.txt (default: from config.json save_directory)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to announcements-summarizer (or set PHOTOSCANNER_ANNOUNCEMENTS_SUMMARIZER)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary to stdout only; do not write a file",
    )
    args = parser.parse_args()

    ann: Path
    if args.path is not None:
        ann = args.path.expanduser().resolve()
    else:
        p = load_announcements_path()
        if p is None:
            print("Missing config.json; pass --path to announcements.txt", file=sys.stderr)
            sys.exit(1)
        ann = p

    if not ann.is_file():
        print(f"Not found: {ann}", file=sys.stderr)
        sys.exit(1)

    binary = args.binary.expanduser().resolve() if args.binary else None
    r = write_announcements_summary(ann, binary=binary, dry_run=args.dry_run)
    if r is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
