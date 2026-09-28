#!/usr/bin/env python3
"""
Write announcements_summary.txt from announcements.txt via Cursor agent CLI
(`agent -p --mode ask`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from photo_scanner.paths import CONFIG_PATH

SUMMARY_NAME = "announcements_summary.txt"

DIGEST_RULES = """\
You write a short announcement digest as flat plaintext bullets.

Output only lines that start with "- " (dash then space). One sentence per item.
Put a blank line between items. No markdown, headings, bold, italics, nested
bullets, numbered lists, links, URLs, or hashtags. No preamble or closing.

Keep only:
1) Album / EP / single releases: artist + title + release date. Include physical
   album release dates even when buried inside a tour post. Skip pre-order
   windows and version lists.
2) Drops in the next ~48 hours (trailer, medley, highlight, etc.): one short line.
3) Live / fansign / offline / ticket news ONLY for Europe:
   - If Poland (Warsaw, Kraków, etc.) is named, you MUST keep it: artist + city + date.
   - Else if Europe tour with no Poland, one Europe line.
   - Drop Japan, Korea, US, and every other non-Europe region entirely.

Drop everything else: TV/livestream, season's greetings, merch calendars,
concept-photo schedules, tracklists, fan sentiment, TikToks, slogans, sources.
"""


def default_agent() -> Path | None:
    env = os.environ.get("PHOTOSCANNER_AGENT")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_file() else None
    found = shutil.which("agent")
    return Path(found).resolve() if found else None


def load_announcements_path() -> Path | None:
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    save_dir = cfg.get("save_directory", "~/Pictures/TwitterImages")
    base = Path(save_dir).expanduser()
    return base / "announcements" / "announcements.txt"


def build_prompt(raw: str) -> str:
    return (
        f"{DIGEST_RULES}\n"
        "Summarize the announcements below. Output only the digest bullets.\n\n"
        f"{raw.strip()}\n"
    )


def normalize_digest(text: str) -> str:
    """Force flat spaced bullets even if the model nests or packs lines."""
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            items.append(line)
        elif line.startswith("-"):
            rest = line[1:].strip()
            items.append(f"- {rest}" if rest else "-")
        elif line[0].isdigit() and "." in line[:4]:
            dot = line.index(".")
            if line[:dot].isdigit():
                rest = line[dot + 1 :].strip()
                items.append(f"- {rest}" if rest else f"- {line}")
            else:
                items.append(f"- {line}")
        else:
            items.append(f"- {line}")
    if not items:
        return ""
    return "\n\n".join(items) + "\n"


def run_agent_digest(raw: str, *, agent: Path) -> str:
    prompt = build_prompt(raw)
    proc = subprocess.run(
        [
            str(agent),
            "-p",
            "--mode",
            "ask",
            "--trust",
            "--output-format",
            "text",
            prompt,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(err)
    body = (proc.stdout or "").strip()
    if not body:
        raise RuntimeError("agent returned empty digest")
    return normalize_digest(body)


def write_announcements_summary(
    announcements_path: Path,
    *,
    agent: Path | None = None,
    dry_run: bool = False,
) -> bool | None:
    """
    Write <same-dir>/announcements_summary.txt from the announcement log, or print when dry_run.

    Returns:
        True if a summary was written (or printed in dry_run),
        None if nothing to do (no file, or empty),
        False on failure (I/O, missing agent, agent process error).
    """
    if not announcements_path.is_file():
        return None
    try:
        raw = announcements_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"⚠️  Could not read {announcements_path}: {e}", file=sys.stderr)
        return False
    if not raw.strip():
        return None
    agent_path = agent or default_agent()
    if agent_path is None or not agent_path.is_file():
        print(
            "⚠️  Cursor agent CLI not found; install/login so `agent` is on PATH "
            "(or set PHOTOSCANNER_AGENT)",
            file=sys.stderr,
        )
        return False
    try:
        body = run_agent_digest(raw, agent=agent_path)
    except RuntimeError as e:
        print(f"⚠️  Summarizer failed: {e}", file=sys.stderr)
        return False
    if dry_run:
        sys.stdout.write(body)
        return True
    out = announcements_path.parent / SUMMARY_NAME
    out.write_text(body, encoding="utf-8")
    print(f"📝 Wrote {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write announcements_summary.txt from announcements.txt (Cursor agent CLI)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Path to announcements.txt (default: from config.json save_directory)",
    )
    parser.add_argument(
        "--agent",
        type=Path,
        help="Path to agent binary (or set PHOTOSCANNER_AGENT; default: agent on PATH)",
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

    agent = args.agent.expanduser().resolve() if args.agent else None
    r = write_announcements_summary(ann, agent=agent, dry_run=args.dry_run)
    if r is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
