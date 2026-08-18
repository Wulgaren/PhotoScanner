#!/usr/bin/env python3
"""
Finish the current web-review session without the UI:
  - save kept photos as training positives (+ feedback history)
  - write confirmed_delete_*.txt
  - add marked deletes to the Photos "To Delete" album
  - mark the session finished
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from review_gui import (
    SESSION_FILE,
    ReviewState,
    load_photos_library_uuids,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export review session: kept → training, deletes → To Delete album"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing or touching Photos",
    )
    parser.add_argument(
        "--no-album",
        action="store_true",
        help="Skip adding deletes to the Photos album",
    )
    args = parser.parse_args()

    console.print("\n[bold blue]PhotoScanner — Export Review Session[/bold blue]\n")

    if not SESSION_FILE.exists():
        console.print(f"[red]No session file at {SESSION_FILE}[/red]")
        return

    try:
        session = json.loads(SESSION_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Could not read session: {e}[/red]")
        return

    if session.get("finished"):
        console.print("[yellow]Session already finished.[/yellow]")
        fr = session.get("finish_result") or {}
        if fr:
            console.print(f"  Deletes: {fr.get('confirmed_delete_count', 0)}")
            console.print(f"  Kept saved: {fr.get('kept_saved', fr.get('kept_count', '?'))}")
            console.print(f"  Album added: {fr.get('album_added')}")
        return

    results_path = Path(session.get("results_path") or "")
    if not results_path.exists():
        console.print(f"[red]Scan results not found: {results_path}[/red]")
        return

    console.print(f"Session: [cyan]{SESSION_FILE}[/cyan]")
    console.print(f"Results: [cyan]{results_path}[/cyan]")
    console.print(f"Mode: [cyan]{session.get('mode', 'grouped')}[/cyan]")
    console.print(f"Decided units: {len(session.get('decided', []))}")
    console.print(f"Marked delete: {len(session.get('confirmed_delete', []))}")

    with open(results_path) as f:
        results = json.load(f)

    threshold = float(session.get("threshold", 0.5))
    mode = session.get("mode") or "grouped"
    page_size = int(session.get("page_size") or 3)
    library_uuids = load_photos_library_uuids()
    state = ReviewState(
        results,
        results_path,
        threshold,
        mode=mode,
        page_size=page_size,
        library_uuids=library_uuids,
    )

    if state.finished:
        console.print("[yellow]Session finished while loading (unexpected).[/yellow]")
        return

    kept = state.kept_photos()
    kept_uuids = [p["uuid"] for p in kept if p.get("uuid")]
    console.print(f"Kept (training positives): [green]{len(kept_uuids)}[/green]")
    console.print(f"Deletes: [red]{len(state.confirmed_delete)}[/red]")

    if args.dry_run:
        console.print("\n[yellow]Dry run — no changes written.[/yellow]")
        return

    add_album = not args.no_album
    console.print(
        "\n[yellow]Finishing session"
        + (" and adding deletes to “To Delete”…" if add_album else " (no album)…")
        + "[/yellow]"
    )
    out = state.finish(add_to_album=add_album)
    r = out.get("result") or {}

    console.print(f"\n[green]✓[/green] Session finished")
    console.print(f"  Kept saved for training: [green]{r.get('kept_saved', 0)}[/green] "
                  f"(of {r.get('kept_count', 0)})")
    console.print(f"  Deletes listed: {r.get('confirmed_delete_count', 0)}")
    if r.get("output_file"):
        console.print(f"  List file: {r['output_file']}")
    if add_album:
        if r.get("album_added"):
            console.print(f"  Added to album: [green]{r.get('album_count', 0)}[/green]")
        else:
            console.print("  [red]Album update failed or no UUIDs[/red]")

    console.print("\n[bold]Next:[/bold]")
    console.print("  1. In Photos, sort the “To Delete” album (remove any you want to keep)")
    console.print("  2. Run [cyan]python src/learn_from_feedback.py[/cyan] or [cyan]./photoscanner.sh[/cyan] (Learn)")
    console.print("  3. Run [cyan]./photoscanner.sh[/cyan] (Train) to retrain")


if __name__ == "__main__":
    main()
