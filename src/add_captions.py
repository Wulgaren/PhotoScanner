#!/usr/bin/env python3
"""Fill empty Photos captions from a username map learned from the library."""

from __future__ import annotations

import argparse

from rich.console import Console

from photo_scanner.caption_map import (
    extract_username,
    learn_from_teachers,
    load_map,
    resolve_caption,
    save_map,
)
from photo_scanner.paths import USERNAME_CAPTIONS_PATH, ensure_data_dirs
from photo_scanner.photos_captions import list_smart_album, iter_library_teachers, set_description

console = Console()

ensure_data_dirs()

DEFAULT_ALBUM = "Non-added photos"


def add_captions(album_name: str = DEFAULT_ALBUM, dry_run: bool = False) -> None:
    console.print("\n[bold]Add captions[/bold]")
    console.print(f"Album: {album_name}")
    if dry_run:
        console.print("[yellow]Dry run, no Photos writes, map not saved[/yellow]")
    console.print()

    photos = list_smart_album(album_name)
    need = [p for p in photos if not p.description.strip()]
    skipped = len(photos) - len(need)
    console.print(f"Listed {len(photos)} photos ({skipped} already captioned, {len(need)} empty)")

    mapping = load_map()
    original = dict(mapping)
    labeled = [(photo, extract_username(photo.filename)) for photo in need]
    usernames_needed = {u for _, u in labeled if u}
    missing_users = {u for u in usernames_needed if u not in mapping}

    conflicts: dict[str, set[str]] = {}
    if missing_users:
        console.print(f"\n{len(missing_users)} usernames missing from the map, learning from the library…")
        teachers = iter_library_teachers()
        learned = learn_from_teachers(teachers, existing=mapping)
        for username in missing_users:
            if username in learned.mapping:
                mapping[username] = learned.mapping[username]
        conflicts = {
            username: captions
            for username, captions in learned.conflicts.items()
            if username in missing_users
        }
        if not dry_run:
            save_map(mapping)
            console.print(f"Saved {USERNAME_CAPTIONS_PATH}")

    new_entries = {u: c for u, c in mapping.items() if u not in original}
    if new_entries:
        label = "Learned (not saved)" if dry_run else "Learned"
        console.print(f"\n{label}:")
        for username, caption in sorted(new_entries.items()):
            console.print(f"  {username}={caption}")

    if conflicts:
        console.print()
        for username, captions in sorted(conflicts.items()):
            console.print(f"[yellow]Conflict {username}[/yellow]")
            for caption in sorted(captions):
                console.print(f"  {caption}")
            for photo, u in labeled:
                if u == username:
                    console.print(f"  {photo.filename}")

    applied = 0
    unmapped: list[str] = []
    for photo, username in labeled:
        if username is None or username in conflicts:
            unmapped.append(photo.filename)
            continue
        caption = resolve_caption(username, mapping)
        if not caption:
            unmapped.append(photo.filename)
            continue
        if not dry_run:
            set_description(photo.applescript_id, caption)
            applied += 1
            if applied % 25 == 0:
                console.print(f"[dim]Wrote {applied} captions[/dim]")
        else:
            applied += 1

    console.print()
    verb = "Would caption" if dry_run else "Captioned"
    console.print(f"{verb}: {applied}")
    console.print(f"Already captioned: {skipped}")
    console.print(f"Unmapped: {len(unmapped)}")
    if unmapped:
        for filename in unmapped:
            console.print(f"  {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill empty Photos captions from a username map learned from the library"
    )
    parser.add_argument(
        "--album",
        type=str,
        default=DEFAULT_ALBUM,
        help=f'Smart album to caption (default: "{DEFAULT_ALBUM}")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not write captions or the map",
    )
    args = parser.parse_args()

    try:
        add_captions(album_name=args.album, dry_run=args.dry_run)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
