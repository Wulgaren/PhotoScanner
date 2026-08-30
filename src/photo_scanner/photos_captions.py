"""Apple Photos caption I/O via AppleScript and read-only Photos.sqlite."""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"
_BATCH = 25

_COUNT_SCRIPT = """
on run argv
    tell application "Photos"
        set albumName to item 1 of argv
        if not (exists album albumName) then
            error "Album not found: " & albumName
        end if
        return count of media items of album albumName
    end tell
end run
"""

_SLICE_SCRIPT = """
on run argv
    tell application "Photos"
        set a to album (item 1 of argv)
        set startIdx to item 2 of argv as integer
        set endIdx to item 3 of argv as integer
        set US to ASCII character 31
        set RS to ASCII character 30
        set out to ""
        repeat with i from startIdx to endIdx
            set m to media item i of a
            set uid to id of m as text
            set fn to filename of m as text
            set desc to ""
            try
                set rawDesc to description of m
                if rawDesc is not missing value then set desc to rawDesc as text
            end try
            set out to out & uid & US & fn & US & desc & RS
        end repeat
        return out
    end tell
end run
"""

_SET_SCRIPT = """
on run argv
    tell application "Photos"
        set theItem to media item id (item 1 of argv)
        set description of theItem to (item 2 of argv)
    end tell
end run
"""


@dataclass
class AlbumPhoto:
    uuid: str  # bare UUID without /L0/001
    applescript_id: str  # full id for scripting
    filename: str
    description: str  # "" if missing


def _run_osascript(script: str, *args: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script, *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "osascript failed").strip()
        raise RuntimeError(err)
    return result.stdout


def _parse_item(record: str) -> AlbumPhoto:
    parts = record.split("\x1f")
    if len(parts) < 3:
        raise RuntimeError(f"Bad AppleScript record: {record!r}")
    applescript_id = parts[0].strip()
    filename = parts[1]
    description = "\x1f".join(parts[2:])
    if not applescript_id:
        raise RuntimeError(f"Missing media item id in record: {record!r}")
    return AlbumPhoto(
        uuid=applescript_id.split("/", 1)[0],
        applescript_id=applescript_id,
        filename=filename,
        description=description,
    )


def _parse_records(raw: str) -> list[AlbumPhoto]:
    photos: list[AlbumPhoto] = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\r\n")
        if not rec:
            continue
        photos.append(_parse_item(rec))
    return photos


def _list_slice(album_name: str, start: int, end: int) -> list[AlbumPhoto]:
    raw = _run_osascript(_SLICE_SCRIPT, album_name, str(start), str(end))
    return _parse_records(raw)


def list_smart_album(album_name: str = "Non-added photos") -> list[AlbumPhoto]:
    """Enumerate via AppleScript index. Raise clear error if album missing."""
    n = int(_run_osascript(_COUNT_SCRIPT, album_name).strip())
    if n == 0:
        return []
    photos: list[AlbumPhoto] = []
    for start in range(1, n + 1, _BATCH):
        end = min(start + _BATCH - 1, n)
        if n > _BATCH:
            console.print(f"[dim]Listing {end}/{n}[/dim]")
        try:
            photos.extend(_list_slice(album_name, start, end))
        except RuntimeError:
            for i in range(start, end + 1):
                photos.extend(_list_slice(album_name, i, i))
    if len(photos) != n:
        raise RuntimeError(
            f"Album {album_name!r} listed {len(photos)} items, expected {n}"
        )
    return photos


def set_description(applescript_id: str, caption: str) -> None:
    """Set description of media item id via AppleScript. Raise on failure."""
    _run_osascript(_SET_SCRIPT, applescript_id, caption)


def iter_library_teachers(
    library_path: Path | None = None,
) -> list[tuple[str, str]]:
    """
    Read-only SQL over Photos.sqlite.
    Return list of (original_filename, description) for assets with non-empty description.
    Default library_path = ~/Pictures/Photos Library.photoslibrary
    """
    lib = Path(library_path) if library_path is not None else DEFAULT_LIBRARY
    db_path = lib / "database" / "Photos.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Photos.sqlite not found at {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only=ON")
        rows = con.execute(
            """
            SELECT aaa.ZORIGINALFILENAME, d.ZLONGDESCRIPTION
            FROM ZASSET a
            JOIN ZADDITIONALASSETATTRIBUTES aaa
              ON aaa.Z_PK = a.ZADDITIONALATTRIBUTES
            JOIN ZASSETDESCRIPTION d
              ON d.Z_PK = aaa.ZASSETDESCRIPTION
            WHERE d.ZLONGDESCRIPTION IS NOT NULL
              AND trim(d.ZLONGDESCRIPTION) != ''
            """
        )
        return [(filename or "", description) for filename, description in rows]
    finally:
        con.close()
