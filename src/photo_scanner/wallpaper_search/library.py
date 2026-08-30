from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


@dataclass
class Photo:
    uuid: str
    original_filename: str | None
    title: str | None
    created: datetime | None
    added: datetime | None
    width: int
    height: int
    favorite: bool
    albums: list[str]
    original_path: Path | None
    preview_path: Path | None


def default_library() -> Path:
    pictures = Path.home() / "Pictures"
    preferred = pictures / "Photos Library.photoslibrary"
    if (preferred / "database" / "Photos.sqlite").is_file():
        return preferred
    libraries = [
        p
        for p in pictures.glob("*.photoslibrary")
        if (p / "database" / "Photos.sqlite").is_file()
    ]
    if not libraries:
        raise FileNotFoundError(f"no Photos library under {pictures}")
    libraries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return libraries[0]


def _apple_date(value: float | None) -> datetime | None:
    if value is None:
        return None
    return APPLE_EPOCH + timedelta(seconds=float(value))


def connect(library: Path) -> sqlite3.Connection:
    db = library / "database" / "Photos.sqlite"
    if not db.is_file():
        raise FileNotFoundError(f"missing {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def preview_path(library: Path, uuid: str) -> Path | None:
    p = library / "resources" / "derivatives" / uuid[0] / f"{uuid}_1_105_c.jpeg"
    return p if p.is_file() else None


def original_path(library: Path, directory: str, filename: str) -> Path | None:
    p = library / "originals" / directory / filename
    return p if p.is_file() else None


def iter_search_targets(library: Path) -> list[tuple[str, str, bool]]:
    """(uuid, preview_path, favorite) favorites first, then the rest."""
    con = connect(library)
    rows = con.execute(
        """
        SELECT ZUUID, ZFAVORITE
        FROM ZASSET
        WHERE ZTRASHEDSTATE = 0 AND ZHIDDEN = 0 AND ZUUID IS NOT NULL
        ORDER BY ZFAVORITE DESC
        """
    ).fetchall()
    out: list[tuple[str, str, bool]] = []
    for row in rows:
        uuid = row["ZUUID"]
        preview = preview_path(library, uuid)
        if preview is None:
            continue
        out.append((uuid, str(preview), bool(row["ZFAVORITE"])))
    return out


def lookup(library: Path, uuid: str) -> Photo:
    con = connect(library)
    row = con.execute(
        """
        SELECT a.ZUUID, a.ZDIRECTORY, a.ZFILENAME, a.ZWIDTH, a.ZHEIGHT, a.ZFAVORITE,
               a.ZDATECREATED, a.ZADDEDDATE, aa.ZORIGINALFILENAME, aa.ZTITLE
        FROM ZASSET a
        LEFT JOIN ZADDITIONALASSETATTRIBUTES aa ON aa.ZASSET = a.Z_PK
        WHERE a.ZUUID = ? COLLATE NOCASE
        """,
        (uuid,),
    ).fetchone()
    if row is None:
        raise KeyError(uuid)
    albums = [
        r[0]
        for r in con.execute(
            """
            SELECT alb.ZTITLE
            FROM ZASSET a
            JOIN Z_34ASSETS j ON j.Z_3ASSETS = a.Z_PK
            JOIN ZGENERICALBUM alb ON alb.Z_PK = j.Z_34ALBUMS
            WHERE a.ZUUID = ? COLLATE NOCASE AND alb.ZTITLE IS NOT NULL AND alb.ZTITLE != ''
            """,
            (uuid,),
        )
    ]
    return Photo(
        uuid=row["ZUUID"],
        original_filename=row["ZORIGINALFILENAME"],
        title=row["ZTITLE"],
        created=_apple_date(row["ZDATECREATED"]),
        added=_apple_date(row["ZADDEDDATE"]),
        width=int(row["ZWIDTH"] or 0),
        height=int(row["ZHEIGHT"] or 0),
        favorite=bool(row["ZFAVORITE"]),
        albums=albums,
        original_path=original_path(library, row["ZDIRECTORY"] or "", row["ZFILENAME"]),
        preview_path=preview_path(library, uuid),
    )


def spotlight(uuid: str) -> None:
    item_id = f"{uuid}/L0/001"
    subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "Photos" to spotlight media item id "{item_id}"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
