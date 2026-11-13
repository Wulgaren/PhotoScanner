#!/usr/bin/env python3
"""
Export one chronological review window to the private phone-site repo.

Window 1 starts at the scan's `after_date`. Each window holds at most
`--window-size` photos, oldest first. Every photo in the window is included;
photos scoring below the threshold are pre-marked for deletion. The site is
replaced per window; the phone keeps selections by UUID in localStorage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console

from learn_from_feedback import normalize_uuid
from photo_scanner.paths import REPO_ROOT
from review_gui import (
    best_photo_index,
    default_photos_library,
    load_latest_results_path,
    load_photos_library_uuids,
    photo_key,
    resolve_readable_image,
)

console = Console()

SITE_THUMB_MAX = 720
SITE_THUMB_QUALITY = 72
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW_SIZE = 1000
SITE_DIR_NAME = "interactivephotoreview"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def is_video_path(path: str | None) -> bool:
    return Path(path or "").suffix.lower() in VIDEO_EXTENSIONS


def load_video_uuids(library: Path | None = None) -> set[str] | None:
    """
    UUIDs that Photos marks as video assets (ZKIND=1).

    Scan can store a JPEG poster frame for a video, so the file extension is not
    enough. Returns None when the library DB cannot be read.
    """
    lib = library or default_photos_library()
    if lib is None:
        return None
    db_path = lib / "database" / "Photos.sqlite"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT ZUUID FROM ZASSET WHERE ZKIND = 1 AND ZUUID IS NOT NULL"
            )
            return {normalize_uuid(u) for (u,) in rows if u}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        console.print(f"[yellow]Photos DB unreadable ({exc}); video filter skipped[/yellow]")
        return None


def load_hidden_uuids(library: Path | None = None) -> set[str] | None:
    """UUIDs in the Photos Hidden album. None when the library DB is unreadable."""
    lib = library or default_photos_library()
    if lib is None:
        return None
    db_path = lib / "database" / "Photos.sqlite"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT ZUUID FROM ZASSET WHERE ZHIDDEN = 1 AND ZUUID IS NOT NULL"
            )
            return {normalize_uuid(u) for (u,) in rows if u}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        console.print(f"[yellow]Photos DB unreadable ({exc}); hidden filter skipped[/yellow]")
        return None


def load_captions(library: Path | None = None) -> dict[str, str]:
    """UUID -> non-empty Photos description. Empty when the library DB is unreadable."""
    lib = library or default_photos_library()
    if lib is None:
        return {}
    db_path = lib / "database" / "Photos.sqlite"
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT a.ZUUID, d.ZLONGDESCRIPTION
                FROM ZASSET a
                JOIN ZADDITIONALASSETATTRIBUTES aaa
                  ON aaa.Z_PK = a.ZADDITIONALATTRIBUTES
                JOIN ZASSETDESCRIPTION d
                  ON d.Z_PK = aaa.ZASSETDESCRIPTION
                WHERE d.ZLONGDESCRIPTION IS NOT NULL
                  AND trim(d.ZLONGDESCRIPTION) != ''
                """
            )
            captions: dict[str, str] = {}
            for uuid, description in rows:
                key = normalize_uuid(uuid)
                text = (description or "").strip()
                if key and text:
                    captions[key] = text
            return captions
        finally:
            conn.close()
    except sqlite3.Error as exc:
        console.print(f"[yellow]Photos captions unreadable ({exc}); captions skipped[/yellow]")
        return {}


def photo_id(photo: dict) -> str:
    return normalize_uuid(photo.get("uuid"))


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def chronological_photos(photos: list[dict]) -> list[dict]:
    """Dated photos oldest first, with ties ordered by score."""
    rows: list[tuple[datetime, float, dict]] = []
    for photo in photos:
        when = parse_datetime(photo.get("date"))
        if when is None:
            continue
        score = photo.get("score")
        rows.append((when, float(score) if score is not None else 0.0, photo))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [photo for _, _, photo in rows]


def window_start_index(
    photos: list[dict],
    *,
    start: datetime | None,
    after_uuid: str | None,
) -> int:
    """Index of the next window's first photo in a chronological list."""
    if after_uuid:
        key = normalize_uuid(after_uuid)
        for index, photo in enumerate(photos):
            if photo_id(photo) == key:
                return index + 1
        if start is None:
            raise ValueError(f"previous window photo {key} is not in this scan")
    if start is not None:
        for index, photo in enumerate(photos):
            when = parse_datetime(photo.get("date"))
            if when is not None and when >= start:
                return index
        return len(photos)
    return 0


def plan_window(
    photos: list[dict],
    *,
    index: int,
    size: int,
    start: datetime | None = None,
    after_uuid: str | None = None,
    offset: int | None = None,
) -> dict:
    """Pick at most `size` photos and build the metadata stored in data.json."""
    if index < 1:
        raise ValueError("window index must be 1 or greater")
    if size < 1:
        raise ValueError("window size must be 1 or greater")
    ordered = chronological_photos(photos)
    begin = offset
    if begin is None:
        begin = window_start_index(ordered, start=start, after_uuid=after_uuid)
    if begin < 0 or begin >= len(ordered):
        raise ValueError("no photos left for this window")
    remaining = len(ordered) - begin
    chunk = ordered[begin : begin + size]
    first = parse_datetime(chunk[0].get("date"))
    if first is None:
        raise ValueError("window photo has no date")
    if begin + size < len(ordered):
        end = parse_datetime(ordered[begin + size].get("date"))
    else:
        end = parse_datetime(chunk[-1].get("date")) + timedelta(seconds=1)
    if end is None:
        raise ValueError("window end has no date")
    last = parse_datetime(chunk[-1].get("date"))
    if last is None:
        raise ValueError("window photo has no date")
    window = {
        "index": index,
        "total": (index - 1) + ceil(remaining / size),
        "size": size,
        "start": first.isoformat(),
        "end": end.isoformat(),
        "label": format_window_label(first, last + timedelta(days=1)),
        "cursor": photo_id(chunk[-1]),
    }
    return {"photos": chunk, "window": window}


def format_window_label(start: datetime, end: datetime) -> str:
    last = end - timedelta(days=1)
    if start.year == last.year:
        return (
            f"{start.strftime('%b')} {start.day} – "
            f"{last.strftime('%b')} {last.day}, {last.year}"
        )
    return (
        f"{start.strftime('%b')} {start.day}, {start.year} – "
        f"{last.strftime('%b')} {last.day}, {last.year}"
    )


def site_thumb_name(item_id: str) -> str:
    key = hashlib.sha1(
        f"site-v1|{item_id}|{SITE_THUMB_MAX}|{SITE_THUMB_QUALITY}".encode("utf-8")
    ).hexdigest()
    return f"{key}.jpg"


def ensure_site_thumb(photo: dict, dest: Path) -> bool:
    """Create the 720px site thumbnail if needed. False when no source image."""
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    src = resolve_readable_image(photo.get("path") or "", uuid=photo.get("uuid"))
    if src is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((SITE_THUMB_MAX, SITE_THUMB_MAX), Image.Resampling.LANCZOS)
            im.save(
                dest,
                "JPEG",
                quality=SITE_THUMB_QUALITY,
                optimize=True,
                progressive=True,
            )
        return True
    except (OSError, ValueError) as exc:
        console.print(f"[yellow]thumb failed for {photo_id(photo)}: {exc}[/yellow]")
        return False


def is_best_in_series(photo: dict, series: dict[int, list[dict]]) -> bool:
    group = series.get(int(photo["series_id"]), [])
    if len(group) <= 1:
        return False
    return photo_key(photo) == photo_key(group[best_photo_index(group)])


def photos_by_series(photos: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for photo in photos:
        groups.setdefault(int(photo["series_id"]), []).append(photo)
    for group in groups.values():
        group.sort(
            key=lambda p: (p.get("score") is not None, p.get("score") or 0),
            reverse=True,
        )
    return groups


def should_premark(score: Any, threshold: float) -> bool:
    return score is not None and float(score) < threshold


def build_item(
    photo: dict,
    series: dict[int, list[dict]],
    threshold: float,
    thumbs_dir: Path,
    captions: dict[str, str],
) -> dict | None:
    item_id = photo_id(photo)
    thumb_name = site_thumb_name(item_id)
    if not ensure_site_thumb(photo, thumbs_dir / thumb_name):
        return None
    score = photo.get("score")
    item = {
        "id": item_id,
        "uuid": item_id,
        "score": round(float(score), 6) if score is not None else None,
        "date": photo.get("date"),
        "thumb": f"thumbs/{thumb_name}",
        "is_best": is_best_in_series(photo, series),
        "default_delete": should_premark(score, threshold),
    }
    caption = captions.get(item_id)
    if caption:
        item["caption"] = caption
    return item


def embedding_backbone() -> str:
    """Backbone tag the saved preference model uses, else the default."""
    from photo_scanner.feature_extractor import DEFAULT_BACKBONE, saved_backbone_from_model

    return saved_backbone_from_model() or DEFAULT_BACKBONE


def capture_features(
    photos: list[dict],
    window_index: int,
    thumbs_dir: Path,
    *,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Embed every photo in a window and store the vector by UUID.

    Both sides of a pair need a vector, so unmarked photos are captured too.
    A photo you delete later leaves Photos, but the vector stays usable.
    Returns (stored, missing).
    """
    from photo_scanner.feedback_features import record_flagged_batch
    from photo_scanner.feature_extractor import FeatureExtractor
    from photo_scanner.thumbs import EMBED_PREPROCESS

    backbone = embedding_backbone()
    console.print(f"[dim]Embedding window {window_index} with {backbone}...[/dim]")
    extractor = FeatureExtractor(model_name=backbone)

    uuid_by_path: dict[str, str] = {}
    score_by_uuid: dict[str, float] = {}
    resolvable = 0
    for photo in photos:
        item_id = photo_id(photo)
        if not item_id:
            continue
        resolvable += 1
        src = resolve_readable_image(photo.get("path") or "", uuid=photo.get("uuid"))
        if src is None:
            thumb = thumbs_dir / site_thumb_name(item_id)
            src = thumb if thumb.is_file() else None
        if src is None:
            continue
        uuid_by_path[str(src)] = item_id
        score = photo.get("score")
        score_by_uuid[item_id] = float(score) if score is not None else 0.0

    features = extractor.extract_batch(list(uuid_by_path), batch_size=batch_size)
    entries = [
        (uuid_by_path[path], feature, score_by_uuid[uuid_by_path[path]])
        for path, feature in features.items()
        if path in uuid_by_path
    ]
    stored = record_flagged_batch(
        entries, backbone, EMBED_PREPROCESS, window=window_index
    )
    # A photo with no readable source, or one whose vector already existed, is
    # not a failure; report only the photos that produced no vector at all.
    missing = resolvable - len({uuid for uuid, _, _ in entries})
    return stored, max(missing, 0)


def prune_thumbs(site_dir: Path, items: list[dict]) -> int:
    """Delete generated thumbnails no longer referenced by data.json."""
    thumbs_dir = site_dir / "thumbs"
    if not thumbs_dir.is_dir():
        return 0
    keep = {
        Path(str(item.get("thumb") or "")).name
        for item in items
        if item.get("thumb")
    }
    removed = 0
    for path in thumbs_dir.glob("*.jpg"):
        if path.name not in keep:
            path.unlink()
            removed += 1
    return removed


def write_data(
    out_dir: Path,
    items: list[dict],
    *,
    scan_date: str,
    results_name: str,
    threshold: float,
    window: dict,
) -> Path:
    data_path = out_dir / "data.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_date": scan_date,
        "results_file": results_name,
        "threshold": threshold,
        "window": window,
        "count": len(items),
        "photos": items,
    }
    if data_path.is_file():
        try:
            prior = json.loads(data_path.read_text())
            if {k: v for k, v in prior.items() if k != "generated_at"} == {
                k: v for k, v in payload.items() if k != "generated_at"
            }:
                payload["generated_at"] = prior.get(
                    "generated_at", payload["generated_at"]
                )
        except (json.JSONDecodeError, OSError):
            pass
    tmp = data_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(data_path)
    return data_path


def current_window(data_path: Path) -> dict | None:
    if not data_path.is_file():
        return None
    try:
        payload = json.loads(data_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    window = payload.get("window")
    return window if isinstance(window, dict) else None


def photos_for_backfill(data_path: Path, all_photos: list[dict]) -> list[dict]:
    """Scan photo records for the window currently published in data.json."""
    try:
        payload = json.loads(data_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    uuids = {
        normalize_uuid(item.get("uuid") or item.get("id"))
        for item in payload.get("photos") or []
    }
    uuids.discard("")
    if not uuids:
        return []
    return [
        photo for photo in all_photos if normalize_uuid(photo.get("uuid")) in uuids
    ]


def window_start_before(
    all_photos: list[dict], start: datetime, size: int, steps: int
) -> datetime:
    """Walk `steps` windows of `size` photos backwards from `start`.

    Windows are counted in photos, not months, so earlier windows are rebuilt
    by taking the last `size` photos before the next window's start. Photos
    deleted since the review are already gone, which is why this can only
    reproduce a window approximately; the date range is the stable part.
    """
    ordered = chronological_photos(all_photos)
    cursor = start
    for _ in range(steps):
        earlier = [
            photo
            for photo in ordered
            if (when := parse_datetime(photo.get("date"))) is not None and when < cursor
        ]
        if not earlier:
            raise ValueError("no photos left before this window")
        cursor = parse_datetime(earlier[-size:][0].get("date"))
        if cursor is None:
            raise ValueError("window photo has no date")
    return cursor


def photos_for_past_window(
    all_photos: list[dict],
    *,
    target: int,
    published: dict,
    size: int,
) -> list[dict]:
    """Rebuild the photo list for an earlier review window.

    Anchors on the published window's start and steps back by whole windows, so
    a past window can be re-embedded after its site thumbnails were pruned.
    """
    current = int(published.get("index") or 0)
    if target >= current:
        raise ValueError(
            f"window {target} is not before the published window {current}"
        )
    anchor = parse_datetime(published.get("start"))
    if anchor is None:
        raise ValueError("published window has no start date")
    start = window_start_before(all_photos, anchor, size, current - target)
    plan = plan_window(all_photos, index=target, size=size, start=start)
    return plan["photos"]


def resolve_window(
    existing: dict | None,
    *,
    requested_index: int | None,
    next_window: bool,
    size: int,
) -> tuple[int, int]:
    """Pick the window index and size, advancing from the current data."""
    if next_window:
        if not existing:
            raise ValueError("no current window to advance from")
        index = int(existing.get("index", 0)) + 1
        size = int(existing.get("size", size))
        return index, size
    if requested_index is not None:
        if existing and requested_index == int(existing.get("index", 0)):
            size = int(existing.get("size", size))
        return requested_index, size
    if existing:
        raise ValueError(
            f"site already holds window {existing.get('index')}; "
            "use --next-window or --window-index"
        )
    return 1, size


def commit_and_push(out_dir: Path, message: str, *, squash: bool = False) -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=out_dir,
            check=False,
            capture_output=True,
            text=True,
        )

    if not (out_dir / ".git").exists():
        console.print("[red]--push needs an initialized git repo[/red]")
        return

    if squash:
        branch = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or "main"
        orphan = f"review-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        checkout = run("checkout", "--orphan", orphan)
        if checkout.returncode != 0:
            console.print(f"[red]git orphan checkout failed: {checkout.stderr.strip()}[/red]")
            return
        run("add", "-A")
        commit = run("commit", "-m", message)
        if commit.returncode != 0:
            console.print(f"[red]git commit failed: {commit.stderr.strip()}[/red]")
            return
        run("branch", "-M", branch)
        push = run("push", "--force", "origin", branch)
        if push.returncode != 0:
            console.print(f"[red]git push failed: {push.stderr.strip()}[/red]")
            return
        run("reflog", "expire", "--expire=now", "--all")
        run("gc", "--prune=now", "--quiet")
        console.print("[green]✓[/green] Pushed squashed site history")
        return

    add = run("add", "-A")
    if add.returncode != 0:
        console.print(f"[red]git add failed: {add.stderr.strip()}[/red]")
        return
    staged = run("diff", "--cached", "--quiet")
    if staged.returncode == 0:
        console.print("[dim]No site changes to commit[/dim]")
        return
    commit = run("commit", "-m", message)
    if commit.returncode != 0:
        console.print(f"[red]git commit failed: {commit.stderr.strip()}[/red]")
        return
    push = run("push")
    if push.returncode != 0:
        console.print(f"[red]git push failed: {push.stderr.strip()}[/red]")
        return
    console.print("[green]✓[/green] Pushed site repo")


def eligible_photos(results: dict) -> list[dict]:
    """Scan photos that can be reviewed: dated, still present, not video or hidden."""
    all_photos = [
        photo
        for photo in results.get("photos") or []
        if normalize_uuid(photo.get("uuid"))
        and not is_video_path(photo.get("path"))
        and not is_video_path(photo.get("path_edited"))
    ]

    library_uuids = load_photos_library_uuids()
    if library_uuids is not None:
        before = len(all_photos)
        all_photos = [
            photo
            for photo in all_photos
            if normalize_uuid(photo.get("uuid")) in library_uuids
        ]
        dropped = before - len(all_photos)
        if dropped:
            console.print(f"[dim]Dropped {dropped:,} photos gone from Photos[/dim]")

    video_uuids = load_video_uuids()
    if video_uuids is not None:
        before = len(all_photos)
        all_photos = [
            photo
            for photo in all_photos
            if normalize_uuid(photo.get("uuid")) not in video_uuids
        ]
        dropped = before - len(all_photos)
        if dropped:
            console.print(f"[dim]Dropped {dropped:,} videos[/dim]")

    hidden_uuids = load_hidden_uuids()
    if hidden_uuids is not None:
        before = len(all_photos)
        all_photos = [
            photo
            for photo in all_photos
            if normalize_uuid(photo.get("uuid")) not in hidden_uuids
        ]
        dropped = before - len(all_photos)
        if dropped:
            console.print(f"[dim]Dropped {dropped:,} hidden photos[/dim]")

    return chronological_photos(all_photos)


def export_window(
    results_path: Path,
    out_dir: Path,
    *,
    threshold: float,
    plan: dict,
    all_photos: list[dict],
    capture: bool = True,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text())
    photos = plan["photos"]
    window = plan["window"]
    series = photos_by_series(all_photos)

    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    captions = load_captions()
    items: list[dict] = []
    skipped = 0
    for number, photo in enumerate(photos, start=1):
        item = build_item(photo, series, threshold, thumbs_dir, captions)
        if item is None:
            skipped += 1
        else:
            items.append(item)
        if number % 100 == 0 or number == len(photos):
            console.print(f"[dim]Thumbs {number}/{len(photos)}[/dim]")

    stored = missing = 0
    if capture:
        stored, missing = capture_features(
            photos, int(window["index"]), thumbs_dir
        )
        console.print(
            f"[green]✓[/green] Embedded {stored} photos "
            f"({missing} without a readable source)"
        )

    data_path = write_data(
        out_dir,
        items,
        scan_date=str(results.get("scan_date") or "unknown"),
        results_name=results_path.name,
        threshold=threshold,
        window=window,
    )
    pruned = prune_thumbs(out_dir, items)
    return {
        "data_path": data_path,
        "items": items,
        "window": window,
        "skipped": skipped,
        "pruned": pruned,
        "embedded": stored,
        "unembedded": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one chronological phone-review window"
    )
    parser.add_argument("--results", type=str, default=None, help="Scan results JSON")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT.parent / SITE_DIR_NAME),
        help=f"Site repo folder (default: ../{SITE_DIR_NAME})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Pre-mark photos below this score (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Photos per window for a fresh export (default {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=None,
        help="Use this 1-based window instead of advancing",
    )
    parser.add_argument(
        "--next-window",
        action="store_true",
        help="Advance from the window currently in data.json",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="git add/commit/push the site repo after writing",
    )
    parser.add_argument(
        "--squash-history",
        action="store_true",
        help="Replace the site repo's history with a single window commit",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Do not embed the window (no pairwise training signal is kept)",
    )
    parser.add_argument(
        "--backfill-features",
        action="store_true",
        help="Embed the window already in data.json, without rewriting the site",
    )
    parser.add_argument(
        "--backfill-window",
        type=int,
        default=None,
        help="Embed an earlier review window (recovers marks from pruned thumbs)",
    )
    parser.add_argument(
        "--no-build-pairs",
        action="store_true",
        help="Skip refreshing the pair dataset after this export",
    )
    args = parser.parse_args()

    console.print("\n[bold blue]PhotoScanner — Export Phone Review[/bold blue]\n")

    if args.results:
        results_path = Path(args.results)
    else:
        results_path = load_latest_results_path()
        if results_path is None:
            console.print("[red]No scan results found.[/red]")
            raise SystemExit(1)

    out_dir = Path(args.out).expanduser().resolve()
    if not (out_dir / "index.html").is_file():
        console.print(
            "[red]Site app missing.[/red] Point --out at the "
            "interactivephotoreview repo (it needs index.html)."
        )
        raise SystemExit(1)

    results = json.loads(results_path.read_text())
    all_photos = eligible_photos(results)
    if args.window_size < 1:
        console.print("[red]--window-size must be 1 or greater[/red]")
        raise SystemExit(1)

    existing = current_window(out_dir / "data.json")
    # Windows are a fixed photo count; take it from the published window so a
    # replay lines up with how it was originally cut.
    size = int((existing or {}).get("size") or args.window_size)

    if args.backfill_window is not None:
        if existing is None:
            console.print("[red]No data.json window to anchor the walk from.[/red]")
            raise SystemExit(1)
        try:
            photos = photos_for_past_window(
                all_photos,
                target=args.backfill_window,
                published=existing,
                size=size,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1)
        stored, missing = capture_features(
            photos, args.backfill_window, out_dir / "thumbs"
        )
        console.print(
            f"[green]✓[/green] Backfilled {stored} features for window "
            f"{args.backfill_window} ({missing} without a readable source)"
        )
        return

    if args.backfill_features:
        if existing is None:
            console.print("[red]No data.json window to backfill.[/red]")
            raise SystemExit(1)
        photos = photos_for_backfill(out_dir / "data.json", all_photos)
        if not photos:
            console.print("[red]No matching photos for the published window.[/red]")
            raise SystemExit(1)
        stored, missing = capture_features(
            photos, int(existing.get("index", 0)), out_dir / "thumbs"
        )
        console.print(
            f"[green]✓[/green] Backfilled {stored} features for window "
            f"{existing.get('index')} ({missing} without a readable source)"
        )
        return

    try:
        index, size = resolve_window(
            existing,
            requested_index=args.window_index,
            next_window=args.next_window,
            size=args.window_size,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    start = None
    after_uuid = None
    offset = None
    if args.next_window:
        start = parse_datetime((existing or {}).get("end"))
        after_uuid = (existing or {}).get("cursor")
    elif args.window_index is not None:
        if existing and index == int(existing.get("index", 0)):
            start = parse_datetime(existing.get("start"))
        else:
            offset = (index - 1) * size

    try:
        plan = plan_window(
            all_photos,
            index=index,
            size=size,
            start=start,
            after_uuid=after_uuid,
            offset=offset,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    window = plan["window"]
    threshold = args.threshold
    console.print(f"Results: [cyan]{results_path}[/cyan]")
    console.print(f"Site: [cyan]{out_dir}[/cyan]")
    console.print(f"Window [bold]{window['index']}/{window['total']}[/bold] · {window['label']}")
    console.print(f"Threshold: {threshold}  size: {size}")

    result = export_window(
        results_path,
        out_dir,
        threshold=args.threshold,
        plan=plan,
        all_photos=all_photos,
        capture=not args.skip_features,
    )
    marked = sum(1 for item in result["items"] if item["default_delete"])
    console.print(
        f"\n[green]✓[/green] {len(result['items'])} photos, "
        f"{marked} below {args.threshold} pre-marked, {result['skipped']} skipped"
    )
    if result["pruned"]:
        console.print(f"[dim]Pruned {result['pruned']} stale thumbs[/dim]")
    console.print(f"Wrote [cyan]{result['data_path']}[/cyan]")

    if args.push:
        commit_and_push(
            out_dir,
            f"Window {index}: {result['window']['label']}",
            squash=args.squash_history,
        )
    else:
        console.print(
            "\n[dim]Next: git push the site repo, or rerun with --push[/dim]"
        )

    if not args.skip_features and not args.no_build_pairs:
        rebuild_pair_dataset(previous_window=index - 1)


def rebuild_pair_dataset(previous_window: int) -> None:
    """Rebuild the frozen pair dataset after exporting a new window.

    The window just finished is the natural held-out exam: its marks are fresh
    and never entered training. If that window has no marks yet (their IDs were
    not imported), fall back to a random split so the dataset still refreshes.
    Failures here never fail the export, because the site is already published.
    """
    from build_pair_dataset import build_dataset

    if previous_window < 1:
        return
    console.print(
        f"\n[bold]Refreshing pair dataset (holdout: window {previous_window})[/bold]"
    )
    try:
        meta = build_dataset(holdout_window=previous_window, quiet=True)
    except ValueError as exc:
        console.print(
            f"[yellow]No holdout for window {previous_window} ({exc}); "
            "using a random split.[/yellow]"
        )
        try:
            meta = build_dataset(quiet=True)
        except ValueError as fallback_exc:
            console.print(f"[yellow]Pair dataset skipped: {fallback_exc}[/yellow]")
            return
    if meta is None:
        console.print(
            "[yellow]Pair dataset skipped: no usable pairs yet.[/yellow]"
        )
        return
    console.print(
        f"[green]✓[/green] Pairs: {meta['n_pairs']:,} across windows "
        f"{meta['windows']} (holdout {meta['test_windows']})"
    )
    console.print(
        "[dim]Marks for the window just reviewed count only after you import "
        "them with move_to_album --uuids-file and rebuild.[/dim]"
    )


if __name__ == "__main__":
    main()
