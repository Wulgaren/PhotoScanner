#!/usr/bin/env python3
"""
Export one chronological review window to the private phone-site repo.

Window 1 starts at the scan's `after_date` and covers `--window-months` months.
Every photo in the window is included, oldest first. Photos scoring below the
threshold are pre-marked for deletion. The site is replaced per window; the
phone keeps selections by UUID in localStorage.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
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
DEFAULT_WINDOW_MONTHS = 1
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


def add_months(value: datetime, months: int) -> datetime:
    """Calendar-month addition that clamps to the target month's last day."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def window_start(results: dict, photos: list[dict]) -> datetime:
    """Scan cutoff when present, otherwise the earliest dated photo."""
    cutoff = parse_datetime(results.get("after_date"))
    if cutoff is not None:
        return cutoff
    dates = [parse_datetime(photo.get("date")) for photo in photos]
    present = [date for date in dates if date is not None]
    if not present:
        return datetime.now(timezone.utc)
    return min(present).replace(hour=0, minute=0, second=0, microsecond=0)


def last_photo_date(photos: list[dict]) -> datetime | None:
    dates = [parse_datetime(photo.get("date")) for photo in photos]
    present = [date for date in dates if date is not None]
    return max(present) if present else None


def window_bounds(
    start: datetime, months: int, index: int
) -> tuple[datetime, datetime]:
    """Inclusive start / exclusive end for a 1-based window index."""
    if index < 1:
        raise ValueError("window index must be 1 or greater")
    begin = add_months(start, (index - 1) * months)
    end = add_months(start, index * months)
    return begin, end


def total_windows(start: datetime, months: int, last: datetime | None) -> int:
    if last is None or last < start:
        return 0
    count = 0
    cursor = start
    while cursor <= last:
        count += 1
        cursor = add_months(cursor, months)
    return count


def select_window(
    photos: list[dict], start: datetime, end: datetime
) -> list[dict]:
    """Every dated photo in [start, end), oldest first."""
    rows: list[tuple[datetime, float, dict]] = []
    for photo in photos:
        when = parse_datetime(photo.get("date"))
        if when is None or when < start or when >= end:
            continue
        score = photo.get("score")
        rows.append((when, float(score) if score is not None else 0.0, photo))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [photo for _, _, photo in rows]


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


def resolve_window(
    data_path: Path,
    *,
    requested_index: int | None,
    next_window: bool,
    months: int,
) -> tuple[int, int]:
    """Pick the window index and month size, advancing from the current data."""
    existing = current_window(data_path)
    if next_window:
        if not existing:
            raise ValueError("no current window to advance from")
        index = int(existing.get("index", 0)) + 1
        months = int(existing.get("months", months))
        return index, months
    if requested_index is not None:
        return requested_index, months
    if existing:
        raise ValueError(
            f"site already holds window {existing.get('index')}; "
            "use --next-window or --window-index"
        )
    return 1, months


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


def export_window(
    results_path: Path,
    out_dir: Path,
    *,
    threshold: float,
    months: int,
    index: int,
) -> dict[str, Any]:
    results = json.loads(results_path.read_text())
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

    start = window_start(results, all_photos)
    last = last_photo_date(all_photos)
    total = total_windows(start, months, last)
    if total == 0:
        raise ValueError("scan has no dated photos")
    if index > total:
        raise ValueError(f"window {index} is past the last window ({total})")

    begin, end = window_bounds(start, months, index)
    photos = select_window(all_photos, begin, end)
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

    window = {
        "index": index,
        "total": total,
        "months": months,
        "start": begin.isoformat(),
        "end": end.isoformat(),
        "label": format_window_label(begin, end),
    }
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
        "--window-months",
        type=int,
        default=DEFAULT_WINDOW_MONTHS,
        help=f"Months per window for a fresh export (default {DEFAULT_WINDOW_MONTHS})",
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
    photos = [
        photo
        for photo in results.get("photos") or []
        if normalize_uuid(photo.get("uuid"))
    ]
    start = window_start(results, photos)
    last = last_photo_date(photos)
    if args.window_months < 1:
        console.print("[red]--window-months must be 1 or greater[/red]")
        raise SystemExit(1)

    try:
        index, months = resolve_window(
            out_dir / "data.json",
            requested_index=args.window_index,
            next_window=args.next_window,
            months=args.window_months,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    total = total_windows(start, months, last)
    if index < 1 or index > total:
        console.print(
            f"[red]Window {index} is outside 1–{total}.[/red]"
        )
        raise SystemExit(1)

    begin, end = window_bounds(start, months, index)
    threshold = args.threshold
    console.print(f"Results: [cyan]{results_path}[/cyan]")
    console.print(f"Site: [cyan]{out_dir}[/cyan]")
    console.print(
        f"Window [bold]{index}/{total}[/bold] · {format_window_label(begin, end)}"
    )
    console.print(f"Threshold: {threshold}  months: {months}")

    result = export_window(
        results_path,
        out_dir,
        threshold=args.threshold,
        months=months,
        index=index,
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


if __name__ == "__main__":
    main()
