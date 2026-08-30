from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

from photo_scanner.wallpaper_search.capture import capture_wallpaper, mapped_originals
from photo_scanner.wallpaper_search.hashing import query_hashes
from photo_scanner.wallpaper_search.library import default_library, iter_search_targets, lookup, spotlight
from photo_scanner.wallpaper_search.scan import scan, score_file


def _fmt_dt(value) -> str:
    if value is None:
        return ""
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def print_photo(photo, distance: int) -> None:
    name = photo.original_filename or photo.title or photo.uuid
    print(f"match  (dhash distance {distance})")
    print(f"  file     {name}")
    created = _fmt_dt(photo.created)
    if created:
        print(f"  created  {created}")
    albums = list(photo.albums)
    if photo.favorite and "Favorites" not in albums:
        albums.insert(0, "Favorites")
    if albums:
        print(f"  albums   {', '.join(albums)}")
    print(f"  uuid     {photo.uuid}")
    if photo.original_path:
        print(f"  original {photo.original_path}")


def uuid_from_path(path: Path) -> str | None:
    stem = path.stem
    if len(stem) == 36 and stem.count("-") == 4:
        return stem
    return None


def try_mapped(library: Path, target_full: int, target_center: int, tw: int, th: int):
    hits = []
    for path in mapped_originals():
        uuid = uuid_from_path(path)
        if uuid is None:
            continue
        dist = score_file(path, target_full, target_center, tw, th)
        hits.append((dist, uuid, str(path)))
    hits.sort(key=lambda row: row[0])
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-search",
        description="Find the current macOS wallpaper, or a screenshot, in Apple Photos.",
    )
    parser.add_argument("image", nargs="?", help="screenshot or photo to search for")
    parser.add_argument("--library", type=Path, help="Photos library package")
    parser.add_argument("--no-open", action="store_true", help="print the match, do not open Photos")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=15,
        help="refuse to open Photos above this dhash distance (default 15)",
    )
    args = parser.parse_args(argv)

    library = args.library or default_library()
    if args.image is None and sys.platform != "darwin":
        print("live wallpaper capture is macOS-only; pass an image file", file=sys.stderr)
        return 2
    if args.image:
        query_path = Path(args.image).expanduser()
        if not query_path.is_file():
            print(f"not a file: {query_path}", file=sys.stderr)
            return 2
        query = ImageOps.exif_transpose(Image.open(query_path)).convert("RGB")
    else:
        try:
            query = capture_wallpaper()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    target_full, target_center, tw, th = query_hashes(query)

    ranked: list[tuple[int, str, str]]
    if not args.image:
        mapped_hits = try_mapped(library, target_full, target_center, tw, th)
        if mapped_hits and mapped_hits[0][0] <= 8:
            ranked = mapped_hits
        else:
            print("checking Photos previews...", flush=True)
            ranked = scan(iter_search_targets(library), target_full, target_center, tw, th)
            if mapped_hits:
                ranked = sorted(ranked + mapped_hits, key=lambda row: row[0])
    else:
        print("checking Photos previews...", flush=True)
        ranked = scan(iter_search_targets(library), target_full, target_center, tw, th)

    if not ranked:
        print("no Photos previews to search", file=sys.stderr)
        return 1

    best_dist, uuid, _path = ranked[0]
    try:
        photo = lookup(library, uuid)
    except KeyError:
        print(f"matched preview {uuid} but Photos.sqlite has no row", file=sys.stderr)
        return 1
    print_photo(photo, best_dist)

    if best_dist > args.max_distance:
        print(
            f"weak match (distance {best_dist} > {args.max_distance}); not opening Photos",
            file=sys.stderr,
        )
        print("next:")
        for dist, other_uuid, _ in ranked[1:6]:
            try:
                other = lookup(library, other_uuid)
            except KeyError:
                continue
            label = other.original_filename or other.uuid
            print(f"  {dist:3d}  {label}")
        return 1

    if not args.no_open:
        try:
            spotlight(photo.uuid)
        except Exception as exc:
            print(f"could not open Photos: {exc}", file=sys.stderr)
            return 1
    return 0
