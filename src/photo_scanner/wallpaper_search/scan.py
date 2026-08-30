from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from photo_scanner.paths import SRC_DIR
from photo_scanner.wallpaper_search.hashing import center_box, dhash, fill_crop, hamming

_TARGET_FULL = 0
_TARGET_CENTER = 0
_TW = 320
_TH = 200

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _init_worker(target_full: int, target_center: int, tw: int, th: int) -> None:
    global _TARGET_FULL, _TARGET_CENTER, _TW, _TH
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    _TARGET_FULL = target_full
    _TARGET_CENTER = target_center
    _TW = tw
    _TH = th


def score_preview(path: str) -> int:
    try:
        im = Image.open(path)
    except (OSError, UnidentifiedImageError):
        return 999
    fitted = fill_crop(im, _TW, _TH)
    box = center_box(_TW, _TH)
    full = hamming(dhash(fitted), _TARGET_FULL)
    center = hamming(dhash(fitted.crop(box)), _TARGET_CENTER)
    return min(full, center)


def score_file(path: Path, target_full: int, target_center: int, tw: int, th: int) -> int:
    try:
        im = Image.open(path)
    except (OSError, UnidentifiedImageError):
        return 999
    fitted = fill_crop(ImageOps.exif_transpose(im), tw, th)
    box = center_box(tw, th)
    return min(hamming(dhash(fitted), target_full), hamming(dhash(fitted.crop(box)), target_center))


def scan(
    items: list[tuple[str, str, bool]],
    target_full: int,
    target_center: int,
    tw: int,
    th: int,
    workers: int = 10,
    stop_at: int = 2,
    batch_size: int = 4000,
) -> list[tuple[int, str, str]]:
    """Return (distance, uuid, path) sorted best-first. Stops early on a near-exact hit."""
    if not items:
        return []
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(SRC_DIR) if not existing else str(SRC_DIR) + os.pathsep + existing
    best: list[tuple[int, str, str]] = []
    n = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(target_full, target_center, tw, th),
    ) as pool:
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            futures = {pool.submit(score_preview, path): (uuid, path) for uuid, path, _fav in chunk}
            for fut in as_completed(futures):
                uuid, path = futures[fut]
                dist = fut.result()
                n += 1
                best.append((dist, uuid, path))
            print(f"scanned {n}/{len(items)}", flush=True)
            if best and min(row[0] for row in best) <= stop_at:
                break
    best.sort(key=lambda row: row[0])
    return best
