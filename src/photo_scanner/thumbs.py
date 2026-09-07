"""Review thumbnail helpers shared by the review GUI and training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from photo_scanner.paths import OUTPUT_DIR, SESSION_FILE, THUMB_DIR

# Must stay in sync with review GUI thumbnail generation.
THUMB_MAX = 900

# Downscale full-res training/scan images to this max edge before CLIP so
# embeddings match soft review thumbs (iCloud Optimize / no originals on disk).
EMBED_MAX_EDGE = THUMB_MAX

# Feature-cache tag: bump when embed preprocessing changes incompatibly.
EMBED_PREPROCESS = f"max_edge_{EMBED_MAX_EDGE}"


def thumb_cache_key(path: str, uuid: str | None = None) -> str:
    return hashlib.sha1(f"{path}|{uuid or ''}|{THUMB_MAX}".encode("utf-8")).hexdigest()


def expected_thumb_path(path: str, uuid: str | None = None) -> Path:
    return THUMB_DIR / f"{thumb_cache_key(path, uuid)}.jpg"


def is_placeholder_thumb(path: Path | str) -> bool:
    name = Path(path).name
    return name.startswith("missing_")


def is_real_thumb(path: Path | str) -> bool:
    p = Path(path)
    return p.is_file() and not is_placeholder_thumb(p)


def _normalize_uuid(uuid: str | None) -> str | None:
    if not uuid:
        return None
    from learn_from_feedback import normalize_uuid

    key = normalize_uuid(uuid)
    return key or None


def iter_scan_result_photos() -> Iterable[dict]:
    """Yield photo dicts from all scan_results_*.json (newest files first)."""
    files = sorted(OUTPUT_DIR.glob("scan_results_*.json"), reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        photos = data.get("photos") or []
        if isinstance(photos, list):
            for photo in photos:
                if isinstance(photo, dict):
                    yield photo


def load_session_photo_refs() -> list[dict]:
    """UUID/path refs from the active review session (keeps + deletes)."""
    if not SESSION_FILE.exists():
        return []
    try:
        data = json.loads(SESSION_FILE.read_text())
    except Exception:
        return []

    refs: list[dict] = []
    for key in ("kept_from_flat", "confirmed_delete"):
        items = data.get(key) or []
        if isinstance(items, list):
            refs.extend(p for p in items if isinstance(p, dict))
    return refs


def build_uuid_to_path_index() -> dict[str, tuple[str, str]]:
    """
    Map normalized UUID → (filesystem path, uuid string used for thumb hashing).

    Thumb filenames are sha1(path|uuid|THUMB_MAX); the uuid must match the
    string the review GUI hashed (usually the scan_results form). Newest scan
    results first; session refs overlay as the freshest signal.
    """
    index: dict[str, tuple[str, str]] = {}

    for photo in iter_scan_result_photos():
        raw_uuid = photo.get("uuid")
        uid = _normalize_uuid(raw_uuid)
        path = photo.get("path")
        if uid and path and uid not in index:
            index[uid] = (path, str(raw_uuid))

    for photo in load_session_photo_refs():
        raw_uuid = photo.get("uuid")
        uid = _normalize_uuid(raw_uuid)
        path = photo.get("path")
        if uid and path:
            index[uid] = (path, str(raw_uuid))

    return index


def find_real_thumb_for(
    uuid: str | None,
    path: str | None = None,
    thumb_uuid: str | None = None,
) -> Path | None:
    """
    Return an existing real review thumb for path/uuid, or None.

    Tries the exact thumb_uuid (scan/session form) first, then normalized and
    empty variants — review GUI hashes whatever uuid string it was given.
    """
    if not path:
        return None

    candidates: list[str | None] = []
    for value in (thumb_uuid, uuid, _normalize_uuid(uuid), ""):
        if value not in candidates:
            candidates.append(value)

    for candidate in candidates:
        thumb = expected_thumb_path(path, candidate)
        if is_real_thumb(thumb):
            return thumb
    return None


def resolve_thumb_for_uuid(
    uuid: str,
    uuid_to_path: dict[str, tuple[str, str]] | None = None,
) -> Path | None:
    """Locate a real review thumb for a UUID via scan/session path index."""
    uid = _normalize_uuid(uuid)
    if not uid:
        return None
    index = uuid_to_path if uuid_to_path is not None else build_uuid_to_path_index()
    entry = index.get(uid)
    if not entry:
        return None
    path, thumb_uuid = entry
    return find_real_thumb_for(uid, path, thumb_uuid=thumb_uuid)


def pick_largest_readable(paths: Iterable[str | Path | None]) -> Path | None:
    """Return the largest existing file among paths, or None."""
    best: Path | None = None
    best_size = -1
    for raw in paths:
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if size > best_size:
            best = candidate
            best_size = size
    return best


def resolve_scorable_image(
    uuid: str | None,
    *,
    path: str | None = None,
    path_edited: str | None = None,
    derivatives: Iterable[str | Path | None] | None = None,
    uuid_to_path: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str, str] | None:
    """
    Prefer on-disk original/edited; else largest Photos derivative; else review thumb.

    Returns (score_path, library_path, source) where:
      - score_path: readable file to embed / phash
      - library_path: path to store in scan JSON (library path when known,
        so review_gui can find the same thumb via sha1(path|uuid|900))
      - source: 'original' | 'edited' | 'derivative' | 'thumb'
    """
    for candidate, source in ((path, "original"), (path_edited, "edited")):
        if candidate and Path(candidate).is_file():
            return candidate, candidate, source

    deriv = pick_largest_readable(derivatives or ())
    if deriv is not None:
        library_path = path or path_edited or str(deriv)
        return str(deriv), library_path, "derivative"

    uid = _normalize_uuid(uuid)
    index = uuid_to_path
    if index is None and uid:
        index = build_uuid_to_path_index()
    entry = index.get(uid) if (index and uid) else None
    # Prefer the uuid string that was hashed when the thumb was created
    thumb_uuid_hint = entry[1] if entry else uuid

    # Thumb keyed by Photos path strings even when the files are cloud-only
    for candidate in (path, path_edited):
        if not candidate:
            continue
        thumb = find_real_thumb_for(uuid, candidate, thumb_uuid=thumb_uuid_hint)
        if thumb is not None:
            return str(thumb), candidate, "thumb"

    # Prior scan/session path → thumb
    if entry:
        lib_path, thumb_uuid = entry
        thumb = find_real_thumb_for(uid, lib_path, thumb_uuid=thumb_uuid)
        if thumb is not None:
            return str(thumb), lib_path, "thumb"
    elif uid and index is not None:
        thumb = resolve_thumb_for_uuid(uid, index)
        if thumb is not None:
            return str(thumb), str(thumb), "thumb"

    return None
