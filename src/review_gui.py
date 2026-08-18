#!/usr/bin/env python3
"""
Browser review UI for photo deletion suggestions.

Modes:
  flat    — all score < threshold, worst-first, paginated (--page-size)
  grouped — one similar-photo series per screen (best kept, others culled)

Keyboard-first commit; at end of a threshold tier the UI asks to raise +0.1
(up to 0.9). Finish writes a delete list and can add UUIDs to “To Delete”.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sqlite3
import subprocess
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image, ImageDraw
from rich.console import Console

from interactive_review import add_photos_to_album, open_in_preview
from learn_from_feedback import normalize_uuid, record_added_photos, record_review_kept
from photo_scanner.paths import (
    OUTPUT_DIR,
    SESSION_FILE,
    STATIC_DIR,
    THUMB_DIR,
    ensure_data_dirs,
)

console = Console()

ensure_data_dirs()
SLOTS = 1
THUMB_MAX = 900
THRESHOLD_CAP = 0.9
THRESHOLD_STEP = 0.1
PREMARK_MARGIN = 0.1
VALID_MODES = ("flat", "grouped")


def photo_key(photo: dict) -> str:
    uid = (photo.get("uuid") or "").strip().upper()
    if uid:
        return f"u:{uid}"
    return f"p:{photo.get('path') or ''}"


def is_soft_protected(photo: dict, photos_by_series: dict[int, list[dict]]) -> bool:
    """Best-in-series or sole shot — eligible in flat lists but not pre-marked."""
    sid = int(photo["series_id"])
    series = photos_by_series.get(sid, [])
    if len(series) <= 1:
        return True
    best = series[best_photo_index(series)]
    return photo_key(photo) == photo_key(best)


def load_latest_results_path() -> Path | None:
    files = sorted(OUTPUT_DIR.glob("scan_results_*.json"), reverse=True)
    return files[0] if files else None


def best_photo_index(photos: list[dict]) -> int:
    """Index of highest-scoring photo (already sorted desc by score in state)."""
    if not photos:
        return 0
    best_i = 0
    best_score = photos[0].get("score")
    if best_score is None:
        best_score = float("-inf")
    for i, p in enumerate(photos):
        score = p.get("score")
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def deletable_indices(photos: list[dict], threshold: float) -> list[int]:
    """Indices suggested for deletion at threshold (never best, never sole)."""
    if len(photos) <= 1:
        return []
    best_i = best_photo_index(photos)
    out = []
    for i, p in enumerate(photos):
        if i == best_i:
            continue
        score = p.get("score")
        if score is not None and score < threshold:
            out.append(i)
    return out


def series_eligible(photos: list[dict], threshold: float) -> bool:
    return bool(deletable_indices(photos, threshold))


def default_photos_library() -> Path | None:
    guess = Path.home() / "Pictures" / "Photos Library.photoslibrary"
    return guess if guess.exists() else None


def load_photos_library_uuids(library: Path | None = None) -> set[str] | None:
    """
    UUIDs currently in Photos.sqlite (including Recently Deleted).

    Returns None if the library DB cannot be read — caller should skip filtering.
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
            rows = conn.execute("SELECT ZUUID FROM ZASSET WHERE ZUUID IS NOT NULL")
            return {str(u).upper() for (u,) in rows if u}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        console.print(f"[yellow]Photos DB unreadable ({exc}); not dropping gone assets[/yellow]")
        return None


def filter_photos_in_library(photos: list[dict], library_uuids: set[str]) -> tuple[list[dict], int]:
    """Keep only scan photos whose UUID still exists in Photos. Returns (kept, dropped_count)."""
    kept: list[dict] = []
    dropped = 0
    for photo in photos:
        uid = (photo.get("uuid") or "").strip().upper()
        if not uid or uid not in library_uuids:
            dropped += 1
            continue
        kept.append(photo)
    return kept, dropped


class ReviewState:
    """In-memory review session backed by scan JSON + session file."""

    def __init__(
        self,
        results: dict,
        results_path: Path,
        threshold: float,
        *,
        mode: str = "flat",
        page_size: int = 3,
        library_uuids: set[str] | None = None,
    ):
        mode = (mode or "flat").strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self.results_path = str(results_path.resolve())
        self.scan_date = results.get("scan_date", "unknown")
        self.threshold = round(float(threshold), 2)
        self.mode = mode
        self.page_size = max(1, int(page_size))
        self.photos_by_series: dict[int, list[dict]] = {}
        self.pages: dict[int, list[dict]] = {}
        self.path_set: set[str] = set()
        self.decided: set[int] = set()
        self.reviewed_keys: set[str] = set()
        self.kept_from_flat: list[dict] = []
        self.confirmed_delete: list[dict] = []
        self.undo_stack: list[dict] = []
        self.active_slots: list[int | None] = [None] * SLOTS
        self.queue: list[int] = []
        self.finished = False
        self.finish_result: dict | None = None
        self.dropped_gone = 0
        self._page_seq = 0
        self._lock = threading.Lock()

        photos = results["photos"]
        if library_uuids is not None:
            photos, self.dropped_gone = filter_photos_in_library(photos, library_uuids)

        for photo in photos:
            sid = int(photo["series_id"])
            self.photos_by_series.setdefault(sid, []).append(photo)
            if photo.get("path"):
                self.path_set.add(photo["path"])

        self.photos_by_series = {
            sid: plist for sid, plist in self.photos_by_series.items() if plist
        }

        for series_photos in self.photos_by_series.values():
            series_photos.sort(
                key=lambda p: (p.get("score") is not None, p.get("score") or 0),
                reverse=True,
            )

        self._load_session_or_init()

    def _units(self) -> dict[int, list[dict]]:
        return self.pages if self.mode == "flat" else self.photos_by_series

    def _session_payload(self) -> dict:
        payload: dict[str, Any] = {
            "results_path": self.results_path,
            "mode": self.mode,
            "page_size": self.page_size,
            "threshold": self.threshold,
            "decided": sorted(self.decided),
            "confirmed_delete": self.confirmed_delete,
            "active_slots": self.active_slots,
            "finished": self.finished,
            "finish_result": self.finish_result,
        }
        if self.mode == "flat":
            payload["reviewed_keys"] = sorted(self.reviewed_keys)
            payload["kept_from_flat"] = self.kept_from_flat
        return payload

    def save_session(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(self._session_payload(), indent=2))

    def _filter_alive_deletes(self, deletes: list[dict]) -> list[dict]:
        alive_paths = self.path_set
        alive_uuids = {
            (p.get("uuid") or "").upper()
            for photos in self.photos_by_series.values()
            for p in photos
        }
        return [
            d for d in deletes
            if (d.get("path") in alive_paths)
            or ((d.get("uuid") or "").upper() in alive_uuids)
        ]

    def _load_session_or_init(self) -> None:
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                data = None
            session_mode = (data or {}).get("mode") or "grouped"
            if (
                data
                and data.get("results_path") == self.results_path
                and not data.get("finished")
                and session_mode == self.mode
            ):
                self.threshold = round(float(data.get("threshold", self.threshold)), 2)
                if self.mode == "flat":
                    saved_ps = data.get("page_size")
                    if saved_ps is not None:
                        self.page_size = max(1, int(saved_ps))
                    self.reviewed_keys = set(str(k) for k in data.get("reviewed_keys", []))
                    self.kept_from_flat = list(data.get("kept_from_flat") or [])
                    self.confirmed_delete = self._filter_alive_deletes(
                        data.get("confirmed_delete", [])
                    )
                    # Drop reviewed keys / kept entries for gone assets
                    alive = {photo_key(p) for ps in self.photos_by_series.values() for p in ps}
                    self.reviewed_keys &= alive
                    self.kept_from_flat = [
                        p for p in self.kept_from_flat if photo_key(p) in alive
                    ]
                    self.decided = set()
                    self.active_slots = [None] * SLOTS
                    self._rebuild_queue()
                    self._refill_slots()
                    console.print(
                        f"[green]Resumed flat session[/green] @ threshold {self.threshold} "
                        f"(page-size {self.page_size})"
                    )
                    return

                self.decided = set(int(x) for x in data.get("decided", []))
                self.confirmed_delete = self._filter_alive_deletes(
                    data.get("confirmed_delete", [])
                )
                slots = data.get("active_slots") or []
                self.active_slots = [(int(s) if s is not None else None) for s in slots]
                while len(self.active_slots) < SLOTS:
                    self.active_slots.append(None)
                self.active_slots = self.active_slots[:SLOTS]
                self.decided = {sid for sid in self.decided if sid in self.photos_by_series}
                self._rebuild_queue()
                self._refill_slots()
                console.print(f"[green]Resumed grouped session[/green] @ threshold {self.threshold}")
                return

        self._rebuild_queue()
        self._refill_slots()
        self.save_session()

    def _flat_candidates(self, threshold: float) -> list[dict]:
        t = round(threshold, 2)
        out: list[dict] = []
        for series_photos in self.photos_by_series.values():
            for p in series_photos:
                if photo_key(p) in self.reviewed_keys:
                    continue
                score = p.get("score")
                if score is not None and score < t:
                    out.append(p)
        out.sort(key=lambda p: float(p["score"]))
        return out

    def _rebuild_flat_pages(self) -> None:
        candidates = self._flat_candidates(self.threshold)
        self.pages = {}
        self._page_seq = 0
        for i in range(0, len(candidates), self.page_size):
            self._page_seq += 1
            self.pages[self._page_seq] = candidates[i : i + self.page_size]
        active = {s for s in self.active_slots if s is not None}
        self.queue = [pid for pid in sorted(self.pages) if pid not in active]
        self.decided = set()

    def _rebuild_queue(self) -> None:
        if self.mode == "flat":
            self._rebuild_flat_pages()
            return
        eligible = []
        for sid, photos in self.photos_by_series.items():
            if sid in self.decided:
                continue
            if series_eligible(photos, self.threshold):
                n = len(deletable_indices(photos, self.threshold))
                eligible.append((n, sid))
        eligible.sort(key=lambda t: (-t[0], t[1]))
        active = {s for s in self.active_slots if s is not None}
        self.queue = [sid for _, sid in eligible if sid not in active]

    def _unit_still_eligible(self, unit_id: int) -> bool:
        units = self._units()
        if unit_id not in units:
            return False
        if self.mode == "flat":
            return unit_id not in self.decided
        return series_eligible(units[unit_id], self.threshold)

    def _refill_slots(self) -> None:
        for i in range(SLOTS):
            if self.active_slots[i] is not None:
                uid = self.active_slots[i]
                if uid in self.decided or not self._unit_still_eligible(uid):
                    self.active_slots[i] = None
        for i in range(SLOTS):
            if self.active_slots[i] is None and self.queue:
                self.active_slots[i] = self.queue.pop(0)

    def count_new_at_threshold(self, threshold: float) -> int:
        t = round(threshold, 2)
        if self.mode == "flat":
            return len(self._flat_candidates(t))
        n = 0
        for sid, photos in self.photos_by_series.items():
            if sid in self.decided:
                continue
            if series_eligible(photos, t):
                n += 1
        return n

    def progress(self) -> dict[str, Any]:
        if self.mode == "flat":
            remaining_pages = len(self.pages)
            remaining_photos = sum(len(p) for p in self.pages.values())
            reviewed_at_or_below = 0
            for ps in self.photos_by_series.values():
                for p in ps:
                    score = p.get("score")
                    if score is None or score >= self.threshold:
                        continue
                    if photo_key(p) in self.reviewed_keys:
                        reviewed_at_or_below += 1
            return {
                "done": reviewed_at_or_below,
                "total": reviewed_at_or_below + remaining_photos,
                "remaining": remaining_pages,
                "confirmed_delete_count": len(self.confirmed_delete),
            }

        eligible_ids = {
            sid for sid, photos in self.photos_by_series.items()
            if sid not in self.decided and series_eligible(photos, self.threshold)
        }
        remaining = len(eligible_ids)
        decided_relevant = 0
        for sid in self.decided:
            photos = self.photos_by_series.get(sid, [])
            if len(photos) > 1 and any(
                (p.get("score") is not None and p["score"] < self.threshold)
                for p in photos
            ):
                decided_relevant += 1
        return {
            "done": decided_relevant,
            "total": remaining + decided_relevant,
            "remaining": remaining,
            "confirmed_delete_count": len(self.confirmed_delete),
        }

    def serialize_unit(self, unit_id: int) -> dict:
        photos = self._units()[unit_id]
        items = []
        marks: set[int] = set()

        if self.mode == "grouped":
            best_i = best_photo_index(photos)
            marks = set(deletable_indices(photos, self.threshold))
            for i, p in enumerate(photos):
                items.append({
                    "index": i,
                    "uuid": p.get("uuid"),
                    "path": p.get("path"),
                    "score": p.get("score"),
                    "date": p.get("date"),
                    "filename": Path(p["path"]).name if p.get("path") else "",
                    "is_best": i == best_i,
                    "marked_delete": i in marks,
                })
        else:
            premark_cut = self.threshold - PREMARK_MARGIN
            for i, p in enumerate(photos):
                protected = is_soft_protected(p, self.photos_by_series)
                score = p.get("score")
                marked = (
                    not protected
                    and score is not None
                    and score < premark_cut
                )
                if marked:
                    marks.add(i)
                series = self.photos_by_series.get(int(p["series_id"]), [])
                best_i = best_photo_index(series) if series else 0
                is_best = bool(series) and photo_key(p) == photo_key(series[best_i])
                items.append({
                    "index": i,
                    "uuid": p.get("uuid"),
                    "path": p.get("path"),
                    "score": p.get("score"),
                    "date": p.get("date"),
                    "filename": Path(p["path"]).name if p.get("path") else "",
                    "is_best": is_best,
                    "marked_delete": marked,
                })

        return {
            "series_id": unit_id,
            "photos": items,
            "deletable_count": len(marks),
        }

    def public_state(self) -> dict:
        with self._lock:
            return self._unlocked_public_state()

    def commit(self, series_id: int, delete_indices: list[int]) -> dict:
        with self._lock:
            if self.finished:
                return {"ok": False, "error": "Session already finished"}
            uid = int(series_id)
            units = self._units()
            if uid not in units:
                return {"ok": False, "error": "Unknown unit"}
            if uid in self.decided:
                return {"ok": False, "error": "Already decided"}
            if uid not in self.active_slots:
                return {"ok": False, "error": "Not in active slots"}

            photos = units[uid]
            deleted = []
            delete_set = {int(i) for i in delete_indices}
            for i in sorted(delete_set):
                if i < 0 or i >= len(photos):
                    return {"ok": False, "error": f"Invalid photo index {i}"}
                p = photos[i]
                entry = {
                    "uuid": p.get("uuid"),
                    "path": p.get("path"),
                    "score": p.get("score"),
                    "series_id": int(p.get("series_id", uid)),
                    "unit_id": uid,
                }
                deleted.append(entry)
                self.confirmed_delete.append(entry)

            kept_here = []
            if self.mode == "flat":
                for i, p in enumerate(photos):
                    self.reviewed_keys.add(photo_key(p))
                    if i not in delete_set:
                        kept_here.append(p)
                        self.kept_from_flat.append(p)

            self.undo_stack.append({
                "mode": self.mode,
                "series_id": uid,
                "deleted": deleted,
                "kept": kept_here,
                "photos": list(photos) if self.mode == "flat" else None,
            })
            self.decided.add(uid)

            if self.mode == "flat":
                self.pages.pop(uid, None)
                if uid in self.queue:
                    self.queue.remove(uid)

            for i, s in enumerate(self.active_slots):
                if s == uid:
                    self.active_slots[i] = None
            self._refill_slots()
            self.save_session()
            return {"ok": True, "state": self._unlocked_public_state()}

    def _unlocked_public_state(self) -> dict:
        slots = []
        for uid in self.active_slots:
            if uid is None:
                slots.append(None)
            else:
                slots.append(self.serialize_unit(uid))
        prog = self.progress()
        tier_empty = prog["remaining"] == 0 and not self.finished
        next_t = round(self.threshold + THRESHOLD_STEP, 2)
        next_count = 0
        can_raise = False
        if tier_empty and next_t <= THRESHOLD_CAP + 1e-9:
            next_count = self.count_new_at_threshold(next_t)
            can_raise = next_count > 0

        return {
            "mode": self.mode,
            "page_size": self.page_size,
            "threshold": self.threshold,
            "scan_date": self.scan_date,
            "slots": slots,
            "progress": prog,
            "tier_empty": tier_empty,
            "can_raise": can_raise,
            "next_threshold": next_t if next_t <= THRESHOLD_CAP + 1e-9 else None,
            "next_tier_count": next_count,
            "can_undo": bool(self.undo_stack),
            "finished": self.finished,
            "finish_result": self.finish_result,
        }

    def keep_all(self, series_id: int) -> dict:
        return self.commit(series_id, [])

    def undo(self) -> dict:
        with self._lock:
            if self.finished:
                return {"ok": False, "error": "Session already finished"}
            if not self.undo_stack:
                return {"ok": False, "error": "Nothing to undo"}
            last = self.undo_stack.pop()
            uid = int(last["series_id"])
            deleted = last["deleted"]
            remove_paths = {d.get("path") for d in deleted}
            remove_keys = {photo_key(d) for d in deleted}
            # Also match by unit_id for flat commits
            self.confirmed_delete = [
                d for d in self.confirmed_delete
                if not (
                    (d.get("unit_id") == uid or d.get("series_id") == uid)
                    and (d.get("path") in remove_paths or photo_key(d) in remove_keys)
                )
            ]

            if self.mode == "flat" and last.get("mode") == "flat":
                photos = last.get("photos") or []
                for p in photos:
                    self.reviewed_keys.discard(photo_key(p))
                kept_keys = {photo_key(p) for p in (last.get("kept") or [])}
                self.kept_from_flat = [
                    p for p in self.kept_from_flat if photo_key(p) not in kept_keys
                ]
                if photos:
                    self.pages[uid] = photos
                self.decided.discard(uid)
            else:
                self.decided.discard(uid)

            displaced = None
            if uid in self.active_slots:
                pass
            elif None in self.active_slots:
                self.active_slots[self.active_slots.index(None)] = uid
            else:
                for i, s in enumerate(self.active_slots):
                    if s is not None:
                        displaced = s
                        self.active_slots[i] = uid
                        break

            if self.mode == "flat":
                active = {s for s in self.active_slots if s is not None}
                self.queue = [pid for pid in sorted(self.pages) if pid not in active]
            else:
                self._rebuild_queue()
            if displaced is not None:
                if displaced in self.queue:
                    self.queue.remove(displaced)
                self.queue.insert(0, displaced)
            self.save_session()
            return {"ok": True, "state": self._unlocked_public_state()}

    def raise_threshold(self) -> dict:
        with self._lock:
            if self.finished:
                return {"ok": False, "error": "Session already finished"}
            next_t = round(self.threshold + THRESHOLD_STEP, 2)
            if next_t > THRESHOLD_CAP + 1e-9:
                return {"ok": False, "error": "Threshold cap reached"}
            self.threshold = next_t
            self.active_slots = [None] * SLOTS
            if self.mode == "flat":
                self.undo_stack = []  # page ids rebuild; undos would be stale
            self._rebuild_queue()
            self._refill_slots()
            self.save_session()
            return {"ok": True, "state": self._unlocked_public_state()}

    def kept_photos(self) -> list[dict]:
        """Photos reviewed and not marked for deletion."""
        delete_uuids = set()
        delete_paths = set()
        for d in self.confirmed_delete:
            uid = normalize_uuid(d.get("uuid"))
            if uid:
                delete_uuids.add(uid)
            if d.get("path"):
                delete_paths.add(d["path"])

        if self.mode == "flat":
            kept: list[dict] = []
            seen: set[str] = set()
            for p in self.kept_from_flat:
                key = photo_key(p)
                if key in seen:
                    continue
                uid = normalize_uuid(p.get("uuid"))
                if uid and uid in delete_uuids:
                    continue
                if p.get("path") and p["path"] in delete_paths:
                    continue
                seen.add(key)
                kept.append(p)
            return kept

        kept = []
        for sid in self.decided:
            for p in self.photos_by_series.get(sid, []):
                uid = normalize_uuid(p.get("uuid"))
                if uid and uid in delete_uuids:
                    continue
                if p.get("path") and p["path"] in delete_paths:
                    continue
                kept.append(p)
        return kept

    def finish(self, add_to_album: bool = False) -> dict:
        with self._lock:
            if self.finished and self.finish_result:
                result = dict(self.finish_result)
                if add_to_album and not result.get("album_added") and self.confirmed_delete:
                    uuids = [p["uuid"] for p in self.confirmed_delete if p.get("uuid")]
                    ok = add_photos_to_album(uuids, "To Delete") if uuids else False
                    result["album_added"] = bool(ok)
                    result["album_count"] = len(uuids) if ok else 0
                    if ok and uuids:
                        record_added_photos(uuids, self.results_path)
                    self.finish_result = result
                    self.save_session()
                return {"ok": True, "result": result}

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_file = None
            if self.confirmed_delete:
                out_file = OUTPUT_DIR / f"confirmed_delete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(out_file, "w") as f:
                    for p in self.confirmed_delete:
                        if p.get("path"):
                            f.write(f"{p['path']}\n")

            album_added = False
            album_count = 0
            if add_to_album and self.confirmed_delete:
                uuids = [p["uuid"] for p in self.confirmed_delete if p.get("uuid")]
                if uuids:
                    album_added = bool(add_photos_to_album(uuids, "To Delete"))
                    album_count = len(uuids) if album_added else 0
                    if album_added:
                        record_added_photos(uuids, self.results_path)

            kept = self.kept_photos()
            kept_uuids = [p["uuid"] for p in kept if p.get("uuid")]
            kept_saved = record_review_kept(kept_uuids) if kept_uuids else 0

            result = {
                "confirmed_delete_count": len(self.confirmed_delete),
                "output_file": str(out_file) if out_file else None,
                "album_added": album_added,
                "album_count": album_count,
                "kept_count": len(kept_uuids),
                "kept_saved": kept_saved,
            }
            self.finished = True
            self.finish_result = result
            self.save_session()
            return {"ok": True, "result": result}


# Global state set in main()
STATE: ReviewState | None = None


def photos_library_root(path: str | Path) -> Path | None:
    """Return *.photoslibrary root from an originals path, if present."""
    p = Path(path).resolve() if Path(path).exists() else Path(path)
    for parent in [p, *p.parents]:
        if parent.suffix == ".photoslibrary" or parent.name.endswith(".photoslibrary"):
            return parent
    return None


def resolve_readable_image(path: str, uuid: str | None = None) -> Path | None:
    """
    Find a local image file for a scan path.

    Favorites often live in iCloud only — originals missing on disk.
    Fall back to Photos derivatives when present.
    """
    src = Path(path)
    if src.exists() and src.is_file():
        return src

    uid = uuid or src.stem
    if not uid:
        return None

    lib = photos_library_root(path)
    if lib is None:
        # Path may point at a deleted/moved library; try default Pictures location
        guess = Path.home() / "Pictures" / "Photos Library.photoslibrary"
        lib = guess if guess.exists() else None
    if lib is None:
        return None

    folder = uid[0].upper()
    search_dirs = [
        lib / "resources" / "derivatives" / "masters" / folder,
        lib / "resources" / "derivatives" / folder,
        lib / "originals" / folder,
    ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        # Exact original-style name first, then derivative prefixes
        exact = list(directory.glob(f"{uid}.*"))
        prefixed = [p for p in directory.glob(f"{uid}_*") if p.is_file()]
        # Prefer larger master derivatives over tiny thumbs when both exist
        candidates = exact + sorted(prefixed, key=lambda p: p.stat().st_size, reverse=True)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def placeholder_thumb(label: str) -> Path:
    """Cached placeholder JPEG when original + derivatives are unavailable."""
    key = hashlib.sha1(f"missing:{label}".encode("utf-8")).hexdigest()
    dest = THUMB_DIR / f"missing_{key}.jpg"
    if dest.exists():
        return dest
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    # Darkroom-style missing plate
    im = Image.new("RGB", (720, 480), (40, 38, 34))
    draw = ImageDraw.Draw(im)
    draw.rectangle((16, 16, 703, 463), outline=(180, 80, 60), width=3)
    sub = (label[:36] + "…") if len(label) > 36 else label
    draw.text((40, 140), "NOT ON DISK", fill=(220, 120, 90))
    draw.text((40, 180), sub, fill=(160, 150, 130))
    draw.text((40, 220), "iCloud / deleted / no derivative", fill=(120, 110, 95))
    im.save(dest, "JPEG", quality=85)
    return dest


def ensure_thumb(path: str, uuid: str | None = None) -> Path | None:
    """Create or return cached thumbnail; derivative fallback; placeholder last."""
    key = hashlib.sha1(f"{path}|{uuid or ''}|{THUMB_MAX}".encode("utf-8")).hexdigest()
    dest = THUMB_DIR / f"{key}.jpg"
    if dest.exists():
        return dest

    src = resolve_readable_image(path, uuid=uuid)
    if src is None:
        return placeholder_thumb(uuid or Path(path).name)

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
            im.save(dest, "JPEG", quality=82, optimize=True)
        return dest
    except Exception:
        return placeholder_thumb(uuid or Path(path).name)


def open_in_photos(uuid: str) -> bool:
    """Reveal a photo in the Photos app by UUID when filesystem path is gone."""
    if not uuid:
        return False
    script = f'''
    tell application "Photos"
        activate
        try
            set theItem to media item id "{uuid}"
            spotlight theItem
            return "ok"
        on error
            return "missing"
        end try
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and "ok" in (result.stdout or "")
    except Exception:
        return False


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "PhotoScannerReview/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Quiet default access logs; errors still useful via console on failures
        if args and len(args) >= 2 and str(args[1]).startswith("4"):
            console.print(f"[dim]{self.address_string()} {fmt % args}[/dim]")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, rel: str) -> None:
        rel = rel.lstrip("/")
        if not rel or rel == "/":
            rel = "index.html"
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            self._send_json(404, {"error": "Not found"})
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
            return
        if path == "/api/state":
            assert STATE is not None
            self._send_json(200, STATE.public_state())
            return
        if path == "/thumb":
            assert STATE is not None
            qs = parse_qs(parsed.query)
            photo_path = unquote(qs.get("path", [""])[0])
            photo_uuid = unquote(qs.get("uuid", [""])[0]) or None
            if not photo_path or photo_path not in STATE.path_set:
                self._send_json(403, {"error": "Forbidden"})
                return
            thumb = ensure_thumb(photo_path, uuid=photo_uuid)
            if not thumb:
                self._send_json(404, {"error": "Thumb failed"})
                return
            data = thumb.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        assert STATE is not None
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON"})
            return

        if path == "/api/commit":
            sid = payload.get("series_id")
            idxs = payload.get("delete_indices", [])
            if sid is None:
                self._send_json(400, {"ok": False, "error": "series_id required"})
                return
            self._send_json(200, STATE.commit(int(sid), list(idxs)))
            return
        if path == "/api/keep-all":
            sid = payload.get("series_id")
            if sid is None:
                self._send_json(400, {"ok": False, "error": "series_id required"})
                return
            self._send_json(200, STATE.keep_all(int(sid)))
            return
        if path == "/api/undo":
            self._send_json(200, STATE.undo())
            return
        if path == "/api/raise-threshold":
            self._send_json(200, STATE.raise_threshold())
            return
        if path == "/api/finish":
            add_album = bool(payload.get("add_to_album", False))
            self._send_json(200, STATE.finish(add_to_album=add_album))
            return
        if path == "/api/preview":
            paths = payload.get("paths") or []
            single = payload.get("path")
            uuid = payload.get("uuid")
            prefer_photos = bool(payload.get("prefer_photos"))
            if single:
                paths = [single]
            opened = 0

            def try_photos(uid: str | None, path_list: list) -> int:
                if uid and open_in_photos(str(uid)):
                    return 1
                n = 0
                for p in path_list:
                    if p in STATE.path_set and open_in_photos(Path(p).stem):
                        n += 1
                return n

            if prefer_photos:
                opened = try_photos(str(uuid) if uuid else None, paths)
                self._send_json(200, {"ok": True, "opened": opened, "app": "photos"})
                return

            local_paths = []
            for p in paths:
                if p not in STATE.path_set:
                    continue
                resolved = resolve_readable_image(p, uuid=uuid if len(paths) == 1 else Path(p).stem)
                if resolved:
                    local_paths.append(str(resolved))
            if local_paths:
                open_in_preview(local_paths)
                opened = len(local_paths)
            else:
                opened = try_photos(str(uuid) if uuid else None, paths)
            self._send_json(200, {"ok": True, "opened": opened})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})


def main() -> None:
    global STATE

    parser = argparse.ArgumentParser(
        description="Browser review for deletion suggestions (flat pages or grouped series)"
    )
    parser.add_argument("--results", type=str, default=None, help="Path to scan results JSON")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="flat",
        help="flat: threshold list paginated; grouped: similar-photo series (default flat)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Starting score threshold (default 0.5)")
    parser.add_argument(
        "--page-size",
        type=int,
        default=3,
        help="Photos per page in flat mode (default 3; ignored for grouped)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Local port (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    console.print("\n[bold blue]PhotoScanner — Review[/bold blue]\n")

    if args.results:
        results_path = Path(args.results)
    else:
        results_path = load_latest_results_path()
        if not results_path:
            console.print("[red]No scan results found. Run ./photoscanner.sh (Scan) first.[/red]")
            return

    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        return

    console.print(f"Loading [cyan]{results_path}[/cyan] ...")
    with open(results_path) as f:
        results = json.load(f)

    console.print(f"Scan date: {results.get('scan_date', 'unknown')}")
    console.print(f"Photos: {results.get('total_photos', len(results.get('photos', [])))}")
    console.print(f"Mode: [cyan]{args.mode}[/cyan]  threshold: {args.threshold}"
                  + (f"  page-size: {args.page_size}" if args.mode == "flat" else ""))

    library_uuids = load_photos_library_uuids()
    if library_uuids is None:
        console.print("[yellow]No Photos library UUID index — keeping all scan assets[/yellow]")
    else:
        console.print(f"Photos library assets: [cyan]{len(library_uuids):,}[/cyan]")

    STATE = ReviewState(
        results,
        results_path,
        args.threshold,
        mode=args.mode,
        page_size=args.page_size,
        library_uuids=library_uuids,
    )
    if STATE.dropped_gone:
        console.print(
            f"[dim]Dropped {STATE.dropped_gone:,} scan photos gone from Photos library[/dim]"
        )
    prog = STATE.progress()
    unit = "pages" if STATE.mode == "flat" else "series"
    console.print(
        f"Queue @ {STATE.threshold}: [yellow]{prog['remaining']}[/yellow] {unit} "
        f"(done {prog['done']} / {prog['total']})"
    )

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    url = f"http://127.0.0.1:{args.port}/"
    console.print(f"\n[green]Serving[/green] {url}")
    console.print("[dim]Keys: ←/→ photo · Space toggle · Enter commit · s/d mark all keep/delete · u undo · Done early[/dim]")
    console.print("[dim]Ctrl+C to stop server[/dim]\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
        if STATE and not STATE.finished and STATE.confirmed_delete:
            console.print(
                f"[dim]Session saved with {len(STATE.confirmed_delete)} pending deletes. "
                f"Relaunch to resume, or finish from the UI.[/dim]"
            )
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
