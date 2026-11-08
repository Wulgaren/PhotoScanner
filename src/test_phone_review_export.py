"""Unit tests for chronological phone-review window helpers (no torch)."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from export_review_site import (
    build_item,
    format_window_label,
    is_video_path,
    load_captions,
    load_hidden_uuids,
    load_video_uuids,
    plan_window,
    resolve_window,
    should_premark,
)
from learn_from_feedback import parse_uuid_text


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def photo(uuid: str, date: str, score: float) -> dict:
    return {
        "uuid": uuid,
        "path": f"/library/{uuid}.jpg",
        "date": date,
        "score": score,
        "series_id": 0,
    }


def test_plan_window_starts_after_previous_end_and_takes_size():
    photos = [
        photo("A", "2024-01-01T00:00:00+00:00", 0.9),
        photo("B", "2024-01-02T00:00:00+00:00", 0.9),
        photo("C", "2024-01-03T00:00:00+00:00", 0.9),
        photo("D", "2024-01-04T00:00:00+00:00", 0.9),
        photo("E", "2024-01-05T00:00:00+00:00", 0.9),
    ]
    plan = plan_window(
        photos,
        index=3,
        size=2,
        start=dt("2024-01-03T00:00:00+00:00"),
    )
    assert [p["uuid"] for p in plan["photos"]] == ["C", "D"]
    assert plan["window"]["total"] == 4
    assert plan["window"]["label"] == "Jan 3 – Jan 4, 2024"
    assert plan["window"]["end"] == "2024-01-05T00:00:00+00:00"


def test_plan_window_resumes_after_cursor_on_identical_timestamps():
    when = "2024-01-01T00:00:00+00:00"
    photos = [
        photo("A", when, 0.9),
        photo("B", when, 0.8),
        photo("C", when, 0.7),
        photo("D", when, 0.6),
    ]
    plan = plan_window(
        photos,
        index=4,
        size=2,
        start=dt(when),
        after_uuid="d",
    )
    assert [p["uuid"] for p in plan["photos"]] == ["C", "B"]
    assert plan["window"]["cursor"] == "B"


def test_should_premark_uses_plain_threshold_even_for_best_shots():
    assert should_premark(0.499, 0.5) is True
    assert should_premark(0.5, 0.5) is False
    assert should_premark(None, 0.5) is False


def test_format_window_label_same_and_cross_year():
    assert (
        format_window_label(
            dt("2023-11-18T00:00:00+00:00"), dt("2023-12-18T00:00:00+00:00")
        )
        == "Nov 18 – Dec 17, 2023"
    )
    assert (
        format_window_label(
            dt("2023-12-18T00:00:00+00:00"), dt("2024-01-18T00:00:00+00:00")
        )
        == "Dec 18, 2023 – Jan 17, 2024"
    )


def test_resolve_window_starts_at_one_then_advances():
    assert resolve_window(
        None, requested_index=None, next_window=False, size=1000
    ) == (1, 1000)

    existing = {"index": 2, "size": 1000}
    with pytest.raises(ValueError):
        resolve_window(
            existing, requested_index=None, next_window=False, size=1000
        )
    assert resolve_window(
        existing, requested_index=None, next_window=True, size=1000
    ) == (3, 1000)
    assert resolve_window(
        {"index": 2}, requested_index=None, next_window=True, size=1000
    ) == (3, 1000)
    assert resolve_window(
        existing, requested_index=4, next_window=False, size=500
    ) == (4, 500)


def test_video_paths_are_rejected():
    assert is_video_path("/library/A.MOV")
    assert is_video_path("/library/B.mp4")
    assert not is_video_path("/library/B.HEIC")
    assert not is_video_path(None)


def test_load_video_uuids_reads_photos_media_kind(tmp_path):
    lib = tmp_path / "Photos Library.photoslibrary"
    db_dir = lib / "database"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "Photos.sqlite")
    conn.execute("CREATE TABLE ZASSET (ZUUID TEXT, ZKIND INTEGER)")
    conn.executemany(
        "INSERT INTO ZASSET VALUES (?, ?)",
        [("video-1", 1), ("photo-1", 0), ("video-2", 1)],
    )
    conn.commit()
    conn.close()
    assert load_video_uuids(lib) == {"VIDEO-1", "VIDEO-2"}


def test_load_hidden_uuids_reads_photos_hidden_flag(tmp_path):
    lib = tmp_path / "Photos Library.photoslibrary"
    db_dir = lib / "database"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "Photos.sqlite")
    conn.execute("CREATE TABLE ZASSET (ZUUID TEXT, ZHIDDEN INTEGER)")
    conn.executemany(
        "INSERT INTO ZASSET VALUES (?, ?)",
        [("hidden-1", 1), ("photo-1", 0), ("hidden-2", 1), ("photo-2", None)],
    )
    conn.commit()
    conn.close()
    assert load_hidden_uuids(lib) == {"HIDDEN-1", "HIDDEN-2"}


def test_load_captions_reads_non_empty_photos_descriptions(tmp_path):
    lib = tmp_path / "Photos Library.photoslibrary"
    db_dir = lib / "database"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "Photos.sqlite")
    conn.execute("CREATE TABLE ZASSET (Z_PK INTEGER, ZUUID TEXT, ZADDITIONALATTRIBUTES INTEGER)")
    conn.execute(
        "CREATE TABLE ZADDITIONALASSETATTRIBUTES (Z_PK INTEGER, ZASSETDESCRIPTION INTEGER)"
    )
    conn.execute("CREATE TABLE ZASSETDESCRIPTION (Z_PK INTEGER, ZLONGDESCRIPTION TEXT)")
    conn.executemany(
        "INSERT INTO ZASSET VALUES (?, ?, ?)",
        [(1, "aaaa-1", 10), (2, "bbbb-2", 20), (3, "cccc-3", 30)],
    )
    conn.executemany(
        "INSERT INTO ZADDITIONALASSETATTRIBUTES VALUES (?, ?)",
        [(10, 100), (20, 200), (30, 300)],
    )
    conn.executemany(
        "INSERT INTO ZASSETDESCRIPTION VALUES (?, ?)",
        [(100, "  aespa karina  "), (200, "   "), (300, None)],
    )
    conn.commit()
    conn.close()

    assert load_captions(lib) == {"AAAA-1": "aespa karina"}


def test_build_item_includes_caption_only_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "export_review_site.ensure_site_thumb", lambda photo, dest: True
    )
    photo = {
        "uuid": "aaaa-1/L0/001",
        "date": "2023-11-18T08:23:18+01:00",
        "score": 0.2,
        "series_id": 0,
    }
    item = build_item(photo, {}, 0.5, tmp_path, {"AAAA-1": "aespa karina"})
    assert item is not None
    assert item["caption"] == "aespa karina"

    bare = build_item(photo, {}, 0.5, tmp_path, {})
    assert bare is not None
    assert "caption" not in bare


def test_parse_uuid_text_accepts_lines_commas_json_comments_and_dedupes():
    text = """
    # saved on phone
    aaaa-1,
    BBBB-2
    {"ids": ["cccc-3", "aaaa-1"]}
    "dddd-4"
    """
    assert parse_uuid_text(text) == ["AAAA-1", "BBBB-2", "CCCC-3", "DDDD-4"]


def test_parse_uuid_text_strips_points_suffix_and_ignores_empty():
    assert parse_uuid_text("abc/L0/001\n\n,,") == ["ABC"]
