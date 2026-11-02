"""Unit tests for chronological phone-review window helpers (no torch)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest

from export_review_site import (
    add_months,
    format_window_label,
    is_video_path,
    load_video_uuids,
    resolve_window,
    select_window,
    should_premark,
    total_windows,
    window_bounds,
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


def test_add_months_clamps_to_last_day():
    assert add_months(dt("2024-01-31T00:00:00+00:00"), 1) == dt(
        "2024-02-29T00:00:00+00:00"
    )
    assert add_months(dt("2023-12-18T00:00:00+00:00"), 1) == dt(
        "2024-01-18T00:00:00+00:00"
    )


def test_window_bounds_and_total():
    start = dt("2023-11-18T00:00:00+00:00")
    assert window_bounds(start, 1, 1) == (start, dt("2023-12-18T00:00:00+00:00"))
    assert window_bounds(start, 1, 2) == (
        dt("2023-12-18T00:00:00+00:00"),
        dt("2024-01-18T00:00:00+00:00"),
    )
    assert total_windows(start, 1, dt("2024-01-17T23:00:00+00:00")) == 2
    assert total_windows(start, 1, dt("2024-01-18T00:00:00+00:00")) == 3


def test_select_window_is_inclusive_start_exclusive_end_and_chronological():
    start = dt("2023-11-18T00:00:00+00:00")
    end = dt("2023-12-18T00:00:00+00:00")
    photos = [
        photo("C", "2023-12-17T23:00:00+00:00", 0.9),
        photo("A", "2023-11-18T00:00:00+00:00", 0.2),
        photo("B", "2023-12-01T10:00:00+00:00", 0.8),
        photo("X", "2023-12-18T00:00:00+00:00", 0.1),
        photo("Y", "2023-11-17T23:59:59+00:00", 0.1),
    ]
    assert [p["uuid"] for p in select_window(photos, start, end)] == ["A", "B", "C"]


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


def test_resolve_window_starts_at_one_then_advances(tmp_path):
    data_path = tmp_path / "data.json"
    assert resolve_window(
        data_path, requested_index=None, next_window=False, months=1
    ) == (1, 1)

    data_path.write_text(json.dumps({"window": {"index": 1, "months": 1}}))
    with pytest.raises(ValueError):
        resolve_window(data_path, requested_index=None, next_window=False, months=1)
    assert resolve_window(
        data_path, requested_index=None, next_window=True, months=1
    ) == (2, 1)
    assert resolve_window(
        data_path, requested_index=4, next_window=False, months=2
    ) == (4, 2)


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
