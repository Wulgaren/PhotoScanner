"""Unit tests for review-thumb training helpers (no torch required)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photo_scanner import thumbs as th


def test_thumb_key_matches_review_gui_formula(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path)
    path = "/Photos Library.photoslibrary/originals/A/ABC.jpg"
    uuid = "ABCDEF12-3456-7890-ABCD-EF1234567890"
    key = th.thumb_cache_key(path, uuid)
    assert key == hashlib.sha1(f"{path}|{uuid}|{th.THUMB_MAX}".encode()).hexdigest()
    assert th.expected_thumb_path(path, uuid) == tmp_path / f"{key}.jpg"


def test_placeholder_thumbs_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path)
    real = tmp_path / "abc123.jpg"
    missing = tmp_path / "missing_deadbeef.jpg"
    real.write_bytes(b"real")
    missing.write_bytes(b"placeholder")
    assert th.is_real_thumb(real)
    assert not th.is_real_thumb(missing)
    assert th.is_placeholder_thumb(missing)


def test_resolve_thumb_for_uuid_via_scan_index(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path)
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "review_session.json")

    # Scan stores mixed-case UUID; thumb hash must use that exact string
    uuid_raw = "AbCdEf12-3456-7890-AbCd-Ef1234567890"
    path = "/library/originals/X/photo.jpg"
    thumb = th.expected_thumb_path(path, uuid_raw)
    thumb.write_bytes(b"jpeg-bytes")

    scan = {
        "photos": [
            {"uuid": uuid_raw, "path": path, "score": 0.2},
        ]
    }
    (tmp_path / "scan_results_20260101_120000.json").write_text(json.dumps(scan))

    # Placeholder must not win
    ph = tmp_path / f"missing_{hashlib.sha1(b'x').hexdigest()}.jpg"
    ph.write_bytes(b"nope")

    found = th.resolve_thumb_for_uuid(uuid_raw.upper())
    assert found == thumb
    assert th.is_real_thumb(found)


def test_missing_original_without_thumb_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path)
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "no_session.json")
    (tmp_path / "scan_results_1.json").write_text(
        json.dumps({"photos": [{"uuid": "AAAA", "path": "/gone.jpg"}]})
    )
    assert th.resolve_thumb_for_uuid("AAAA") is None


def test_embed_preprocess_tag():
    assert th.EMBED_MAX_EDGE == th.THUMB_MAX == 900
    assert th.EMBED_PREPROCESS == "max_edge_900"


def test_resolve_scorable_prefers_original(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "out" / "session.json")
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "out").mkdir()

    original = tmp_path / "original.jpg"
    original.write_bytes(b"orig")
    uuid = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    # Thumb exists too, but original should win
    thumb = th.expected_thumb_path(str(original), uuid)
    thumb.write_bytes(b"thumb")

    result = th.resolve_scorable_image(uuid, path=str(original))
    assert result is not None
    score_path, library_path, source = result
    assert source == "original"
    assert score_path == str(original)
    assert library_path == str(original)


def test_resolve_scorable_falls_back_to_thumb(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "out" / "session.json")
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "out").mkdir()

    lib_path = "/Photos/originals/A/cloud-only.jpg"
    # Same casing Photos/scan would use for both thumb hash and lookup
    uuid = "BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB"
    thumb = th.expected_thumb_path(lib_path, uuid)
    thumb.write_bytes(b"jpeg")
    (tmp_path / "thumbs" / "missing_dead.jpg").write_bytes(b"nope")

    result = th.resolve_scorable_image(
        uuid,
        path=lib_path,  # not on disk
    )
    assert result is not None
    score_path, library_path, source = result
    assert source == "thumb"
    assert score_path == str(thumb)
    assert library_path == lib_path  # keep library path for review_gui keys


def test_resolve_scorable_uses_index_uuid_casing(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "out" / "session.json")
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "out").mkdir()

    lib_path = "/Photos/originals/A/cloud-only.jpg"
    uuid_raw = "AaAaAaAa-AaAa-AaAa-AaAa-AaAaAaAaAaAa"
    thumb = th.expected_thumb_path(lib_path, uuid_raw)
    thumb.write_bytes(b"jpeg")
    (tmp_path / "out" / "scan_results_1.json").write_text(
        json.dumps({"photos": [{"uuid": uuid_raw, "path": lib_path}]})
    )

    result = th.resolve_scorable_image(uuid_raw.upper(), path=lib_path)
    assert result is not None
    assert result[0] == str(thumb)
    assert result[2] == "thumb"


def test_resolve_scorable_skips_placeholder_only(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "THUMB_DIR", tmp_path / "thumbs")
    monkeypatch.setattr(th, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(th, "SESSION_FILE", tmp_path / "out" / "session.json")
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "thumbs" / "missing_abc.jpg").write_bytes(b"placeholder")

    result = th.resolve_scorable_image(
        "CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC",
        path="/gone.jpg",
    )
    assert result is None
