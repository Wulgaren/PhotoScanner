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
