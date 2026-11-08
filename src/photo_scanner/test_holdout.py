"""Unit tests for the deterministic holdout split (no torch)."""

from __future__ import annotations

from photo_scanner import holdout


def test_split_is_deterministic():
    key = holdout.good_key("ABCD-1234", "/some/path.jpg")
    assert holdout.is_holdout(key) == holdout.is_holdout(key)


def test_good_key_normalizes_uuid_and_prefers_it():
    a = holdout.good_key("abcd-1234", "/one.jpg")
    b = holdout.good_key("ABCD-1234", "/two.jpg")
    assert a == b
    assert "abcd" in a.lower()


def test_good_key_falls_back_to_path():
    assert holdout.good_key(None, "/one.jpg") == holdout.good_key("", "/one.jpg")
    assert holdout.good_key(None, "/one.jpg") != holdout.good_key(None, "/two.jpg")


def test_bad_key_is_content_based():
    assert holdout.bad_key("abc") == holdout.bad_key("abc")
    assert holdout.bad_key("abc") != holdout.bad_key("def")


def test_bucket_distribution_is_roughly_one_in_ten():
    keys = [holdout.bad_key(f"image-{i}") for i in range(10_000)]
    share = sum(holdout.is_holdout(k) for k in keys) / len(keys)
    assert 0.08 < share < 0.12
