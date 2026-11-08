"""Unit tests for the persisted negative-example store (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from photo_scanner import negative_features as nf

BACKBONE = "vit_base_patch16_clip_224.openai"
PREPROCESS = "max_edge_900"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(nf, "NEGATIVE_FEATURES_FILE", tmp_path / "negative_features.pkl")
    return nf


def _image(tmp_path, name: str, payload: bytes = b"fake-jpeg-bytes"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_record_and_load_roundtrip(store, tmp_path):
    path = _image(tmp_path, "a.jpg")
    feat = np.arange(8, dtype=np.float32)
    assert store.record_negative_batch([(str(path), feat)], BACKBONE, PREPROCESS) == 1

    paths, features = store.persisted_negatives(BACKBONE, PREPROCESS)
    assert paths[str(path)] == store.content_hash(path)
    np.testing.assert_array_equal(features[paths[str(path)]], feat)


def test_negative_survives_source_deletion(store, tmp_path):
    path = _image(tmp_path, "gone.jpg")
    store.record_negative_batch([(str(path), np.ones(4))], BACKBONE, PREPROCESS)
    digest = store.content_hash(path)
    path.unlink()

    _, features = store.persisted_negatives(BACKBONE, PREPROCESS)
    assert digest in features


def test_duplicate_content_stored_once(store, tmp_path):
    first = _image(tmp_path, "one.jpg", b"same-bytes")
    second = _image(tmp_path, "two.jpg", b"same-bytes")
    store.record_negative_batch(
        [(str(first), np.ones(4)), (str(second), np.ones(4))], BACKBONE, PREPROCESS
    )

    paths, features = store.persisted_negatives(BACKBONE, PREPROCESS)
    assert len(paths) == 2
    assert len(features) == 1


def test_backbones_are_namespaced(store, tmp_path):
    path = _image(tmp_path, "a.jpg")
    store.record_negative_batch([(str(path), np.zeros(4))], "backboneA", PREPROCESS)
    store.record_negative_batch([(str(path), np.ones(4))], "backboneB", PREPROCESS)

    _, a = store.persisted_negatives("backboneA", PREPROCESS)
    _, b = store.persisted_negatives("backboneB", PREPROCESS)
    assert all(np.array_equal(v, np.zeros(4)) for v in a.values())
    assert all(np.array_equal(v, np.ones(4)) for v in b.values())


def test_missing_file_is_skipped(store, tmp_path):
    written = store.record_negative_batch(
        [(str(tmp_path / "nope.jpg"), np.ones(4))], BACKBONE, PREPROCESS
    )
    assert written == 0
    _, features = store.persisted_negatives(BACKBONE, PREPROCESS)
    assert features == {}


def test_corrupt_store_is_ignored(store, tmp_path):
    (tmp_path / "negative_features.pkl").write_bytes(b"not-a-pickle")
    assert store.load_negative_features() == {}
    assert store.persisted_negatives(BACKBONE, PREPROCESS) == ({}, {})


def test_total_counts_every_namespace(store, tmp_path):
    a = _image(tmp_path, "a.jpg", b"aaa")
    b = _image(tmp_path, "b.jpg", b"bbb")
    store.record_negative_batch([(str(a), np.ones(4))], BACKBONE, PREPROCESS)
    store.record_negative_batch([(str(b), np.ones(4))], "other", PREPROCESS)
    assert store.total_persisted_negatives() == 2
