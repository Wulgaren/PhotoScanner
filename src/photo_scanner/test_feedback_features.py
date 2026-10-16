"""Unit tests for flagged-feature storage and threshold calibration (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from photo_scanner import feedback_features as ff


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "FLAGGED_FEATURES_FILE", tmp_path / "flagged_features.pkl")
    return ff


def test_record_and_load_roundtrip(store):
    feat = np.arange(768, dtype=np.float32)
    n = store.record_flagged_batch([("UUID-1", feat, 0.2)], "backboneA", "max_edge_900")
    assert n == 1
    loaded = store.flagged_for_backbone("backboneA", "max_edge_900")
    assert set(loaded) == {"UUID-1"}
    np.testing.assert_array_equal(loaded["UUID-1"]["feature"], feat)
    assert loaded["UUID-1"]["score"] == pytest.approx(0.2)


def test_backbones_are_namespaced(store):
    store.record_flagged_batch([("U1", np.zeros(4), 0.1)], "backboneA", "max_edge_900")
    store.record_flagged_batch([("U1", np.ones(4), 0.9)], "backboneB", "max_edge_900")
    a = store.flagged_for_backbone("backboneA", "max_edge_900")["U1"]
    b = store.flagged_for_backbone("backboneB", "max_edge_900")["U1"]
    assert a["score"] == pytest.approx(0.1)
    assert b["score"] == pytest.approx(0.9)


def test_newest_write_wins(store):
    store.record_flagged_batch([("U1", np.zeros(4), 0.1)], "b", "p")
    # A later batch replaces the record (same uuids rescanned with a new model)
    store.record_flagged_batch([("U1", np.ones(4), 0.8)], "b", "p")
    loaded = store.flagged_for_backbone("b", "p")["U1"]
    assert loaded["score"] == pytest.approx(0.8)


def test_corrupt_store_is_ignored(store, tmp_path):
    (tmp_path / "flagged_features.pkl").write_bytes(b"not-a-pickle")
    assert store.load_flagged_features() == {}


def test_calibrate_threshold_separates_clean_case(store):
    keep = [0.9, 0.8, 0.7, 0.95]
    delete = [0.1, 0.2, 0.15]
    t = store.calibrate_threshold(keep, delete)
    assert t is not None
    assert all(s >= t for s in keep)
    assert all(s < t for s in delete)


def test_calibrate_threshold_needs_both_sides(store):
    assert store.calibrate_threshold([], [0.2]) is None
    assert store.calibrate_threshold([0.9], []) is None


def test_calibrate_threshold_prefers_useful_over_degenerate(store):
    # 90 rescues at 0.5+, 10 deletions at 0.45+: a threshold that flags
    # everything (below 0.9) has recall 1.0 but terrible precision.
    keep = [0.55 + 0.004 * i for i in range(90)]
    delete = [0.45 + 0.004 * i for i in range(10)]
    t = store.calibrate_threshold(keep, delete)
    assert t is not None
    flagged_keep = sum(s < t for s in keep)
    flagged_delete = sum(s < t for s in delete)
    assert flagged_delete >= 8          # catches nearly all real deletions
    assert flagged_keep <= 5            # without drowning them in rescues
