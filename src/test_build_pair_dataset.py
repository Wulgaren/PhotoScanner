"""Pair dataset building: within-window pairing and group-safe splits."""

from __future__ import annotations

import numpy as np

from build_pair_dataset import build_pairs, split_groups


def _record(vector, window=None):
    record = {"feature": np.asarray(vector, dtype=np.float32)}
    if window is not None:
        record["window"] = window
    return record


def test_marked_photo_pairs_only_with_unmarked_in_same_window():
    features = {
        "LOSER": _record([1.0, 0.0], window=3),
        "WINNER": _record([0.0, 1.0], window=3),
        "OTHER-WINDOW": _record([5.0, 5.0], window=4),
    }
    diffs, labels, groups = build_pairs({"save-a.txt": {"LOSER"}}, features)

    assert len(diffs) == 2  # one pair -> two symmetric rows
    assert labels.tolist() == [0, 1]
    assert np.allclose(diffs[0], -diffs[1])
    assert len(set(groups.tolist())) == 1


def test_photo_marked_in_another_save_never_becomes_a_winner():
    features = {
        "LOSER-A": _record([1.0, 0.0], window=3),
        "LOSER-B": _record([0.5, 0.5], window=3),
    }
    diffs, labels, _ = build_pairs(
        {"save-a.txt": {"LOSER-A"}, "save-b.txt": {"LOSER-B"}}, features
    )

    # Both photos were marked for deletion, so neither can be the kept side.
    assert len(diffs) == 0


def test_pair_group_is_the_review_window_not_the_save_session():
    # Two separate saves inside one window compare against the same unmarked
    # photos, so their pairs are not independent: one group must hold both.
    features = {
        "LOSER-A": _record([1.0, 0.0], window=3),
        "LOSER-B": _record([0.25, 0.75], window=3),
        "WINNER": _record([0.0, 1.0], window=3),
    }
    diffs, labels, groups = build_pairs(
        {"save-a.txt": {"LOSER-A"}, "save-b.txt": {"LOSER-B"}}, features
    )

    assert len(diffs) == 4
    assert set(groups.tolist()) == {3}


def test_unwindowed_photos_are_ignored():
    features = {
        "LOSER": _record([1.0, 0.0], window=3),
        "NO-WINDOW": _record([0.0, 1.0]),
    }
    diffs, labels, _ = build_pairs({"save-a.txt": {"LOSER"}}, features)

    # A photo with no window has no known review session to compare against.
    assert len(diffs) == 0


def test_split_keeps_groups_whole():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    train_mask, test_mask = split_groups(groups, test_groups=1, seed=7)

    assert not (train_mask & test_mask).any()
    assert train_mask.sum() + test_mask.sum() == len(groups)
    test_groups = set(groups[test_mask].tolist())
    assert len(test_groups) == 1
    # Every row of the held-out group is on the test side.
    assert groups[test_mask].tolist() == [test_groups.pop()] * 2


def test_split_never_empties_training():
    groups = np.array([0, 0, 1, 1])
    train_mask, test_mask = split_groups(groups, test_groups=99, seed=1)

    assert train_mask.any()
    assert test_mask.any()


def test_holdout_window_puts_that_window_in_test_only():
    groups = np.array([5, 5, 6, 6, 7, 7])
    train_mask, test_mask = split_groups(groups, test_groups=1, holdout=6)

    assert set(groups[test_mask].tolist()) == {6}
    assert set(groups[train_mask].tolist()) == {5, 7}


def test_holdout_window_absent_from_dataset_is_an_error():
    groups = np.array([5, 5, 6, 6])
    try:
        split_groups(groups, test_groups=1, holdout=9)
    except ValueError as exc:
        assert "window 9" in str(exc)
    else:
        raise AssertionError("expected a missing holdout window to fail")


def test_pairs_in_same_window_land_on_one_side_only():
    features = {
        "LOSER-A": _record([1.0, 0.0], window=3),
        "LOSER-B": _record([0.5, 0.2], window=4),
        "WINNER-3": _record([0.0, 1.0], window=3),
        "WINNER-4": _record([0.1, 0.9], window=4),
    }
    diffs, labels, groups = build_pairs(
        {"a.txt": {"LOSER-A"}, "b.txt": {"LOSER-B"}}, features
    )
    train_mask, test_mask = split_groups(groups, test_groups=1, seed=3)

    for window in np.unique(groups):
        in_train = bool(np.unique(train_mask[groups == window]))
        in_test = bool(np.unique(test_mask[groups == window]))
        # A window is wholly on one side; never split across both.
        assert in_train != in_test
