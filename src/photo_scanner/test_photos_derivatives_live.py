"""Live smoke: ismissing favorites resolve via Photos derivatives."""

from __future__ import annotations

import pytest

from photo_scanner.thumbs import resolve_scorable_image

osxphotos = pytest.importorskip("osxphotos")

SAMPLE = 25


def test_live_ismissing_favorites_resolve_via_derivatives():
    db = osxphotos.PhotosDB()
    sample = []
    for photo in db.photos():
        if not photo.favorite or photo.screenshot or not photo.ismissing:
            continue
        sample.append(photo)
        if len(sample) >= SAMPLE:
            break

    assert sample, "expected ismissing favorites in the Photos library"

    resolved_as_derivative = 0
    for photo in sample:
        try:
            derivs = list(photo.path_derivatives or [])
        except Exception:
            derivs = []
        result = resolve_scorable_image(
            photo.uuid,
            path=photo.path,
            path_edited=photo.path_edited,
            derivatives=derivs,
        )
        assert result is not None, f"unresolved uuid={photo.uuid}"
        score_path, _library_path, source = result
        assert source == "derivative", (
            f"uuid={photo.uuid} expected derivative, got {source} path={score_path}"
        )
        resolved_as_derivative += 1

    assert resolved_as_derivative == len(sample)
