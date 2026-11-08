"""Persist negative-example features so BadPhotos/ can be emptied safely.

Train rebuilds its negative list from ``BadPhotos/`` on every run. Deleting a
file therefore dropped its negative example even though its embedding still
sat in the feature cache. This store keeps negatives independently of the
filesystem:

  * keyed by backbone+preprocess, so stale embeddings are ignored after a
    backbone change;
  * one record per image content hash, so duplicates collapse and a moved or
    renamed file still maps to its feature;
  * written atomically, and a corrupt file degrades to an empty store (the
    next Train run repopulates it from whatever is on disk).
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from photo_scanner.paths import CACHE_DIR

NEGATIVE_FEATURES_FILE = CACHE_DIR / "negative_features.pkl"

# store_key -> {"features": {content_hash: ndarray}, "paths": {path: hash}}
NegativeStore = dict


def store_key(backbone: str, preprocess: str) -> str:
    """Namespace records by backbone+preprocess so stale features are ignored."""
    return f"{backbone}|{preprocess}"


def content_hash(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """SHA-1 of the file bytes, used as a path-independent identity."""
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_negative_features() -> NegativeStore:
    """Load the negative store. Corrupt/absent file -> empty store."""
    if not NEGATIVE_FEATURES_FILE.exists():
        return {}
    try:
        with open(NEGATIVE_FEATURES_FILE, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(store: NegativeStore) -> None:
    tmp = NEGATIVE_FEATURES_FILE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(store, f)
    tmp.replace(NEGATIVE_FEATURES_FILE)


def _slot(store: NegativeStore, backbone: str, preprocess: str) -> dict:
    slot = store.setdefault(store_key(backbone, preprocess), {})
    if not isinstance(slot.get("features"), dict):
        slot["features"] = {}
    if not isinstance(slot.get("paths"), dict):
        slot["paths"] = {}
    return slot


def persisted_negatives(
    backbone: str, preprocess: str
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Return (path -> content hash, content hash -> feature) for this backbone."""
    slot = load_negative_features().get(store_key(backbone, preprocess), {})
    if not isinstance(slot, dict):
        return {}, {}
    paths = slot.get("paths")
    features = slot.get("features")
    return (
        paths if isinstance(paths, dict) else {},
        features if isinstance(features, dict) else {},
    )


def total_persisted_negatives() -> int:
    """Number of stored negative features across every backbone namespace."""
    store = load_negative_features()
    return sum(
        len(slot.get("features") or {})
        for slot in store.values()
        if isinstance(slot, dict)
    )


def record_negative_batch(
    entries: Iterable[tuple[str, np.ndarray]],
    backbone: str,
    preprocess: str,
) -> int:
    """Store features for negative images.

    entries: (path, feature_vector). The path is hashed to a content identity,
    so identical images share one record. Files that vanished since the
    caller read them are skipped. Returns how many entries were written.
    """
    entries = list(entries)
    if not entries:
        return 0
    store = load_negative_features()
    slot = _slot(store, backbone, preprocess)
    written = 0
    for path, feature in entries:
        try:
            digest = content_hash(path)
        except OSError:
            continue
        slot["paths"][str(path)] = digest
        if digest not in slot["features"]:
            slot["features"][digest] = np.asarray(feature, dtype=np.float32)
        written += 1
    _write_store(store)
    return written
