"""Persist features for flagged photos so user decisions become training signal.

Scan suggests deletions; the user later rescues or deletes those photos in
Review / the "To Delete" album. Before this store existed, that feedback was
unusable at Train time:

  * deleted photos leave the Photos library, so Train could not resolve them;
  * rescued photos are already favorited, so they added nothing new.

Features are cheap (768 floats), so Scan now saves one per flagged photo,
keyed by backbone+preprocess. Train reads them back: confirmed deletions become
hard negatives (same population the scanner ranks), rescues stay positives.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np

from photo_scanner.paths import CACHE_DIR

FLAGGED_FEATURES_FILE = CACHE_DIR / "flagged_features.pkl"

# uuid -> record dict
FlaggedRecord = dict
# store_key -> uuid -> record
FlaggedStore = dict


def store_key(backbone: str, preprocess: str) -> str:
    """Namespace records by backbone+preprocess so stale features are ignored."""
    return f"{backbone}|{preprocess}"


def load_flagged_features() -> FlaggedStore:
    """Load the flagged-features store. Corrupt/absent file -> empty store."""
    if not FLAGGED_FEATURES_FILE.exists():
        return {}
    try:
        with open(FLAGGED_FEATURES_FILE, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_store(store: FlaggedStore) -> None:
    tmp = FLAGGED_FEATURES_FILE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(store, f)
    tmp.replace(FLAGGED_FEATURES_FILE)


def record_flagged_batch(
    entries: Iterable[tuple[str, np.ndarray, float]],
    backbone: str,
    preprocess: str,
) -> int:
    """Append flagged-photo features for the current backbone.

    entries: (uuid, feature_vector, score). Newest write per uuid wins.
    Returns the number of records stored.
    """
    entries = list(entries)
    if not entries:
        return 0
    store = load_flagged_features()
    slot = store.setdefault(store_key(backbone, preprocess), {})
    now = datetime.now(timezone.utc).isoformat()
    for uuid, feature, score in entries:
        if not uuid:
            continue
        existing = slot.get(uuid)
        if existing is not None and str(existing.get("when", "")) >= now:
            continue
        slot[uuid] = {
            "feature": np.asarray(feature, dtype=np.float32),
            "score": float(score),
            "when": now,
        }
    _write_store(store)
    return len(entries)


def flagged_for_backbone(backbone: str, preprocess: str) -> dict[str, FlaggedRecord]:
    """Records usable with the given backbone/preprocess (others ignored)."""
    return load_flagged_features().get(store_key(backbone, preprocess), {})


def calibrate_threshold(
    keep_scores: Iterable[float],
    delete_scores: Iterable[float],
) -> Optional[float]:
    """Pick the flag threshold with the best F1 for catching real deletions.

    Flag rule: score < t. `keep_scores` are photos the user rescued (must NOT
    be flagged), `delete_scores` are photos the user actually deleted (should
    be flagged). Returns None when either side is empty.
    """
    keep = np.asarray(list(keep_scores), dtype=float)
    dele = np.asarray(list(delete_scores), dtype=float)
    keep = keep[np.isfinite(keep)]
    dele = dele[np.isfinite(dele)]
    if keep.size == 0 or dele.size == 0:
        return None

    values = np.unique(np.concatenate([keep, dele]))
    candidates = np.concatenate([[values[0] - 1e-6], (values[:-1] + values[1:]) / 2.0, [values[-1]]])
    best_t, best_f1 = None, -1.0
    for t in candidates:
        tp = int(np.sum(dele < t))
        fp = int(np.sum(keep < t))
        if tp == 0 and fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / dele.size
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t
