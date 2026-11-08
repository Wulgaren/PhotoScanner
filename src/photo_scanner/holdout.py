"""Deterministic photo-level holdout for honest evaluation.

Every photo is assigned to a bucket by hashing a stable identity: the asset
UUID for library photos, the file content hash for negatives. The same photo
always lands in the same bucket, so the holdout survives re-runs, changed row
order, and newly added photos.

Training code filters holdout photos out of the training matrix; the evaluation
pass scores them once. That keeps the encoder and the head from ever seeing the
photos whose AUC is reported.
"""

from __future__ import annotations

import hashlib

from learn_from_feedback import normalize_uuid

# ~1/10 of photos: buckets 0..9, hold out bucket 0.
HOLDOUT_MODULUS = 10
HOLDOUT_BUCKET = 0


def _bucket(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % HOLDOUT_MODULUS


def good_key(uuid: str | None, path: str = "") -> str:
    """Stable identity for a library photo, preferring its UUID."""
    ident = normalize_uuid(uuid) if uuid else None
    return f"good:{ident or path}"


def bad_key(content_digest: str) -> str:
    """Stable identity for a negative, keyed by file content."""
    return f"bad:{content_digest}"


def is_holdout(key: str) -> bool:
    return _bucket(key) == HOLDOUT_BUCKET
