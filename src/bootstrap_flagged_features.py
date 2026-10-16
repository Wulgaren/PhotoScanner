#!/usr/bin/env python3
"""
Seed .cache/flagged_features.pkl from surviving review thumbs.

Problem this solves: a photo you delete disappears from Photos, so Train used
to lose every "confirmed deletion" signal. When the review thumb is still on
disk, its features can be extracted now and stored, so deletions (and rescues)
become usable hard negatives / positives immediately — no need to wait for the
next Scan → Review → Learn cycle.

Idempotent: safe to run repeatedly.

    python src/bootstrap_flagged_features.py --dry-run
    python src/bootstrap_flagged_features.py
    python src/bootstrap_flagged_features.py --backbone vit_base_patch16_clip_224.openai
"""

import argparse
import json

from rich.console import Console

from learn_from_feedback import normalize_uuid
from photo_scanner.feedback_features import (
    flagged_for_backbone,
    record_flagged_batch,
)
from photo_scanner.feature_extractor import DEFAULT_BACKBONE, FeatureExtractor
from photo_scanner.paths import CACHE_DIR, OUTPUT_DIR, ensure_data_dirs
from photo_scanner.thumbs import (
    EMBED_PREPROCESS,
    build_uuid_to_path_index,
    find_real_thumb_for,
    resolve_thumb_for_uuid,
)

console = Console()
ensure_data_dirs()


def _load_uuids(name: str) -> set:
    path = CACHE_DIR / name
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            return {normalize_uuid(u) for u in json.load(f) if normalize_uuid(u)}
    except Exception:
        return set()


def _scan_scores_by_uuid() -> dict:
    """Latest recorded score per uuid from scan results (if any)."""
    scores: dict[str, tuple[str, float]] = {}
    for path in sorted(OUTPUT_DIR.glob("scan_results_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for photo in data.get("photos", []):
            uid = normalize_uuid(photo.get("uuid"))
            if not uid:
                continue
            score = photo.get("score")
            if score is None:
                continue
            prev = scores.get(uid)
            if prev is None or path.name >= prev[0]:
                scores[uid] = (path.name, float(score))
    return {uid: s for uid, (_, s) in scores.items()}


def bootstrap(backbone: str, dry_run: bool = False) -> int:
    uuid_to_path = build_uuid_to_path_index()
    rescued = _load_uuids("rescued_photos.json")
    deleted = _load_uuids("feedback_bad_photos.json")
    scores = _scan_scores_by_uuid()
    already = flagged_for_backbone(backbone, EMBED_PREPROCESS)

    targets = [(uid, "rescued") for uid in sorted(rescued)] + \
              [(uid, "deleted") for uid in sorted(deleted)]
    console.print(
        f"Looking for review thumbs: {len(rescued)} rescued, {len(deleted)} deleted "
        f"({len(already)} already stored for this backbone)"
    )

    uuid_to_thumb = {}
    for uid, kind in targets:
        entry = uuid_to_path.get(uid)
        if entry:
            scan_path, thumb_uuid = entry
            thumb = find_real_thumb_for(uid, scan_path, thumb_uuid=thumb_uuid)
        else:
            thumb = resolve_thumb_for_uuid(uid, uuid_to_path)
        if thumb is not None:
            uuid_to_thumb[uid] = (str(thumb), kind)

    console.print(f"[green]✓[/green] {len(uuid_to_thumb)} feedback photos have real thumbs on disk")
    if not uuid_to_thumb:
        return 0
    if dry_run:
        for uid, (thumb, kind) in list(uuid_to_thumb.items())[:5]:
            console.print(f"  {kind:8s} {uid} → {thumb}")
        console.print("[yellow]Dry run — nothing written[/yellow]")
        return len(uuid_to_thumb)

    console.print("[dim]Extracting CLIP features (once per thumb)...[/dim]")
    extractor = FeatureExtractor(model_name=backbone, max_edge=900)
    paths = [t for t, _ in uuid_to_thumb.values()]
    features = extractor.extract_batch(paths, batch_size=32, show_progress=False)

    entries = []
    for uid, (thumb, kind) in uuid_to_thumb.items():
        feat = features.get(thumb)
        if feat is None:
            continue
        entries.append((uid, feat, scores.get(uid, 0.0)))
    stored = record_flagged_batch(entries, backbone, EMBED_PREPROCESS)
    n_rescued = sum(1 for _, (_, k) in uuid_to_thumb.items() if k == "rescued")
    n_deleted = sum(1 for _, (_, k) in uuid_to_thumb.items() if k == "deleted")
    console.print(
        f"[green]✓[/green] Stored features for {stored} flagged photos "
        f"({n_deleted} deleted → hard negatives, {n_rescued} rescued → positives) "
        f"for backbone {backbone}"
    )
    console.print(
        "[dim]Next: python src/train_model.py — deleted feedback now trains as "
        "hard negatives[/dim]"
    )
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help=f"Backbone tag (default: {DEFAULT_BACKBONE})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bootstrap(args.backbone, args.dry_run)


if __name__ == "__main__":
    main()
