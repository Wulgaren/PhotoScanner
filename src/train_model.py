#!/usr/bin/env python3
"""
Train the photo preference model on your curated photos.

Training signal, in priority order:
  1. Favorited photos before the cutoff (positives)
  2. BadPhotos/ images (negatives — out-of-distribution junk)
  3. Flagged-photo feedback (the important part):
       * photos you deleted after they were flagged  → hard negatives
       * photos you rescued after they were flagged  → positives
     Scan persists CLIP features for flagged photos (.cache/flagged_features.pkl)
     because a deleted photo's image is gone from Photos by Train time.

Every run also reports honest cross-validated AUC/PR-AUC and writes
.cache/train_metrics.json so runs are comparable, and derives the scan
"flag below" threshold from your rescue/delete decisions (best F1).
"""

import osxphotos
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import pickle
import random
import shutil
from rich.console import Console
from tqdm import tqdm

from photo_scanner.feature_extractor import (
    DEFAULT_BACKBONE,
    AestheticScorer,
    FeatureExtractor,
    make_preference_head,
)
from photo_scanner.paths import BAD_PHOTOS_DIR, CACHE_DIR, MODEL_PATH, ensure_data_dirs
from photo_scanner.thumbs import (
    EMBED_MAX_EDGE,
    EMBED_PREPROCESS,
    build_uuid_to_path_index,
    find_real_thumb_for,
    resolve_scorable_image,
    resolve_thumb_for_uuid,
)
from photo_scanner.feedback_features import (
    calibrate_threshold,
    flagged_for_backbone,
)
from learn_from_feedback import normalize_uuid

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Video extensions to skip
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
# Image extensions to include from BadPhotos folder
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tiff', '.tif'}

console = Console()

ensure_data_dirs()
RESCUED_PHOTOS_FILE = CACHE_DIR / 'rescued_photos.json'
METRICS_FILE = CACHE_DIR / 'train_metrics.json'

# Hard negatives (photos you actually deleted) describe the exact decision
# boundary the scanner ranks, so they outrank the junk in BadPhotos/.
DEFAULT_HARD_NEG_WEIGHT = 3.0
DEFAULT_CV_FOLDS = 5


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(path: Path) -> Path | None:
    """Copy a file aside with a timestamp; returns the backup path or None."""
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak_{_timestamp()}")
    shutil.copy2(path, backup)
    return backup


def get_photo_paths(photos, desc="Getting paths"):
    """Resolve readable paths: original/edited, Photos derivatives, else review thumbs."""
    paths = []
    skipped = 0
    skipped_videos = 0
    used_derivatives = 0
    used_thumbs = 0
    uuid_to_path = build_uuid_to_path_index()

    for photo in tqdm(photos, desc=desc):
        path = photo.path
        path_edited = photo.path_edited
        if path and Path(path).suffix.lower() in VIDEO_EXTENSIONS and not (
            path_edited and Path(path_edited).suffix.lower() not in VIDEO_EXTENSIONS
        ):
            skipped_videos += 1
            continue

        try:
            derivs = list(photo.path_derivatives or [])
        except Exception:
            derivs = []

        resolved = resolve_scorable_image(
            photo.uuid,
            path=path if path and Path(path).suffix.lower() not in VIDEO_EXTENSIONS else None,
            path_edited=(
                path_edited
                if path_edited and Path(path_edited).suffix.lower() not in VIDEO_EXTENSIONS
                else None
            ),
            derivatives=derivs,
            uuid_to_path=uuid_to_path,
        )
        if resolved is None:
            skipped += 1
            continue

        score_path, _library_path, source = resolved
        if Path(score_path).suffix.lower() in VIDEO_EXTENSIONS:
            skipped_videos += 1
            continue
        if source == "derivative":
            used_derivatives += 1
        elif source == "thumb":
            used_thumbs += 1
        paths.append((photo.uuid, score_path, photo.date))

    if skipped > 0:
        console.print(
            f"[yellow]Skipped {skipped} photos "
            f"(no local original, derivative, or real thumb)[/yellow]"
        )
    if used_derivatives:
        console.print(
            f"[green]✓[/green] Using {used_derivatives:,} Photos derivatives "
            f"(iCloud originals not on disk)"
        )
    if used_thumbs:
        console.print(
            f"[green]✓[/green] Using {used_thumbs:,} review thumbs "
            f"(iCloud originals not on disk)"
        )
    if skipped_videos > 0:
        console.print(f"[dim]Skipped {skipped_videos} videos[/dim]")

    return paths


def fill_missing_with_review_thumbs(
    missing_uuids: set,
    uuid_to_path: dict,
    date_by_uuid: dict | None = None,
) -> list[tuple]:
    """
    For UUIDs with no local original, use real review GUI thumbs when available.

    Placeholders (missing_*.jpg) are never used. Returns (uuid, thumb_path, date).
    """
    filled = []
    date_by_uuid = date_by_uuid or {}
    for raw_uuid in missing_uuids:
        uid = normalize_uuid(raw_uuid)
        if not uid:
            continue
        entry = uuid_to_path.get(uid)
        if entry:
            scan_path, thumb_uuid = entry
            thumb = find_real_thumb_for(uid, scan_path, thumb_uuid=thumb_uuid)
        else:
            thumb = resolve_thumb_for_uuid(uid, uuid_to_path)
        if thumb is None:
            continue
        filled.append((uid, str(thumb), date_by_uuid.get(uid)))
    return filled


def get_bad_photo_paths():
    """Get paths from the BadPhotos folder if it exists."""
    if not BAD_PHOTOS_DIR.exists():
        return []
    
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(BAD_PHOTOS_DIR.glob(f'*{ext}'))
        paths.extend(BAD_PHOTOS_DIR.glob(f'*{ext.upper()}'))
    
    # Also search subdirectories
    for ext in IMAGE_EXTENSIONS:
        paths.extend(BAD_PHOTOS_DIR.glob(f'**/*{ext}'))
        paths.extend(BAD_PHOTOS_DIR.glob(f'**/*{ext.upper()}'))
    
    # Remove duplicates and convert to strings
    unique_paths = list(set(str(p) for p in paths))
    return unique_paths


def get_rescued_photo_uuids():
    """Get UUIDs of photos that were rescued (user marked as good via feedback)."""
    if not RESCUED_PHOTOS_FILE.exists():
        return set()
    
    try:
        with open(RESCUED_PHOTOS_FILE) as f:
            from learn_from_feedback import normalize_uuid_set
            return normalize_uuid_set(json.load(f))
    except Exception:
        return set()


def get_feedback_bad_photo_uuids():
    """Get UUIDs of photos marked as bad via feedback."""
    bad_file = CACHE_DIR / 'feedback_bad_photos.json'
    if not bad_file.exists():
        return set()
    
    try:
        with open(bad_file) as f:
            from learn_from_feedback import normalize_uuid_set
            return normalize_uuid_set(json.load(f))
    except Exception:
        return set()


def get_session_kept_uuids() -> set:
    """UUIDs marked keep in the active review session (may not be in rescued yet)."""
    from learn_from_feedback import normalize_uuid_set
    from photo_scanner.paths import SESSION_FILE

    if not SESSION_FILE.exists():
        return set()
    try:
        data = json.loads(SESSION_FILE.read_text())
    except Exception:
        return set()
    kept = data.get("kept_from_flat") or []
    return normalize_uuid_set(
        p.get("uuid") for p in kept if isinstance(p, dict) and p.get("uuid")
    )


def collect_training_data(cutoff_date: datetime, sample_size: int = None) -> dict:
    """Gather every labeled training image (favorites+, BadPhotos, feedback bads).

    Returns good_paths [(uuid, path, date)], bad_paths [path...] and the uuid
    sets Train needs to wire in flagged-photo feedback. Prints progress.
    """
    # Check for BadPhotos folder
    bad_photo_paths = get_bad_photo_paths()
    has_bad_photos = len(bad_photo_paths) > 0
    
    # Check for feedback data
    rescued_uuids = get_rescued_photo_uuids()
    session_kept_uuids = get_session_kept_uuids()
    positive_uuids = set(rescued_uuids) | set(session_kept_uuids)
    feedback_bad_uuids = get_feedback_bad_photo_uuids()
    has_rescued = len(rescued_uuids) > 0
    has_session_kept = len(session_kept_uuids - rescued_uuids) > 0
    has_feedback_bad = len(feedback_bad_uuids) > 0
    
    if has_bad_photos or has_feedback_bad:
        total_bad = len(bad_photo_paths) + len(feedback_bad_uuids)
        console.print(f"[yellow]📁 Found {total_bad:,} bad photo examples[/yellow]")
        if has_bad_photos:
            console.print(f"   • {len(bad_photo_paths):,} from BadPhotos folder")
        if has_feedback_bad:
            console.print(f"   • {len(feedback_bad_uuids):,} from feedback")
        console.print("   Will train binary classifier (good vs bad)\n")
    else:
        console.print(f"[dim]No bad photos found - will use one-class learning[/dim]")
        console.print(f"[dim]Tip: Create a 'BadPhotos' folder or use feedback learning for better results[/dim]\n")
    
    if has_rescued:
        console.print(f"[green]📚 Found {len(rescued_uuids)} rescued photos from feedback[/green]")
        console.print("   These will be included as positive examples\n")
    if has_session_kept:
        console.print(
            f"[green]📌 Found {len(session_kept_uuids - rescued_uuids)} additional "
            f"keeps in review session[/green]"
        )
    
    # Connect to Photos
    console.print("Connecting to Apple Photos...")
    photosdb = osxphotos.PhotosDB()
    console.print(f"[green]✓[/green] Connected\n")
    
    # Get training photos (favorited before cutoff date)
    console.print(f"Finding training photos (favorited before {cutoff_date.strftime('%Y-%m-%d')})...")
    
    all_photos = photosdb.photos()
    
    # Favorites as positive examples (ismissing OK — derivatives / thumbs resolve later)
    good_photos = [
        p for p in all_photos 
        if p.favorite and p.date and p.date < cutoff_date
        and not p.screenshot  # Exclude screenshots
    ]
    
    console.print(f"[green]✓[/green] Found {len(good_photos):,} curated photos (good examples)")
    
    # Add rescued / session-kept photos to good examples
    if positive_uuids:
        positive_photos = [p for p in all_photos if normalize_uuid(p.uuid) in positive_uuids]
        # Filter out any already in good_photos
        existing_uuids = {normalize_uuid(p.uuid) for p in good_photos}
        new_positives = [p for p in positive_photos if normalize_uuid(p.uuid) not in existing_uuids]
        good_photos.extend(new_positives)
        console.print(f"[green]✓[/green] Added {len(new_positives)} rescued/session-kept photos to training")
    
    # Get feedback bad photos from Photos library
    feedback_bad_photos = []
    if has_feedback_bad:
        feedback_bad_photos = [
            p for p in all_photos
            if normalize_uuid(p.uuid) in feedback_bad_uuids
        ]
        console.print(f"[red]✓[/red] Found {len(feedback_bad_photos)} feedback bad photos in library")
    
    # Sample if needed
    if sample_size:
        if len(good_photos) > sample_size:
            good_photos = random.sample(good_photos, sample_size)
            console.print(f"  Sampled {sample_size} good photos")
        if has_bad_photos and len(bad_photo_paths) > sample_size:
            bad_photo_paths = random.sample(bad_photo_paths, sample_size)
            console.print(f"  Sampled {sample_size} bad photos")
    
    # Get file paths
    console.print("\nResolving file paths...")

    good_paths = get_photo_paths(good_photos, "Good photos")
    resolved_good_uuids = {normalize_uuid(u) for u, _, _ in good_paths}

    # iCloud Optimize: originals often missing — fall back to real review thumbs
    uuid_to_path = build_uuid_to_path_index()
    date_by_uuid = {
        normalize_uuid(p.uuid): p.date
        for p in good_photos
        if normalize_uuid(p.uuid)
    }
    # Prefer rescued/session-kept + any good photo still unresolved
    candidates_for_thumbs = set(positive_uuids) | {
        normalize_uuid(p.uuid) for p in good_photos if normalize_uuid(p.uuid)
    }
    missing_good = candidates_for_thumbs - resolved_good_uuids
    thumb_goods = fill_missing_with_review_thumbs(
        missing_good, uuid_to_path, date_by_uuid=date_by_uuid
    )
    if thumb_goods:
        good_paths.extend(thumb_goods)
        console.print(
            f"[green]✓[/green] Using {len(thumb_goods):,} review thumbs as good examples "
            f"(iCloud originals not on disk)"
        )
    elif missing_good:
        console.print(
            f"[yellow]No review thumbs found for {len(missing_good):,} missing good photos[/yellow]"
        )
        console.print(
            "[dim]Tip: run Review once so .cache/review_thumbs/ has real (non-placeholder) thumbs[/dim]"
        )

    # Get paths for feedback bad photos
    feedback_bad_paths = []
    if feedback_bad_photos:
        feedback_bad_tuples = get_photo_paths(feedback_bad_photos, "Feedback bad photos")
        feedback_bad_paths = [p[1] for p in feedback_bad_tuples]
        resolved_bad_uuids = {normalize_uuid(u) for u, _, _ in feedback_bad_tuples}
        missing_bad = {
            normalize_uuid(p.uuid) for p in feedback_bad_photos if normalize_uuid(p.uuid)
        } - resolved_bad_uuids
        thumb_bads = fill_missing_with_review_thumbs(missing_bad, uuid_to_path)
        if thumb_bads:
            feedback_bad_paths.extend(t[1] for t in thumb_bads)
            console.print(
                f"[red]✓[/red] Using {len(thumb_bads):,} review thumbs as bad examples"
            )
    
    # Combine all bad photo paths
    all_bad_paths = bad_photo_paths + feedback_bad_paths

    return {
        "good_paths": good_paths,
        "bad_paths": all_bad_paths,
        "rescued_uuids": rescued_uuids,
        "feedback_bad_uuids": feedback_bad_uuids,
    }


def load_feature_cache():
    """Load existing feature cache if available."""
    cache_file = CACHE_DIR / 'feature_cache.pkl'
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load cache: {e}[/yellow]")
    return None


def save_feature_cache(cache_data):
    """Save feature cache to disk."""
    cache_file = CACHE_DIR / 'feature_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)


def cross_validate_head(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                        folds: int = DEFAULT_CV_FOLDS) -> dict:
    """Honest evaluation: out-of-fold scores with per-fold scaling.

    Returns {"oof": probabilities, "roc_auc": mean, "roc_auc_folds": [...],
             "avg_precision": mean, "avg_precision_folds": [...]}.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    aucs, aps = [], []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        head = make_preference_head()
        head.fit(
            scaler.transform(X[train_idx]), y[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        proba = head.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        oof[val_idx] = proba
        aucs.append(roc_auc_score(y[val_idx], proba))
        aps.append(average_precision_score(y[val_idx], proba))
    return {
        "oof": oof,
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_folds": [float(a) for a in aucs],
        "avg_precision": float(np.mean(aps)),
        "avg_precision_folds": [float(a) for a in aps],
    }


def train(
    cutoff_date: datetime,
    sample_size: int = None,
    batch_size: int = 32,
    model_name: str = DEFAULT_BACKBONE,
    hard_neg_weight: float = DEFAULT_HARD_NEG_WEIGHT,
    cv_folds: int = DEFAULT_CV_FOLDS,
):
    """
    Train the model on curated photos.
    
    Args:
        cutoff_date: Photos before this date are considered training data
        sample_size: Limit training samples (None = use all)
        batch_size: Batch size for feature extraction
        model_name: timm backbone for visual embeddings, or a fine-tuned tag
        hard_neg_weight: Sample weight for confirmed deletions (hard negatives)
        cv_folds: Folds for the honest evaluation pass
    """
    console.print("\n[bold blue]📸 PhotoScanner - Model Training[/bold blue]\n")
    console.print(f"[dim]Backbone: {model_name}[/dim]")
    
    data = collect_training_data(cutoff_date, sample_size)
    good_paths = data["good_paths"]
    all_bad_paths = data["bad_paths"]
    rescued_uuids = data["rescued_uuids"]
    feedback_bad_uuids = data["feedback_bad_uuids"]
    has_any_bad = len(all_bad_paths) > 0

    # ---- Flagged-photo feedback: the only decision-relevant signal ----
    flagged = flagged_for_backbone(model_name, EMBED_PREPROCESS)
    flagged_rescued = [u for u in flagged if u in rescued_uuids]
    flagged_deleted = [
        u for u in flagged
        if u in feedback_bad_uuids and u not in rescued_uuids
    ]
    if flagged:
        console.print(
            f"[green]🧲 {len(flagged):,} flagged photos have saved features "
            f"({len(flagged_rescued)} rescued, {len(flagged_deleted)} deleted)"
        )
    else:
        console.print(
            "[dim]No flagged-photo features yet — runs the feedback loop once "
            "(Scan → Review/Move → Learn) so deletions become hard negatives[/dim]"
        )

    console.print(f"\nReady to extract features from:")
    console.print(f"  • [green]{len(good_paths):,}[/green] good photos")
    if has_any_bad:
        console.print(f"  • [red]{len(all_bad_paths):,}[/red] bad photos")
        if flagged_deleted:
            console.print(
                f"  • [red]{len(flagged_deleted):,}[/red] confirmed deletions "
                f"(hard negatives, weight ×{hard_neg_weight:g})"
            )
        console.print(
            f"[dim]Full-res negatives downscale to max edge {EMBED_MAX_EDGE} "
            f"to match review thumbs[/dim]"
        )

    if not good_paths:
        console.print(
            "\n[bold red]No good training images found.[/bold red]\n"
            "Favorites/rescued need local originals, Photos derivatives, or real "
            "review thumbs in .cache/review_thumbs/ (not missing_*.jpg placeholders)."
        )
        return

    # Load existing cache for incremental extraction
    existing_cache = load_feature_cache()
    cached_features = {}
    if existing_cache and existing_cache.get('backbone') not in (None, model_name):
        console.print(
            f"[yellow]Backbone changed "
            f"({existing_cache.get('backbone')} → {model_name}); clearing feature cache[/yellow]"
        )
        existing_cache = None
    elif existing_cache and 'backbone' not in existing_cache:
        # Legacy EfficientNet caches have no backbone tag
        console.print(
            f"[yellow]Legacy feature cache has no backbone tag; clearing for {model_name}[/yellow]"
        )
        existing_cache = None
    elif existing_cache and existing_cache.get('preprocess') not in (None, EMBED_PREPROCESS):
        console.print(
            f"[yellow]Embed preprocess changed "
            f"({existing_cache.get('preprocess')} → {EMBED_PREPROCESS}); "
            f"clearing feature cache[/yellow]"
        )
        existing_cache = None
    elif existing_cache and 'preprocess' not in existing_cache:
        console.print(
            f"[yellow]Legacy feature cache has no preprocess tag; "
            f"clearing for {EMBED_PREPROCESS}[/yellow]"
        )
        existing_cache = None
    if existing_cache is None:
        backup = _backup(CACHE_DIR / 'feature_cache.pkl')
        if backup:
            console.print(f"[dim]Old feature cache backed up to {backup.name}[/dim]")
    if existing_cache and 'feature_cache' in existing_cache:
        cached_features = existing_cache.get('feature_cache', {})
        console.print(f"[dim]Loaded {len(cached_features):,} cached features[/dim]")
    
    # Determine which photos need feature extraction (good photos)
    good_paths_to_extract = [p[1] for p in good_paths if p[1] not in cached_features]
    good_already_cached = len(good_paths) - len(good_paths_to_extract)
    
    # Determine which bad photos need extraction
    bad_paths_to_extract = [p for p in all_bad_paths if p not in cached_features] if has_any_bad else []
    bad_already_cached = len(all_bad_paths) - len(bad_paths_to_extract) if has_any_bad else 0
    
    if good_already_cached > 0 or bad_already_cached > 0:
        console.print(f"[green]✓[/green] Cached: {good_already_cached:,} good, {bad_already_cached:,} bad")
        console.print(f"   To extract: {len(good_paths_to_extract):,} good, {len(bad_paths_to_extract):,} bad")
    
    # Initialize feature extractor (max_edge matches review thumbs)
    console.print("\n[bold]Initializing neural network...[/bold]")
    extractor = FeatureExtractor(model_name=model_name, max_edge=EMBED_MAX_EDGE)
    
    # Combine all paths to extract
    all_paths_to_extract = good_paths_to_extract + bad_paths_to_extract
    
    # Extract features with incremental saving
    if all_paths_to_extract:
        console.print("\n[bold]Extracting features...[/bold]")
        console.print("[dim]Progress is saved every 100 photos - safe to interrupt[/dim]\n")
        
        save_interval = 100
        for i in tqdm(range(0, len(all_paths_to_extract), batch_size), desc="Extracting features"):
            batch_paths = all_paths_to_extract[i:i + batch_size]
            batch_features = extractor.extract_batch(batch_paths, batch_size=batch_size, show_progress=False)
            
            # Add to cache
            cached_features.update(batch_features)
            
            # Save incrementally
            if (i // batch_size + 1) % (save_interval // batch_size) == 0 or i + batch_size >= len(all_paths_to_extract):
                save_feature_cache({
                    'feature_cache': cached_features,
                    'cutoff_date': cutoff_date,
                    'backbone': model_name,
                    'preprocess': EMBED_PREPROCESS,
                })
        
        console.print(f"[green]✓[/green] Extracted features for {len(all_paths_to_extract):,} photos")
    
    # ---- Build the training matrix ----
    # Rows: favorites+rescued+session-kept (w=1), BadPhotos/feedback-bad (w=1),
    # confirmed deletions from flagged feedback (w=hard_neg_weight).
    features_list, labels_list, weights_list = [], [], []
    rescued_row_idx = []      # positives the user actively rescued
    deleted_row_idx = []      # hard negatives: photos the user deleted
    for (uuid, path, date) in good_paths:
        feat = cached_features.get(path)
        if feat is None:
            continue
        uid = normalize_uuid(uuid) if uuid else ""
        if uid and uid in rescued_uuids:
            rescued_row_idx.append(len(features_list))
        features_list.append(feat)
        labels_list.append(1)
        weights_list.append(1.0)

    # Rescued flagged photos the favorites pass missed (e.g. no longer favorited)
    for uuid in flagged_rescued:
        rec = flagged[uuid]
        rescued_row_idx.append(len(features_list))
        features_list.append(rec["feature"])
        labels_list.append(1)
        weights_list.append(1.0)

    n_flagged_deleted_used = 0
    for uuid in flagged_deleted:
        rec = flagged[uuid]
        deleted_row_idx.append(len(features_list))
        features_list.append(rec["feature"])
        labels_list.append(0)
        weights_list.append(hard_neg_weight)
        n_flagged_deleted_used += 1

    n_bad_used = 0
    for path in all_bad_paths:
        feat = cached_features.get(path)
        if feat is None:
            continue
        features_list.append(feat)
        labels_list.append(0)
        weights_list.append(1.0)
        n_bad_used += 1

    X = np.array(features_list)
    y = np.array(labels_list, dtype=int)
    w = np.array(weights_list, dtype=float)

    n_good = int((y == 1).sum())
    n_bad = int((y == 0).sum())
    console.print(f"\n[green]✓[/green] Good features: {n_good:,}")
    if n_bad > 0:
        console.print(f"[red]✓[/red] Bad features: {n_bad:,} "
                      f"({n_flagged_deleted_used:,} confirmed deletions × {hard_neg_weight:g})")

    if n_good == 0:
        console.print("[bold red]No good features extracted — cannot train.[/bold red]")
        return

    # Save final feature cache
    console.print("\nSaving feature cache...")
    save_feature_cache({
        'good_paths': good_paths,
        'good_features': np.array([cached_features[p[1]] for p in good_paths if p[1] in cached_features]),
        'bad_paths': all_bad_paths if has_any_bad else [],
        'bad_features': np.array([cached_features[p] for p in all_bad_paths if p in cached_features]) if has_any_bad else None,
        'feature_cache': cached_features,
        'cutoff_date': cutoff_date,
        'backbone': model_name,
        'preprocess': EMBED_PREPROCESS,
    })

    # ---- Honest evaluation (out-of-fold) ----
    cv = {}
    if n_bad >= 10 and min(np.bincount(y)) >= cv_folds:
        console.print(f"\n[bold]Cross-validating ({cv_folds} folds)...[/bold]")
        cv = cross_validate_head(X, y, w, folds=cv_folds)
        console.print(
            f"  ROC-AUC: {cv['roc_auc']:.3f} "
            f"(folds: {', '.join(f'{a:.3f}' for a in cv['roc_auc_folds'])})"
        )
        console.print(f"  Avg precision: {cv['avg_precision']:.3f}")
    else:
        console.print("[yellow]Too few samples for cross-validation; skipping CV[/yellow]")

    # ---- Train the model ----
    console.print("\n[bold]Training preference scorer...[/bold]")
    scorer = AestheticScorer(extractor)
    scorer.train(
        X[y == 1], X[y == 0] if n_bad > 0 else None,
        sample_weight=w,
    )

    # ---- Threshold calibration on the decision-relevant population ----
    # Flag rule is score < t: rescues must stay above, deletions must fall below.
    threshold_info = {}
    if cv.get("oof") is not None and deleted_row_idx and rescued_row_idx:
        keep_scores = cv["oof"][rescued_row_idx]
        delete_scores = cv["oof"][deleted_row_idx]
        t = calibrate_threshold(keep_scores, delete_scores)
        if t is not None:
            scorer.recommended_threshold = t
            flagged_del = int((delete_scores < t).sum())
            flagged_keep = int((keep_scores < t).sum())
            delete_recall = flagged_del / len(delete_scores)
            flag_precision = (
                flagged_del / (flagged_del + flagged_keep)
                if (flagged_del + flagged_keep) > 0 else None
            )
            keep_safe = 1.0 - flagged_keep / len(keep_scores)
            threshold_info = {
                "threshold": t,
                "delete_recall": delete_recall,
                "flag_precision": flag_precision,   # of flagged, share actually deleted
                "rescued_kept_rate": keep_safe,
                "n_rescued": len(keep_scores),
                "n_deleted": len(delete_scores),
                "source": "cross_validated_feedback",
            }
            precision_note = (
                f"of flagged {flag_precision*100:.0f}% were deletions"
                if flag_precision is not None else "no rescues flagged"
            )
            console.print(
                f"\n[bold]Auto-calibrated flag threshold: {t:.3f}[/bold]"
            )
            console.print(
                f"  On your real decisions: catches {delete_recall*100:.0f}% of deletions, "
                f"keeps {keep_safe*100:.0f}% of rescues ({precision_note})"
            )
    else:
        console.print(
            "\n[dim]Not enough rescue/delete feedback yet to auto-calibrate; "
            "Scan keeps its --threshold default (0.3).[/dim]"
        )

    # Back up the previous model before overwriting
    backup = _backup(MODEL_PATH)
    if backup:
        console.print(f"[dim]Previous model backed up to {backup.name}[/dim]")

    # Save model
    scorer.save(MODEL_PATH)

    # ---- Metrics + comparison with the previous run ----
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "backbone": model_name,
        "preprocess": EMBED_PREPROCESS,
        "n_good": n_good,
        "n_bad": n_bad,
        "n_confirmed_deletions": n_flagged_deleted_used,
        "hard_neg_weight": hard_neg_weight,
        "cv_folds": cv_folds if cv else None,
        "roc_auc": cv.get("roc_auc"),
        "roc_auc_folds": cv.get("roc_auc_folds"),
        "avg_precision": cv.get("avg_precision"),
        "threshold": threshold_info,
        "head": "MLP(512-128)",
    }
    prev = None
    if METRICS_FILE.exists():
        try:
            prev = json.loads(METRICS_FILE.read_text())
        except Exception:
            prev = None
    METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    console.print(f"[dim]Metrics saved to {METRICS_FILE}[/dim]")
    if prev and prev.get("roc_auc") is not None and metrics.get("roc_auc") is not None:
        delta = metrics["roc_auc"] - prev["roc_auc"]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        console.print(
            f"[dim]vs previous run: AUC {prev['roc_auc']:.3f} → "
            f"{metrics['roc_auc']:.3f} ({arrow} {abs(delta):.3f})[/dim]"
        )

    # ---- In-sample summary (sanity, not generalization) ----
    console.print("\n[bold]Summary (in-sample):[/bold]")
    good_scores = scorer.score(X[y == 1])
    console.print(f"  Good photos  - Mean score: {good_scores.mean():.3f}, Std: {good_scores.std():.3f}")
    if n_bad > 0:
        bad_scores = scorer.score(X[y == 0])
        console.print(f"  Bad photos   - Mean score: {bad_scores.mean():.3f}, Std: {bad_scores.std():.3f}")
        separation = good_scores.mean() - bad_scores.mean()
        console.print(f"\n  [bold]Score separation: {separation:.3f}[/bold]")
        if separation > 0.3:
            console.print("  [green]✓ Good separation between good and bad photos![/green]")
        elif separation > 0.1:
            console.print("  [yellow]○ Moderate separation - consider more flagged deletions[/yellow]")
        else:
            console.print("  [red]⚠ Low separation - let Scan flag, then Learn from feedback[/red]")

    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Model saved to: {MODEL_PATH}")
    console.print(f"\nNext step: [cyan]./photoscanner.sh[/cyan] (Scan) or [dim]python src/scan_photos.py --after {cutoff_date.strftime('%Y-%m-%d')}[/dim]")


def main():
    parser = argparse.ArgumentParser(description='Train photo preference model on your curated (favorited) photos')
    parser.add_argument('--cutoff-date', type=str, default='2023-11-18',
                       help='Cutoff date - photos before this are training data (YYYY-MM-DD)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit number of training samples (for testing)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--model', type=str, default=DEFAULT_BACKBONE,
                       help=f'timm backbone or fine-tuned tag (default: {DEFAULT_BACKBONE})')
    parser.add_argument('--hard-neg-weight', type=float, default=DEFAULT_HARD_NEG_WEIGHT,
                       help='Sample weight for confirmed deletions (hard negatives)')
    parser.add_argument('--cv-folds', type=int, default=DEFAULT_CV_FOLDS,
                       help='Folds for cross-validated evaluation')
    
    args = parser.parse_args()
    
    cutoff = datetime.strptime(args.cutoff_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    try:
        train(
            cutoff_date=cutoff,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            model_name=args.model,
            hard_neg_weight=args.hard_neg_weight,
            cv_folds=args.cv_folds,
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
