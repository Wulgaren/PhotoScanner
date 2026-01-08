#!/usr/bin/env python3
"""
Train the photo preference model on your curated photos.
"""

import osxphotos
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm
import pickle

from photo_scanner.feature_extractor import FeatureExtractor, AestheticScorer

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

# Directory to store models and cache
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

# Folder for bad photo examples
BAD_PHOTOS_DIR = Path(__file__).parent / 'BadPhotos'
# File for rescued photos (from feedback learning)
RESCUED_PHOTOS_FILE = CACHE_DIR / 'rescued_photos.json'


def get_photo_paths(photos, desc="Getting paths"):
    """Get file paths for photos, handling both original and edited versions."""
    paths = []
    skipped = 0
    skipped_videos = 0
    
    for photo in tqdm(photos, desc=desc):
        # Try to get the path
        path = photo.path
        if path and Path(path).exists():
            # Skip videos
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                skipped_videos += 1
                continue
            paths.append((photo.uuid, path, photo.date))
        else:
            # Try edited version
            if photo.path_edited and Path(photo.path_edited).exists():
                if Path(photo.path_edited).suffix.lower() in VIDEO_EXTENSIONS:
                    skipped_videos += 1
                    continue
                paths.append((photo.uuid, photo.path_edited, photo.date))
            else:
                skipped += 1
    
    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} photos (files not found)[/yellow]")
    if skipped_videos > 0:
        console.print(f"[dim]Skipped {skipped_videos} videos[/dim]")
    
    return paths


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
        return []
    
    try:
        with open(RESCUED_PHOTOS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def get_feedback_bad_photo_uuids():
    """Get UUIDs of photos marked as bad via feedback."""
    bad_file = CACHE_DIR / 'feedback_bad_photos.json'
    if not bad_file.exists():
        return []
    
    try:
        with open(bad_file) as f:
            return json.load(f)
    except Exception:
        return []


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


def train(cutoff_date: datetime, sample_size: int = None, batch_size: int = 32):
    """
    Train the model on curated photos.
    
    Uses favorited photos as positive examples. If a 'BadPhotos' folder exists,
    uses those as negative examples for better discrimination.
    
    Args:
        cutoff_date: Photos before this date are considered training data
        sample_size: Limit training samples (None = use all)
        batch_size: Batch size for feature extraction
    """
    console.print("\n[bold blue]📸 PhotoScanner - Model Training[/bold blue]\n")
    
    # Check for BadPhotos folder
    bad_photo_paths = get_bad_photo_paths()
    has_bad_photos = len(bad_photo_paths) > 0
    
    # Check for feedback data
    rescued_uuids = get_rescued_photo_uuids()
    feedback_bad_uuids = get_feedback_bad_photo_uuids()
    has_rescued = len(rescued_uuids) > 0
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
    
    # Connect to Photos
    console.print("Connecting to Apple Photos...")
    photosdb = osxphotos.PhotosDB()
    console.print(f"[green]✓[/green] Connected\n")
    
    # Get training photos (favorited before cutoff date)
    console.print(f"Finding training photos (favorited before {cutoff_date.strftime('%Y-%m-%d')})...")
    
    all_photos = photosdb.photos()
    
    # Only use favorited photos as positive training examples
    good_photos = [
        p for p in all_photos 
        if p.favorite and p.date and p.date < cutoff_date
        and not p.screenshot  # Exclude screenshots
        and not p.ismissing
    ]
    
    console.print(f"[green]✓[/green] Found {len(good_photos):,} curated photos (good examples)")
    
    # Add rescued photos to good examples
    if has_rescued:
        rescued_photos = [p for p in all_photos if p.uuid in rescued_uuids]
        # Filter out any already in good_photos
        existing_uuids = {p.uuid for p in good_photos}
        new_rescued = [p for p in rescued_photos if p.uuid not in existing_uuids]
        good_photos.extend(new_rescued)
        console.print(f"[green]✓[/green] Added {len(new_rescued)} rescued photos to training")
    
    # Get feedback bad photos from Photos library
    feedback_bad_photos = []
    if has_feedback_bad:
        feedback_bad_photos = [p for p in all_photos if p.uuid in feedback_bad_uuids and not p.ismissing]
        console.print(f"[red]✓[/red] Found {len(feedback_bad_photos)} feedback bad photos in library")
    
    # Sample if needed
    if sample_size:
        import random
        if len(good_photos) > sample_size:
            good_photos = random.sample(good_photos, sample_size)
            console.print(f"  Sampled {sample_size} good photos")
        if has_bad_photos and len(bad_photo_paths) > sample_size:
            bad_photo_paths = random.sample(bad_photo_paths, sample_size)
            console.print(f"  Sampled {sample_size} bad photos")
    
    # Get file paths
    console.print("\nResolving file paths...")
    good_paths = get_photo_paths(good_photos, "Good photos")
    
    # Get paths for feedback bad photos
    feedback_bad_paths = []
    if feedback_bad_photos:
        feedback_bad_paths = get_photo_paths(feedback_bad_photos, "Feedback bad photos")
        # Convert to just paths (not tuples)
        feedback_bad_paths = [p[1] for p in feedback_bad_paths]
    
    # Combine all bad photo paths
    all_bad_paths = bad_photo_paths + feedback_bad_paths
    has_any_bad = len(all_bad_paths) > 0
    
    console.print(f"\nReady to extract features from:")
    console.print(f"  • [green]{len(good_paths):,}[/green] good photos")
    if has_any_bad:
        console.print(f"  • [red]{len(all_bad_paths):,}[/red] bad photos")
    
    # Load existing cache for incremental extraction
    existing_cache = load_feature_cache()
    cached_features = {}
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
    
    # Initialize feature extractor
    console.print("\n[bold]Initializing neural network...[/bold]")
    extractor = FeatureExtractor(model_name='efficientnet_b0')
    
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
                })
        
        console.print(f"[green]✓[/green] Extracted features for {len(all_paths_to_extract):,} photos")
    
    # Build final feature arrays
    good_features = np.array([cached_features[p[1]] for p in good_paths if p[1] in cached_features])
    bad_features = None
    if has_any_bad:
        bad_features = np.array([cached_features[p] for p in all_bad_paths if p in cached_features])
    
    console.print(f"\n[green]✓[/green] Good features: {len(good_features):,}")
    if bad_features is not None and len(bad_features) > 0:
        console.print(f"[red]✓[/red] Bad features: {len(bad_features):,}")
    
    # Save final feature cache
    console.print("\nSaving feature cache...")
    save_feature_cache({
        'good_paths': good_paths,
        'good_features': good_features,
        'bad_paths': all_bad_paths if has_any_bad else [],
        'bad_features': bad_features,
        'feature_cache': cached_features,
        'cutoff_date': cutoff_date,
    })
    
    # Train the model
    console.print("\n[bold]Training aesthetic scorer...[/bold]")
    scorer = AestheticScorer(extractor)
    scorer.train(good_features, bad_features)
    
    # Save model
    model_path = CACHE_DIR / 'aesthetic_model.pkl'
    scorer.save(model_path)
    
    # Quick validation
    console.print("\n[bold]Validation:[/bold]")
    good_scores = scorer.score(good_features)
    console.print(f"  Good photos  - Mean score: {good_scores.mean():.3f}, Std: {good_scores.std():.3f}")
    
    if bad_features is not None and len(bad_features) > 0:
        bad_scores = scorer.score(bad_features)
        console.print(f"  Bad photos   - Mean score: {bad_scores.mean():.3f}, Std: {bad_scores.std():.3f}")
        
        # Show separation
        separation = good_scores.mean() - bad_scores.mean()
        console.print(f"\n  [bold]Score separation: {separation:.3f}[/bold]")
        if separation > 0.3:
            console.print("  [green]✓ Good separation between good and bad photos![/green]")
        elif separation > 0.1:
            console.print("  [yellow]○ Moderate separation - consider adding more bad examples[/yellow]")
        else:
            console.print("  [red]⚠ Low separation - add more diverse bad examples[/red]")
    
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Model saved to: {model_path}")
    console.print(f"\nNext step: Run [cyan]python scan_photos.py --after {cutoff_date.strftime('%Y-%m-%d')}[/cyan]")


def main():
    parser = argparse.ArgumentParser(description='Train photo preference model on your curated (favorited) photos')
    parser.add_argument('--cutoff-date', type=str, default='2023-11-18',
                       help='Cutoff date - photos before this are training data (YYYY-MM-DD)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit number of training samples (for testing)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    
    args = parser.parse_args()
    
    cutoff = datetime.strptime(args.cutoff_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    try:
        train(
            cutoff_date=cutoff,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
