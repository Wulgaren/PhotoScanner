#!/usr/bin/env python3
"""
Train the photo preference model on your curated photos.
"""

import osxphotos
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
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

console = Console()

# Directory to store models and cache
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)


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
    Train the model on curated photos (favorited photos only).
    
    Args:
        cutoff_date: Photos before this date are considered training data
        sample_size: Limit training samples (None = use all)
        batch_size: Batch size for feature extraction
    """
    console.print("\n[bold blue]📸 PhotoScanner - Model Training[/bold blue]\n")
    
    # Connect to Photos
    console.print("Connecting to Apple Photos...")
    photosdb = osxphotos.PhotosDB()
    console.print(f"[green]✓[/green] Connected\n")
    
    # Get training photos (favorited before cutoff date)
    console.print(f"Finding training photos (favorited before {cutoff_date.strftime('%Y-%m-%d')})...")
    
    all_photos = photosdb.photos()
    
    # Only use favorited photos as training examples
    good_photos = [
        p for p in all_photos 
        if p.favorite and p.date and p.date < cutoff_date
        and not p.screenshot  # Exclude screenshots
        and not p.ismissing
    ]
    
    console.print(f"[green]✓[/green] Found {len(good_photos):,} curated photos for training")
    
    # Sample if needed
    if sample_size:
        import random
        if len(good_photos) > sample_size:
            good_photos = random.sample(good_photos, sample_size)
            console.print(f"  Sampled {sample_size} photos")
    
    # Get file paths
    console.print("\nResolving file paths...")
    good_paths = get_photo_paths(good_photos, "Photos")
    
    console.print(f"\nReady to extract features from [green]{len(good_paths):,}[/green] photos")
    
    # Load existing cache for incremental extraction
    existing_cache = load_feature_cache()
    cached_features = {}
    if existing_cache and 'feature_cache' in existing_cache:
        cached_features = existing_cache.get('feature_cache', {})
        console.print(f"[dim]Loaded {len(cached_features):,} cached features[/dim]")
    
    # Determine which photos need feature extraction
    paths_to_extract = [p[1] for p in good_paths if p[1] not in cached_features]
    already_cached = len(good_paths) - len(paths_to_extract)
    
    if already_cached > 0:
        console.print(f"[green]✓[/green] {already_cached:,} photos already cached, {len(paths_to_extract):,} to extract")
    
    # Initialize feature extractor
    console.print("\n[bold]Initializing neural network...[/bold]")
    extractor = FeatureExtractor(model_name='efficientnet_b0')
    
    # Extract features with incremental saving
    if paths_to_extract:
        console.print("\n[bold]Extracting features...[/bold]")
        console.print("[dim]Progress is saved every 100 photos - safe to interrupt[/dim]\n")
        
        save_interval = 100
        for i in tqdm(range(0, len(paths_to_extract), batch_size), desc="Extracting features"):
            batch_paths = paths_to_extract[i:i + batch_size]
            batch_features = extractor.extract_batch(batch_paths, batch_size=batch_size, show_progress=False)
            
            # Add to cache
            cached_features.update(batch_features)
            
            # Save incrementally
            if (i // batch_size + 1) % (save_interval // batch_size) == 0 or i + batch_size >= len(paths_to_extract):
                save_feature_cache({
                    'feature_cache': cached_features,
                    'cutoff_date': cutoff_date,
                })
        
        console.print(f"[green]✓[/green] Extracted features for {len(paths_to_extract):,} photos")
    
    # Build final feature array
    good_features = np.array([cached_features[p[1]] for p in good_paths if p[1] in cached_features])
    
    console.print(f"\n[green]✓[/green] Total features: {len(good_features):,}")
    
    # Save final feature cache
    console.print("\nSaving feature cache...")
    save_feature_cache({
        'good_paths': good_paths,
        'good_features': good_features,
        'feature_cache': cached_features,
        'cutoff_date': cutoff_date,
    })
    
    # Train the model (one-class, since we only have good examples)
    console.print("\n[bold]Training aesthetic scorer...[/bold]")
    scorer = AestheticScorer(extractor)
    scorer.train(good_features, None)  # No negative examples
    
    # Save model
    model_path = CACHE_DIR / 'aesthetic_model.pkl'
    scorer.save(model_path)
    
    # Quick validation
    console.print("\n[bold]Validation:[/bold]")
    good_scores = scorer.score(good_features)
    console.print(f"  Training photos - Mean score: {good_scores.mean():.3f}, Std: {good_scores.std():.3f}")
    
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
