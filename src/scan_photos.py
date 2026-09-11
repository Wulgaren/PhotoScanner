#!/usr/bin/env python3
"""
Scan new photos and get deletion suggestions.
"""

import osxphotos
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm
import pickle
import json

from photo_scanner.feature_extractor import (
    DEFAULT_BACKBONE,
    AestheticScorer,
    FeatureExtractor,
)
from photo_scanner.paths import CACHE_DIR, OUTPUT_DIR, ensure_data_dirs
from photo_scanner.thumbs import (
    EMBED_MAX_EDGE,
    build_uuid_to_path_index,
    resolve_scorable_image,
)

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Video extensions to skip
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
from photo_scanner.series_detector import (
    PhotoInfo, PhotoSeries, SeriesDetector, get_deletion_suggestions
)

console = Console()

ensure_data_dirs()


def _is_video_path(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def scan(after_date: datetime, score_threshold: float = 0.3,
         batch_size: int = 32, limit: int = None,
         model_name: str = DEFAULT_BACKBONE):
    """
    Scan photos after the given date and suggest deletions.
    
    Args:
        after_date: Only scan photos after this date
        score_threshold: Suggest deletion for photos below this score
        batch_size: Batch size for feature extraction
        limit: Limit number of photos to scan (for testing)
        model_name: timm backbone (must match the trained preference model)
    """
    console.print("\n[bold blue]📸 PhotoScanner - Photo Analysis[/bold blue]\n")
    
    # Load trained model
    model_path = CACHE_DIR / 'aesthetic_model.pkl'
    if not model_path.exists():
        console.print("[red]Error: Model not found. Run ./photoscanner.sh (Train) first.[/red]")
        return
    
    # Initialize (max_edge matches training / review thumbs)
    console.print("Loading model...")
    extractor = FeatureExtractor(model_name=model_name, max_edge=EMBED_MAX_EDGE)
    scorer = AestheticScorer(extractor)
    scorer.load(model_path)
    
    # Connect to Photos
    console.print("Connecting to Apple Photos...")
    photosdb = osxphotos.PhotosDB()
    
    # Get photos to scan — include ismissing (iCloud Optimize); resolve thumbs later
    console.print(f"Finding favorited photos after {after_date.strftime('%Y-%m-%d')}...")
    
    all_photos = photosdb.photos()
    photos_to_scan = [
        p for p in all_photos
        if p.favorite 
        and p.date and p.date >= after_date
        and not p.screenshot
    ]
    missing_marked = sum(1 for p in photos_to_scan if getattr(p, "ismissing", False))
    
    console.print(f"[green]✓[/green] Found {len(photos_to_scan):,} favorited photos to consider")
    if missing_marked:
        console.print(
            f"[dim]{missing_marked:,} marked ismissing by Photos "
            f"(will use review thumbs when available)[/dim]"
        )
    
    if limit:
        photos_to_scan = photos_to_scan[:limit]
        console.print(f"  (Limited to {limit} for testing)")
    
    # Resolve readable images: original/edited, Photos derivatives, else review thumbs
    console.print("\nResolving file paths...")
    uuid_to_path = build_uuid_to_path_index()
    photo_data = []
    skipped_videos = 0
    used_derivatives = 0
    used_thumbs = 0
    unresolved = 0
    for photo in tqdm(photos_to_scan, desc="Getting paths"):
        # Skip obvious videos when the Photos path is a video file
        if _is_video_path(photo.path) and not (
            photo.path_edited and not _is_video_path(photo.path_edited)
        ):
            skipped_videos += 1
            continue

        try:
            derivs = list(photo.path_derivatives or [])
        except Exception:
            derivs = []

        resolved = resolve_scorable_image(
            photo.uuid,
            path=photo.path if not _is_video_path(photo.path) else None,
            path_edited=photo.path_edited if not _is_video_path(photo.path_edited) else None,
            derivatives=derivs,
            uuid_to_path=uuid_to_path,
        )
        if resolved is None:
            unresolved += 1
            continue

        score_path, library_path, source = resolved
        if source in ("original", "edited") and _is_video_path(score_path):
            skipped_videos += 1
            continue
        if source == "derivative":
            used_derivatives += 1
        elif source == "thumb":
            used_thumbs += 1

        photo_data.append({
            'uuid': photo.uuid,
            'path': library_path,       # for scan JSON / review_gui thumb keys
            'score_path': score_path,   # readable file for CLIP + phash
            'date': photo.date,
            'filename': photo.filename,
            'source': source,
        })
    
    if skipped_videos > 0:
        console.print(f"[dim]Skipped {skipped_videos} videos[/dim]")
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
    if unresolved:
        console.print(
            f"[yellow]Skipped {unresolved:,} photos with no local original, "
            f"derivative, or real thumb[/yellow]"
        )
    
    console.print(f"[green]✓[/green] Found {len(photo_data):,} scorable photos")
    
    if not photo_data:
        console.print("[yellow]No photos to scan.[/yellow]")
        console.print(
            "[dim]Tip: Photos needs local derivatives (or Review thumbs) for "
            "cloud-only favorites[/dim]"
        )
        return
    
    # Extract features from readable score_path (thumb or original)
    console.print("\n[bold]Extracting visual features...[/bold]")
    features_dict = extractor.extract_batch(
        [p['score_path'] for p in photo_data],
        batch_size=batch_size
    )
    
    # Create PhotoInfo objects (path = library path for JSON / review)
    photo_infos = []
    score_path_by_lib = {}
    for p in photo_data:
        feats = features_dict.get(p['score_path'])
        if feats is None:
            continue
        info = PhotoInfo(
            path=p['path'],
            uuid=p['uuid'],
            date=p['date'],
            features=feats,
        )
        photo_infos.append(info)
        score_path_by_lib[id(info)] = p['score_path']
    
    console.print(f"[green]✓[/green] Extracted features for {len(photo_infos):,} photos")
    
    # Score photos
    console.print("\n[bold]Scoring photos...[/bold]")
    all_features = np.array([p.features for p in photo_infos])
    scores = scorer.score(all_features)
    
    for photo, score in zip(photo_infos, scores):
        photo.score = float(score)
    
    # Perceptual hashes: hash the readable file (thumb OK; None on failure is fine)
    console.print("\n[bold]Computing perceptual hashes...[/bold]")
    detector = SeriesDetector(
        time_threshold_seconds=60,
        similarity_threshold=0.85,
        phash_threshold=10
    )
    
    for photo in tqdm(photo_infos, desc="Computing hashes"):
        readable = score_path_by_lib.get(id(photo), photo.path)
        photo.phash = detector.compute_phash(readable)
    
    # Detect series
    console.print("\n[bold]Detecting photo series...[/bold]")
    series_list = detector.detect_series(photo_infos)
    series_list = detector.rank_series(series_list)
    
    # Get deletion suggestions
    deletable = get_deletion_suggestions(series_list, score_threshold=score_threshold)
    
    # Statistics
    console.print("\n" + "=" * 60)
    console.print("[bold]📊 Analysis Results[/bold]")
    console.print("=" * 60)
    
    score_distribution = {
        'excellent (0.8-1.0)': len([p for p in photo_infos if p.score >= 0.8]),
        'good (0.6-0.8)': len([p for p in photo_infos if 0.6 <= p.score < 0.8]),
        'average (0.4-0.6)': len([p for p in photo_infos if 0.4 <= p.score < 0.6]),
        'below avg (0.2-0.4)': len([p for p in photo_infos if 0.2 <= p.score < 0.4]),
        'poor (0-0.2)': len([p for p in photo_infos if p.score < 0.2]),
    }
    
    table = Table(title="Score Distribution")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    
    for category, count in score_distribution.items():
        pct = count / len(photo_infos) * 100 if photo_infos else 0
        table.add_row(category, str(count), f"{pct:.1f}%")
    
    console.print(table)
    
    console.print(f"\n[bold]Series Analysis:[/bold]")
    console.print(f"  • Total photos: {len(photo_infos):,}")
    console.print(f"  • Photo series detected: {len(series_list):,}")
    console.print(f"  • Single photos (no series): {len([s for s in series_list if len(s.photos) == 1]):,}")
    console.print(f"  • Photos in series of 2+: {sum(len(s.photos) for s in series_list if len(s.photos) > 1):,}")
    
    console.print(f"\n[bold]Deletion Suggestions:[/bold]")
    console.print(f"  • Suggested for deletion: [red]{len(deletable):,}[/red] photos")
    console.print(f"  • Safe to keep: [green]{len(photo_infos) - len(deletable):,}[/green] photos")
    
    if deletable:
        console.print(f"\n[bold yellow]⚠️  Photos suggested for deletion (score < {score_threshold}):[/bold yellow]")
        
        # Show worst 20
        for i, photo in enumerate(deletable[:20]):
            console.print(f"  {i+1}. Score: {photo.score:.3f} | {Path(photo.path).name}")
        
        if len(deletable) > 20:
            console.print(f"  ... and {len(deletable) - 20} more")
    
    # Save detailed results — path is library path when known (review thumb keys)
    results = {
        'scan_date': datetime.now().isoformat(),
        'after_date': after_date.isoformat(),
        'total_photos': len(photo_infos),
        'total_series': len(series_list),
        'score_threshold': score_threshold,
        'suggested_deletions': len(deletable),
        'photos': [
            {
                'uuid': p.uuid,
                'path': p.path,
                'date': p.date.isoformat() if p.date else None,
                'score': p.score,
                'series_id': p.series_id,
                'suggested_delete': p in deletable,
            }
            for p in photo_infos
        ],
        'deletion_suggestions': [
            {
                'uuid': p.uuid,
                'path': p.path,
                'score': p.score,
                'series_id': p.series_id,
            }
            for p in deletable
        ]
    }
    
    output_file = OUTPUT_DIR / f'scan_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n[green]✓[/green] Detailed results saved to: {output_file}")
    
    # Create a simple text list of deletable photos
    if deletable:
        delete_list_file = OUTPUT_DIR / f'delete_suggestions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(delete_list_file, 'w') as f:
            f.write(f"# Photos suggested for deletion\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Threshold: {score_threshold}\n")
            f.write(f"# Total: {len(deletable)} photos\n\n")
            for p in deletable:
                f.write(f"{p.score:.3f}\t{p.path}\n")
        
        console.print(f"[green]✓[/green] Delete suggestions saved to: {delete_list_file}")
    
    console.print("\n[bold green]✓ Scan complete![/bold green]")
    console.print("\n[dim]Note: Always manually review before deleting. The AI suggestions are just hints![/dim]")


def main():
    parser = argparse.ArgumentParser(description='Scan photos for deletion suggestions')
    parser.add_argument('--after', type=str, default='2023-11-18',
                       help='Scan photos after this date (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Score threshold for deletion suggestions (0-1)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of photos to scan (for testing)')
    parser.add_argument('--model', type=str, default=DEFAULT_BACKBONE,
                       help=f'timm backbone (must match training; default: {DEFAULT_BACKBONE})')
    
    args = parser.parse_args()
    
    after = datetime.strptime(args.after, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    try:
        scan(
            after_date=after,
            score_threshold=args.threshold,
            batch_size=args.batch_size,
            limit=args.limit,
            model_name=args.model,
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
