#!/usr/bin/env python3
"""
Move photos below a score threshold to a 'To Delete' album in Apple Photos.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console

from learn_from_feedback import parse_uuid_text, record_added_photos
from photo_scanner.feature_extractor import recommended_threshold_from_model
from photo_scanner.paths import OUTPUT_DIR, ensure_data_dirs

console = Console()

ensure_data_dirs()


def recommended_threshold() -> float | None:
    """Train's calibrated flag threshold, if the saved model has one."""
    return recommended_threshold_from_model()


def add_photos_to_album(uuids: list, album_name: str, batch_size: int = 50) -> int:
    """
    Add photos to an album in Apple Photos by UUID.
    Creates album if it doesn't exist. Processes in batches.
    Returns number of successfully added photos.
    """
    if not uuids:
        return 0
    
    success_count = 0
    
    # Process in batches to avoid AppleScript limits
    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i + batch_size]
        uuid_list = ', '.join(f'"{uuid}"' for uuid in batch)
        
        script = f'''
        tell application "Photos"
            -- Create album if it doesn't exist
            if not (exists album "{album_name}") then
                make new album named "{album_name}"
            end if
            set theAlbum to album "{album_name}"
            
            -- Add each photo to the album
            set uuidList to {{{uuid_list}}}
            set addedCount to 0
            repeat with theUUID in uuidList
                try
                    set theItem to media item id theUUID
                    add {{theItem}} to theAlbum
                    set addedCount to addedCount + 1
                end try
            end repeat
            return addedCount
        end tell
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', script], 
                check=True, capture_output=True, text=True
            )
            # Parse the count from AppleScript output
            try:
                success_count += int(result.stdout.strip())
            except ValueError:
                success_count += len(batch)  # Assume all succeeded if can't parse
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Batch error: {e.stderr.strip()}[/red]")
    
    return success_count


def load_latest_results():
    """Load the most recent scan results."""
    result_files = sorted(OUTPUT_DIR.glob('scan_results_*.json'), reverse=True)
    if not result_files:
        return None, None
    
    with open(result_files[0]) as f:
        return json.load(f), result_files[0]


def resolve_threshold(results: dict, threshold: float | None) -> tuple[float, str]:
    """Pick the album cutoff: explicit arg > scan's own > model calibration > 0.3."""
    if threshold is not None:
        return threshold, "explicit"
    scan_threshold = results.get("score_threshold")
    if isinstance(scan_threshold, (int, float)):
        return float(scan_threshold), "the scan's own threshold"
    auto = recommended_threshold()
    if auto is not None:
        return float(auto), "auto-calibrated from your rescue/delete feedback"
    return 0.3, "default"


def move_uuids_to_album(
    uuids: list[str],
    album_name: str = "To Delete",
    source: str = "phone review",
    dry_run: bool = False,
) -> int:
    """Add a pasted UUID list to an album and record it as feedback."""
    console.print("\n[bold blue]📸 PhotoScanner - Add IDs to Album[/bold blue]\n")
    console.print(f"IDs: [cyan]{len(uuids)}[/cyan]")
    console.print(f"Album: [cyan]{album_name}[/cyan]")
    console.print(f"Source: {source}")

    if not uuids:
        console.print("[yellow]No IDs found.[/yellow]")
        return 0

    if dry_run:
        console.print("\n[yellow]Dry run — nothing added.[/yellow]")
        for uuid in uuids[:20]:
            console.print(f"  {uuid}")
        if len(uuids) > 20:
            console.print(f"  ... and {len(uuids) - 20} more")
        return 0

    console.print(f"\n[yellow]Adding to '{album_name}'...[/yellow]")
    added = add_photos_to_album(uuids, album_name)

    console.print(f"\n[green]✓[/green] Added {added}/{len(uuids)} to '{album_name}'")
    if added < len(uuids):
        console.print(
            f"[yellow]{len(uuids) - added} were already gone or not addable.[/yellow]"
        )
    if added:
        record_added_photos(uuids, source)
        console.print(
            f"[dim]Open Photos → '{album_name}' → Select All → Delete[/dim]"
        )
    return added


def move_to_album(threshold: float = None, album_name: str = "To Delete", 
                  results_file: str = None, dry_run: bool = False):
    """Move all photos below threshold to an album."""
    
    console.print("\n[bold blue]📸 PhotoScanner - Move to Album[/bold blue]\n")
    
    # Load results
    if results_file:
        with open(results_file) as f:
            results = json.load(f)
        source = results_file
    else:
        results, source = load_latest_results()
        if not results:
            console.print("[red]No scan results found. Run ./photoscanner.sh (Scan) first.[/red]")
            return
    
    threshold, source_label = resolve_threshold(results, threshold)
    console.print(f"Loaded: {source}")
    console.print(f"Total photos in scan: {results['total_photos']}")
    console.print(f"Threshold: {threshold} ({source_label})")
    
    # Find photos below threshold
    photos_to_move = [
        p for p in results['photos'] 
        if p['score'] < threshold and p.get('uuid')
    ]
    
    console.print(f"\nPhotos below {threshold}: [yellow]{len(photos_to_move)}[/yellow]")
    
    if not photos_to_move:
        console.print("[green]No photos to move![/green]")
        return
    
    # Show score distribution of photos to move
    console.print("\n[dim]Score distribution of photos to move:[/dim]")
    ranges = [
        (0.0, 0.2, "0.0-0.2"),
        (0.2, 0.4, "0.2-0.4"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
    ]
    for low, high, label in ranges:
        count = len([p for p in photos_to_move if low <= p['score'] < high])
        if count > 0:
            console.print(f"  {label}: {count}")
    
    if dry_run:
        console.print("\n[yellow]Dry run - not moving any photos[/yellow]")
        console.print("\nWould move these photos:")
        for p in photos_to_move[:20]:
            console.print(f"  {p['score']:.3f} - {Path(p['path']).name}")
        if len(photos_to_move) > 20:
            console.print(f"  ... and {len(photos_to_move) - 20} more")
        return
    
    # Confirm
    console.print(f"\n[bold]Will add {len(photos_to_move)} photos to '[cyan]{album_name}[/cyan]' album[/bold]")
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    # Move photos
    console.print(f"\n[yellow]Adding photos to '{album_name}' album...[/yellow]")
    
    uuids = [p['uuid'] for p in photos_to_move]
    success_count = add_photos_to_album(uuids, album_name)
    
    console.print(f"\n[green]✓[/green] Added {success_count}/{len(photos_to_move)} photos to '{album_name}'")
    
    # Record for feedback learning
    record_added_photos(uuids, str(source))
    console.print(f"\n[dim]Open Photos app → '{album_name}' album → Select All → Delete[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description='Move photos below score threshold to an album in Apple Photos'
    )
    parser.add_argument('--threshold', type=float, default=None,
                       help='Score threshold - photos below this are moved '
                            '(default: the scan\'s own threshold, else Train\'s '
                            'auto-calibrated value)')
    parser.add_argument('--album', type=str, default='To Delete',
                       help='Album name to move photos to (default: "To Delete")')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to scan results JSON file (default: latest)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be moved without actually moving')
    parser.add_argument('--uuids', type=str, default=None,
                       help='Add specific photo UUIDs (comma/space/newline separated)')
    parser.add_argument('--uuids-file', type=str, default=None,
                       help='Read photo UUIDs from a file ("-" for stdin)')

    args = parser.parse_args()

    try:
        if args.uuids or args.uuids_file:
            if args.uuids_file == "-":
                text = sys.stdin.read()
                source = "stdin"
            elif args.uuids_file:
                path = Path(args.uuids_file).expanduser()
                text = path.read_text()
                source = path.name
            else:
                text = args.uuids or ""
                source = "command line"
            move_uuids_to_album(
                parse_uuid_text(text),
                album_name=args.album,
                source=f"phone review: {source}",
                dry_run=args.dry_run,
            )
        else:
            move_to_album(
                threshold=args.threshold,
                album_name=args.album,
                results_file=args.results,
                dry_run=args.dry_run
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
