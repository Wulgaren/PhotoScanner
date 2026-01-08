#!/usr/bin/env python3
"""
Move photos below a score threshold to a 'To Delete' album in Apple Photos.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from rich.console import Console
from rich.progress import track

console = Console()

OUTPUT_DIR = Path(__file__).parent / 'output'
CACHE_DIR = Path(__file__).parent / '.cache'
FEEDBACK_FILE = CACHE_DIR / 'feedback_history.json'


def record_added_photos(uuids: list, scan_file: str):
    """Record that these photos were added to To Delete album for feedback tracking."""
    # Load existing history
    history = {'added_to_album': {}, 'rescued': [], 'confirmed_delete': []}
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE) as f:
            history = json.load(f)
    
    timestamp = datetime.now().isoformat()
    
    for uuid in uuids:
        if uuid not in history['added_to_album']:
            history['added_to_album'][uuid] = {
                'added_date': timestamp,
                'scan_file': scan_file,
            }
    
    CACHE_DIR.mkdir(exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    console.print(f"[dim]Recorded {len(uuids)} photos for feedback tracking[/dim]")


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


def move_to_album(threshold: float = 0.8, album_name: str = "To Delete", 
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
            console.print("[red]No scan results found. Run scan_photos.py first.[/red]")
            return
    
    console.print(f"Loaded: {source}")
    console.print(f"Total photos in scan: {results['total_photos']}")
    console.print(f"Threshold: {threshold}")
    
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
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Score threshold - photos below this are moved (default: 0.8)')
    parser.add_argument('--album', type=str, default='To Delete',
                       help='Album name to move photos to (default: "To Delete")')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to scan results JSON file (default: latest)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be moved without actually moving')
    
    args = parser.parse_args()
    
    try:
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
