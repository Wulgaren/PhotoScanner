#!/usr/bin/env python3
"""
Learn from user feedback by checking which photos were rescued from the 'To Delete' album.
Photos that were removed from the album are considered "good" and added to training.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
from rich.console import Console
import pickle

console = Console()

OUTPUT_DIR = Path(__file__).parent / 'output'
CACHE_DIR = Path(__file__).parent / '.cache'
FEEDBACK_FILE = CACHE_DIR / 'feedback_history.json'


def get_album_photo_uuids(album_name: str) -> set:
    """Get all photo UUIDs currently in an album."""
    script = f'''
    tell application "Photos"
        if not (exists album "{album_name}") then
            return ""
        end if
        set theAlbum to album "{album_name}"
        set uuidList to {{}}
        repeat with theItem in media items of theAlbum
            set end of uuidList to id of theItem
        end repeat
        set AppleScript's text item delimiters to ","
        return uuidList as text
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            check=True, capture_output=True, text=True
        )
        uuid_text = result.stdout.strip()
        if not uuid_text:
            return set()
        return set(uuid_text.split(','))
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error reading album: {e.stderr}[/red]")
        return set()


def check_photos_exist(uuids: list) -> tuple[set, set]:
    """
    Check which photos still exist in the library.
    Returns (existing_uuids, deleted_uuids)
    """
    if not uuids:
        return set(), set()
    
    existing = set()
    deleted = set()
    
    # Check in batches to avoid AppleScript limits
    batch_size = 50
    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i + batch_size]
        uuid_list = ', '.join(f'"{uuid}"' for uuid in batch)
        
        script = f'''
        tell application "Photos"
            set uuidList to {{{uuid_list}}}
            set existingList to {{}}
            repeat with theUUID in uuidList
                try
                    set theItem to media item id theUUID
                    set end of existingList to theUUID as text
                end try
            end repeat
            set AppleScript's text item delimiters to ","
            return existingList as text
        end tell
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                check=True, capture_output=True, text=True
            )
            existing_text = result.stdout.strip()
            if existing_text:
                batch_existing = set(existing_text.split(','))
                existing.update(batch_existing)
        except subprocess.CalledProcessError:
            pass  # If error, assume none exist
    
    deleted = set(uuids) - existing
    return existing, deleted


def load_feedback_history() -> dict:
    """Load history of photos we've added to the To Delete album."""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return {'added_to_album': {}, 'rescued': [], 'confirmed_delete': []}


def save_feedback_history(history: dict):
    """Save feedback history."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def record_added_photos(uuids: list, scan_file: str):
    """Record that these photos were added to To Delete album."""
    history = load_feedback_history()
    timestamp = datetime.now().isoformat()
    
    for uuid in uuids:
        if uuid not in history['added_to_album']:
            history['added_to_album'][uuid] = {
                'added_date': timestamp,
                'scan_file': scan_file,
            }
    
    save_feedback_history(history)
    console.print(f"[dim]Recorded {len(uuids)} photos in feedback history[/dim]")


def check_feedback(album_name: str = "To Delete"):
    """
    Check which photos were rescued from the album.
    Rescued photos = user decided they're good.
    """
    console.print("\n[bold blue]📸 PhotoScanner - Learn from Feedback[/bold blue]\n")
    
    # Load history of what we added
    history = load_feedback_history()
    added_photos = history.get('added_to_album', {})
    
    if not added_photos:
        console.print("[yellow]No feedback history found.[/yellow]")
        console.print("[dim]Run move_to_album.py first, then remove good photos from the album.[/dim]")
        return
    
    console.print(f"Photos previously added to '{album_name}': {len(added_photos)}")
    
    # Get current photos in album
    console.print(f"Checking current '{album_name}' album...")
    current_in_album = get_album_photo_uuids(album_name)
    console.print(f"Photos currently in album: {len(current_in_album)}")
    
    # Find photos no longer in album
    added_uuids = set(added_photos.keys())
    not_in_album = added_uuids - current_in_album
    still_in_album = added_uuids & current_in_album
    
    # Check which of those still exist in the library (rescued) vs deleted
    previously_rescued = set(history.get('rescued', []))
    previously_deleted = set(history.get('confirmed_delete', []))
    
    # Only check photos we haven't processed before
    to_check = not_in_album - previously_rescued - previously_deleted
    
    if to_check:
        console.print(f"Checking if {len(to_check)} photos still exist in library...")
        existing, deleted = check_photos_exist(list(to_check))
        new_rescued = existing  # Still exists but not in album = rescued
        new_deleted = deleted   # No longer exists = was deleted
    else:
        new_rescued = set()
        new_deleted = set()
    
    # Photos still in album are also bad (user is done sorting)
    new_bad = still_in_album - previously_deleted
    
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  • Rescued (you kept them): [green]{len(new_rescued)}[/green]")
    console.print(f"  • Still in album (bad): [red]{len(new_bad)}[/red]")
    console.print(f"  • Already deleted (bad): [red]{len(new_deleted)}[/red]")
    console.print(f"  • Previously processed: {len(previously_rescued) + len(previously_deleted)}")
    
    # Combine all bad photos (still in album + deleted)
    all_new_bad = new_bad | new_deleted
    
    if not new_rescued and not all_new_bad:
        console.print("\n[yellow]No new feedback to process.[/yellow]")
        return
    
    # Show summary
    if new_rescued:
        console.print(f"\n[green]Rescued photos ({len(new_rescued)}) - will be positive examples[/green]")
        for uuid in list(new_rescued)[:5]:
            console.print(f"  [dim]{uuid}[/dim]")
        if len(new_rescued) > 5:
            console.print(f"  [dim]... and {len(new_rescued) - 5} more[/dim]")
    
    if all_new_bad:
        console.print(f"\n[red]Bad photos ({len(all_new_bad)}) - will be negative examples[/red]")
        for uuid in list(all_new_bad)[:5]:
            console.print(f"  [dim]{uuid}[/dim]")
        if len(all_new_bad) > 5:
            console.print(f"  [dim]... and {len(all_new_bad) - 5} more[/dim]")
    
    # Confirm learning
    console.print(f"\n[bold]Training update:[/bold]")
    if new_rescued:
        console.print(f"  • [green]+{len(new_rescued)} positive examples[/green] (rescued)")
    if all_new_bad:
        console.print(f"  • [red]+{len(all_new_bad)} negative examples[/red] (bad)")
    
    response = input("\nApply feedback? [y/N]: ").strip().lower()
    if response != 'y':
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    # Update history
    history['rescued'].extend(list(new_rescued))
    history['confirmed_delete'].extend(list(all_new_bad))
    save_feedback_history(history)
    
    # Save for training
    if new_rescued:
        add_rescued_to_training(new_rescued, added_photos)
        console.print(f"[green]✓[/green] Added {len(new_rescued)} rescued photos as positive examples")
    
    if all_new_bad:
        add_bad_to_training(all_new_bad, added_photos)
        console.print(f"[red]✓[/red] Added {len(all_new_bad)} bad photos as negative examples")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Run [cyan]python train_model.py[/cyan] to retrain with feedback")
    console.print("  2. Run [cyan]python scan_photos.py[/cyan] to rescan with improved model")


def add_rescued_to_training(rescued_uuids: set, added_photos: dict):
    """Add rescued photos to the positive training examples."""
    
    # Save rescued UUIDs for training to pick up
    rescued_file = CACHE_DIR / 'rescued_photos.json'
    existing_rescued = []
    if rescued_file.exists():
        with open(rescued_file) as f:
            existing_rescued = json.load(f)
    
    # Add new rescued UUIDs
    for uuid in rescued_uuids:
        if uuid not in existing_rescued:
            existing_rescued.append(uuid)
    
    with open(rescued_file, 'w') as f:
        json.dump(existing_rescued, f, indent=2)
    
    console.print(f"[dim]Saved {len(existing_rescued)} total rescued photos for training[/dim]")


def add_bad_to_training(bad_uuids: set, added_photos: dict):
    """Add bad photos (from feedback) as negative training examples."""
    
    # Save bad UUIDs for training to pick up
    bad_file = CACHE_DIR / 'feedback_bad_photos.json'
    existing_bad = []
    if bad_file.exists():
        with open(bad_file) as f:
            existing_bad = json.load(f)
    
    # Add new bad UUIDs
    for uuid in bad_uuids:
        if uuid not in existing_bad:
            existing_bad.append(uuid)
    
    with open(bad_file, 'w') as f:
        json.dump(existing_bad, f, indent=2)
    
    console.print(f"[dim]Saved {len(existing_bad)} total bad photos for training[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description='Learn from feedback - rescued photos become positive training examples'
    )
    parser.add_argument('--album', type=str, default='To Delete',
                       help='Album name to check (default: "To Delete")')
    
    args = parser.parse_args()
    
    try:
        check_feedback(album_name=args.album)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
