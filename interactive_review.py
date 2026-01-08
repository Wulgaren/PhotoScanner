#!/usr/bin/env python3
"""
Interactive review tool for photo deletion suggestions.
Shows photos side by side within series for easy comparison.
"""

import json
from pathlib import Path
from datetime import datetime
import subprocess
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()

OUTPUT_DIR = Path(__file__).parent / 'output'


def add_photos_to_album(uuids: list, album_name: str) -> bool:
    """Add photos to an album in Apple Photos by UUID. Creates album if it doesn't exist."""
    if not uuids:
        return False
    
    # Build the UUID list for AppleScript
    uuid_list = ', '.join(f'"{uuid}"' for uuid in uuids)
    
    script = f'''
    tell application "Photos"
        -- Create album if it doesn't exist
        if not (exists album "{album_name}") then
            make new album named "{album_name}"
        end if
        set theAlbum to album "{album_name}"
        
        -- Add each photo to the album
        set uuidList to {{{uuid_list}}}
        repeat with theUUID in uuidList
            try
                set theItem to media item id theUUID
                add {{theItem}} to theAlbum
            end try
        end repeat
    end tell
    '''
    
    try:
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]AppleScript error: {e.stderr}[/red]")
        return False


def open_in_preview(paths: list):
    """Open images in Preview app."""
    subprocess.run(['open', '-a', 'Preview'] + paths)


def open_in_finder(path: str):
    """Reveal file in Finder."""
    subprocess.run(['open', '-R', path])


def load_latest_results():
    """Load the most recent scan results."""
    result_files = sorted(OUTPUT_DIR.glob('scan_results_*.json'), reverse=True)
    if not result_files:
        return None
    
    with open(result_files[0]) as f:
        return json.load(f)


def interactive_review(results_file: str = None):
    """Interactive review of deletion suggestions."""
    
    console.print("\n[bold blue]📸 PhotoScanner - Interactive Review[/bold blue]\n")
    
    # Load results
    if results_file:
        with open(results_file) as f:
            results = json.load(f)
    else:
        results = load_latest_results()
        if not results:
            console.print("[red]No scan results found. Run scan_photos.py first.[/red]")
            return
    
    console.print(f"Loaded results from: {results.get('scan_date', 'unknown')}")
    console.print(f"Total photos: {results['total_photos']}")
    console.print(f"Suggested deletions: {results['suggested_deletions']}")
    
    # Group by series
    photos_by_series = {}
    for photo in results['photos']:
        series_id = photo['series_id']
        if series_id not in photos_by_series:
            photos_by_series[series_id] = []
        photos_by_series[series_id].append(photo)
    
    # Find series with deletion suggestions
    series_with_deletions = []
    for series_id, photos in photos_by_series.items():
        deletable = [p for p in photos if p['suggested_delete']]
        if deletable:
            series_with_deletions.append({
                'series_id': series_id,
                'photos': sorted(photos, key=lambda p: p['score'], reverse=True),
                'deletable_count': len(deletable)
            })
    
    # Sort by number of deletable photos
    series_with_deletions.sort(key=lambda s: s['deletable_count'], reverse=True)
    
    console.print(f"\nSeries with deletion suggestions: {len(series_with_deletions)}")
    
    # Review loop
    reviewed = []
    confirmed_delete = []
    confirmed_keep = []
    
    for i, series in enumerate(series_with_deletions):
        console.print("\n" + "=" * 60)
        console.print(f"[bold]Series {i+1}/{len(series_with_deletions)}[/bold] (ID: {series['series_id']})")
        console.print("=" * 60)
        
        photos = series['photos']
        
        # Show table of photos in series
        table = Table()
        table.add_column("#", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        table.add_column("Filename")
        
        for j, photo in enumerate(photos):
            score = photo['score']
            status = "[red]DELETE?[/red]" if photo['suggested_delete'] else "[green]KEEP[/green]"
            if j == 0:
                status += " [cyan](BEST)[/cyan]"
            
            table.add_row(
                str(j + 1),
                f"{score:.3f}",
                status,
                Path(photo['path']).name
            )
        
        console.print(table)
        
        # Options
        console.print("\n[dim]Commands:[/dim]")
        console.print("  [cyan]v[/cyan]   - View all photos in this series (opens Preview)")
        console.print("  [cyan]v #[/cyan] - View specific photo (e.g., v 2)")
        console.print("  [cyan]f[/cyan]   - Show in Finder")
        console.print("  [cyan]k #[/cyan] - Keep only photo # (delete others)")
        console.print("  [cyan]d #[/cyan] - Delete specific photos (e.g., d 2 3)")
        console.print("  [cyan]da[/cyan]  - Delete ALL photos in this series")
        console.print("  [cyan]y[/cyan]   - Confirm deletion suggestions")
        console.print("  [cyan]n[/cyan]   - Keep all photos in this series")
        console.print("  [cyan]s[/cyan]   - Skip this series")
        console.print("  [cyan]q[/cyan]   - Quit review")
        
        while True:
            choice = Prompt.ask("\nAction", default="s")
            choice = choice.strip().lower()
            
            if choice == 'v':
                paths = [p['path'] for p in photos]
                open_in_preview(paths)
            elif choice.startswith('v '):
                # View specific photo
                try:
                    idx = int(choice[2:].strip()) - 1
                    if 0 <= idx < len(photos):
                        open_in_preview([photos[idx]['path']])
                    else:
                        console.print(f"[red]Invalid photo number. Choose 1-{len(photos)}[/red]")
                except ValueError:
                    console.print("[red]Invalid format. Use: v 2[/red]")
            elif choice == 'f':
                open_in_finder(photos[0]['path'])
            elif choice.startswith('k '):
                # Keep only specific photo(s), delete others
                try:
                    keep_indices = [int(x.strip()) - 1 for x in choice[2:].split()]
                    for j, p in enumerate(photos):
                        if j in keep_indices:
                            confirmed_keep.append(p)
                        else:
                            confirmed_delete.append(p)
                    kept = len(keep_indices)
                    deleted = len(photos) - kept
                    console.print(f"[green]✓ Keeping {kept}, deleting {deleted}[/green]")
                    break
                except ValueError:
                    console.print("[red]Invalid format. Use: k 1 (or k 1 3 to keep multiple)[/red]")
            elif choice == 'da':
                # Delete ALL photos in this series
                for p in photos:
                    confirmed_delete.append(p)
                console.print(f"[red]✓ Deleting all {len(photos)} photos in this series[/red]")
                break
            elif choice.startswith('d '):
                # Delete specific photo(s), keep others
                try:
                    delete_indices = [int(x.strip()) - 1 for x in choice[2:].split()]
                    for j, p in enumerate(photos):
                        if j in delete_indices:
                            confirmed_delete.append(p)
                        else:
                            confirmed_keep.append(p)
                    deleted = len(delete_indices)
                    kept = len(photos) - deleted
                    console.print(f"[green]✓ Keeping {kept}, deleting {deleted}[/green]")
                    break
                except ValueError:
                    console.print("[red]Invalid format. Use: d 2 3[/red]")
            elif choice == 'y':
                for p in photos:
                    if p['suggested_delete']:
                        confirmed_delete.append(p)
                    else:
                        confirmed_keep.append(p)
                break
            elif choice == 'n':
                for p in photos:
                    confirmed_keep.append(p)
                break
            elif choice == 's':
                break
            elif choice == 'q':
                console.print("\n[yellow]Review interrupted.[/yellow]")
                break
            else:
                console.print("[dim]Unknown command. Use v, f, k #, d #, y, n, s, or q[/dim]")
        
        if choice == 'q':
            break
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Review Summary[/bold]")
    console.print("=" * 60)
    console.print(f"Confirmed for deletion: [red]{len(confirmed_delete)}[/red]")
    console.print(f"Confirmed to keep: [green]{len(confirmed_keep)}[/green]")
    
    if confirmed_delete:
        # Save confirmed deletions
        output_file = OUTPUT_DIR / f'confirmed_delete_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(output_file, 'w') as f:
            for p in confirmed_delete:
                f.write(f"{p['path']}\n")
        
        console.print(f"\n[green]✓[/green] Confirmed deletions saved to: {output_file}")
        
        if Confirm.ask("\nDo you want to add these photos to 'To Delete' album in Photos?"):
            console.print("\n[yellow]Adding to 'To Delete' album in Apple Photos...[/yellow]")
            
            # Collect all UUIDs
            uuids = [p.get('uuid') for p in confirmed_delete if p.get('uuid')]
            
            if not uuids:
                console.print("[red]No valid UUIDs found[/red]")
                return
            
            success = add_photos_to_album(uuids, "To Delete")
            
            if success:
                console.print(f"\n[green]✓[/green] Added {len(uuids)} photos to 'To Delete' album")
                console.print("[dim]Open Photos app and review the 'To Delete' album, then select all and delete[/dim]")
            else:
                console.print("[red]Failed to add photos to album[/red]")


def main():
    parser = argparse.ArgumentParser(description='Interactive review of deletion suggestions')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to scan results JSON file')
    
    args = parser.parse_args()
    
    try:
        interactive_review(results_file=args.results)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
