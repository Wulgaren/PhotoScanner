#!/usr/bin/env python3
"""
Analyze Apple Photos library to understand the data structure.
This script helps understand what photos are available for training.
"""

import osxphotos
from datetime import datetime, timezone
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import argparse

console = Console()

def analyze_library(cutoff_date: datetime = None):
    """Analyze the Apple Photos library."""
    
    console.print("\n[bold blue]📸 PhotoScanner - Library Analysis[/bold blue]\n")
    
    # Connect to Photos library
    console.print("Connecting to Apple Photos library...")
    photosdb = osxphotos.PhotosDB()
    
    console.print(f"[green]✓[/green] Connected to library: {photosdb.library_path}\n")
    
    # Get all photos
    all_photos = photosdb.photos()
    console.print(f"Total photos in library: [bold]{len(all_photos):,}[/bold]")
    
    # Analyze favorites
    favorites = [p for p in all_photos if p.favorite]
    non_favorites = [p for p in all_photos if not p.favorite]
    
    console.print(f"Favorited photos: [bold green]{len(favorites):,}[/bold green]")
    console.print(f"Non-favorited photos: [bold yellow]{len(non_favorites):,}[/bold yellow]")
    
    # If cutoff date provided, split by date
    if cutoff_date:
        console.print(f"\n[bold]Analysis with cutoff date: {cutoff_date.strftime('%Y-%m-%d')}[/bold]\n")
        
        fav_before = [p for p in favorites if p.date and p.date < cutoff_date]
        fav_after = [p for p in favorites if p.date and p.date >= cutoff_date]
        
        non_fav_before = [p for p in non_favorites if p.date and p.date < cutoff_date]
        non_fav_after = [p for p in non_favorites if p.date and p.date >= cutoff_date]
        
        # Create summary table
        table = Table(title="Photo Distribution")
        table.add_column("Category", style="cyan")
        table.add_column(f"Before {cutoff_date.strftime('%Y-%m-%d')}", justify="right")
        table.add_column(f"After {cutoff_date.strftime('%Y-%m-%d')}", justify="right")
        
        table.add_row("Favorited", f"{len(fav_before):,}", f"{len(fav_after):,}")
        table.add_row("Non-favorited", f"{len(non_fav_before):,}", f"{len(non_fav_after):,}")
        table.add_row("Total", f"{len(fav_before) + len(non_fav_before):,}", 
                     f"{len(fav_after) + len(non_fav_after):,}")
        
        console.print(table)
        
        console.print("\n[bold]Training Data Summary:[/bold]")
        console.print(f"  • [green]Positive examples (curated favorites before cutoff):[/green] {len(fav_before):,}")
        console.print(f"  • [yellow]Potential negatives (non-favorites):[/yellow] {len(non_favorites):,}")
        console.print(f"  • [blue]Photos to classify (favorites after cutoff):[/blue] {len(fav_after):,}")
        
        # Check if we have enough negative examples
        if len(non_favorites) < 100:
            console.print("\n[yellow]⚠️  Few non-favorited photos available.[/yellow]")
            console.print("   Will use one-class classification (anomaly detection) approach.")
        else:
            console.print(f"\n[green]✓[/green] Sufficient non-favorited photos for binary classification.")
    
    # Analyze photo types
    console.print("\n[bold]Photo Types:[/bold]")
    
    screenshots = len([p for p in all_photos if p.screenshot])
    selfies = len([p for p in all_photos if p.selfie])
    live_photos = len([p for p in all_photos if p.live_photo])
    bursts = len([p for p in all_photos if p.burst])
    
    console.print(f"  • Screenshots: {screenshots:,}")
    console.print(f"  • Selfies: {selfies:,}")
    console.print(f"  • Live Photos: {live_photos:,}")
    console.print(f"  • Burst photos: {bursts:,}")
    
    # Albums analysis
    albums = photosdb.album_info
    console.print(f"\n[bold]Albums:[/bold] {len(albums)}")
    
    # Date range
    dates = [p.date for p in all_photos if p.date]
    if dates:
        console.print(f"\n[bold]Date Range:[/bold]")
        console.print(f"  • Oldest photo: {min(dates).strftime('%Y-%m-%d')}")
        console.print(f"  • Newest photo: {max(dates).strftime('%Y-%m-%d')}")
    
    # Return stats for further use
    return {
        'total': len(all_photos),
        'favorites': len(favorites),
        'non_favorites': len(non_favorites),
        'library_path': photosdb.library_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze Apple Photos library')
    parser.add_argument('--cutoff-date', type=str, default='2023-11-18',
                       help='Cutoff date for training data (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    cutoff = datetime.strptime(args.cutoff_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    try:
        stats = analyze_library(cutoff_date=cutoff)
        console.print("\n[bold green]✓ Analysis complete![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        console.print("\nMake sure you have granted Terminal/Python access to Photos.")
        console.print("Go to: System Preferences → Privacy & Security → Photos")
        raise


if __name__ == '__main__':
    main()
