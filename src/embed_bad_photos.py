#!/usr/bin/env python3
"""
Embed every BadPhotos/ image into the negative store so the folder can be emptied.

Train already reads negatives from .cache/negative_features.pkl, not the folder,
so a stored vector survives the file's deletion. This fills the gaps first:
files added since the last Train run, and anything that never got embedded.

Run with --dry-run to see what is missing. The folder is never modified; delete
it yourself once coverage is complete.

    python src/embed_bad_photos.py --dry-run
    python src/embed_bad_photos.py
    python src/embed_bad_photos.py --backbone vit_base_patch16_clip_224.openai
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from photo_scanner.feature_extractor import (
    DEFAULT_BACKBONE,
    FeatureExtractor,
    saved_backbone_from_model,
)
from photo_scanner.negative_features import (
    persisted_negatives,
    record_negative_batch,
    store_key,
)
from photo_scanner.paths import BAD_PHOTOS_DIR
from photo_scanner.thumbs import EMBED_MAX_EDGE, EMBED_PREPROCESS

console = Console()


def image_files(root: Path) -> list[Path]:
    """Every image in BadPhotos/, including subdirectories, case-insensitive."""
    if not root.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )


def missing_files(root: Path, paths: dict[str, str]) -> list[Path]:
    """Images with no stored vector for this backbone."""
    covered = set(paths)
    return [path for path in image_files(root) if str(path.resolve()) not in covered]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed all BadPhotos images into the negative feature store"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone to embed with (default: the trained model's)",
    )
    parser.add_argument("--dir", type=str, default=str(BAD_PHOTOS_DIR), help="Folder to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="Report coverage, write nothing")
    args = parser.parse_args()

    console.print("\n[bold blue]PhotoScanner — Embed BadPhotos[/bold blue]\n")

    root = Path(args.dir).expanduser().resolve()
    backbone = args.backbone or saved_backbone_from_model() or DEFAULT_BACKBONE
    console.print(f"Folder: [cyan]{root}[/cyan]")
    console.print(f"Backbone: [cyan]{backbone}[/cyan]")
    console.print(f"Store key: [cyan]{store_key(backbone, EMBED_PREPROCESS)}[/cyan]")
    if not root.is_dir():
        console.print("[red]Folder not found.[/red]")
        raise SystemExit(1)

    files = image_files(root)
    paths, features = persisted_negatives(backbone, EMBED_PREPROCESS)
    missing = missing_files(root, paths)

    console.print(f"\nFiles on disk: {len(files):,}")
    console.print(f"Already stored: {len(files) - len(missing):,}")
    console.print(f"Missing: {len(missing):,}")
    console.print(f"Unique vectors in store: {len(features):,}")

    if not missing:
        console.print(
            "\n[green]✓[/green] Every file has a stored vector. "
            "The folder can be deleted safely."
        )
        return

    if args.dry_run:
        console.print("\n[yellow]Dry run — nothing written.[/yellow]")
        for path in missing[:10]:
            console.print(f"  {path.name}")
        if len(missing) > 10:
            console.print(f"  ... and {len(missing) - 10} more")
        return

    console.print(f"\nEmbedding {len(missing):,} images with {backbone}...")
    extractor = FeatureExtractor(model_name=backbone, max_edge=EMBED_MAX_EDGE)
    written = 0
    step = max(args.batch_size, 1)
    for index in range(0, len(missing), step):
        batch = missing[index : index + step]
        batch_features = extractor.extract_batch(
            [str(path) for path in batch], batch_size=step, show_progress=False
        )
        entries = [(path, feature) for path, feature in batch_features.items()]
        written += record_negative_batch(entries, backbone, EMBED_PREPROCESS)
        console.print(f"[dim]{min(index + step, len(missing)):,}/{len(missing):,}[/dim]")

    paths, features = persisted_negatives(backbone, EMBED_PREPROCESS)
    still_missing = missing_files(root, paths)
    console.print(
        f"\n[green]✓[/green] Wrote {written:,} entries; unique vectors now {len(features):,}"
    )
    if still_missing:
        console.print(
            f"[yellow]{len(still_missing)} files still lack a vector "
            "(unreadable or removed mid-run). Do not delete the folder yet.[/yellow]"
        )
        for path in still_missing[:10]:
            console.print(f"  {path.name}")
        raise SystemExit(1)
    console.print(
        "[green]✓[/green] Coverage complete for this backbone. "
        "The folder can be deleted."
    )
    console.print(
        "[dim]Note: changing backbone needs the images again, because vectors "
        "are namespaced per backbone.[/dim]"
    )


if __name__ == "__main__":
    main()
