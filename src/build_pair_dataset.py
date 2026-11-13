#!/usr/bin/env python3
"""
Freeze the pairwise preference dataset from phone-review markings.

A photo marked for delete loses to an unmarked photo from the same save
session. Only photos you saw together are compared, which keeps the label
meaningful: the marked one was on screen next to the unmarked one.

Feature vectors come from .cache/flagged_features.pkl (written by the phone
export). Deleted photos leave Photos, so this store is the only place their
embeddings survive.

Groups are save sessions, never windows alone: the split must not put the same
judgement on both sides of the train/test line.

    python src/build_pair_dataset.py --dry-run
    python src/build_pair_dataset.py
    python src/build_pair_dataset.py --test-groups 1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console

from learn_from_feedback import normalize_uuid
from photo_scanner.feedback_features import flagged_for_backbone, store_key
from photo_scanner.feature_extractor import DEFAULT_BACKBONE, saved_backbone_from_model
from photo_scanner.paths import CACHE_DIR
from photo_scanner.thumbs import EMBED_PREPROCESS

console = Console()

PAIR_DATASET_FILE = CACHE_DIR / "pair_dataset.npz"
PAIR_DATASET_META = CACHE_DIR / "pair_dataset.json"
DATASET_VERSION = 1


def load_marked_groups() -> dict[str, set[str]]:
    """Save session name -> marked-for-delete UUIDs, from feedback history."""
    history_path = CACHE_DIR / "feedback_history.json"
    if not history_path.is_file():
        return {}
    try:
        history = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    groups: dict[str, set[str]] = defaultdict(set)
    for uuid, record in (history.get("added_to_album") or {}).items():
        scan_file = str((record or {}).get("scan_file") or "")
        if not scan_file:
            continue
        group = scan_file.split("/")[-1]
        key = normalize_uuid(uuid)
        if key:
            groups[group].add(key)
    return dict(groups)


def build_pairs(
    marked: dict[str, set[str]],
    features: dict[str, dict],
    max_winners_per_loser: int = 40,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(differences, labels, window ids) for every usable within-window pair.

    A marked photo loses to an unmarked photo captured in the same review
    window. Photos marked under any save session are never winners, so a
    deletion cannot be counted as the one you kept.

    The group id is the review WINDOW, not the save session: two sessions in
    one window compare against the same unmarked photos, so their pairs are
    not independent and must never straddle a train/test split.

    Each pair contributes two rows: loser-minus-winner labelled 0 and its
    negation labelled 1, so a linear head learns a symmetric preference.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    group_ids: list[int] = []

    window_by_uuid = {
        uuid: int(record["window"])
        for uuid, record in features.items()
        if record.get("window") is not None
    }
    all_marked = set().union(*marked.values()) if marked else set()
    unmarked_by_window: dict[int, list[str]] = defaultdict(list)
    for uuid, window in window_by_uuid.items():
        if uuid not in all_marked:
            unmarked_by_window[window].append(uuid)

    for window in sorted(unmarked_by_window):
        losers = sorted(
            uuid
            for uuid in all_marked
            if window_by_uuid.get(uuid) == window
        )
        winners_pool = unmarked_by_window[window]
        if not losers or not winners_pool:
            continue
        for loser in losers:
            if len(winners_pool) > max_winners_per_loser:
                chosen = rng.choice(
                    winners_pool, size=max_winners_per_loser, replace=False
                )
            else:
                chosen = winners_pool
            loser_vec = np.asarray(features[loser]["feature"], dtype=np.float32)
            for winner in chosen:
                winner_vec = np.asarray(features[winner]["feature"], dtype=np.float32)
                diff = loser_vec - winner_vec
                rows.append(diff)
                labels.append(0)
                group_ids.append(window)
                rows.append(-diff)
                labels.append(1)
                group_ids.append(window)

    if not rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
    )


def split_groups(
    group_ids: np.ndarray,
    test_groups: int,
    seed: int = 42,
    holdout: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, test), split by group so no pair leaks across.

    `holdout` picks one specific group (a review window) as the test set.
    Otherwise `test_groups` groups are chosen at random.
    """
    unique = np.unique(group_ids)
    if len(unique) < 2:
        return np.ones(len(group_ids), dtype=bool), np.zeros(len(group_ids), dtype=bool)
    if holdout is not None:
        if holdout not in set(unique.tolist()):
            raise ValueError(f"window {holdout} has no pairs to hold out")
        test_mask = group_ids == holdout
        return ~test_mask, test_mask
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    test_count = max(1, min(int(test_groups), len(unique) - 1))
    test_set = set(shuffled[:test_count].tolist())
    test_mask = np.isin(group_ids, list(test_set))
    return ~test_mask, test_mask


def build_dataset(
    *,
    model: str | None = None,
    holdout_window: int | None = None,
    test_groups: int = 1,
    seed: int = 42,
    max_winners: int = 40,
    min_pairs: int = 20,
    allow_single_window: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict | None:
    """Build and write the frozen pair dataset.

    Returns the metadata dict, or None when there is nothing usable.
    Raises ValueError when the data cannot support a split.
    """
    def say(*args, **kwargs):
        if not quiet:
            console.print(*args, **kwargs)

    say("\n[bold blue]PhotoScanner — Build Pair Dataset[/bold blue]\n")

    backbone = model or saved_backbone_from_model() or DEFAULT_BACKBONE
    features = flagged_for_backbone(backbone, EMBED_PREPROCESS)
    say(f"Store: [cyan]{store_key(backbone, EMBED_PREPROCESS)}[/cyan]")
    say(f"Photos with stored features: {len(features)}")
    with_window = sum(1 for r in features.values() if r.get("window") is not None)
    say(f"  tagged with a review window: {with_window}")

    marked = load_marked_groups()
    total_marked = len(set().union(*marked.values())) if marked else 0
    say(f"Marked photos in history: {total_marked:,} across {len(marked)} saves")

    diffs, labels, group_ids = build_pairs(
        marked, features, max_winners_per_loser=max_winners, seed=seed
    )
    if len(diffs) == 0:
        say("\n[yellow]No usable pairs.[/yellow]")
        say(
            "Pairs need two things: photos tagged with a review window, and at "
            "least one marked photo and one unmarked photo in that window."
        )
        say(
            "[dim]Run the phone export (or --backfill-features) on a reviewed "
            "window first, so features exist for both sides.[/dim]"
        )
        return None

    train_mask, test_mask = split_groups(
        group_ids, test_groups, seed, holdout=holdout_window
    )
    windows = sorted(int(w) for w in np.unique(group_ids))

    say("\n[bold]Dataset[/bold]")
    say(f"  pairs (2 rows each): {len(diffs) // 2:,}")
    say(f"  review windows: {len(windows)} {windows}")
    say(f"  train rows: {int(train_mask.sum()):,}")
    say(f"  test rows: {int(test_mask.sum()):,}")
    say(f"  feature dim: {diffs.shape[1]}")

    per_window = defaultdict(int)
    for gid in group_ids.tolist():
        per_window[gid] += 1
    say(f"  rows per window: { {w: per_window[w] for w in windows} }")

    if len(diffs) // 2 < min_pairs:
        say(
            f"\n[yellow]Only {len(diffs)//2} pairs; below {min_pairs}. "
            "Results will be noisy.[/yellow]"
        )

    if dry_run:
        say("\n[dim]--dry-run: nothing written[/dim]")
        return None

    if len(windows) < 2 and not allow_single_window:
        raise ValueError(
            "Only one review window has stored features; a train/test split "
            "needs two. Review the next window and rebuild."
        )

    np.savez_compressed(
        PAIR_DATASET_FILE,
        diffs=diffs,
        labels=labels,
        groups=group_ids,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    meta = {
        "version": DATASET_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "preprocess": EMBED_PREPROCESS,
        "n_pairs": int(len(diffs) // 2),
        "n_rows": int(len(diffs)),
        "windows": windows,
        "test_windows": sorted(
            set(windows) - set(int(w) for w in np.unique(group_ids[train_mask]))
        ),
        "test_groups": test_groups,
        "split_unit": "review_window",
        "max_winners_per_loser": max_winners,
        "seed": seed,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "feature_dim": int(diffs.shape[1]),
        "groups": sorted(marked),
    }
    PAIR_DATASET_META.write_text(json.dumps(meta, indent=2))

    say(f"\n[green]✓[/green] Wrote [cyan]{PAIR_DATASET_FILE}[/cyan]")
    say(f"[green]✓[/green] Wrote [cyan]{PAIR_DATASET_META}[/cyan]")
    say("[dim]Frozen for comparison: rebuild only when you mean to change the exam.[/dim]")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen pairwise dataset from phone-review marks"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing")
    parser.add_argument(
        "--test-groups",
        type=int,
        default=1,
        help="Number of save sessions held out entirely for the locked test (default 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Group split seed")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Backbone tag whose features to use (default: the trained model's)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Warn when the dataset has fewer pairs than this (default 20)",
    )
    parser.add_argument(
        "--max-winners",
        type=int,
        default=40,
        help="Cap unmarked photos sampled per marked photo (default 40)",
    )
    parser.add_argument(
        "--allow-single-window",
        action="store_true",
        help="Write the dataset even with one window (nothing can be held out)",
    )
    parser.add_argument(
        "--holdout-window",
        type=int,
        default=None,
        help="Hold out this review window for testing instead of a random one",
    )
    args = parser.parse_args()

    try:
        meta = build_dataset(
            model=args.model,
            holdout_window=args.holdout_window,
            test_groups=args.test_groups,
            seed=args.seed,
            max_winners=args.max_winners,
            min_pairs=args.min_pairs,
            allow_single_window=args.allow_single_window,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise SystemExit(1)
    if meta is None:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
