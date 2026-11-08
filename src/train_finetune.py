#!/usr/bin/env python3
"""
Fine-tune the CLIP backbone on your library (optional, Stage 3).

Frozen CLIP features + the MLP head (Train) are the recommended default. This
script is for the extra jump: adapt the backbone itself to your photo domain.

What it does:
  * positives: favorited photos before the cutoff + rescues (rescues weigh more:
    you actively kept them when the scanner questioned them)
  * negatives: BadPhotos/ images and any feedback-bad photos still in the library
  * full fine-tune of CLIP ViT-B/16 with a small preference head (BCE loss,
    per-sample weights, AdamW, early stopping on a 10% validation split)

Confirmed deletions are NOT usable here: their images are gone from Photos by
now. Train uses their saved features as hard negatives instead.

Output: a weights file in .cache/ plus a registry tag, e.g. "finetuned_clip_v1".
Then run Train with --model finetuned_clip_v1: features get re-extracted with
the fine-tuned backbone (old cache is backed up) and the MLP head trains on top.

Time: roughly 10-20 min/epoch on Apple Silicon for ~20-40k images at batch 64.
"""

import argparse
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
import timm
from timm.data import resolve_model_data_config, create_transform

from photo_scanner.feature_extractor import (
    DEFAULT_BACKBONE,
    load_rgb_image,
    register_finetuned_backbone,
    resolve_backbone_spec,
)
from photo_scanner.holdout import bad_key, good_key, is_holdout
from photo_scanner.negative_features import content_hash
from photo_scanner.paths import CACHE_DIR, ensure_data_dirs
from photo_scanner.thumbs import EMBED_MAX_EDGE
from train_model import collect_training_data
from learn_from_feedback import normalize_uuid

console = Console()
ensure_data_dirs()

DEFAULT_TAG = "finetuned_clip_v1"


class PreferenceDataset(Dataset):
    """(image_tensor, label, weight) triples for the preference fine-tune."""

    def __init__(self, rows, transform, max_edge=EMBED_MAX_EDGE):
        self.rows = rows  # list of (path, label, weight)
        self.transform = transform
        self.max_edge = max_edge

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, weight = self.rows[idx]
        img = load_rgb_image(path)
        if img is None:
            raise RuntimeError(f"Could not decode training image: {path}")
        if self.max_edge and max(img.size) > self.max_edge:
            img = img.copy()
            img.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)


def build_rows(cutoff: datetime, sample_size: int, rescue_weight: float = 2.0,
               bad_weight: float = 1.0, exclude_holdout: bool = True) -> list[tuple]:
    """Collect (path, label, weight) rows: favorites+rescues vs BadPhotos/feedback-bad.

    With exclude_holdout, photos in the deterministic evaluation holdout are
    dropped so the backbone cannot learn them.
    """
    data = collect_training_data(cutoff, sample_size)
    rescued = data["rescued_uuids"]
    rows = []
    skipped = 0
    for uuid, path, _date in data["good_paths"]:
        if exclude_holdout and is_holdout(good_key(uuid, str(path))):
            skipped += 1
            continue
        weight = rescue_weight if normalize_uuid(uuid) in rescued else 1.0
        rows.append((str(path), 1, weight))
    for path in data["bad_paths"]:
        if exclude_holdout and is_holdout(bad_key(content_hash(path))):
            skipped += 1
            continue
        rows.append((str(path), 0, bad_weight))
    if skipped:
        console.print(f"[dim]Held out {skipped:,} photos from fine-tuning[/dim]")
    return rows


class PreferenceModel(nn.Module):
    """CLIP backbone + small MLP preference head (binary)."""

    def __init__(self, backbone_name: str, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        dim = 768  # ViT-B/16 pooled dim; robust enough for the default backbone
        with torch.no_grad():
            dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (loss, AUC) on a validation loader."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    losses, scores, labels = [], [], []
    with torch.no_grad():
        for images, y, _w in loader:
            images = images.to(device)
            y = y.to(device).float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("mps", "cuda")):
                logits = model(images)
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits.float(), y).item())
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy())
            labels.extend(y.cpu().numpy())
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return float(np.mean(losses)), auc


def fine_tune(args):
    cutoff = datetime.strptime(args.cutoff_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    console.print(f"\n[bold blue]🎛 PhotoScanner - Fine-tune backbone[/bold blue]\n")
    console.print(f"Device: {device} · Base backbone: {args.backbone}")

    # --- rows ---
    rows = build_rows(cutoff, args.sample_size, args.rescue_weight, args.bad_weight,
                      exclude_holdout=not args.include_holdout)
    if not rows:
        console.print("[red]No training images found.[/red]")
        return
    n_pos = sum(1 for r in rows if r[1] == 1)
    n_neg = len(rows) - n_pos
    console.print(f"[green]✓[/green] {n_pos:,} positives · [red]{n_neg:,} negatives")

    # --- model + transforms ---
    base_name, _ = resolve_backbone_spec(args.backbone)
    model = PreferenceModel(base_name).to(device)
    data_config = resolve_model_data_config(model.backbone)
    train_tf = create_transform(**data_config, is_training=True)
    val_tf = create_transform(**data_config, is_training=False)

    # stratified 90/10 split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    n_val = max(1, int(0.1 * len(rows)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_ds = PreferenceDataset(train_rows, train_tf)
    val_ds = PreferenceDataset(val_rows, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    # --- training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_auc, best_state, patience = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for images, labels, weights in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device).float()
            weights = weights.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type in ("mps", "cuda")):
                logits = model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), labels, weight=weights
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(images)
            n_seen += len(images)
        scheduler.step()
        val_loss, val_auc = evaluate(model, val_loader, device)
        console.print(
            f"Epoch {epoch}: train_loss={running/max(1,n_seen):.4f} "
            f"val_loss={val_loss:.4f} val_AUC={val_auc:.4f} "
            f"({time.time()-t0:.0f}s)"
        )
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, patience = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                console.print("[yellow]Early stopping (no val-AUC improvement)[/yellow]")
                break

    if best_state is None:
        console.print("[red]Training produced no usable checkpoint.[/red]")
        return

    # --- save backbone weights + registry tag ---
    weights_path = CACHE_DIR / f"{args.tag}.pth"
    if weights_path.exists() and not args.overwrite:
        console.print(f"[red]{weights_path.name} exists — pass --overwrite to replace it.[/red]")
        return
    torch.save(best_state, weights_path)
    register_finetuned_backbone(args.tag, weights_path, base_name)
    console.print(f"\n[bold green]✓ Fine-tuned backbone saved ({args.tag}, val AUC {best_auc:.3f})[/bold green]")
    console.print(f"  weights: {weights_path}")
    console.print(
        f"\nNext: [cyan]python src/train_model.py --model {args.tag}[/cyan] "
        f"— re-extracts features with this backbone and trains the MLP head.\n"
        f"[dim]Compare train_metrics.json AUC against the frozen-backbone run; "
        f"if it does not improve, retrain with the default backbone.[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the CLIP backbone on your preference data")
    parser.add_argument("--cutoff-date", type=str, default="2023-11-18",
                        help="Favorites before this date are positives (YYYY-MM-DD)")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        help=f"timm base backbone (default: {DEFAULT_BACKBONE})")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG,
                        help=f"Registry tag for the fine-tuned backbone (default: {DEFAULT_TAG})")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="AdamW learning rate for the whole backbone (heads learn implicitly)")
    parser.add_argument("--patience", type=int, default=1,
                        help="Early-stop patience in epochs")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Cap training samples for a quick smoke test")
    parser.add_argument("--rescue-weight", type=float, default=2.0,
                        help="Loss weight for rescued photos (active keeps)")
    parser.add_argument("--bad-weight", type=float, default=1.0,
                        help="Loss weight for BadPhotos negatives")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing weights file for --tag")
    parser.add_argument("--include-holdout", action="store_true",
                        help="Train on the evaluation holdout too (disables clean evaluation)")
    args = parser.parse_args()
    fine_tune(args)


if __name__ == "__main__":
    main()
