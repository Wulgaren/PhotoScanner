# PhotoScanner scripts

Day-to-day: run `./photoscanner.sh` from the repo root. This file is the command list if you want to call a script yourself.

Run from the **repo root**, with the venv active (or `venv/bin/python`). Data stays at the root: `.cache/`, `output/`, `BadPhotos/`, `config.json`.

```bash
source venv/bin/activate
```

## Train

Favorited photos before the cutoff are positive examples. Optional `BadPhotos/` images (repo root) are negatives. Feedback matters too: photos you **deleted** after Scan flagged them become hard negatives (weight ×3 by default — they describe the exact decision boundary the scanner ranks), and photos you **rescued** stay positives. Scan persists a 768-float feature vector per flagged photo (`.cache/flagged_features.pkl`) because a deleted photo's image is gone from Photos by Train time.

Each run prints cross-validated ROC-AUC / avg-precision (5 folds) and writes `.cache/train_metrics.json`, so runs are comparable. At the end it derives the Scan flag threshold from your rescue/delete decisions (best F1: catch deletions, keep rescues).

```bash
python src/train_model.py --cutoff-date 2023-11-18
python src/train_model.py --cutoff-date 2023-11-18 --sample-size 200 --batch-size 32
python src/train_model.py --cutoff-date 2023-11-18 --model vit_base_patch16_clip_224.openai
python src/train_model.py --hard-neg-weight 5 --cv-folds 3
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--cutoff-date` | `2023-11-18` | Favorites before this date (YYYY-MM-DD) |
| `--sample-size` | none | Cap training samples (testing) |
| `--batch-size` | `32` | Feature-extraction batch size (try `64` on M3 Pro if memory allows) |
| `--model` | `vit_base_patch16_clip_224.openai` | timm backbone or a fine-tuned tag (see Fine-tune below) |
| `--hard-neg-weight` | `3` | Sample weight for confirmed deletions (hard negatives) |
| `--cv-folds` | `5` | Folds for cross-validated evaluation |

The preference head is an MLP (512→128) — it beat the previous RandomForest by ~0.13 AUC on the same data. Changing `--model` clears the feature cache (the old one is copied to `feature_cache.pkl.bak_<timestamp>`) and requires a full re-extract + retrain. The previous model is likewise backed up before each save.

**iCloud Optimize / no originals on disk:** Train still prefers local originals when present. If a good photo’s file is missing, it looks up the UUID in `output/scan_results_*.json` / `output/review_session.json`, then uses a real thumb from `.cache/review_thumbs/` (`sha1(path|uuid|900).jpg`). Placeholder `missing_*.jpg` thumbs are excluded. BadPhotos and other full-res negatives are downscaled to max edge 900 before CLIP so they match thumb sharpness. Feature cache key includes `preprocess=max_edge_900` — old caches are cleared automatically.

## Bootstrap flagged features (one-off, recommended once)

Deleted feedback photos normally can’t be recovered once Photos removes them — but their **review thumbs** usually still exist on disk. This extracts those features now, so the very next Train already has hard negatives instead of waiting a full Scan → Review → Learn cycle. Idempotent.

```bash
python src/bootstrap_flagged_features.py --dry-run
python src/bootstrap_flagged_features.py
```

## Scan

Scores photos after a date. Needs a trained model (`.cache/aesthetic_model.pkl`).

```bash
python src/scan_photos.py --after 2023-11-18
python src/scan_photos.py --after 2023-11-18 --threshold 0.5 --batch-size 32 --limit 50
python src/scan_photos.py --model finetuned_clip_v1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--after` | `2023-11-18` | Scan photos after this date |
| `--threshold` | **auto** | Flag photos below this score. Default: the threshold Train calibrated from your rescue/delete feedback; falls back to 0.3 when none exists yet. |
| `--batch-size` | `32` | Feature-extraction batch size |
| `--limit` | none | Cap how many photos to scan (for testing) |
| `--model` | `vit_base_patch16_clip_224.openai` | Must match the backbone used for Train |

Scan saves the features of everything it flags to `.cache/flagged_features.pkl` — that's what turns your later rescues/deletes into training signal.

**iCloud Optimize:** Scan includes favorites even when Photos marks them `ismissing`. Local originals/edited win when present; otherwise a real review thumb is scored (same helpers as Train). JSON `path` keeps the library path when known so Review can reopen the same thumb key. pHash runs on the readable file (thumb OK; failure → no phash, series still uses embeddings/time).

Writes `output/scan_results_*.json` and a text suggestion list.

## Fine-tune the backbone (advanced, optional)

Frozen CLIP + the MLP head is the recommended default. For the extra jump, this adapts the CLIP backbone itself to your library (positives: favorites, rescues weighed higher; negatives: BadPhotos). Full fine-tune, AdamW, early stopping on a 10% validation split, MPS/CUDA autocast.

```bash
python src/train_finetune.py --cutoff-date 2023-11-18 --epochs 3
python src/train_finetune.py --sample-size 200 --epochs 1        # quick test
python src/train_model.py --model finetuned_clip_v1              # then retrain on top
```

Roughly 10–30 min/epoch on Apple Silicon for 20–40k images. The script registers a backbone **tag** (default `finetuned_clip_v1`) in `.cache/finetuned_backbones.json`; Train/Scan accept that tag via `--model`. Afterwards compare the new AUC in `.cache/train_metrics.json` — if it doesn't beat the frozen backbone, retrain with the default. (Confirmed deletions have no images left, so they're only usable by Train's head, not here.)

## Review (browser)

```bash
python src/review_gui.py
python src/review_gui.py --mode flat --threshold 0.5 --page-size 3
python src/review_gui.py --mode grouped --threshold 0.5
python src/review_gui.py --results output/scan_results_YYYYMMDD_HHMMSS.json --port 8765 --no-open
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--results` | latest `output/scan_results_*.json` | Scan JSON |
| `--mode` | `flat` | `flat` (paginated list) or `grouped` (one series) |
| `--threshold` | **auto** | Starting score threshold (default: Train's auto-calibrated value, else 0.5) |
| `--page-size` | `3` | Photos per page in flat mode |
| `--port` | `8765` | Local server port |
| `--no-open` | off | Do not auto-open the browser |

Keys: `←` `→` focus · `Space` toggle delete/keep · `1`–`9` jump+toggle · `Enter` commit · `s`/`d` all keep/delete · `u` undo · **Done** finish early.

Progress: `output/review_session.json`. Finish writes `confirmed_delete_*.txt` and can add UUIDs to the Photos “To Delete” album.

## Phone review site (static)

A three-column chronological review grid. The private `interactivephotoreview`
repo holds the app, `data.json`, and 720px thumbnails; Netlify deploys it on
push. Nothing personal enters the public PhotoScanner repo.

Start with the first window, then advance one window at a time:

```bash
python src/export_review_site.py --window-index 1 --push --squash-history
python src/export_review_site.py --next-window --push --squash-history
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--results` | latest scan | Scan JSON |
| `--out` | `../interactivephotoreview` | Site repo |
| `--threshold` | `0.5` | Photos below this are pre-marked for deletion |
| `--window-months` | `1` | Months per fresh window |
| `--window-index` | first run only | Use a specific 1-based window |
| `--next-window` | off | Advance from the window in `data.json` |
| `--push` | off | Commit and push the site repo |
| `--squash-history` | off | Replace the site repo history with one commit |

Window 1 starts at the scan's `after_date` and includes every photo in range,
oldest first. Videos and photos in the Hidden album are excluded (Scan already
skips videos; the exporter checks the Photos DB for both). Tap toggles delete,
hold opens fullscreen, and **Save IDs** copies UUIDs / downloads a `.txt` for the
photos up to the furthest card you have tapped. Selections live in
`localStorage`, so revisiting a window keeps them.

Cross-device sync uses Netlify Blobs. Tap **Local only** in the header, then
**Create sync link**; open that link on the other device once. State is stored
per window and writes are last-write-wins. Reloading resumes at the last toggled
photo. **Export progress** / **Import progress** in the same sheet provide a
manual JSON backup.

`--squash-history` keeps the private repo at roughly one window's size instead
of accumulating every batch in Git history.

## Move to album

Alternative to browser review: dump low scores into Photos. The default cutoff is the scan's own threshold (the value Scan used), or Train's auto-calibrated one for older scans.

```bash
python src/move_to_album.py
python src/move_to_album.py --threshold 0.6
python src/move_to_album.py --threshold 0.6 --dry-run
python src/move_to_album.py --album "To Delete" --results path/to/scan_results.json
```

Add the IDs saved from the phone site:

```bash
python src/move_to_album.py --uuids-file ~/Downloads/to-delete-YYYYMMDD-HHMMSS.txt
python src/move_to_album.py --uuids "UUID1, UUID2" --dry-run
```

The UUID path skips scan loading, adds to “To Delete”, and records feedback.

## Learn from feedback

After you sort the “To Delete” album: rescued photos become positives; leftover/deleted become negatives. Then retrain.

```bash
python src/learn_from_feedback.py
python src/learn_from_feedback.py --album "To Delete"
```

## Add captions

Sets empty descriptions in a Photos smart album from `.cache/username_captions.txt`. Usernames missing from that file are learned from other library photos that already have a short caption (not `Author:` / `Source:`).

```bash
python src/add_captions.py
python src/add_captions.py --dry-run
python src/add_captions.py --album "Non-added photos"
```

`--dry-run` writes nothing: no Photos captions, no map update. A username with disagreeing library captions is left unmapped and printed.

## Find wallpaper

Finds the current desktop wallpaper, or a screenshot of it, in Apple Photos. Uses the same venv (Pillow is already in `requirements.txt`). First live capture needs Screen Recording for the terminal.

```bash
python src/find_wallpaper.py
python src/find_wallpaper.py --no-open
python src/find_wallpaper.py ~/Desktop/Screenshot.png
```

Shell alias (from `~/.zshrc`): `image-search` and `image-search shot.png`.

## Twitter curator

Needs `config.json` at the repo root (copy `config.example.json`). Discord bot + Message Content Intent.

```bash
python src/twitter_curator.py --hours 24 --no-listen
python src/twitter_curator.py --hours 24
python src/twitter_curator.py
python src/twitter_curator.py --threshold 0.8
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--hours` | `0` | Backfill last N hours (`0` = live only) |
| `--no-listen` | off | Exit after backfill |
| `--threshold` | `config.json` `score_threshold` | Curation cutoff |

Images go to `~/Pictures/TwitterImages/` (`all/`, `videos/`, `curated/`, `announcements/`) unless `save_directory` is set.

## Summarize announcements

Uses the Cursor agent CLI (`agent` on PATH, or `PHOTOSCANNER_AGENT`). Requires `agent login`.

```bash
python src/summarize_announcements.py
python src/summarize_announcements.py --path /path/to/announcements.txt
python src/summarize_announcements.py --dry-run
python src/summarize_announcements.py --agent /path/to/agent
```

Also runs when the Twitter bot exits.

## Not on the wizard menu

Library stats:

```bash
python src/analyze_library.py --cutoff-date 2023-11-18
```

Terminal series review (not the browser UI):

```bash
python src/interactive_review.py
python src/interactive_review.py --results path/to/scan_results.json
```

Commands: `v` view all · `v #` view one · `f` Finder · `k #` keep only # · `d #` delete those · `da` delete all in series · `y` confirm · `n` keep all · `s` skip · `q` quit.

Finish a web-review session without the UI:

```bash
python src/export_review_session.py
python src/export_review_session.py --dry-run
python src/export_review_session.py --no-album
```

## Layout

| Path | Role |
|------|------|
| `photoscanner.sh` | Launcher |
| `src/photoscanner.py` | Menu / cycle |
| `src/*.py` | CLI scripts (`train_model.py`, `train_finetune.py`, `bootstrap_flagged_features.py`, …) |
| `src/photo_scanner/` | Shared model / series code |
| `src/review_gui/static/` | Browser review assets |
| `.cache/` | Model, features, flagged features, launcher last-used params, `train_metrics.json` |
| `output/` | Scan results, delete lists, review session |
| `BadPhotos/` | Optional negative training images |
| `config.json` | Discord / Twitter settings (gitignored) |
