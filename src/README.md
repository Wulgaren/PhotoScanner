# PhotoScanner scripts

Day-to-day: run `./photoscanner.sh` from the repo root. This file is the command list if you want to call a script yourself.

Run from the **repo root**, with the venv active (or `venv/bin/python`). Data stays at the root: `.cache/`, `output/`, `BadPhotos/`, `config.json`.

```bash
source venv/bin/activate
```

## Train

Favorited photos before the cutoff are positive examples. Optional `BadPhotos/` images (repo root) are negatives.

```bash
python src/train_model.py --cutoff-date 2023-11-18
python src/train_model.py --cutoff-date 2023-11-18 --sample-size 200 --batch-size 32
python src/train_model.py --cutoff-date 2023-11-18 --model vit_base_patch16_clip_224.openai
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--cutoff-date` | `2023-11-18` | Favorites before this date (YYYY-MM-DD) |
| `--sample-size` | none | Cap training samples (testing) |
| `--batch-size` | `32` | Feature-extraction batch size (try `64` on M3 Pro if memory allows) |
| `--model` | `vit_base_patch16_clip_224.openai` | timm backbone for embeddings (CLIP ViT-B/16) |

Changing `--model` clears the feature cache and requires a full re-extract + retrain.

## Scan

Scores photos after a date. Needs a trained model (`.cache/aesthetic_model.pkl`).

```bash
python src/scan_photos.py --after 2023-11-18 --threshold 0.8
python src/scan_photos.py --after 2023-11-18 --threshold 0.3 --batch-size 32 --limit 50
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--after` | `2023-11-18` | Scan photos after this date |
| `--threshold` | `0.3` | Suggest deletion below this score |
| `--batch-size` | `32` | Feature-extraction batch size |
| `--limit` | none | Cap how many photos to scan |
| `--model` | `vit_base_patch16_clip_224.openai` | Must match the backbone used for Train |

Writes `output/scan_results_*.json` and a text suggestion list.

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
| `--threshold` | `0.5` | Starting score threshold |
| `--page-size` | `3` | Photos per page in flat mode |
| `--port` | `8765` | Local server port |
| `--no-open` | off | Do not auto-open the browser |

Keys: `←` `→` focus · `Space` toggle delete/keep · `1`–`9` jump+toggle · `Enter` commit · `s`/`d` all keep/delete · `u` undo · **Done** finish early.

Progress: `output/review_session.json`. Finish writes `confirmed_delete_*.txt` and can add UUIDs to the Photos “To Delete” album.

## Move to album

Alternative to browser review: dump low scores into Photos.

```bash
python src/move_to_album.py --threshold 0.8
python src/move_to_album.py --threshold 0.8 --dry-run
python src/move_to_album.py --threshold 0.8 --album "To Delete" --results path/to/scan_results.json
```

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

macOS + Apple Intelligence helper. Build once:

```bash
cd src/tools/AnnouncementsSummarizer && swift build -c release
```

```bash
python src/summarize_announcements.py
python src/summarize_announcements.py --path /path/to/announcements.txt
python src/summarize_announcements.py --dry-run
python src/summarize_announcements.py --binary /path/to/announcements-summarizer
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
| `src/*.py` | CLI scripts |
| `src/photo_scanner/` | Shared model / series code |
| `src/review_gui/static/` | Browser review assets |
| `src/tools/AnnouncementsSummarizer/` | Swift summarizer |
| `.cache/` | Model, features, launcher last-used params |
| `output/` | Scan results, delete lists, review session |
| `BadPhotos/` | Optional negative training images |
| `config.json` | Discord / Twitter settings (gitignored) |
