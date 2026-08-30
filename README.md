# PhotoScanner

AI-powered photo curation that learns your taste from your Apple Photos library. Twitter/X image curation uses the same model. Everything runs locally on your Mac.

## Setup (once)

Python **3.10–3.13** (not 3.14+; `osxphotos` cannot use 3.14 yet).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Grant Terminal/Python access to Photos: System Settings → Privacy & Security → Photos.

Optional: put photos you consider bad in `BadPhotos/` so training has negative examples.

Twitter curator: copy `config.example.json` to `config.json` and fill in the Discord token and channel IDs. Full flags are in [`src/README.md`](src/README.md).

## Run

```bash
./photoscanner.sh
```

That uses `venv` and opens a numbered menu:

1. **Guided cycle** — Train → Scan → Review → Learn (skip any step)
2. **Pick a tool** — one job (including move-to-album, captions, Twitter, summarize)

Each job shows the equivalent `python` command, then runs it. Last dates and thresholds are remembered. After a cycle or a single tool, you are back at the shell.

## What it does

- Trains on favorited photos (and `BadPhotos/` / feedback)
- Scores later photos and groups bursts/series
- Browser review for keep/delete (never auto-deletes the best shot in a series)
- Learns from what you rescue vs leave in “To Delete”
- Optional Twitter/X curator via Discord

Scripts, flags, and internals: [`src/README.md`](src/README.md).
